# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 FLASH MoE layer with DSpark expert-parallel dispatch and combine."""


# Sub-kernels freeze EP / n_routed_experts into their shapes at import
# time: read --ep / --experts-per-rank from argv and override config before
# importing them below.
from math import gcd

import dataclasses
import functools
import sys

import config

_EP_CHOICES = (2, 4, 8, 16)
_EP_DEFAULT = 2
# Fixed reference for the default density -- not config.EP, which a caller
# (e.g. dspark_drafter.py) may already have overwritten before importing this.
_CHECKPOINT_N_ROUTED_EXPERTS = config.FLASH.n_routed_experts


def _parse_int_argv(name, default):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(name + "="):
            return int(tok.split("=", 1)[1])
    return default


EP = _parse_int_argv("--ep", _EP_DEFAULT)
# Capped by EP so the default density (32/rank) can't overflow the checkpoint's
# 256 experts past EP8 -- gate.py's score buffers are fixed at that width.
_EXPERTS_PER_RANK_DEFAULT = min(_CHECKPOINT_N_ROUTED_EXPERTS // 8, _CHECKPOINT_N_ROUTED_EXPERTS // EP)
EXPERTS_PER_RANK = _parse_int_argv("--experts-per-rank", _EXPERTS_PER_RANK_DEFAULT)
if EXPERTS_PER_RANK <= 0:
    raise ValueError(f"--experts-per-rank must be a positive integer (got {EXPERTS_PER_RANK})")
config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=EXPERTS_PER_RANK * EP)
config.EP = EP
MOE_TOKEN_CAPACITY = _parse_int_argv("--token-capacity", config.MOE_TOKENS)
TOKEN_CAPACITY_SPECIFIED = any(arg == "--token-capacity" or arg.startswith("--token-capacity=") for arg in sys.argv)
if TOKEN_CAPACITY_SPECIFIED and (MOE_TOKEN_CAPACITY <= 0 or (MOE_TOKEN_CAPACITY > 64 and MOE_TOKEN_CAPACITY % 64 != 0)):
    raise ValueError("--token-capacity must be positive and, above 64, a multiple of 64")
config.MOE_TOKENS = MOE_TOKEN_CAPACITY
_RECV_ALIGNMENT = 16 if MOE_TOKEN_CAPACITY <= 8 else 64
config.RECV_MAX = ((EP * config.MOE_TOKENS + _RECV_ALIGNMENT - 1) // _RECV_ALIGNMENT) * _RECV_ALIGNMENT

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

from config import FLASH as M, MOE_TOKENS, RECV_MAX, INT8_AMAX_EPS, INT8_SCALE_MAX
from expert_routed import RECV_TILE, TILES_PER_EXPERT, expert_routed_tile
from expert_shared import expert_shared
from gate import gate
from hc_post import hc_post
from hc_pre import hc_pre
from prefill_cp_token_allgather import (
    CP_KV_T_DYN as PREFILL_GROUP_T_DYN,
    CP_Q_T_DYN as PREFILL_LOCAL_T_DYN,
    PREFILL_GROUP_CAP,
    TP_SIZE,
    prefill_cp_token_allgather_step,
)


# model config
T = MOE_TOKENS
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size

HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

N_RANKS = EP
N_EXPERTS_GLOBAL = M.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
# TaskAllocator ring heap, scaled from the 256 MiB runtime default per 16 local experts.
MOE_RING_HEAP = (1 << 28) * max(1, N_LOCAL // 16)
N_ROUTES = T * TOPK

# recv_x/recv_aux laid out [expert, source, slot], flattened to
# [N_LOCAL * RECV_MAX, D]. Lane (e, src, slot) flat row = e * RECV_MAX +
# src * MAX_PER_SRC + slot. One source sends <= T rows to a local expert.
MAX_PER_SRC = T
PUSH_GROUPS = max(1, N_LOCAL // N_RANKS) if T <= 8 else N_LOCAL
FP32_PER_CACHE_LINE = 16
INT64_PER_CACHE_LINE = 8
AUX_PAD = 8  # FP32 pack tile width (32 B min tile); cols: 0=scale 1=weight
AUX_SCALE = 0
AUX_W = 1
IDX_PAD = 8  # INT32 route tile width; route rides a separate window from scale/w
             # (an FP32 tile can't hold it: INDEX->FP32 casts are unsupported).

# tiling
PREFILL_INPUT_ID_TILE = 4
COMBINE_TOKEN_TILE = gcd(T, 4)

PREFILL_RECV_TILE = 16
PREFILL_K_TILE = 512
PREFILL_INTER_K_TILE = 512
PREFILL_MM_INTER_TILE = 256
PREFILL_ACT_INTER_TILE = 128
PREFILL_D_OUT_TILE = 256
PREFILL_QUANT_TILE = 512
PREFILL_D_OUT_ACT_TILE = 512
PREFILL_W2_INNER = 1
PREFILL_W2_ACT_INNER = 1
PREFILL_MOE_SCALE_PAD = 16
PREFILL_MOE_EXPERT_SCALE_PAD = 16
PREFILL_MOE_ROUTE_MAP_PAD = 16
PREFILL_MOE_FINALIZE_TOKEN_TILE = gcd(T, 16)
PREFILL_MOE_GROUPED_EXPERT_TILE = 16
PREFILL_MOE_ROUTES_PER_SRC = T * TOPK
PREFILL_MOE_PEER_CAP = PREFILL_MOE_ROUTES_PER_SRC
PREFILL_MOE_TOTAL_CAP = N_RANKS * PREFILL_MOE_PEER_CAP
PREFILL_MOE_GROUPED_ROWS = PREFILL_MOE_TOTAL_CAP + N_LOCAL * (PREFILL_RECV_TILE - 1)
PREFILL_MOE_GROUPED_TOTAL_CAP = (PREFILL_MOE_GROUPED_ROWS + PREFILL_RECV_TILE - 1) // PREFILL_RECV_TILE * PREFILL_RECV_TILE
PREFILL_MOE_SEND_SCALE_PAD = 16
PREFILL_SOURCE_PACK_BLOCKS = N_RANKS
PREFILL_RESORT_BLOCKS_PER_EXPERT = 1
PREFILL_COUNT_TOKEN_TILE = min(T, 128)
PREFILL_COUNT_SHARDS = (T + PREFILL_COUNT_TOKEN_TILE - 1) // PREFILL_COUNT_TOKEN_TILE
PREFILL_DISPATCH_ROW_TILE = 1
PREFILL_REVERSE_ROW_TILE = 1

# Scalar stores into recv_* land a whole 64-byte line at a time, so an expert
# row and a token block must each own whole lines.
assert RECV_MAX % FP32_PER_CACHE_LINE == 0


@pl.jit.inline
def clear_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's MoE signal windows after its final MoE completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear"):
        # The final MoE output depends on this rank observing every peer's final
        # meta, payload, and combine notify. No peer can issue another MoE notify
        # to this rank in the current forward after this dependency is satisfied.
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(arrived, [src, 0], zero)
            pl.write(data_arrived, [src, 0], zero)
            pl.write(combine_arrived, [src, 0], zero)


@pl.jit.inline
def clear_prefill_moe_signals(
    stage_token: pl.Tensor[[1], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Clear retained prefill-MoE epochs after the final wave completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_signal_clear") as cleared_tid:
        # Read the wave-barrier epoch: orders the clear after the final wave.
        # The always-true guard keeps the read as a dependency edge.
        completed_epoch = pl.read(stage_token, [0])
        if completed_epoch >= 0:
            zero = pl.cast(0, pl.INT32)
            for src in pl.range(N_RANKS):
                pl.write(arrived, [src, 0], zero)
                pl.write(data_arrived, [src, 0], zero)
                pl.write(combine_arrived, [src, 0], zero)
                pl.write(stage_done, [src, 0], zero)
    return cleared_tid


# === Dispatch ================================================================
# Exchange route counts, push payload lanes, defer payload waits, and compact rows by expert.
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
                pld.system.notify(target=arrived, peer=dst, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    # A deferred waiter must be a registration-only task. It cannot follow the
    # route counting and publication work in the producer task above.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait") as _meta_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=arrived, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)

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

    # Push source-major lanes with one block per destination and expert group.
    with pl.spmd(N_RANKS * PUSH_GROUPS, name_hint="dispatch_push", allow_early_resolve=True) as push_tid:
        push_block = pl.tile.get_block_idx()
        dst = push_block // PUSH_GROUPS
        push_group = push_block - dst * PUSH_GROUPS
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        for loc_e in pl.range(push_group, N_LOCAL, PUSH_GROUPS):
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
        # destination after its own puts, so a peer sees PUSH_GROUPS notifies per
        # source per epoch and the wait below expects PUSH_GROUPS * moe_epoch. The
        # count must bump only after this block's puts issue in program order:
        # recv_aux / recv_route ride a non-draining remote_store and a PIPE_ALL
        # barrier is not a cross-rank DDR fence (PTOAS#872).
        if dst != my_rank:
            pld.system.notify(target=data_arrived, peer=dst, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[push_tid]) as _wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=data_arrived,
                    offsets=[src, 0],
                    expected=pl.cast(moe_epoch * PUSH_GROUPS, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    if T <= 8:
        # Compact source-major payloads with one block per local expert.
        with pl.spmd(
            N_LOCAL,
            name_hint="dispatch_gather",
            deps=[_wait_tid, _meta_tid],
            allow_early_resolve=False,
        ) as gather_tid:
            e = pl.tile.get_block_idx()
            e_base_row = e * RECV_MAX
            source_begin = pl.cast(0, pl.INDEX)
            for src in pl.range(N_RANKS):
                n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
                src_base_row = e_base_row + src * MAX_PER_SRC
                for slot in pl.range(n):
                    in_row = src_base_row + slot
                    out_col = source_begin + slot
                    out_row = e_base_row + out_col
                    recv_x_out_flat[out_row : out_row + 1, :] = recv_x[in_row : in_row + 1, :]
                    pl.write(recv_scale_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_SCALE]))
                    pl.write(recv_w_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_W]))
                    pl.write(recv_r_route_out, [e, out_col], pl.read(recv_route, [in_row, 0]))
                source_begin = source_begin + n
        gather_done = gather_tid

    else:
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

        _gather_tid = pl.system.task_dummy(deps=[gather_tids[e] for e in range(N_LOCAL)])
        gather_done = _gather_tid

    return gather_done


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
                acc = pl.cast(sh[t:t + 1, :], target_type=pl.FP32)
                for k in pl.range(TOPK):
                    r = t * TOPK + k
                    acc = pl.add(acc, pl.cast(routed_y_buf[r:r + 1, :], target_type=pl.FP32))
                ffn_out[t:t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")
            else:
                ffn_out[t:t + 1, :] = sh[t:t + 1, :]
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
) -> pl.Scalar[pl.TASK_ID]:
    """Run one fixed-capacity gate, expert dispatch, and combine tile."""
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, num_tokens, tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    shared_out = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        shared_out,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32, manual_dep=True)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_meta_local = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
    dispatch_done = dispatch(
        indices, x_norm_i8, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out, recv_meta_local,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        num_tokens, my_rank, moe_epoch,
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
                recv_x_out, recv_scale_out, recv_w_out,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
                recv_y_tile,
                local_e, tile_row, valid_rows, dispatch_done,
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

    scatter_done = pl.system.task_dummy(deps=[expert_completion_tids[local_e] for local_e in range(N_LOCAL)])

    completion_tid = combine(
        shared_out, ffn_out,
        routed_y_buf, combine_arrived, scatter_done,
        num_tokens, my_rank, moe_epoch,
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
            norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            ffn_out,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            layer_id, num_tokens, my_rank, moe_epoch,
        )
        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next


def _make_expert_gate_up_quant_tile(grouped_capacity: int, row_tile: int):
    """Build a gate/up projection, SwiGLU, and INT8 quantization tile function."""
    ROW_TILE = row_tile
    MM_COL_TILE = 128 if row_tile == 192 else PREFILL_MM_INTER_TILE
    VEC_ROW_TILE = PREFILL_RECV_TILE
    ACT_TILE = PREFILL_ACT_INTER_TILE
    H_QUANT_TILE = PREFILL_QUANT_TILE

    @pl.jit.inline(auto_scope=False)
    def expert_gate_up_quant_tile(
        expert_x: pl.Tensor[[grouped_capacity, D], pl.INT8],
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        h_i8: pl.Tensor[[grouped_capacity, MOE_INTER], pl.INT8],
        h_scale_dq: pl.Tensor[[grouped_capacity, 1], pl.FP32],
        tile_row_start: pl.Scalar[pl.INDEX],
        local_expert_id: pl.Scalar[pl.INDEX],
        valid_rows: pl.Scalar[pl.INDEX],
        layout_tid: pl.Scalar[pl.TASK_ID],
        routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
        routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
        routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    ):
        """Compute gate/up projections, apply SwiGLU, and quantize a token tile."""
        weight_e = local_expert_id
        with pl.scope():
            w13_tile_i32 = pl.create_tensor([ROW_TILE, 2 * MOE_INTER], dtype=pl.INT32)

            with pl.spmd((2 * MOE_INTER) // MM_COL_TILE, name_hint="prefill_exp_w13_mm", deps=[layout_tid]) as w13_tid:
                block = pl.tile.get_block_idx()
                n0 = block * MM_COL_TILE
                if n0 < MOE_INTER:
                    w13_acc = pl.create_tensor([1, ROW_TILE, MM_COL_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, PREFILL_K_TILE, stage=2):
                        x_chunk = expert_x[tile_row_start : tile_row_start + ROW_TILE, k0 : k0 + PREFILL_K_TILE]
                        w13_chunk = routed_w1[weight_e : weight_e + 1, n0 : n0 + MM_COL_TILE, k0 : k0 + PREFILL_K_TILE]
                        w13_acc = pl.matmul_acc(w13_acc, x_chunk, w13_chunk, b_trans=True, init_cond=k0 == 0)
                    w13_tile_i32[:, n0 : n0 + MM_COL_TILE] = pl.reshape(w13_acc, [ROW_TILE, MM_COL_TILE])
                else:
                    up_n0 = (block % (MOE_INTER // MM_COL_TILE)) * MM_COL_TILE
                    w13_acc = pl.create_tensor([1, ROW_TILE, MM_COL_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, PREFILL_K_TILE, stage=2):
                        x_chunk = expert_x[tile_row_start : tile_row_start + ROW_TILE, k0 : k0 + PREFILL_K_TILE]
                        w13_chunk = routed_w3[weight_e : weight_e + 1, up_n0 : up_n0 + MM_COL_TILE, k0 : k0 + PREFILL_K_TILE]
                        w13_acc = pl.matmul_acc(w13_acc, x_chunk, w13_chunk, b_trans=True, init_cond=k0 == 0)
                    w13_tile_i32[:, n0 : n0 + MM_COL_TILE] = pl.reshape(w13_acc, [ROW_TILE, MM_COL_TILE])

            h_tile_fp32 = pl.create_tensor([ROW_TILE, MOE_INTER], dtype=pl.FP32)
            h_tile_i8 = h_i8[tile_row_start : tile_row_start + ROW_TILE]
            h_tile_scale_dq = h_scale_dq[tile_row_start : tile_row_start + ROW_TILE]
            with pl.spmd((ROW_TILE // VEC_ROW_TILE), name_hint="prefill_exp_gate_up_quant", deps=[w13_tid]):
                vector_row = pl.tile.get_block_idx() * VEC_ROW_TILE
                vector_flat_row = tile_row_start + vector_row
                vector_valid_rows = pl.min(VEC_ROW_TILE, pl.max(valid_rows - vector_row, 0))
                x_scale_padded = expert_scale[vector_flat_row : vector_flat_row + VEC_ROW_TILE, 0:PREFILL_MOE_EXPERT_SCALE_PAD]
                x_scale_transposed = pl.transpose(x_scale_padded, axis1=0, axis2=1)
                x_scale_tile = pl.reshape(x_scale_transposed[0:1, :], [VEC_ROW_TILE, 1])
                row_amax = pl.full([1, VEC_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for inter0 in pl.pipeline(0, MOE_INTER, ACT_TILE, stage=2):
                    gate_i32 = w13_tile_i32[vector_row : vector_row + VEC_ROW_TILE, inter0 : inter0 + ACT_TILE]
                    up_i32 = w13_tile_i32[vector_row : vector_row + VEC_ROW_TILE, MOE_INTER + inter0 : MOE_INTER + inter0 + ACT_TILE]
                    gate_fp32 = pl.cast(gate_i32, pl.FP32, mode="none")
                    gate_scaled = pl.row_expand_mul(gate_fp32, x_scale_tile)
                    gate_weight_scale = routed_w1_scale[local_expert_id : local_expert_id + 1, inter0 : inter0 + ACT_TILE]
                    gate_fp32 = pl.col_expand_mul(gate_scaled, gate_weight_scale)
                    up_fp32 = pl.cast(up_i32, pl.FP32, mode="none")
                    up_scaled = pl.row_expand_mul(up_fp32, x_scale_tile)
                    up_weight_scale = routed_w3_scale[local_expert_id : local_expert_id + 1, inter0 : inter0 + ACT_TILE]
                    up_fp32 = pl.col_expand_mul(up_scaled, up_weight_scale)
                    if SWIGLU_LIMIT > 0.0:
                        gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                        up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                    activated_chunk = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                    h_valid = pl.set_validshape(activated_chunk, vector_valid_rows, ACT_TILE)
                    h_padded = pl.fillpad(h_valid, pad_value=pl.PadValue.zero)
                    h_tile_fp32[vector_row : vector_row + VEC_ROW_TILE, inter0 : inter0 + ACT_TILE] = h_padded
                    h_abs = pl.maximum(h_padded, pl.neg(h_padded))
                    row_amax = pl.maximum(row_amax, pl.reshape(pl.row_max(h_abs), [1, VEC_ROW_TILE]))
                quant_scale = pl.div(pl.full([1, VEC_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), row_amax)
                h_tile_scale_dq[vector_row : vector_row + VEC_ROW_TILE, :] = pl.reshape(pl.recip(quant_scale), [VEC_ROW_TILE, 1])
                quant_scale_col = pl.reshape(quant_scale, [VEC_ROW_TILE, 1])
                for k0 in pl.pipeline(0, MOE_INTER, H_QUANT_TILE, stage=2):
                    h_quant_chunk = h_tile_fp32[vector_row : vector_row + VEC_ROW_TILE, k0 : k0 + H_QUANT_TILE]
                    h_scaled = pl.row_expand_mul(h_quant_chunk, quant_scale_col)
                    h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                    h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                    h_tile_i8[vector_row : vector_row + VEC_ROW_TILE, k0 : k0 + H_QUANT_TILE] = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")

    return expert_gate_up_quant_tile


def _make_expert_down_proj_tile(grouped_capacity: int, row_tile: int):
    """Build an expert down-projection tile function with routing weights."""
    ROW_TILE = row_tile
    OUTPUT_MM_TILE = 128 if row_tile == 192 else PREFILL_D_OUT_TILE
    VEC_ROW_TILE = PREFILL_RECV_TILE
    OUTPUT_ACT_TILE = PREFILL_D_OUT_ACT_TILE

    @pl.jit.inline(auto_scope=False)
    def expert_down_proj_tile(
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        h_i8: pl.Tensor[[grouped_capacity, MOE_INTER], pl.INT8],
        h_scale_dq: pl.Tensor[[grouped_capacity, 1], pl.FP32],
        tile_row_start: pl.Scalar[pl.INDEX],
        local_expert_id: pl.Scalar[pl.INDEX],
        routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
        routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
        expert_y: pl.Tensor[[grouped_capacity, D], pl.BF16],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Write routing-weighted BF16 down-projection output and return its TaskId."""
        weight_e = local_expert_id
        h_tile_i8 = h_i8[tile_row_start : tile_row_start + ROW_TILE]
        h_tile_scale_dq = h_scale_dq[tile_row_start : tile_row_start + ROW_TILE]

        y_i32 = pl.create_tensor([ROW_TILE, D], dtype=pl.INT32)
        with pl.spmd(D // (PREFILL_W2_INNER * OUTPUT_MM_TILE), name_hint="prefill_exp_w2_mm", allow_early_resolve=True) as w2_tid:
            block = pl.tile.get_block_idx()
            d_base = block * (PREFILL_W2_INNER * OUTPUT_MM_TILE)
            for inner in pl.range(PREFILL_W2_INNER):
                d0 = d_base + inner * OUTPUT_MM_TILE
                y_acc = pl.create_tensor([1, ROW_TILE, OUTPUT_MM_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, MOE_INTER, PREFILL_INTER_K_TILE, stage=2):
                    h_w2_chunk = h_tile_i8[:, k0 : k0 + PREFILL_INTER_K_TILE]
                    w2_chunk = routed_w2[weight_e : weight_e + 1, d0 : d0 + OUTPUT_MM_TILE, k0 : k0 + PREFILL_INTER_K_TILE]
                    y_acc = pl.matmul_acc(y_acc, h_w2_chunk, w2_chunk, b_trans=True, init_cond=k0 == 0)
                y_i32[:, d0 : d0 + OUTPUT_MM_TILE] = pl.reshape(y_acc, [ROW_TILE, OUTPUT_MM_TILE])

        expert_y_tile = expert_y[tile_row_start : tile_row_start + ROW_TILE, :]
        with pl.spmd(
            (ROW_TILE // VEC_ROW_TILE) * (D // (PREFILL_W2_ACT_INNER * OUTPUT_ACT_TILE)),
            name_hint="prefill_exp_w2_act",
            deps=[w2_tid],
            allow_early_resolve=False,
        ) as w2_act_tid:
            grid_block = pl.tile.get_block_idx()
            vector_row = (grid_block // (D // (PREFILL_W2_ACT_INNER * OUTPUT_ACT_TILE))) * VEC_ROW_TILE
            block = grid_block % (D // (PREFILL_W2_ACT_INNER * OUTPUT_ACT_TILE))
            vector_flat_row = tile_row_start + vector_row
            d_base = block * (PREFILL_W2_ACT_INNER * OUTPUT_ACT_TILE)
            scale_padded = expert_scale[vector_flat_row : vector_flat_row + VEC_ROW_TILE, 0:PREFILL_MOE_EXPERT_SCALE_PAD]
            scale_transposed = pl.transpose(scale_padded, axis1=0, axis2=1)
            weight_col = pl.reshape(scale_transposed[1:2, :], [VEC_ROW_TILE, 1])
            row_scale = pl.mul(h_tile_scale_dq[vector_row : vector_row + VEC_ROW_TILE, :], weight_col)
            for inner in pl.pipeline(PREFILL_W2_ACT_INNER, stage=2):
                d0 = d_base + inner * OUTPUT_ACT_TILE
                y_fp32 = pl.cast(y_i32[vector_row : vector_row + VEC_ROW_TILE, d0 : d0 + OUTPUT_ACT_TILE], target_type=pl.FP32, mode="none")
                y_scaled = pl.row_expand_mul(y_fp32, row_scale)
                output_weight_scale = routed_w2_scale[local_expert_id : local_expert_id + 1, d0 : d0 + OUTPUT_ACT_TILE]
                y_fp32 = pl.col_expand_mul(y_scaled, output_weight_scale)
                y_bf16 = pl.cast(y_fp32, pl.BF16, mode="rint")
                expert_y_tile[vector_row : vector_row + VEC_ROW_TILE, d0 : d0 + OUTPUT_ACT_TILE] = y_bf16
        return w2_act_tid

    return expert_down_proj_tile


def make_prefill_expert_grouped(grouped_capacity: int):
    """Specialize count-driven expert compute for an aligned scratch capacity."""
    if grouped_capacity <= 0 or grouped_capacity % PREFILL_RECV_TILE:
        raise ValueError("grouped expert capacity must be a positive multiple of RECV_TILE")
    grouped_tile_capacity = grouped_capacity // PREFILL_RECV_TILE
    gate_up_quant_m192 = _make_expert_gate_up_quant_tile(grouped_capacity, 192)
    gate_up_quant_m128 = _make_expert_gate_up_quant_tile(grouped_capacity, 128)
    gate_up_quant_m96 = _make_expert_gate_up_quant_tile(grouped_capacity, 96)
    gate_up_quant_m64 = _make_expert_gate_up_quant_tile(grouped_capacity, 64)
    gate_up_quant_m32 = _make_expert_gate_up_quant_tile(grouped_capacity, 32)
    gate_up_quant_m16 = _make_expert_gate_up_quant_tile(grouped_capacity, PREFILL_RECV_TILE)
    down_proj_m192 = _make_expert_down_proj_tile(grouped_capacity, 192)
    down_proj_m128 = _make_expert_down_proj_tile(grouped_capacity, 128)
    down_proj_m96 = _make_expert_down_proj_tile(grouped_capacity, 96)
    down_proj_m64 = _make_expert_down_proj_tile(grouped_capacity, 64)
    down_proj_m32 = _make_expert_down_proj_tile(grouped_capacity, 32)
    down_proj_m16 = _make_expert_down_proj_tile(grouped_capacity, PREFILL_RECV_TILE)

    @pl.jit.inline(auto_scope=False)
    def prefill_expert_grouped(
        expert_x: pl.Tensor[[grouped_capacity, D], pl.INT8],
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
        routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
        routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
        routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
        routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
        expert_y: pl.Tensor[[grouped_capacity, D], pl.BF16],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Compute aligned expert slabs with count-driven row tiles and weighted BF16 outputs."""
        expert_bases = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
        h_i8 = pl.create_tensor([grouped_capacity, MOE_INTER], dtype=pl.INT8)
        h_scale_dq = pl.create_tensor([grouped_capacity, 1], dtype=pl.FP32, manual_dep=True)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_exp_group_layout", allow_early_resolve=True) as layout_tid:
            grouped_base = pl.cast(0, pl.INDEX)
            for local_expert_id in pl.range(N_LOCAL):
                pl.write(expert_bases, [local_expert_id, 0], pl.cast(grouped_base, pl.INT32))
                rows = pl.cast(pl.read(expert_counts, [local_expert_id, 0]), pl.INDEX)
                aligned_rows = ((rows + PREFILL_RECV_TILE - 1) // PREFILL_RECV_TILE) * PREFILL_RECV_TILE
                grouped_base = grouped_base + aligned_rows

        expert_completion_tids = pl.array.create(N_LOCAL, pl.TASK_ID)

        with pl.scope():
            for local_expert_id in pl.parallel(N_LOCAL):
                flat_base = pl.cast(pl.read(expert_bases, [local_expert_id, 0]), pl.INDEX)
                n_rows = pl.read(expert_counts, [local_expert_id, 0])
                tile192 = pl.cast(((n_rows + PREFILL_RECV_TILE - 1) // PREFILL_RECV_TILE) * PREFILL_RECV_TILE == 192, pl.INDEX)
                main_tiles = (n_rows // 128) * (1 - tile192)
                aligned_tail = (((n_rows % 128) + PREFILL_RECV_TILE - 1) // PREFILL_RECV_TILE) * PREFILL_RECV_TILE * (1 - tile192)
                tail96 = pl.cast(aligned_tail == 96, pl.INDEX)
                remaining_tail = aligned_tail - tail96 * 96
                tail64 = remaining_tail // 64
                tail32 = (remaining_tail % 64) // 32
                tail16 = (remaining_tail % 32) // PREFILL_RECV_TILE
                for tile in pl.parallel(tile192):
                    gate_up_quant_m192(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        flat_base, local_expert_id, pl.cast(n_rows, pl.INDEX), layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(main_tiles):
                    tile_row_start = flat_base + tile * 128
                    gate_up_quant_m128(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, pl.cast(128, pl.INDEX), layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail96):
                    tile_row = main_tiles * 128 + tile * 96
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(96, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m96(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail64):
                    tile_row = main_tiles * 128 + tail96 * 96 + tile * 64
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(64, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m64(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail32):
                    tile_row = main_tiles * 128 + tail96 * 96 + tail64 * 64 + tile * 32
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(32, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m32(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail16):
                    tile_row = main_tiles * 128 + tail96 * 96 + tail64 * 64 + tail32 * 32 + tile * 16
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(16, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m16(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )

            for local_expert_id in pl.parallel(N_LOCAL):
                w2_act_tids = pl.array.create(grouped_tile_capacity, pl.TASK_ID)
                flat_base = pl.cast(pl.read(expert_bases, [local_expert_id, 0]), pl.INDEX)
                n_rows = pl.read(expert_counts, [local_expert_id, 0])
                tile192 = pl.cast(((n_rows + PREFILL_RECV_TILE - 1) // PREFILL_RECV_TILE) * PREFILL_RECV_TILE == 192, pl.INDEX)
                main_tiles = (n_rows // 128) * (1 - tile192)
                aligned_tail = (((n_rows % 128) + PREFILL_RECV_TILE - 1) // PREFILL_RECV_TILE) * PREFILL_RECV_TILE * (1 - tile192)
                tail96 = pl.cast(aligned_tail == 96, pl.INDEX)
                remaining_tail = aligned_tail - tail96 * 96
                tail64 = remaining_tail // 64
                tail32 = (remaining_tail % 64) // 32
                tail16 = (remaining_tail % 32) // PREFILL_RECV_TILE
                for tile in pl.parallel(tile192):
                    w2_act_tid = down_proj_m192(
                        expert_scale, h_i8, h_scale_dq, flat_base, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y,
                    )
                    w2_act_tids[tile] = w2_act_tid
                for tile in pl.parallel(main_tiles):
                    tile_row_start = flat_base + tile * 128
                    w2_act_tid = down_proj_m128(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y,
                    )
                    w2_act_tids[tile192 + tile] = w2_act_tid
                for tile in pl.parallel(tail96):
                    tile_row_start = flat_base + main_tiles * 128 + tile * 96
                    w2_act_tid = down_proj_m96(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[tile192 + main_tiles + tile] = w2_act_tid
                for tile in pl.parallel(tail64):
                    tile_row_start = flat_base + main_tiles * 128 + tail96 * 96 + tile * 64
                    w2_act_tid = down_proj_m64(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[tile192 + main_tiles + tail96 + tile] = w2_act_tid
                for tile in pl.parallel(tail32):
                    tile_row_start = flat_base + main_tiles * 128 + tail96 * 96 + tail64 * 64 + tile * 32
                    w2_act_tid = down_proj_m32(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[tile192 + main_tiles + tail96 + tail64 + tile] = w2_act_tid
                for tile in pl.parallel(tail16):
                    tile_row_start = flat_base + main_tiles * 128 + tail96 * 96 + tail64 * 64 + tail32 * 32 + tile * 16
                    w2_act_tid = down_proj_m16(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[tile192 + main_tiles + tail96 + tail64 + tail32 + tile] = w2_act_tid

                expert_completion_tid = pl.system.task_dummy(deps=[w2_act_tids])
                expert_completion_tids[local_expert_id] = expert_completion_tid

        completion_tid = pl.system.task_dummy(deps=[expert_completion_tids[local_expert_id] for local_expert_id in range(N_LOCAL)])

        return completion_tid

    return prefill_expert_grouped


prefill_expert_grouped = make_prefill_expert_grouped(PREFILL_MOE_GROUPED_TOTAL_CAP)


@pl.jit.inline
def prefill_moe_dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    expert_x_out: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, D], pl.INT8],
    expert_scale_out: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
    expert_counts_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_expert_counts_out: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    send_counts_out: pl.Tensor[[N_RANKS, 1], pl.INT32],
    route_to_packed_out: pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD], pl.INT32],
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scale_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    prior_dep: pl.Scalar[pl.TASK_ID],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Pack active routes, exchange counts and payloads, and form aligned expert slabs."""

    send_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)

    x_send = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, D], dtype=pl.INT8, manual_dep=True)
    scale_send = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_SEND_SCALE_PAD], dtype=pl.FP32, manual_dep=True,)

    partial_counts = pl.create_tensor([PREFILL_COUNT_SHARDS, N_RANKS * N_LOCAL], dtype=pl.INT32)
    packed_offsets = pl.create_tensor([PREFILL_COUNT_SHARDS + 1, N_RANKS * N_LOCAL], dtype=pl.INT32)
    with pl.spmd(PREFILL_COUNT_SHARDS, name_hint="prefill_moe_route_count", allow_early_resolve=True) as partial_count_tid:
        shard = pl.tile.get_block_idx()
        token_begin = shard * PREFILL_COUNT_TOKEN_TILE
        active_tokens = pl.min(T, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
        token_rows = pl.min(PREFILL_COUNT_TOKEN_TILE, pl.max(active_tokens - token_begin, 0))
        counts = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for expert_id in pl.range(N_RANKS * N_LOCAL):
            counts[expert_id] = 0
        if token_rows > 0:
            route_indices = pl.tile.load(indices, [token_begin, 0], [PREFILL_COUNT_TOKEN_TILE, 8], valid_shape=[token_rows, TOPK])
            for token in pl.range(token_rows):
                for topk in pl.range(TOPK):
                    route_expert_id = pl.tile.read(route_indices, [token, topk])
                    counts[route_expert_id] = counts[route_expert_id] + 1
        partial_count_row = pl.tile.full([1, N_RANKS * N_LOCAL], dtype=pl.INT32, value=0)
        for expert_id in pl.range(N_RANKS * N_LOCAL):
            pl.tile.write(partial_count_row, [0, expert_id], counts[expert_id])
        pl.tile.store(partial_count_row, [shard, 0], partial_counts)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_route_prefix",
               deps=[partial_count_tid], allow_early_resolve=True) as prefix_tid:
        counts_tile = pl.tile.load(partial_counts, [0, 0], [PREFILL_COUNT_SHARDS, N_RANKS * N_LOCAL])
        offsets_tile = pl.tile.full([PREFILL_COUNT_SHARDS + 1, N_RANKS * N_LOCAL], dtype=pl.INT32, value=0)
        packed_prefix = pl.const(0, pl.INT32)
        for dst in pl.range(N_RANKS):
            rank_total = pl.const(0, pl.INT32)
            for local_e in pl.range(N_LOCAL):
                expert_id = dst * N_LOCAL + local_e
                pl.tile.write(offsets_tile, [PREFILL_COUNT_SHARDS, expert_id], packed_prefix)
                expert_total = pl.const(0, pl.INT32)
                for shard in pl.range(PREFILL_COUNT_SHARDS):
                    pl.tile.write(offsets_tile, [shard, expert_id], packed_prefix)
                    count = pl.tile.read(counts_tile, [shard, expert_id])
                    packed_prefix = packed_prefix + count
                    expert_total = expert_total + count
                pl.write(send_expert_counts, [dst, local_e], expert_total)
                rank_total = rank_total + expert_total
            pl.write(send_counts_out, [dst, 0], rank_total)
        pl.tile.store(offsets_tile, [0, 0], packed_offsets)

    with pl.spmd(PREFILL_COUNT_SHARDS, name_hint="prefill_moe_route_positions", deps=[prefix_tid], allow_early_resolve=True) as count_tid:
        shard = pl.tile.get_block_idx()
        token_begin = shard * PREFILL_COUNT_TOKEN_TILE
        active_tokens = pl.min(T, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
        token_rows = pl.min(PREFILL_COUNT_TOKEN_TILE, pl.max(active_tokens - token_begin, 0))
        if token_rows > 0:
            shard_offsets = pl.tile.load(packed_offsets, [shard, 0], [1, N_RANKS * N_LOCAL])
            expert_starts = pl.tile.load(packed_offsets, [PREFILL_COUNT_SHARDS, 0], [1, N_RANKS * N_LOCAL])
            packed_cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
            destination_prefix = pl.array.create(N_RANKS, pl.INT32)
            for expert_id in pl.range(N_RANKS * N_LOCAL):
                packed_cursor[expert_id] = pl.tile.read(shard_offsets, [0, expert_id])
            for dst in pl.range(N_RANKS):
                destination_prefix[dst] = pl.tile.read(expert_starts, [0, dst * N_LOCAL])
            route_indices = pl.tile.load(indices, [token_begin, 0], [PREFILL_COUNT_TOKEN_TILE, 8], valid_shape=[token_rows, TOPK])
            route_positions = pl.tile.full([PREFILL_COUNT_TOKEN_TILE * TOPK, PREFILL_MOE_ROUTE_MAP_PAD], dtype=pl.INT32, value=0)
            for token in pl.range(token_rows):
                for topk in pl.range(TOPK):
                    route_expert_id = pl.tile.read(route_indices, [token, topk])
                    packed_slot = packed_cursor[route_expert_id]
                    destination = route_expert_id // N_LOCAL
                    return_row = destination * PREFILL_MOE_PEER_CAP + packed_slot - destination_prefix[destination]
                    pl.tile.write(route_positions, [token * TOPK + topk, 0], packed_slot)
                    pl.tile.write(route_positions, [token * TOPK + topk, 1], pl.cast(return_row, pl.INT32))
                    packed_cursor[route_expert_id] = packed_slot + 1
            valid_positions = pl.set_validshape(route_positions, token_rows * TOPK, PREFILL_MOE_ROUTE_MAP_PAD)
            pl.tile.store(valid_positions, [token_begin * TOPK, 0], route_to_packed_out)

    token_scales_row = pl.reshape(x_norm_scale, [1, T])
    with pl.spmd(PREFILL_SOURCE_PACK_BLOCKS, name_hint="prefill_moe_dispatch_pack", deps=[count_tid], allow_early_resolve=True,) as pack_tid:
        pack_block = pl.tile.get_block_idx()
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        scale_row = pl.tile.full([1, PREFILL_MOE_SEND_SCALE_PAD], dtype=pl.FP32, value=0.0)
        token_scales = pl.tile.load(token_scales_row, [0, 0], [1, T])
        route_weights = pl.tile.load(weights, [0, 0], [T, 8], valid_shape=[T, TOPK])
        for token in pl.range(pack_block, active_tokens, PREFILL_SOURCE_PACK_BLOCKS):
            route_map = pl.tile.load(route_to_packed_out, [token * TOPK, 0], [TOPK, PREFILL_MOE_ROUTE_MAP_PAD])
            hidden_row = x_norm_i8[token:token + 1, :]
            token_scale = pl.tile.read(token_scales, [0, token])
            for topk in pl.range(TOPK):
                packed_slot = pl.tile.read(route_map, [topk, 0])
                packed_row = pl.cast(packed_slot, pl.INDEX)
                x_send[packed_row:packed_row + 1, :] = hidden_row
                pl.tile.write(scale_row, [0, 0], token_scale)
                pl.tile.write(scale_row, [0, 1], pl.tile.read(route_weights, [token, topk]))
                pl.tile.store(scale_row, [packed_row, 0], scale_send)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_moe_dispatch_count_exchange",
        deps=[count_tid, prior_dep],
    ) as count_publish_tid:
        for dest in pl.range(N_RANKS):
            pld.tensor.put(
                dst=count_target, peer=dest, src=send_expert_counts,
                dst_offsets=[my_rank, 0], src_offsets=[dest, 0], shape=[1, N_LOCAL],
            )
            if dest != my_rank:
                pld.system.notify(target=count_signal, peer=dest, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_count_wait") as count_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=count_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_count_finalize", deps=[count_publish_tid, count_wait_tid]) as count_exchange_tid:
        for local_e in pl.range(N_LOCAL):
            expert_total = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(count_target, [src, local_e])
                pl.write(recv_expert_counts_out, [src, local_e], count)
                expert_total = expert_total + count
            pl.write(expert_counts_out, [local_e, 0], expert_total)

    with pl.spmd(
        N_RANKS * 4,
        name_hint="prefill_moe_dispatch_payload_put",
        deps=[pack_tid, count_exchange_tid],
        allow_early_resolve=False,
    ) as payload_put_tid:
        transfer_block = pl.tile.get_block_idx()
        dest = transfer_block // 4
        row_group = transfer_block % 4
        n_rows = pl.cast(pl.read(send_counts_out, [dest, 0]), pl.INDEX)
        if n_rows < 0:
            n_rows = pl.cast(0, pl.INDEX)
        if n_rows > PREFILL_MOE_PEER_CAP:
            n_rows = pl.cast(PREFILL_MOE_PEER_CAP, pl.INDEX)
        src_base = pl.cast(0, pl.INDEX)
        for prior_dst in pl.range(dest):
            src_base = src_base + pl.cast(pl.read(send_counts_out, [prior_dst, 0]), pl.INDEX)
        dst_base = pl.cast(my_rank, pl.INDEX) * PREFILL_MOE_PEER_CAP
        full_rows = (n_rows // PREFILL_DISPATCH_ROW_TILE) * PREFILL_DISPATCH_ROW_TILE
        for row in pl.range(row_group * PREFILL_DISPATCH_ROW_TILE, full_rows, 4 * PREFILL_DISPATCH_ROW_TILE):
            pld.tensor.put(
                dst=x_target,
                peer=dest,
                src=x_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[PREFILL_DISPATCH_ROW_TILE, D],
            )
            pld.tensor.put(
                dst=scale_target,
                peer=dest,
                src=scale_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[PREFILL_DISPATCH_ROW_TILE, PREFILL_MOE_SCALE_PAD],
            )
        for row in pl.range(full_rows + row_group, n_rows, 4):
            pld.tensor.put(
                dst=x_target,
                peer=dest,
                src=x_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[1, D],
            )
            pld.tensor.put(
                dst=scale_target,
                peer=dest,
                src=scale_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[1, PREFILL_MOE_SCALE_PAD],
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_dispatch_payload_notify", deps=[payload_put_tid]) as payload_notify_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(target=x_signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_dispatch_payload_wait", deps=[payload_notify_tid]) as payload_exchange_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=x_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    with pl.spmd(
        N_LOCAL * PREFILL_RESORT_BLOCKS_PER_EXPERT,
        name_hint="prefill_moe_dispatch_resort",
        deps=[count_exchange_tid, payload_exchange_tid],
        allow_early_resolve=False,
    ) as resort_tid:
        resort_block = pl.tile.get_block_idx()
        local_e = resort_block // PREFILL_RESORT_BLOCKS_PER_EXPERT
        row_group = resort_block % PREFILL_RESORT_BLOCKS_PER_EXPERT
        if pl.read(expert_counts_out, [local_e, 0]) > 0:
            expert_scale_row = pl.tile.full([1, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32, value=0.0)

            expert_base = pl.cast(0, pl.INDEX)
            for prior_e in pl.range(local_e):
                prior_rows = pl.cast(pl.read(expert_counts_out, [prior_e, 0]), pl.INDEX)
                prior_tiles = (prior_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1) // PREFILL_MOE_GROUPED_EXPERT_TILE
                expert_base = expert_base + prior_tiles * PREFILL_MOE_GROUPED_EXPERT_TILE

            source_prefix = pl.cast(0, pl.INDEX)
            for src in pl.range(N_RANKS):
                source_expert_base = pl.cast(0, pl.INDEX)
                for prior_e in pl.range(local_e):
                    prior_count = pl.cast(pl.read(recv_expert_counts_out, [src, prior_e]), pl.INDEX)
                    source_expert_base = source_expert_base + prior_count
                n_rows = pl.cast(pl.read(recv_expert_counts_out, [src, local_e]), pl.INDEX)
                input_base = src * PREFILL_MOE_PEER_CAP + source_expert_base
                output_base = expert_base + source_prefix
                full_rows = (n_rows // 16) * 16
                for row in pl.range(row_group * 16, full_rows, PREFILL_RESORT_BLOCKS_PER_EXPERT * 16):
                    input_row = input_base + row
                    output_row = output_base + row
                    expert_x_out[output_row : output_row + 16, :] = x_target[input_row : input_row + 16, :]
                    for inner in pl.range(16):
                        pl.tile.write(expert_scale_row, [0, 0], pl.read(scale_target, [input_row + inner, 0]))
                        pl.tile.write(expert_scale_row, [0, 1], pl.read(scale_target, [input_row + inner, 1]))
                        pl.tile.store(expert_scale_row, [output_row + inner, 0], expert_scale_out)
                for row in pl.range(full_rows + row_group, n_rows, PREFILL_RESORT_BLOCKS_PER_EXPERT):
                    input_row = input_base + row
                    output_row = output_base + row
                    expert_x_out[output_row : output_row + 1, :] = x_target[input_row : input_row + 1, :]
                    pl.tile.write(expert_scale_row, [0, 0], pl.read(scale_target, [input_row, 0]))
                    pl.tile.write(expert_scale_row, [0, 1], pl.read(scale_target, [input_row, 1]))
                    pl.tile.store(expert_scale_row, [output_row, 0], expert_scale_out)
                source_prefix = source_prefix + n_rows

            expert_rows = pl.cast(pl.read(expert_counts_out, [local_e, 0]), pl.INDEX)
            expert_tiles = (expert_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1) // PREFILL_MOE_GROUPED_EXPERT_TILE
            aligned_rows = expert_tiles * PREFILL_MOE_GROUPED_EXPERT_TILE
            zero_x = pl.cast(pl.full([1, D], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc")
            zero_scale = pl.full([1, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32, value=0.0)
            for row in pl.range(expert_rows + row_group, aligned_rows, PREFILL_RESORT_BLOCKS_PER_EXPERT):
                output_row = expert_base + row
                expert_x_out[output_row : output_row + 1, :] = zero_x
                expert_scale_out[output_row : output_row + 1, :] = zero_scale

    return resort_tid


@pl.jit.inline
def _prefill_moe_reverse_exchange(
    expert_y: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, D], pl.BF16],
    expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_expert_counts: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    reverse_send_counts: pl.Tensor[[N_RANKS, 1], pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    expert_tid: pl.Scalar[pl.TASK_ID],
    counts_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    with pl.spmd(
        N_RANKS * N_LOCAL,
        name_hint="prefill_moe_combine_y_put",
        deps=[expert_tid, counts_tid],
        allow_early_resolve=False,
    ) as reverse_put_tid:
        transfer_block = pl.tile.get_block_idx()
        dest = transfer_block // N_LOCAL
        local_e = transfer_block % N_LOCAL
        destination_rows = pl.cast(pl.read(reverse_send_counts, [dest, 0]), pl.INDEX)
        destination_rows = pl.min(PREFILL_MOE_PEER_CAP, pl.max(destination_rows, 0))
        dst_base = pl.cast(my_rank, pl.INDEX) * PREFILL_MOE_PEER_CAP
        send_prefix = pl.cast(0, pl.INDEX)
        expert_base = pl.cast(0, pl.INDEX)
        for prior_e in pl.range(local_e):
            send_prefix = send_prefix + pl.cast(pl.read(recv_expert_counts, [dest, prior_e]), pl.INDEX)
            expert_rows = pl.cast(pl.read(expert_counts, [prior_e, 0]), pl.INDEX)
            expert_tiles = (expert_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1) // PREFILL_MOE_GROUPED_EXPERT_TILE
            expert_base = expert_base + expert_tiles * PREFILL_MOE_GROUPED_EXPERT_TILE
        source_prefix = pl.cast(0, pl.INDEX)
        for prior_src in pl.range(dest):
            source_prefix = source_prefix + pl.cast(pl.read(recv_expert_counts, [prior_src, local_e]), pl.INDEX)
        n_rows = pl.cast(pl.read(recv_expert_counts, [dest, local_e]), pl.INDEX)
        n_rows = pl.min(n_rows, pl.max(destination_rows - send_prefix, 0))
        input_base = expert_base + source_prefix
        output_base = dst_base + send_prefix
        bulk_rows = (n_rows // PREFILL_REVERSE_ROW_TILE) * PREFILL_REVERSE_ROW_TILE
        for row in pl.range(0, bulk_rows, PREFILL_REVERSE_ROW_TILE):
            pld.tensor.put(
                dst=reverse_target,
                peer=dest,
                src=expert_y,
                dst_offsets=[output_base + row, 0],
                src_offsets=[input_base + row, 0],
                shape=[PREFILL_REVERSE_ROW_TILE, D],
            )
        for row in pl.range(bulk_rows, n_rows):
            pld.tensor.put(
                dst=reverse_target,
                peer=dest,
                src=expert_y,
                dst_offsets=[output_base + row, 0],
                src_offsets=[input_base + row, 0],
                shape=[1, D],
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_combine_y_notify", deps=[reverse_put_tid]) as reverse_notify_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(target=reverse_signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_combine_y_wait", deps=[reverse_notify_tid]) as reverse_exchange_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=reverse_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    return reverse_exchange_tid


@pl.jit.inline
def prefill_moe_combine(
    expert_y: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, D], pl.BF16],
    expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_expert_counts: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    route_to_packed: pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD], pl.INT32],
    shared_y: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    dispatch_tid: pl.Scalar[pl.TASK_ID],
    expert_tid: pl.Scalar[pl.TASK_ID],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Return weighted expert rows and combine shared output in top-k order."""

    reverse_send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32, manual_dep=True)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_moe_combine_counts",
        deps=[dispatch_tid],
        allow_early_resolve=True,
    ) as reverse_counts_tid:
        for src in pl.range(N_RANKS):
            source_total = pl.const(0, pl.INT32)
            for local_e in pl.range(N_LOCAL):
                source_total = source_total + pl.read(recv_expert_counts, [src, local_e])
            pl.write(reverse_send_counts, [src, 0], source_total)

    reverse_exchange_tid = _prefill_moe_reverse_exchange(
        expert_y, expert_counts, recv_expert_counts,
        reverse_target, reverse_signal, reverse_send_counts,
        epoch, my_rank, expert_tid, reverse_counts_tid,
    )

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    with pl.spmd(
        T // PREFILL_MOE_FINALIZE_TOKEN_TILE,
        name_hint="prefill_moe_combine_finalize",
        deps=[reverse_exchange_tid],
    ) as finalize_tid:
        token0 = pl.tile.get_block_idx() * PREFILL_MOE_FINALIZE_TOKEN_TILE
        for lane in pl.range(PREFILL_MOE_FINALIZE_TOKEN_TILE):
            token = token0 + lane
            if token < active_tokens:
                acc = pl.cast(shared_y[token : token + 1, :], pl.FP32)
                for topk in pl.range(TOPK):
                    route = token * TOPK + topk
                    return_row = pl.cast(pl.read(route_to_packed, [route, 1]), pl.INDEX)
                    route_y = pl.cast(reverse_target[return_row : return_row + 1, 0:D], pl.FP32)
                    acc = pl.add(acc, route_y)
                ffn_out[token : token + 1, :] = pl.cast(acc, pl.BF16, mode="rint")
            else:
                ffn_out[token : token + 1, :] = shared_y[token : token + 1, :]

    return finalize_tid


@pl.jit.inline(auto_scope=False)
def prefill_moe_tile(
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
    recv_x: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prior_dep: pl.Scalar[pl.TASK_ID],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one DSpark prefill wave with compact expert-major exchanges."""
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(x_mixed, norm_w, gate_w, gate_bias, layer_id, num_tokens, tid2eid, input_ids, x_norm_i8, x_norm_scale, indices, weights,)

    shared_out = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(x_norm_i8, x_norm_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, shared_out,)

    expert_counts = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
    send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32, manual_dep=True)
    route_to_packed = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD], dtype=pl.INT32, manual_dep=True)
    grouped_y = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, D], dtype=pl.BF16)
    dispatch_tids = pl.array.create(1, pl.TASK_ID)
    expert_tids = pl.array.create(1, pl.TASK_ID)
    with pl.scope():
        grouped_x = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, D], dtype=pl.INT8)
        grouped_scale = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32)
        dispatch_tids[0] = prefill_moe_dispatch(
            indices, x_norm_i8, x_norm_scale, weights,
            grouped_x, grouped_scale, expert_counts, recv_expert_counts, send_counts, route_to_packed,
            recv_meta, arrived, recv_x, data_arrived, recv_aux,
            num_tokens, my_rank, prior_dep, moe_epoch,
        )
        expert_tids[0] = prefill_expert_grouped(
            grouped_x, grouped_scale, expert_counts,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
            grouped_y,
        )
    dispatch_tid = pl.system.task_dummy(deps=[dispatch_tids])
    expert_tid = pl.system.task_dummy(deps=[expert_tids])
    completion_tid = prefill_moe_combine(
        grouped_y, expert_counts, recv_expert_counts, route_to_packed, shared_out, ffn_out,
        routed_y_buf, combine_arrived, num_tokens, my_rank, dispatch_tid, expert_tid, moe_epoch,
    )
    return completion_tid


@pl.jit.inline(auto_scope=False)
def prefill_moe(
    attn_out: pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[PREFILL_LOCAL_T_DYN], pl.INT64],
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
    x_hc: pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT, D], pl.FP32],
    x_mixed: pl.Tensor[[PREFILL_GROUP_T_DYN, D], pl.BF16],
    post_ffn: pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT], pl.FP32],
    comb_ffn: pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_out: pl.Tensor[[PREFILL_LOCAL_T_DYN, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_token: pl.Tensor[[1], pl.INT32],
    layer_completion: pl.Tensor[[1], pl.INT32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_GROUP_T_DYN, HC_MULT, D], pl.FP32]:
    """Run one CP-aware prefill MoE layer over a rank-local token shard."""
    with pl.scope():
        x_mixed_step = x_mixed
        post_ffn_step = post_ffn
        comb_ffn_step = comb_ffn
        if group_tokens > 0:
            hc_pre(attn_out, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed_step, post_ffn_step, comb_ffn_step)

    local_rows = pl.tensor.dim(ffn_out, 0)
    # Every EP rank participates in every physical wave, including empty groups.
    num_waves = (local_rows + T - 1) // T
    num_waves_i32 = pl.cast(num_waves, pl.INT32)
    for wave in pl.range(num_waves):
        wave_i32 = pl.cast(wave, pl.INT32)
        moe_epoch = layer_id * num_waves_i32 + wave_i32 + pl.const(1, pl.INT32)
        with pl.scope():
            ffn_out_step = ffn_out
            local_wave_base = wave_i32 * T
            # Physical rows in the fixed-capacity wave.
            wave_rows = pl.min(T, local_rows - local_wave_base)
            if group_tokens <= 0:
                wave_rows = pl.cast(0, pl.INDEX)
            wave_rows_i32 = pl.cast(wave_rows, pl.INT32)
            full_wave_base = tp_rank * local_rows + local_wave_base

            x_mixed_wave = pl.create_tensor([T, D], dtype=pl.BF16)
            with pl.spmd(T, name_hint="prefill_moe_hidden_stage") as wave_ready_tid:
                token = pl.tile.get_block_idx()
                completed_epoch = pl.read(stage_token, [0])
                if completed_epoch >= 0:
                    if token < wave_rows:
                        full_token = full_wave_base + token
                        x_mixed_row = x_mixed[full_token : full_token + 1, :]
                        x_mixed_wave[token : token + 1, :] = x_mixed_row
                    else:
                        x_mixed_zero = pl.full([1, D], dtype=pl.BF16, value=0.0)
                        x_mixed_wave[token : token + 1, :] = x_mixed_zero

            input_id_count = pl.tensor.dim(input_ids, 0)
            input_ids_rows = pl.reshape(input_ids, [input_id_count, 1])
            input_ids_wave_rows = pl.create_tensor([T, 1], dtype=pl.INT64)
            if wave_rows == T and T % INT64_PER_CACHE_LINE == 0:
                for token_block in pl.spmd(T // PREFILL_INPUT_ID_TILE, name_hint="prefill_moe_ids_stage"):
                    token0 = token_block * PREFILL_INPUT_ID_TILE
                    local_token = local_wave_base + token0
                    input_id_tile = input_ids_rows[local_token : local_token + PREFILL_INPUT_ID_TILE, 0:1]
                    input_ids_wave_rows[token0 : token0 + PREFILL_INPUT_ID_TILE, 0:1] = input_id_tile
            else:
                for token_block in pl.spmd((T + INT64_PER_CACHE_LINE - 1) // INT64_PER_CACHE_LINE, name_hint="prefill_moe_ids_stage_tail"):
                    token_begin = token_block * INT64_PER_CACHE_LINE
                    for token_offset in pl.range(INT64_PER_CACHE_LINE):
                        token = token_begin + token_offset
                        input_id = pl.cast(0, pl.INT64)
                        if token < wave_rows:
                            local_token = local_wave_base + token
                            input_id = pl.read(input_ids_rows, [local_token, 0])
                        if token < T:
                            pl.write(input_ids_wave_rows, [token, 0], input_id)
            input_ids_wave = pl.reshape(input_ids_wave_rows, [T])

            if wave_rows == T:
                ffn_wave = pl.slice(ffn_out_step, [T, D], [local_wave_base, 0])
            else:
                ffn_wave = pl.create_tensor([T, D], dtype=pl.BF16)
            completion_tid = prefill_moe_tile(
                x_mixed_wave,
                norm_w, gate_w, gate_bias, tid2eid, input_ids_wave,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                ffn_wave,
                recv_meta, recv_x, recv_aux,
                arrived, data_arrived, routed_y_buf, combine_arrived,
                wave_ready_tid, layer_id, wave_rows_i32, my_rank, moe_epoch,
            )
            output_tids = pl.array.create(1, pl.TASK_ID)
            output_tids[0] = completion_tid
            if wave_rows < T:
                with pl.spmd(T, name_hint="prefill_moe_output_store_tail", deps=[completion_tid]) as output_store_tail_tid:
                    token = pl.tile.get_block_idx()
                    if token < wave_rows:
                        local_token = local_wave_base + token
                        ffn_out_step[local_token : local_token + 1, :] = ffn_wave[token : token + 1, :]
                output_tids[0] = output_store_tail_tid
            output_store_tid = pl.system.task_dummy(deps=[output_tids])
            # Publish one globally complete wave before the windows are reused.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_wave_notify",
                deps=[output_store_tid],
            ) as notify_tid:
                for peer in pl.range(N_RANKS):
                    if peer != my_rank:
                        pld.system.notify(
                            target=stage_done, peer=peer, offsets=[my_rank, 0],
                            value=1, op=pld.NotifyOp.AtomicAdd,
                        )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_wave_wait") as wait_tid:
                for peer in pl.range(N_RANKS):
                    if peer != my_rank:
                        pld.system.defer_wait(
                            signal=stage_done, offsets=[peer, 0],
                            expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                        )

            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_wave_publish",
                deps=[output_store_tid, notify_tid, wait_tid],
            ):
                pl.write(stage_token, [0], moe_epoch)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_layer_complete"):
        completed_epoch = pl.read(stage_token, [0])
        if completed_epoch >= 0:
            pl.write(layer_completion, [0], layer_id + pl.const(1, pl.INT32))

    with pl.scope():
        if group_tokens > 0:
            group_rows = pl.tensor.dim(attn_out, 0)
            ffn_out_full = pl.create_tensor([group_rows, D], dtype=pl.BF16)
            ffn_out_full, _gather_signal = prefill_cp_token_allgather_step(
                ffn_out, ffn_out_full,
                gather_window, gather_signal,
                group_base, tp_rank,
            )
            hc_post(ffn_out_full, attn_out, post_ffn, comb_ffn, x_hc)
    return x_hc


@pl.jit
def moe_test(
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
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
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
    # 1-based MoE call id; multi-layer callers increment it per reused window.
    moe_epoch: pl.Scalar[pl.INT32],
    finalize_moe: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    moe(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        routed_y_buf, combine_arrived,
        layer_id, num_tokens, my_rank, moe_epoch,
    )
    if finalize_moe == 1:
        clear_moe_signals(x_next, arrived, data_arrived, combine_arrived)
    return x_next


# Rounds sharing one window allocation. >1 exercises retained-window reuse
# across MoE epochs; the round axis is kept even at 1.
MOE_ROUNDS = 1


@pl.jit.host
def l3_moe(
    x_hc: pl.Tensor[[MOE_ROUNDS, N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[MOE_ROUNDS, N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[MOE_ROUNDS, N_RANKS, T, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for round_id in pl.range(MOE_ROUNDS):
        moe_epoch = round_id + 1
        finalize_moe = pl.cast(round_id == MOE_ROUNDS - 1, pl.INT32)
        for r in pl.range(pld.world_size()):
            recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
            recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
            recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
            recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
            arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
            data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
            routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
            combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
            moe_test(
                x_hc[round_id, r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
                norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[round_id, r],
                routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
                routed_w2[r], routed_w2_scale[r],
                shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
                shared_w2[r], shared_w2_scale[r],
                x_next[round_id, r],
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived,
                layer_id, num_tokens, r, moe_epoch, finalize_moe,
                device=r,
            )


# === Golden + test ==========================================================
def _golden_moe_single(tensors):
    """Per-rank torch reference. Replays the 4 stages on host. Each rank's
    output depends only on its own inputs because the dispatch+combine round-
    trip is r_route-keyed and shape-preserving (test_l3 pattern).

    The per-route result is invariant to the packing layout (each recv row's
    SwiGLU output depends only on that row's own input), so this src-major host
    packing matches the device's per-source-lane cumsum layout by construction."""
    import torch

    from hc_pre import golden_hc_pre
    from hc_post import golden_hc_post
    from gate import golden_gate_core
    from expert_shared import golden_expert_shared
    from expert_routed import golden_expert_routed

    x_next_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # Stages 1-2: hc_pre + gate per rank. Rank-independent, so compute once and
    # reuse for both the dispatch replay and each rank's local stages.
    all_post = []
    all_comb = []
    all_indices = []
    all_x_i8 = []
    all_scale = []
    all_weights = []
    for src in range(N_RANKS):
        src_x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
        src_post = torch.zeros(T, HC_MULT, dtype=torch.float32)
        src_comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
        golden_hc_pre({
            "x":        tensors["x_hc"][src],
            "hc_fn":    tensors["hc_ffn_fn"][src],
            "hc_scale": tensors["hc_ffn_scale"][src],
            "hc_base":  tensors["hc_ffn_base"][src],
            "x_mixed":  src_x_mixed,
            "post":     src_post,
            "comb":     src_comb,
        })
        src_x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
        src_x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
        src_indices = torch.zeros(T, TOPK, dtype=torch.int32)
        src_weights = torch.zeros(T, TOPK, dtype=torch.float32)
        golden_gate_core({
            "x_mixed":      src_x_mixed,
            "norm_w":       tensors["norm_w"][src],
            "gate_w":       tensors["gate_w"][src],
            "gate_bias":    tensors["gate_bias"][src],
            "layer_id":     tensors["layer_id"],
            "num_tokens":   tensors["num_tokens"],
            "tid2eid":      tensors["tid2eid"][src],
            "input_ids":    tensors["input_ids"][src],
            "x_norm_i8":    src_x_norm_i8,
            "x_norm_scale": src_x_norm_scale,
            "indices":      src_indices,
            "weights":      src_weights,
        })
        all_post.append(src_post)
        all_comb.append(src_comb)
        all_indices.append(src_indices)
        all_x_i8.append(src_x_norm_i8)
        all_scale.append(src_x_norm_scale)
        all_weights.append(src_weights)

    # Route counts per (src, dst, local expert); drives the per-source lane cumsum.
    send_counts = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
    for src in range(N_RANKS):
        for t in range(num_tokens):
            for k in range(TOPK):
                eid = int(all_indices[src][t, k].item())
                send_counts[src, eid // N_LOCAL, eid % N_LOCAL] += 1

    # Stages 4-5: dispatch replay + routed expert per dst. Also rank-independent
    # (each recv row's SwiGLU output depends only on that row), so compute once.
    dst_recv_y = {}
    for dst in range(N_RANKS):
        # Pack onto rank dst in src-major order within each local expert — same
        # convention as dispatch's per-source lane cumsum.
        d_recv_x = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.int8)
        d_recv_scale = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
        d_recv_w = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
        d_recv_count = torch.zeros(N_LOCAL, 1, dtype=torch.int32)
        d_slot_offsets = torch.zeros(N_RANKS, N_LOCAL, dtype=torch.int32)
        d_running = torch.zeros(N_LOCAL, dtype=torch.int32)
        for src in range(N_RANKS):
            d_slot_offsets[src] = d_running.clone()
            d_running = d_running + send_counts[src, dst]
        for e in range(N_LOCAL):
            d_recv_count[e, 0] = int(d_running[e].item())
        for src in range(N_RANKS):
            cursor = torch.zeros(N_LOCAL, dtype=torch.int32)
            for t in range(num_tokens):
                for k in range(TOPK):
                    eid = int(all_indices[src][t, k].item())
                    if eid // N_LOCAL != dst:
                        continue
                    loc_e = eid % N_LOCAL
                    slot = int(d_slot_offsets[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    d_recv_x[loc_e, slot, :] = all_x_i8[src][t, :]
                    d_recv_scale[loc_e, slot] = float(all_scale[src][t, 0].item())
                    d_recv_w[loc_e, slot] = float(all_weights[src][t, k].item())
        d_recv_y = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.bfloat16)
        golden_expert_routed({
            "recv_x":            d_recv_x,
            "recv_scale_dq":     d_recv_scale,
            "recv_weights":      d_recv_w,
            "recv_expert_count": d_recv_count,
            "routed_w1":         tensors["routed_w1"][dst],
            "routed_w1_scale":   tensors["routed_w1_scale"][dst],
            "routed_w3":         tensors["routed_w3"][dst],
            "routed_w3_scale":   tensors["routed_w3_scale"][dst],
            "routed_w2":         tensors["routed_w2"][dst],
            "routed_w2_scale":   tensors["routed_w2_scale"][dst],
            "recv_y":            d_recv_y,
        })
        dst_recv_y[dst] = d_recv_y

    for r in range(N_RANKS):
        x_norm_i8 = all_x_i8[r]
        x_norm_scale = all_scale[r]
        post_t = all_post[r]
        comb_t = all_comb[r]

        # Stage 3: expert_shared (local)
        sh = torch.zeros(T, D, dtype=torch.bfloat16)
        golden_expert_shared({
            "x_local_i8":       x_norm_i8,
            "x_local_scale_dq": x_norm_scale,
            "num_tokens":       tensors["num_tokens"],
            "shared_w1":        tensors["shared_w1"][r],
            "shared_w1_scale":  tensors["shared_w1_scale"][r],
            "shared_w3":        tensors["shared_w3"][r],
            "shared_w3_scale":  tensors["shared_w3_scale"][r],
            "shared_w2":        tensors["shared_w2"][r],
            "shared_w2_scale":  tensors["shared_w2_scale"][r],
            "sh":               sh,
        })

        # Stage 6: combine — for each (src, t, k) that originated on this
        # rank, find the (loc_e, slot) on rank dst where the SwiGLU result
        # landed, then accumulate by r_route = t*TOPK+k.
        my_routes = []
        for t in range(num_tokens):
            for k in range(TOPK):
                eid = int(all_indices[r][t, k].item())
                dst = eid // N_LOCAL
                loc_e = eid % N_LOCAL
                my_routes.append((t, k, dst, loc_e))

        # Rank r's contribution to dst sits at slot offset
        # Sigma_{s<r} send_counts[s, dst, loc_e] plus a running per-(dst, loc_e)
        # cursor over r's own routes in (t, k) order.
        routed_y_buf_r = torch.zeros(N_ROUTES, D, dtype=torch.bfloat16)
        cursors = {}
        for (t, k, dst, loc_e) in my_routes:
            src_off = int(send_counts[:r, dst, loc_e].sum().item())
            cursor = cursors.get((dst, loc_e), 0)
            cursors[(dst, loc_e)] = cursor + 1
            r_route = t * TOPK + k
            routed_y_buf_r[r_route, :] = dst_recv_y[dst][loc_e, src_off + cursor, :]

        # Stage 7: reduce + sh + hc_post
        acc = sh.float().clone()
        for k in range(TOPK):
            for t in range(num_tokens):
                acc[t, :] += routed_y_buf_r[t * TOPK + k, :].float()
        ffn_out = acc.to(torch.bfloat16)
        if "ffn_out" in tensors:
            tensors["ffn_out"][r].copy_(ffn_out)
        x_next_r = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
        golden_hc_post({
            "x":        ffn_out,
            "residual": tensors["x_hc"][r],
            "post":     post_t,
            "comb":     comb_t,
            "y":        x_next_r,
        })
        x_next_out[r] = x_next_r

    tensors["x_next"][:] = x_next_out


def golden_moe(tensors):
    """Evaluate one MoE layer."""
    _golden_moe_single(tensors)


def golden_moe_rounds(tensors):
    """Evaluate every MoE round."""
    for round_id in range(tensors["x_hc"].shape[0]):
        round_tensors = dict(tensors)
        round_tensors["x_hc"] = tensors["x_hc"][round_id]
        round_tensors["input_ids"] = tensors["input_ids"][round_id]
        round_tensors["x_next"] = tensors["x_next"][round_id]
        _golden_moe_single(round_tensors)


# === Routing profile =========================================================
# Balanced routing hands every expert the same row count, so the per-expert
# receive-tile loop never runs a second tile and neither card carries more rows
# than the other. Production routing is uneven on both axes. The table below is
# one captured decode step's route counts over 256 global experts, inlined so
# this file needs nothing on disk. Of 6321 captured decode steps it is the one
# whose EP2 receive-tile count, over-tile expert count, peak rows and busier-card
# rows all sit at the median.
#   EP2 rows per local expert, sorted, one line per card:
#     card 0: 196 184  88  76  64  52  40  24  24  24  24  24  16  12   4   4  sum 856, 23 tiles
#     card 1: 200 108 108  56  40  32  28  24  20  20  16  12   8   8   0   0  sum 680, 19 tiles
REAL_PROFILE = (
    0, 8, 0, 4, 9, 1, 0, 0, 0, 2, 6, 1, 0, 0, 1, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 43, 6, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
    0, 4, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 6, 1, 0, 0, 5, 0, 1, 4, 0, 0, 0, 0, 0, 2, 0,
    0, 14, 0, 0, 4, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 2,
    0, 1, 1, 0, 0, 0, 4, 0, 1, 0, 0, 5, 0, 0, 0, 0,
    1, 0, 9, 0, 3, 0, 3, 0, 2, 40, 0, 0, 0, 1, 3, 0,
    0, 0, 11, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0,
    0, 1, 0, 0, 2, 0, 2, 1, 0, 0, 1, 16, 1, 0, 9, 0,
    0, 0, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 8,
    0, 0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 5, 0,
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 1,
    48, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
)
ROUTING_CHOICES = ("uniform", "balanced", "real")


def _profile_routes(active_tokens):
    """Per-token expert picks reproducing the captured expert-load shape."""
    fold = len(REAL_PROFILE) // N_EXPERTS_GLOBAL
    folded = [sum(REAL_PROFILE[e * fold:(e + 1) * fold]) for e in range(N_EXPERTS_GLOBAL)]
    # Rescale to one rank's route budget. Every rank replays the same shape, so
    # each expert's global row count keeps the captured spread.
    budget = active_tokens * TOPK
    scale = budget / sum(folded)
    exact = [c * scale for c in folded]
    demand = [int(v) for v in exact]
    order = sorted(range(N_EXPERTS_GLOBAL), key=lambda e: exact[e] - demand[e], reverse=True)
    for e in order[: budget - sum(demand)]:
        demand[e] += 1
    assert max(demand) <= active_tokens, "profile routes one token to an expert twice"

    counts = [d * N_RANKS for d in demand]
    tiles = sum((c + RECV_TILE - 1) // RECV_TILE for c in counts)
    over_tile = sum(1 for c in counts if c > RECV_TILE)
    print(f"[moe] routing 'real': {N_EXPERTS_GLOBAL} experts, {N_RANKS * budget} routes, "
          f"rows min={min(counts)} max={max(counts)} "
          f"over_tile={over_tile}/{N_EXPERTS_GLOBAL} recv_tiles={tiles}")
    print(f"[moe] per-expert rows: {counts}")

    # Each token takes the TOPK experts with the most demand left, so they stay
    # distinct and every expert lands on exactly its count.
    routes = []
    for _ in range(active_tokens):
        picked = sorted(range(N_EXPERTS_GLOBAL), key=lambda e: demand[e], reverse=True)[:TOPK]
        for e in picked:
            demand[e] -= 1
        routes.append(picked)
    return routes


# Routed = MXFP4 (gen_routed_weight), shared = MXFP8 (gen_shared_weight). This
# is an integration test whose x_next-equivalent output is dominated by near-zero
# residual+FFN cancellations, so it keeps the smaller *behaviorally-calibrated* magnitude
# (random fixtures blow up the relative metric at the real ~2.5e-2 magnitude); only the
# grid SHAPE (FP4/FP8 discreteness, scale CV) matches the real distribution.
ROUTED_DEQUANT_STD = {"w1": 1.08e-2, "w2": 2.54e-2, "w3": 1.10e-2}
SHARED_DEQUANT_STD = {"w1": 7.65e-3, "w2": 2.39e-2, "w3": 7.39e-3}


@functools.cache
def _routed_weights():
    """Per-rank routed expert weights (different shards), generated once per process."""
    import torch
    from expert_routed import gen_routed_weight

    shapes = {
        "w1": (N_LOCAL, MOE_INTER, D),
        "w3": (N_LOCAL, MOE_INTER, D),
        "w2": (N_LOCAL, D, MOE_INTER),
    }
    weights = {}
    for key, shape in shapes.items():
        per_rank = [gen_routed_weight(shape, ROUTED_DEQUANT_STD[key]) for _ in range(N_RANKS)]
        weights[f"routed_{key}"] = torch.stack([w_i8 for w_i8, _ in per_rank])
        weights[f"routed_{key}_scale"] = torch.stack([w_s for _, w_s in per_rank])
    return weights


@functools.cache
def _shared_weights():
    """Shared expert weights replicated across ranks, generated once per process."""
    from expert_shared import gen_shared_weight

    shapes = {
        "w1": ((MOE_INTER, D), 0.50),
        "w3": ((MOE_INTER, D), 0.50),
        "w2": ((D, MOE_INTER), 0.33),
    }
    weights = {}
    for key, (shape, chan_cv) in shapes.items():
        w_i8, w_s = gen_shared_weight(shape, SHARED_DEQUANT_STD[key], chan_cv=chan_cv)
        weights[f"shared_{key}"] = w_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
        weights[f"shared_{key}_scale"] = w_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    return weights


def _build_tensor_specs(layer_id, num_tokens, routing, fixture_rounds):
    import torch
    from golden import ScalarSpec, TensorSpec

    retain_round_axis = fixture_rounds is not None
    rounds = fixture_rounds if retain_round_axis else 1

    assert routing in ROUTING_CHOICES, f"--routing must be one of {ROUTING_CHOICES}"
    balanced_routing = routing == "balanced"
    real_routes = None
    if routing == "real":
        assert layer_id < M.num_hash_layers, "the routing profile requires a hash-routing layer"
        real_routes = _profile_routes(max(0, min(T, num_tokens)))

    # Shared (replicated) weights are broadcast across ranks; the routed
    # weights are per-rank shards.
    def init_x_hc():
        value = torch.randn(rounds, N_RANKS, T, HC_MULT, D)
        return value if retain_round_axis else value[0]

    # Real layer-0 hc_ffn scale/base (fn synthetic at real magnitude). A synthetic
    # scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling the FFN output and
    # hc residual to near-zero in x_next where W8A8 noise blows up the relative tail.
    def init_hc_ffn_fn():
        x = torch.randn(MIX_HC, HC_DIM) * 0.0635
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_hc_ffn_scale():
        x = torch.tensor([0.11334, 0.035901, 0.058183])
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_hc_ffn_base():
        x = torch.tensor([
            2.4153, -2.0252, -2.0019, -2.1947,
            -1.5430, -3.0228, -6.8248, 0.5894,
            2.1916, -7.2132, -3.0938, -2.1119,
            -3.0161, 3.3293, -3.2224, -4.0226,
            -2.0428, -3.3478, 3.0893, -3.4166,
            -1.8144, -3.8147, -3.1307, 1.7862,
        ])
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_norm_w():
        x = torch.ones(D)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_gate_w():
        x = torch.randn(N_EXPERTS_GLOBAL, D) / D ** 0.5
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_gate_bias():
        x = torch.zeros(N_EXPERTS_GLOBAL)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_tid2eid():
        if balanced_routing:
            token_ids = torch.arange(VOCAB, dtype=torch.int64).unsqueeze(1)
            topk_slots = torch.arange(TOPK, dtype=torch.int64).unsqueeze(0)
            x = (token_ids * TOPK + topk_slots) % N_EXPERTS_GLOBAL
            return x.to(torch.int32).unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
        # Distinct experts per token (sample without replacement) like real top-k,
        # so the route-keyed distributed combine stays unambiguous.
        x = torch.argsort(torch.rand(VOCAB, N_EXPERTS_GLOBAL), dim=1)[:, :TOPK].to(torch.int32)
        if real_routes is not None:
            x[:len(real_routes)] = torch.tensor(real_routes, dtype=torch.int32)
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_input_ids():
        if balanced_routing:
            # Active tokens across ranks consume consecutive tid2eid rows, making
            # their route ids one contiguous round-robin sequence over experts.
            round_starts = torch.arange(rounds, dtype=torch.int64).view(rounds, 1, 1)
            rank_starts = torch.arange(N_RANKS, dtype=torch.int64).view(1, N_RANKS, 1)
            token_offsets = torch.arange(T, dtype=torch.int64).view(1, 1, T)
            round_offsets = round_starts * N_RANKS
            stream_indices = round_offsets + rank_starts
            stream_starts = stream_indices * num_tokens
            value = stream_starts + token_offsets
            return value if retain_round_axis else value[0]
        if real_routes is not None:
            # Every rank replays the same profile, so they share one id block.
            value = torch.arange(T, dtype=torch.int64).view(1, 1, T).expand(rounds, N_RANKS, T)
            value = value.contiguous()
            return value if retain_round_axis else value[0]
        # Distinct per-rank token streams.
        value = torch.randint(0, VOCAB, (rounds, N_RANKS, T), dtype=torch.int64)
        return value if retain_round_axis else value[0]

    if balanced_routing:
        assert layer_id < M.num_hash_layers, "balanced routing requires a hash-routing layer"

    x_hc_shape = [N_RANKS, T, HC_MULT, D]
    input_ids_shape = [N_RANKS, T]
    if retain_round_axis:
        x_hc_shape = [rounds, *x_hc_shape]
        input_ids_shape = [rounds, *input_ids_shape]

    specs = [
        TensorSpec("x_hc", x_hc_shape, torch.float32, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",     [N_RANKS, MIX_HC, HC_DIM],       torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale",  [N_RANKS, 3],                    torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",   [N_RANKS, MIX_HC],               torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",        [N_RANKS, D],                    torch.bfloat16,  init_value=init_norm_w),
        TensorSpec("gate_w",        [N_RANKS, N_EXPERTS_GLOBAL, D],  torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",     [N_RANKS, N_EXPERTS_GLOBAL],     torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",       [N_RANKS, VOCAB, TOPK],          torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids", input_ids_shape, torch.int64, init_value=init_input_ids),
        TensorSpec("routed_w1",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: _routed_weights()["routed_w1"]),
        TensorSpec("routed_w1_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: _routed_weights()["routed_w1_scale"]),
        TensorSpec("routed_w3",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: _routed_weights()["routed_w3"]),
        TensorSpec("routed_w3_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: _routed_weights()["routed_w3_scale"]),
        TensorSpec("routed_w2",        [N_RANKS, N_LOCAL, D, MOE_INTER], torch.int8,    init_value=lambda: _routed_weights()["routed_w2"]),
        TensorSpec("routed_w2_scale",  [N_RANKS, N_LOCAL, D],            torch.float32, init_value=lambda: _routed_weights()["routed_w2_scale"]),
        TensorSpec("shared_w1",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: _shared_weights()["shared_w1"]),
        TensorSpec("shared_w1_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: _shared_weights()["shared_w1_scale"]),
        TensorSpec("shared_w3",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: _shared_weights()["shared_w3"]),
        TensorSpec("shared_w3_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: _shared_weights()["shared_w3_scale"]),
        TensorSpec("shared_w2",        [N_RANKS, D, MOE_INTER],          torch.int8,    init_value=lambda: _shared_weights()["shared_w2"]),
        TensorSpec("shared_w2_scale",  [N_RANKS, D],                     torch.float32, init_value=lambda: _shared_weights()["shared_w2_scale"]),
        TensorSpec("x_next", x_hc_shape, torch.float32),
        ScalarSpec("layer_id",         torch.int32,                      layer_id),
        ScalarSpec("num_tokens",       torch.int32,                      num_tokens),
    ]

    # Keep the static weight parameters device-resident (child_memory), sharded
    # per rank: each shard is a leading-dim-stacked [N_RANKS, *tail] tensor sliced
    # as weight[r] and dispatched to device=r; resident="stacked" uploads shard r
    # to card r once and reuses it across dispatches, skipping the per-dispatch
    # H2D/D2H. Covers the routed/shared expert weights and their scales, the gate,
    # the HC-FFN constants, the RMSNorm gamma, and the static tid2eid route table —
    # but NOT the per-step activation (x_hc), per-step input_ids, or the output.
    # All resident names are pure inputs, so the flag is always valid.
    RESIDENT_WEIGHT_NAMES = frozenset([
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
    ])
    for spec in specs:
        if spec.name in RESIDENT_WEIGHT_NAMES:
            spec.resident = "stacked"

    return specs


def build_tensor_specs(layer_id=0, num_tokens=T, routing="uniform"):
    return _build_tensor_specs(layer_id, num_tokens, routing, fixture_rounds=None)


def build_rounds_tensor_specs(layer_id=0, num_tokens=T, routing="uniform"):
    return _build_tensor_specs(layer_id, num_tokens, routing, fixture_rounds=MOE_ROUNDS)


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--ep", type=int, default=_EP_DEFAULT, choices=list(_EP_CHOICES),
                        help="EP world size / rank count")
    parser.add_argument("--experts-per-rank", type=int, default=EXPERTS_PER_RANK,
                        help=f"routed experts per rank (default {_EXPERTS_PER_RANK_DEFAULT})")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--token-capacity", type=int, default=MOE_TOKEN_CAPACITY,
                        help="compiled token capacity per rank; active tokens may be smaller")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help=f"active token count for MoE dispatch/combine (0..{T})")
    parser.add_argument("--routing", type=str, default="uniform", choices=ROUTING_CHOICES,
                        help="expert load profile: uniform random hash ids, balanced round-robin "
                             "across every expert, or a captured production step")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None,
                        help="dir with cached in/{name}.pt + out/{name}.pt; reuses them "
                             "instead of regenerating inputs + recomputing golden.")
    parser.add_argument("--log-level", type=str, default=None,
                        help="runtime log threshold: debug, v0..v9, info, warn, error, null")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if not 0 <= args.num_tokens <= MOE_TOKEN_CAPACITY:
        parser.error("--num-tokens must be within --token-capacity")

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    golden_data = args.golden_data

    result = run(
        fn=l3_moe,
        specs=build_rounds_tensor_specs(
            layer_id=args.layer_id,
            num_tokens=args.num_tokens,
            routing=args.routing,
        ),
        golden_fn=golden_moe_rounds,
        golden_data=golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            enable_chip_swimlane=args.enable_chip_swimlane,
            log_level=args.log_level,
            ring_heap=MOE_RING_HEAP,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 x_next. Tightened 5e-3 -> 3e-3 with the real layer-0 hc_ffn
            # gate (~2.1% of points > 3e-3). No max_diff_hd (near-zero
            # residual/FFN cancellations blow up relatively).
            "x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
