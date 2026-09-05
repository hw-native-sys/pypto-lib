# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 MoE single-layer (decode), FLASH preset. --ep picks the EP world
size: 2/4/8 run N-rank distributed; each rank keeps 32 experts."""


# Sub-kernels freeze EP_WORLD_SIZE / n_routed_experts into their shapes at import
# time, so read --ep from argv and override config before importing them below.
import dataclasses
import sys

import config

_EP_CHOICES = (2, 4, 8)
_EP_DEFAULT = 2


def _parse_ep_argv():
    for i, tok in enumerate(sys.argv):
        if tok == "--ep" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith("--ep="):
            return int(tok.split("=", 1)[1])
    return _EP_DEFAULT


EP = _parse_ep_argv()

config.EP_WORLD_SIZE = EP
config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=config.FLASH.n_routed_experts // 8 * EP)
config.RECV_MAX = EP * config.MOE_TOKENS

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import FLASH as M, EP_WORLD_SIZE, MOE_TOKENS, RECV_MAX
from hc_pre import hc_pre, prefill_hc_pre
from hc_post import hc_post
from gate import gate
from expert_shared import expert_shared
from expert_routed import expert_routed, prefill_expert_grouped


T = MOE_TOKENS
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size

PREFILL_MOE_T_DYN = pl.dynamic("PREFILL_MOE_T_DYN")

HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MOE_INTER = M.moe_intermediate_size

N_RANKS = EP_WORLD_SIZE
N_EXPERTS_GLOBAL = M.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
N_ROUTES = T * TOPK

# recv_x/recv_aux laid out [expert, source, slot], flattened to
# [N_LOCAL * RECV_MAX, D]. Lane (e, src, slot) flat row = e * RECV_MAX +
# src * MAX_PER_SRC + slot. One source sends <= T rows to a local expert.
MAX_PER_SRC = T
AUX_PAD = 8  # FP32 pack tile width (32 B min tile); cols: 0=scale 1=weight
AUX_SCALE = 0
AUX_W = 1
IDX_PAD = 8  # INT32 route tile width; route rides a separate window from scale/w
             # (an FP32 tile can't hold it: INDEX->FP32 casts are unsupported).

# Recipes-style prefill transport.  One source may legally send all of its
# routes to one destination, so each destination lane has T * TOPK rows.  The
# receiver owns one such lane per source.  Only live rows cross the wire via
# all_to_all_v; the static capacity is the dropless upper bound, not padding
# that should be computed over.
COMPACT_ROUTES_PER_SRC = T * TOPK
COMPACT_PEER_CAP = COMPACT_ROUTES_PER_SRC
COMPACT_TOTAL_CAP = N_RANKS * COMPACT_PEER_CAP
COMPACT_SCALE_PAD = 8  # one 32-byte FP32 transport row; col 0 is the scale
COMPACT_EXPERT_SCALE_PAD = 16  # one 64-byte line per receiver expert row
COMPACT_ROUTE_MAP_PAD = 16  # one 64-byte INT32 line per source-side route
COMPACT_WEIGHT_PAD = 16  # one 32-byte BF16 finalize-routing weight row
COMPACT_RETURN_ROWS_PER_BLOCK = 128
COMPACT_FINALIZE_TOKEN_TILE = 16
COMPACT_GROUPED_EXPERT_TILE = 16
COMPACT_GROUPED_TOTAL_CAP = (
    COMPACT_TOTAL_CAP + N_LOCAL * (COMPACT_GROUPED_EXPERT_TILE - 1)
)

PREFILL_INPUT_ID_TILE = 4

assert N_RANKS in _EP_CHOICES, f"--ep must be one of {_EP_CHOICES} (got {N_RANKS})"
assert N_EXPERTS_GLOBAL == N_RANKS * N_LOCAL
assert RECV_MAX == N_RANKS * MAX_PER_SRC


def check_compact_slab() -> None:
    """Shape preconditions of the compact prefill transport.

    T is the importing entry's MoE slab, and a decode entry carries a slab
    these prefill tiles do not divide. Importing this module must therefore
    stay free of them; an entry that builds the compact graph calls this at
    its own module scope instead.
    """
    assert TOPK == 6
    assert N_LOCAL == 32
    assert COMPACT_PEER_CAP % COMPACT_RETURN_ROWS_PER_BLOCK == 0
    assert T % COMPACT_FINALIZE_TOKEN_TILE == 0
    assert COMPACT_GROUPED_TOTAL_CAP % COMPACT_GROUPED_EXPERT_TILE == 0


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
def clear_compact_moe_signals(
    completion_anchor: pl.Tensor[[1, 1, 8], pl.FP32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """clear_moe_signals for the compact transport's three credit banks.

    Same contract and the same three roles -- counts, payload, combine -- over
    the compact windows and the compact path's small completion anchor.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear"):
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(count_signal, [src, 0], zero)
            pl.write(x_signal, [src, 0], zero)
            pl.write(reverse_signal, [src, 0], zero)


# === Dispatch ================================================================
# Lane push, count publish, arrival wait, and cumsum gather run in one
# pl.at(CORE_GROUP) so program order stays push -> notify -> wait -> gather.
@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    # compact per-expert outputs consumed by expert_routed / combine
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
):
    # Flat 2-D view kept outside the scope so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])

    # Meta and payload arrivals ride two independent windows (`arrived` /
    # `data_arrived`), so the two phases barrier separately and overlap freely.

    # Count routes, publish counts, barrier on meta, cumsum -> recv_count_out.
    # Needs every source's counts but none of the bulk payload.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta",
        allow_early_resolve=True,
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
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
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
                    target=arrived,
                    peer=dst,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        # Wait for every source's meta flag.
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=arrived,
                    offsets=[src, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        # Cumsum recv_meta over sources -> per-expert receive count. The host reads
        # recv_count_out to size the routed-expert tile loop, so producing it here
        # lets routed matmuls submit while the payload is still moving.
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Move the bulk payload (x / aux / route) to each destination lane.
    # Split over LOCAL EXPERT INDEX (N_LOCAL blocks): block loc_e handles expert
    # loc_e on EVERY destination rank, so the blocking cross-rank puts fan out
    # across N_LOCAL cores. One slot counter per destination rank; token-major
    # order matches the meta pass's per-(dst, loc_e) cumulative count.
    with pl.spmd(N_LOCAL, name_hint="dispatch_push", allow_early_resolve=True) as _push_tid:
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

        # Pad tiles zeroed once; used cols overwritten per push, then remote_store.
        aux_tile = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
        route_tile = pl.tile.full([1, IDX_PAD], dtype=pl.INT32, value=0)
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL
                le = eid - dst * N_LOCAL
                if le == loc_e:
                    slot = slot_ctr[dst]
                    slot_ctr[dst] = slot + 1
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
                    pl.tile.write(aux_tile, [0, AUX_SCALE], pl.read(x_norm_scale, [t, 0]))
                    pl.tile.write(aux_tile, [0, AUX_W], pl.read(weights, [t, k]))
                    pld.tile.remote_store(aux_tile, target=recv_aux, peer=dst, offsets=[row, 0])
                    pl.tile.write(route_tile, [0, 0], pl.cast(t * TOPK + k, pl.INT32))
                    pld.tile.remote_store(route_tile, target=recv_route, peer=dst, offsets=[row, 0])

        # Payload-arrival notify folded into the push: each block signals every peer
        # after its own puts, so a peer sees N_LOCAL notifies per source per epoch
        # and the wait below expects N_LOCAL * moe_epoch. Saves the launch of a
        # separate post-push notify task. Each block bumps the count only after its
        # own puts issue in program order, which is what gates the gather -- recv_aux
        # / recv_route ride a non-draining remote_store and a PIPE_ALL barrier is not
        # a cross-rank DDR fence (PTOAS#872).
        for dst in pl.range(N_RANKS):
            if dst != my_rank:
                pld.system.notify(
                    target=data_arrived,
                    peer=dst,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_wait",
        deps=[_push_tid],
        allow_early_resolve=True,
    ) as _wait_tid:
        # Anchor the blocking wait to the local routing producer.
        _idx_anchor = pl.read(indices, [0, 0])
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=data_arrived,
                    offsets=[src, 0],
                    expected=pl.cast(moe_epoch * N_LOCAL, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Gather lanes into the compact per-expert buffers: one SPMD block per local
    # expert. deps on _wait_tid for the incoming payload; this rank's own
    # dst == my_rank puts are already ordered by the local RAW edges on
    # recv_x / recv_aux / recv_route. deps on _meta_tid for recv_meta_local, which is
    # manual_dep and so has no auto edge from the cumsum.
    with pl.spmd(
        N_LOCAL,
        name_hint="dispatch_gather",
        deps=[_wait_tid, _meta_tid],
        # Keep the routed expert tasks off the cores until this gather retires.
        allow_early_resolve=False,
    ) as _gather_tid:
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
                recv_x_out_flat[out_row : out_row + 1, :] = recv_x[in_row : in_row + 1, :]
                pl.write(recv_scale_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_SCALE]))
                pl.write(recv_w_out, [e, out_col], pl.read(recv_aux, [in_row, AUX_W]))
                pl.write(recv_r_route_out, [e, out_col], pl.read(recv_route, [in_row, 0]))
            b = b + n

    return _push_tid


# === Prefill compact dispatch ===============================================
@pl.jit.inline
def dispatch_compact(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    # Receiver-side expert-major tensors.  Only sum(expert_counts_out) leading
    # rows are live; consumers must never compute over the static tail.
    expert_x_out: pl.Tensor[[COMPACT_TOTAL_CAP, D], pl.INT8],
    expert_scale_out: pl.Tensor[
        [COMPACT_TOTAL_CAP, COMPACT_EXPERT_SCALE_PAD], pl.FP32
    ],
    expert_counts_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_expert_counts_out: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    # Source-side state retained for the reverse exchange/combine.  The
    # top-k weights remain source-local and are deliberately not transported.
    send_counts_out: pl.Tensor[[N_RANKS, 1], pl.INT32],
    route_to_packed_out: pl.Tensor[
        [COMPACT_ROUTES_PER_SRC, COMPACT_ROUTE_MAP_PAD], pl.INT32
    ],
    # Counts and the INT8/FP32 payload rails use independent transport windows.
    # The scale rail shares the payload's credit bank, so it needs no signal of
    # its own.
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_target: pld.DistributedTensor[[COMPACT_TOTAL_CAP, D], pl.INT8],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scale_target: pld.DistributedTensor[
        [COMPACT_TOTAL_CAP, COMPACT_SCALE_PAD], pl.FP32
    ],
    num_tokens: pl.Scalar[pl.INT32],
    prior_dep: pl.Scalar[pl.TASK_ID],
    # Monotonic per-forward epoch for the manual transport's barriers, shared
    # with the reverse exchange's convention.
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Recipe-compatible count/payload dispatch and receiver expert re-sort.

    Stable destination/expert packing plus the exchanged ``[source, expert]``
    count matrix makes expert-id and route-id payload rails unnecessary.  The
    only data crossing EP are INT8 hidden rows and their FP32 dequant scales.
    ``route_to_packed_out`` stays on the source for the reverse combine.
    """

    send_expert_counts = pl.create_tensor(
        [N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True
    )

    # Static scratch is capacity-sized for dropless routing, but only the
    # runtime counts are transferred and later consumed.
    x_send = pl.create_tensor(
        [COMPACT_TOTAL_CAP, D], dtype=pl.INT8, manual_dep=True
    )
    scale_send = pl.create_tensor(
        [COMPACT_TOTAL_CAP, COMPACT_SCALE_PAD],
        dtype=pl.FP32,
        manual_dep=True,
    )

    # Count once in a single InCore task.  This avoids cache-line lost updates
    # on the compact INT32 count rows and gives packing stable expert prefixes.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="compact_dispatch_count",
        allow_early_resolve=True,
    ) as count_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        counts = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for dst in pl.range(N_RANKS):
            for local_e in pl.range(N_LOCAL):
                counts[dst * N_LOCAL + local_e] = 0

        for token in pl.range(active_tokens):
            for topk in pl.range(TOPK):
                expert = pl.read(indices, [token, topk])
                dst = expert // N_LOCAL
                local_e = expert - dst * N_LOCAL
                lane = dst * N_LOCAL + local_e
                counts[lane] = counts[lane] + 1

        for dst in pl.range(N_RANKS):
            rank_total = pl.const(0, pl.INT32)
            for local_e in pl.range(N_LOCAL):
                count = counts[dst * N_LOCAL + local_e]
                pl.write(send_expert_counts, [dst, local_e], count)
                rank_total = rank_total + count
            pl.write(send_counts_out, [dst, 0], rank_total)

    # One task per destination builds the source's compact lane in
    # destination->expert->token/topk stable order.  Every route is also mapped
    # to its row in the Recipe-style concatenation of the live rank splits.
    with pl.spmd(
        N_RANKS,
        name_hint="compact_dispatch_pack",
        deps=[count_tid],
        allow_early_resolve=True,
    ) as pack_tid:
        dst_block = pl.tile.get_block_idx()
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        cursor = pl.array.create(N_LOCAL, pl.INT32)
        scale_row = pl.tile.full(
            [1, COMPACT_SCALE_PAD], dtype=pl.FP32, value=0.0
        )
        route_map_row = pl.tile.full(
            [1, COMPACT_ROUTE_MAP_PAD], dtype=pl.INT32, value=0
        )
        compact_rank_base = pl.const(0, pl.INT32)
        for prior_dst in pl.range(dst_block):
            compact_rank_base = compact_rank_base + pl.read(
                send_counts_out, [prior_dst, 0]
            )
        prefix = pl.const(0, pl.INT32)
        for local_e in pl.range(N_LOCAL):
            cursor[local_e] = prefix
            prefix = prefix + pl.read(
                send_expert_counts, [dst_block, local_e]
            )

        lane_base = dst_block * COMPACT_PEER_CAP
        for token in pl.range(active_tokens):
            for topk in pl.range(TOPK):
                expert = pl.read(indices, [token, topk])
                dst = expert // N_LOCAL
                if dst == dst_block:
                    local_e = expert - dst * N_LOCAL
                    slot_i32 = cursor[local_e]
                    slot = pl.cast(slot_i32, pl.INDEX)
                    packed_row = lane_base + slot
                    x_send[packed_row : packed_row + 1, :] = x_norm_i8[
                        token : token + 1, :
                    ]
                    pl.tile.write(
                        scale_row,
                        [0, 0],
                        pl.read(x_norm_scale, [token, 0]),
                    )
                    pl.tile.store(
                        scale_row, [packed_row, 0], scale_send
                    )
                    route = token * TOPK + topk
                    pl.tile.write(
                        route_map_row,
                        [0, 0],
                        compact_rank_base + slot_i32,
                    )
                    pl.tile.store(
                        route_map_row,
                        [route, 0],
                        route_to_packed_out,
                    )
                    cursor[local_e] = slot_i32 + 1

    # Counts and both payload rails ride the same hand-written transport the
    # reverse exchange uses: per-destination puts into the peer's own lane,
    # then a monotonic-epoch notify/wait. Serialize count -> payload like
    # Recipes: besides providing the authoritative receive splits, completing
    # the small count exchange before the payload prevents its scalar consumer
    # from racing a capacity-sized payload rail on CP8.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="compact_dispatch_count_exchange",
        # prior_dep (the caller's attention-done anchor) transitively
        # post-dominates the PREVIOUS layer's combine on this rank. Without it
        # the scheduler may start this layer's exchange on the shared
        # count/payload signal banks while the previous layer's credit barrier
        # is still in flight on a peer — a cross-rank deadlock invisible to
        # local dataflow (cf. #1076).
        deps=[count_tid, prior_dep],
    ) as count_exchange_tid:
        my_rank = pld.system.rank(pld.system.get_comm_ctx(count_target))
        count_row = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
        for dest in pl.range(N_RANKS):
            for local_e in pl.range(N_LOCAL):
                pl.tile.write(
                    count_row,
                    [0, local_e],
                    pl.read(send_expert_counts, [dest, local_e]),
                )
            pld.tile.remote_store(
                count_row, target=count_target, peer=dest, offsets=[my_rank, 0]
            )
            if dest != my_rank:
                pld.system.notify(
                    target=count_signal,
                    peer=dest,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=count_signal,
                    offsets=[src, 0],
                    expected=epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        for local_e in pl.range(N_LOCAL):
            expert_total = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(count_target, [src, local_e])
                pl.write(recv_expert_counts_out, [src, local_e], count)
                expert_total = expert_total + count
            pl.write(expert_counts_out, [local_e, 0], expert_total)

    # One SPMD block per destination pushes its live rows into peer dest's
    # lane, one static row-sized put per rail per row. The scale rail rides the
    # same credit as the payload: it is stored before the notify in program
    # order, exactly as dispatch()'s aux/route rails ride data_arrived.
    with pl.spmd(
        N_RANKS,
        name_hint="compact_dispatch_payload_put",
        deps=[pack_tid, count_exchange_tid],
        allow_early_resolve=False,
    ) as payload_put_tid:
        dest = pl.tile.get_block_idx()
        my_rank = pld.system.rank(pld.system.get_comm_ctx(x_target))
        n_rows = pl.cast(pl.read(send_counts_out, [dest, 0]), pl.INDEX)
        if n_rows < 0:
            n_rows = pl.cast(0, pl.INDEX)
        if n_rows > COMPACT_PEER_CAP:
            n_rows = pl.cast(COMPACT_PEER_CAP, pl.INDEX)
        src_base = dest * COMPACT_PEER_CAP
        dst_base = pl.cast(my_rank, pl.INDEX) * COMPACT_PEER_CAP
        for row in pl.range(n_rows):
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
                shape=[1, COMPACT_SCALE_PAD],
            )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="compact_dispatch_payload_wait",
        deps=[payload_put_tid],
    ) as payload_exchange_tid:
        my_rank = pld.system.rank(pld.system.get_comm_ctx(x_target))
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=x_signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=x_signal,
                    offsets=[src, 0],
                    expected=epoch,
                    cmp=pld.WaitCmp.Ge,
                )

    # Count-driven copies trim every source block before reading the transport
    # windows.  Output order is expert-major, then source-major, preserving the
    # source's token/topk order inside each expert.
    with pl.spmd(
        N_LOCAL,
        name_hint="compact_dispatch_resort",
        deps=[count_exchange_tid, payload_exchange_tid],
        allow_early_resolve=False,
    ) as resort_tid:
        local_e = pl.tile.get_block_idx()
        expert_scale_row = pl.tile.full(
            [1, COMPACT_EXPERT_SCALE_PAD],
            dtype=pl.FP32,
            value=0.0,
        )

        expert_base = pl.cast(0, pl.INDEX)
        for prior_e in pl.range(local_e):
            expert_base = expert_base + pl.cast(
                pl.read(expert_counts_out, [prior_e, 0]), pl.INDEX
            )

        source_prefix = pl.cast(0, pl.INDEX)
        for src in pl.range(N_RANKS):
            source_expert_base = pl.cast(0, pl.INDEX)
            for prior_e in pl.range(local_e):
                source_expert_base = source_expert_base + pl.cast(
                    pl.read(recv_expert_counts_out, [src, prior_e]),
                    pl.INDEX,
                )
            n_rows = pl.cast(
                pl.read(recv_expert_counts_out, [src, local_e]),
                pl.INDEX,
            )
            input_base = src * COMPACT_PEER_CAP + source_expert_base
            output_base = expert_base + source_prefix
            for row in pl.range(n_rows):
                input_row = input_base + row
                output_row = output_base + row
                expert_x_out[output_row : output_row + 1, :] = x_target[
                    input_row : input_row + 1, :
                ]
                pl.tile.write(
                    expert_scale_row,
                    [0, 0],
                    pl.read(scale_target, [input_row, 0]),
                )
                pl.tile.store(
                    expert_scale_row,
                    [output_row, 0],
                    expert_scale_out,
                )
            source_prefix = source_prefix + n_rows

    return resort_tid


@pl.jit.inline
def _reverse_exchange(
    reverse_send: pl.Tensor[[COMPACT_TOTAL_CAP, D], pl.BF16],
    reverse_target: pld.DistributedTensor[[COMPACT_TOTAL_CAP, D], pl.BF16],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    reverse_send_counts: pl.Tensor[[N_RANKS, 1], pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    resort_tid: pl.Scalar[pl.TASK_ID],
    counts_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    # One SPMD block per destination: block dest pushes its live rows back to
    # peer dest's lane, one static [1, D] put per row, then a monotonic-epoch
    # notify/wait publishes the whole grid.
    # pypto all_to_all_v sizes its transfer by the runtime row count, and a
    # partial extent deadlocks 8 ranks under skewed counts (pypto#2536).
    with pl.spmd(
        N_RANKS,
        name_hint="compact_combine_y_put",
        deps=[resort_tid, counts_tid],
        allow_early_resolve=False,
    ) as reverse_put_tid:
        dest = pl.tile.get_block_idx()
        my_rank = pld.system.rank(pld.system.get_comm_ctx(reverse_target))
        n_rows = pl.cast(pl.read(reverse_send_counts, [dest, 0]), pl.INDEX)
        if n_rows < 0:
            n_rows = pl.cast(0, pl.INDEX)
        if n_rows > COMPACT_PEER_CAP:
            n_rows = pl.cast(COMPACT_PEER_CAP, pl.INDEX)
        src_base = dest * COMPACT_PEER_CAP
        dst_base = pl.cast(my_rank, pl.INDEX) * COMPACT_PEER_CAP
        for row in pl.range(n_rows):
            pld.tensor.put(
                dst=reverse_target,
                peer=dest,
                src=reverse_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[1, D],
            )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="compact_combine_y_wait",
        deps=[reverse_put_tid],
    ) as reverse_exchange_tid:
        my_rank = pld.system.rank(pld.system.get_comm_ctx(reverse_target))
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=reverse_signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=reverse_signal,
                    offsets=[src, 0],
                    expected=epoch,
                    cmp=pld.WaitCmp.Ge,
                )
    return reverse_exchange_tid


@pl.jit.inline
def combine_compact(
    expert_y: pl.Tensor[[COMPACT_TOTAL_CAP, D], pl.BF16],
    expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_expert_counts: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    route_to_packed: pl.Tensor[
        [COMPACT_ROUTES_PER_SRC, COMPACT_ROUTE_MAP_PAD], pl.INT32
    ],
    forward_send_counts: pl.Tensor[[N_RANKS, 1], pl.INT32],
    # Recipe casts the source-local FP32 top-k weights to hidden dtype before
    # finalize-routing.  PTOAS 0.60 needs an aligned row for the later tile
    # load, so cols [0:TOPK] are live and the padded tail is ignored.
    weights_bf16: pl.Tensor[[T, COMPACT_WEIGHT_PAD], pl.BF16],
    shared_y: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    returned_y: pl.Tensor[[COMPACT_ROUTES_PER_SRC, D], pl.BF16],
    reverse_target: pld.DistributedTensor[
        [COMPACT_TOTAL_CAP, D], pl.BF16
    ],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    dispatch_tid: pl.Scalar[pl.TASK_ID],
    expert_tid: pl.Scalar[pl.TASK_ID],
    # Monotonic per-forward epoch for the manual transport's barrier (the
    # collective's own barrier is self-clearing and ignores it).
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Recipe-compatible reverse exchange and source-side finalize routing.

    ``expert_y`` is expert-major, then source-major within each expert.  It is
    restored to the forward collective's source-major order before the reverse
    exchange returns each row to the rank that routed it.  Its fixed peer lanes
    are compacted to the Recipe's concatenated live output splits, then the
    source-local packed route map selects rows for the final top-k weighted
    sum.  No route, expert, or weight metadata crosses EP.
    """

    reverse_send = pl.create_tensor(
        [COMPACT_TOTAL_CAP, D], dtype=pl.BF16, manual_dep=True
    )
    reverse_send_counts = pl.create_tensor(
        [N_RANKS, 1], dtype=pl.INT32, manual_dep=True
    )

    # Row count per reverse destination: on an expert rank each destination is
    # one original source, so it receives the sum across this rank's local
    # experts.  The exchange below walks exactly these counts.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="compact_combine_counts",
        deps=[dispatch_tid],
        allow_early_resolve=True,
    ) as reverse_counts_tid:
        for src in pl.range(N_RANKS):
            source_total = pl.const(0, pl.INT32)
            for local_e in pl.range(N_LOCAL):
                source_total = source_total + pl.read(
                    recv_expert_counts, [src, local_e]
                )
            pl.write(reverse_send_counts, [src, 0], source_total)

    # Invert dispatch_compact's receiver re-sort.  Each task owns the complete
    # fixed-capacity lane for one original source; only its live prefix is
    # written and subsequently transferred.
    with pl.spmd(
        N_RANKS,
        name_hint="compact_combine_inverse_resort",
        deps=[dispatch_tid, expert_tid],
        allow_early_resolve=False,
    ) as inverse_resort_tid:
        src = pl.tile.get_block_idx()
        send_base = src * COMPACT_PEER_CAP
        send_prefix = pl.cast(0, pl.INDEX)
        expert_base = pl.cast(0, pl.INDEX)
        for local_e in pl.range(N_LOCAL):
            source_prefix = pl.cast(0, pl.INDEX)
            for prior_src in pl.range(src):
                source_prefix = source_prefix + pl.cast(
                    pl.read(recv_expert_counts, [prior_src, local_e]),
                    pl.INDEX,
                )
            n_rows = pl.cast(
                pl.read(recv_expert_counts, [src, local_e]), pl.INDEX
            )
            input_base = expert_base + source_prefix
            output_base = send_base + send_prefix
            for row in pl.range(n_rows):
                input_row = input_base + row
                output_row = output_base + row
                reverse_send[output_row : output_row + 1, :] = expert_y[
                    input_row : input_row + 1, :
                ]
            send_prefix = send_prefix + n_rows
            expert_base = expert_base + pl.cast(
                pl.read(expert_counts, [local_e, 0]), pl.INDEX
            )

    reverse_exchange_tid = _reverse_exchange(
        reverse_send,
        reverse_target,
        reverse_signal,
        reverse_send_counts,
        epoch,
        inverse_resort_tid,
        reverse_counts_tid,
    )

    # PTO all_to_all_v receives into fixed peer-capacity lanes.  Recipe's
    # all_to_all_single instead returns the known output splits concatenated
    # into exactly T*TOPK rows.  Compact the communication staging window to
    # that source-local layout before finalize-routing.
    return_blocks_per_rank = (
        COMPACT_PEER_CAP // COMPACT_RETURN_ROWS_PER_BLOCK
    )
    with pl.spmd(
        N_RANKS * return_blocks_per_rank,
        name_hint="compact_combine_return_compact",
        deps=[reverse_exchange_tid],
        allow_early_resolve=False,
    ) as return_compact_tid:
        block = pl.tile.get_block_idx()
        expert_rank = block // return_blocks_per_rank
        chunk = block - expert_rank * return_blocks_per_rank
        row0 = chunk * COMPACT_RETURN_ROWS_PER_BLOCK
        compact_base = pl.cast(0, pl.INDEX)
        for prior_rank in pl.range(expert_rank):
            compact_base = compact_base + pl.cast(
                pl.read(forward_send_counts, [prior_rank, 0]), pl.INDEX
            )
        n_rows = pl.cast(
            pl.read(forward_send_counts, [expert_rank, 0]), pl.INDEX
        )
        staging_base = expert_rank * COMPACT_PEER_CAP
        if row0 < n_rows:
            chunk_rows = n_rows - row0
            if chunk_rows > COMPACT_RETURN_ROWS_PER_BLOCK:
                chunk_rows = COMPACT_RETURN_ROWS_PER_BLOCK
            for row in pl.range(chunk_rows):
                returned_y[
                    compact_base + row0 + row : compact_base + row0 + row + 1,
                    :,
                ] = reverse_target[
                    staging_base + row0 + row : staging_base + row0 + row + 1,
                    :,
                ]

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    # Recipe finalize-routing semantics: source-local shared expert plus the
    # six returned expert rows weighted in FP32, rounded once to BF16.
    with pl.spmd(
        T // COMPACT_FINALIZE_TOKEN_TILE,
        name_hint="compact_combine_finalize",
        deps=[return_compact_tid],
    ) as finalize_tid:
        token0 = pl.tile.get_block_idx() * COMPACT_FINALIZE_TOKEN_TILE
        for lane in pl.range(COMPACT_FINALIZE_TOKEN_TILE):
            token = token0 + lane
            if token < active_tokens:
                acc = pl.cast(shared_y[token : token + 1, :], pl.FP32)
                weight_row = pl.cast(
                    pl.tile.load(
                        weights_bf16,
                        [token, 0],
                        [1, COMPACT_WEIGHT_PAD],
                    ),
                    pl.FP32,
                )
                for topk in pl.range(TOPK):
                    route = token * TOPK + topk
                    packed_row = pl.cast(
                        pl.read(route_to_packed, [route, 0]), pl.INDEX
                    )
                    route_y = pl.cast(
                        returned_y[
                            packed_row : packed_row + 1, 0:D
                        ],
                        pl.FP32,
                    )
                    route_weight = pl.tile.read(weight_row, [0, topk])
                    acc = pl.add(acc, pl.mul(route_y, route_weight))
                ffn_out[token : token + 1, :] = pl.cast(
                    acc, pl.BF16, mode="rint"
                )
            else:
                ffn_out[token : token + 1, :] = shared_y[
                    token : token + 1, :
                ]

    return finalize_tid


@pl.jit.inline
def _compact_pack_grouped_experts(
    dense_x: pl.Tensor[[COMPACT_TOTAL_CAP, D], pl.INT8],
    dense_scale: pl.Tensor[
        [COMPACT_TOTAL_CAP, COMPACT_EXPERT_SCALE_PAD], pl.FP32
    ],
    expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    grouped_x: pl.Tensor[[COMPACT_GROUPED_TOTAL_CAP, D], pl.INT8],
    grouped_scale: pl.Tensor[
        [COMPACT_GROUPED_TOTAL_CAP, COMPACT_EXPERT_SCALE_PAD], pl.FP32
    ],
    dispatch_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Pad each dense expert slab to a private 16-row compute boundary."""
    with pl.spmd(
        N_LOCAL,
        name_hint="compact_grouped_pack",
        deps=[dispatch_tid],
        allow_early_resolve=False,
    ) as pack_tid:
        local_e = pl.tile.get_block_idx()
        dense_base = pl.cast(0, pl.INDEX)
        grouped_base = pl.cast(0, pl.INDEX)
        for prior_e in pl.range(local_e):
            prior_rows = pl.cast(
                pl.read(expert_counts, [prior_e, 0]), pl.INDEX
            )
            dense_base = dense_base + prior_rows
            grouped_base = grouped_base + (
                (prior_rows + COMPACT_GROUPED_EXPERT_TILE - 1)
                // COMPACT_GROUPED_EXPERT_TILE
            ) * COMPACT_GROUPED_EXPERT_TILE

        n_rows = pl.cast(
            pl.read(expert_counts, [local_e, 0]), pl.INDEX
        )
        grouped_rows = (
            (n_rows + COMPACT_GROUPED_EXPERT_TILE - 1)
            // COMPACT_GROUPED_EXPERT_TILE
        ) * COMPACT_GROUPED_EXPERT_TILE
        zero_x = pl.cast(
            pl.full([1, D], dtype=pl.FP16, value=0.0),
            pl.INT8,
            mode="trunc",
        )
        zero_scale = pl.full(
            [1, COMPACT_EXPERT_SCALE_PAD], dtype=pl.FP32, value=0.0
        )
        for row in pl.range(grouped_rows):
            grouped_row = grouped_base + row
            if row < n_rows:
                dense_row = dense_base + row
                grouped_x[grouped_row : grouped_row + 1, :] = dense_x[
                    dense_row : dense_row + 1, :
                ]
                grouped_scale[
                    grouped_row : grouped_row + 1, :
                ] = dense_scale[dense_row : dense_row + 1, :]
            else:
                grouped_x[grouped_row : grouped_row + 1, :] = zero_x
                grouped_scale[grouped_row : grouped_row + 1, :] = zero_scale
    return pack_tid


@pl.jit.inline
def _compact_unpack_grouped_experts(
    grouped_y: pl.Tensor[[COMPACT_GROUPED_TOTAL_CAP, D], pl.BF16],
    expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    dense_y: pl.Tensor[[COMPACT_TOTAL_CAP, D], pl.BF16],
    expert_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Remove compute-only row padding before the reverse exchange."""
    with pl.spmd(
        N_LOCAL,
        name_hint="compact_grouped_unpack",
        deps=[expert_tid],
        allow_early_resolve=False,
    ) as unpack_tid:
        local_e = pl.tile.get_block_idx()
        dense_base = pl.cast(0, pl.INDEX)
        grouped_base = pl.cast(0, pl.INDEX)
        for prior_e in pl.range(local_e):
            prior_rows = pl.cast(
                pl.read(expert_counts, [prior_e, 0]), pl.INDEX
            )
            dense_base = dense_base + prior_rows
            grouped_base = grouped_base + (
                (prior_rows + COMPACT_GROUPED_EXPERT_TILE - 1)
                // COMPACT_GROUPED_EXPERT_TILE
            ) * COMPACT_GROUPED_EXPERT_TILE

        n_rows = pl.cast(
            pl.read(expert_counts, [local_e, 0]), pl.INDEX
        )
        for row in pl.range(n_rows):
            dense_y[dense_base + row : dense_base + row + 1, :] = grouped_y[
                grouped_base + row : grouped_base + row + 1, :
            ]
    return unpack_tid


@pl.jit.inline(auto_scope=False)
def prefill_moe_compact_grouped_resident(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w13: pl.Tensor[
        [N_LOCAL, 2 * MOE_INTER, D], pl.INT8
    ],
    routed_w13_scale: pl.Tensor[[N_LOCAL, 2 * MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[
        [N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    smooth_scale_2: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    # Caller-owned, layer-reused workspaces.  The multi-layer forward keeps
    # these resident, matching the DSpark prefill ownership contract instead
    # of allocating capacity-sized tensors in every per-layer scope.
    x_mixed: pl.InOut[pl.Tensor[[T, D], pl.BF16]],
    post_ffn: pl.InOut[pl.Tensor[[T, HC_MULT], pl.FP32]],
    comb_ffn: pl.InOut[pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32]],
    ffn_out: pl.InOut[pl.Tensor[[T, D], pl.BF16]],
    dense_x: pl.InOut[pl.Tensor[[COMPACT_TOTAL_CAP, D], pl.INT8]],
    dense_scale: pl.InOut[
        pl.Tensor[[COMPACT_TOTAL_CAP, COMPACT_EXPERT_SCALE_PAD], pl.FP32]
    ],
    grouped_x: pl.InOut[
        pl.Tensor[[COMPACT_GROUPED_TOTAL_CAP, D], pl.INT8]
    ],
    grouped_scale: pl.InOut[
        pl.Tensor[
            [COMPACT_GROUPED_TOTAL_CAP, COMPACT_EXPERT_SCALE_PAD], pl.FP32
        ]
    ],
    grouped_y: pl.InOut[
        pl.Tensor[[COMPACT_GROUPED_TOTAL_CAP, D], pl.BF16]
    ],
    dense_y: pl.InOut[pl.Tensor[[COMPACT_TOTAL_CAP, D], pl.BF16]],
    returned_y: pl.InOut[
        pl.Tensor[[COMPACT_ROUTES_PER_SRC, D], pl.BF16]
    ],
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_target: pld.DistributedTensor[[COMPACT_TOTAL_CAP, D], pl.INT8],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scale_target: pld.DistributedTensor[
        [COMPACT_TOTAL_CAP, COMPACT_SCALE_PAD], pl.FP32
    ],
    reverse_target: pld.DistributedTensor[
        [COMPACT_TOTAL_CAP, D], pl.BF16
    ],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prior_dep: pl.Scalar[pl.TASK_ID],
    layer_id: pl.Scalar[pl.INT32],
    # 1-based MoE call id within this forward, like the wave path's moe_epoch:
    # the reverse exchange waits for `>= moe_epoch` credits on a window shared
    # by every call, so it counts calls, not layers.
    moe_epoch: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one shape-configured production prefill MoE slab.

    The strict CP8/EP8 production caller configures ``config.MOE_TOKENS`` to
    its 1024 local rows before importing this module. EP2/EP4 and T=128 remain
    valid compile/correctness gates because all capacities are shape-derived.
    The production contract routes the complete physical slab, so callers pass
    ``num_tokens == T``; an inactive physical tail is not initialized here.

    This path implements compact count/A2Av transport, 16-row-aligned grouped
    routed experts, Recipes ``smooth_scale_2`` before the W2 activation
    quantizer, and source-side BF16 top-k weighting. It deliberately still
    shares one token-wise INT8 input/scale across the six routes. Recipes'
    expert-specific ``smooth_scale_1`` input quantization is therefore a
    follow-up quantization-protocol gap, not claimed as aligned here.
    """
    prefill_hc_pre(
        x_hc,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        x_mixed,
        post_ffn,
        comb_ffn,
        prior_dep,
    )

    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
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

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    shared_y = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8,
        x_norm_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        shared_y,
    )

    # Recipes casts source-local top-k weights to hidden dtype immediately
    # before finalize-routing. Keep the weights local and pad each row only for
    # the PTOAS 0.60 aligned load used by combine_compact.
    weights_bf16 = pl.create_tensor(
        [T, COMPACT_WEIGHT_PAD], dtype=pl.BF16
    )
    with pl.spmd(T, name_hint="compact_grouped_weight_pack"):
        token = pl.tile.get_block_idx()
        weight_row_fp32 = pl.tile.full(
            [1, COMPACT_WEIGHT_PAD], dtype=pl.FP32, value=0.0
        )
        if token < active_tokens:
            for topk in pl.range(TOPK):
                pl.tile.write(
                    weight_row_fp32,
                    [0, topk],
                    pl.read(weights, [token, topk]),
                )
        # PTOAS 0.60 can lower the vector FP32->BF16 conversion, while the
        # equivalent scalar cast reaches an unsupported AscendC backend path.
        weight_row = pl.cast(weight_row_fp32, pl.BF16, mode="rint")
        pl.tile.store(weight_row, [token, 0], weights_bf16)

    expert_counts = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_expert_counts = pl.create_tensor(
        [N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True
    )
    send_counts = pl.create_tensor(
        [N_RANKS, 1], dtype=pl.INT32, manual_dep=True
    )
    route_to_packed = pl.create_tensor(
        [COMPACT_ROUTES_PER_SRC, COMPACT_ROUTE_MAP_PAD],
        dtype=pl.INT32,
        manual_dep=True,
    )
    dispatch_tid = dispatch_compact(
        indices,
        x_norm_i8,
        x_norm_scale,
        dense_x,
        dense_scale,
        expert_counts,
        recv_expert_counts,
        send_counts,
        route_to_packed,
        count_target,
        count_signal,
        x_target,
        x_signal,
        scale_target,
        num_tokens,
        prior_dep,
        moe_epoch,
    )

    _compact_pack_grouped_experts(
        dense_x,
        dense_scale,
        expert_counts,
        grouped_x,
        grouped_scale,
        dispatch_tid,
    )

    grouped_expert_tid = prefill_expert_grouped(
        grouped_x,
        grouped_scale,
        expert_counts,
        routed_w13,
        routed_w13_scale,
        routed_w2,
        routed_w2_scale,
        smooth_scale_2,
        grouped_y,
    )

    unpack_tid = _compact_unpack_grouped_experts(
        grouped_y, expert_counts, dense_y, grouped_expert_tid
    )

    combine_tid = combine_compact(
        dense_y,
        expert_counts,
        recv_expert_counts,
        route_to_packed,
        send_counts,
        weights_bf16,
        shared_y,
        ffn_out,
        returned_y,
        reverse_target,
        reverse_signal,
        num_tokens,
        dispatch_tid,
        unpack_tid,
        moe_epoch,
    )

    hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_moe_compact_grouped_complete",
        deps=[combine_tid],
        allow_early_resolve=False,
    ) as completion_tid:
        # The RAW edge on x_next joins the HC-post SPMD before exposing a
        # reusable completion token to the enclosing production forward.
        _completion_anchor = pl.read(x_next, [0, 0, 0])
    return completion_tid


@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    dispatch_push_tid: pl.Scalar[pl.TASK_ID],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL * RECV_MAX, D])
    # One SPMD block per LOCAL EXPERT: block e pushes expert e's compact rows back to
    # their origin rank (= the source lane src they arrived on) at their route offset.
    # Rows are src-major, so the per-(e, src) base is a loop-carried prefix sum over
    # src inside the block (same shape as dispatch_gather). Each route maps to a
    # unique (dst, loc_e) and r_route, so the blocks' puts are write-disjoint.
    with pl.spmd(N_LOCAL, name_hint="combine") as _combine_tid:
        e = pl.tile.get_block_idx()
        e_base_row = e * RECV_MAX
        b = pl.cast(0, pl.INDEX)
        for src in pl.range(N_RANKS):
            n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
            for slot in pl.range(n):
                out_col = b + slot
                r_route = pl.cast(pl.read(recv_r_route_out, [e, out_col]), pl.INDEX)
                pld.tensor.put(
                    dst=routed_y_buf,
                    peer=src,
                    src=recv_y_flat,
                    dst_offsets=[r_route, 0],
                    src_offsets=[e_base_row + out_col, 0],
                    shape=[1, D],
                )
            b = b + n

    # Publish one completion after the complete local scatter grid.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="combine_wait",
        deps=[_combine_tid, dispatch_push_tid],
    ) as _cwait_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_arrived,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=combine_arrived,
                    offsets=[src, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

    # ffn_out[t] = sh[t] + Sigma_k routed_y_buf[t*TOPK+k]. deps on combine_wait for the
    # peers' writes; this rank's own puts ride the local RAW edge on routed_y_buf,
    # which is the only thing ordering them now that the wait is off the scatter.
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    with pl.spmd(
        T,
        name_hint="shared_routed",
        deps=[_cwait_tid],
    ) as _reduce_tid:
        t = pl.tile.get_block_idx()
        if t < active_tokens:
            acc = pl.cast(sh[t:t + 1, :], target_type=pl.FP32)
            for k in pl.range(TOPK):
                r = t * TOPK + k
                acc = pl.add(acc, pl.cast(routed_y_buf[r:r + 1, :], target_type=pl.FP32))
            ffn_out[t:t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")
        else:
            ffn_out[t:t + 1, :] = sh[t:t + 1, :]


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
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
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
    # 1-based MoE call id for the shared flag windows (distinct from layer_id).
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    # Non-output intermediates allocate locally, in their producer's scope.
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32, manual_dep=True)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn, comb_ffn,
    )

    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, num_tokens, tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32, manual_dep=True)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_meta_local = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
    dispatch_push_tid = dispatch(
        indices, x_norm_i8, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out, recv_meta_local,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        num_tokens, my_rank, moe_epoch,
    )

    with pl.scope():
        recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
        expert_routed(
            recv_x_out, recv_scale_out, recv_w_out, recv_count_out,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            recv_y,
        )

        ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
        combine(
            recv_y, recv_r_route_out, sh,
            ffn_out, recv_meta_local,
            routed_y_buf, combine_arrived,
            num_tokens, my_rank, moe_epoch, dispatch_push_tid,
        )

        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next


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
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
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
    clear_moe_signals(x_next, arrived, data_arrived, combine_arrived)
    return x_next


@pl.jit.host
def l3_moe(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
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
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
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
            x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            x_next[r],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            layer_id, num_tokens, r, pl.const(1, pl.INT32),
            device=r,
        )


# === Golden + test ==========================================================
def golden_moe(tensors):
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


def build_tensor_specs(layer_id=0, num_tokens=T, balanced_routing=False):
    import torch
    from golden import ScalarSpec, TensorSpec
    from expert_routed import gen_routed_weight
    from expert_shared import gen_shared_weight

    # Routed = MXFP4 (gen_routed_weight), shared = MXFP8 (gen_shared_weight). This
    # is an integration test whose x_next-equivalent output is dominated by near-zero
    # residual+FFN cancellations, so it keeps the smaller *behaviorally-calibrated* magnitude
    # (random fixtures blow up the relative metric at the real ~2.5e-2 magnitude); only the
    # grid SHAPE (FP4/FP8 discreteness, scale CV) matches the real distribution.
    ROUTED_DEQUANT_STD = {"w1": 1.08e-2, "w2": 2.54e-2, "w3": 1.10e-2}
    SHARED_DEQUANT_STD = {"w1": 7.65e-3, "w2": 2.39e-2, "w3": 7.39e-3}

    # Shared (replicated) weights are broadcast across ranks; the routed
    # weights are per-rank shards.
    def init_x_hc():
        return torch.randn(N_RANKS, T, HC_MULT, D)

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
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_input_ids():
        if balanced_routing:
            # Active tokens across ranks consume consecutive tid2eid rows, making
            # their route ids one contiguous round-robin sequence over experts.
            rank_starts = torch.arange(N_RANKS, dtype=torch.int64).unsqueeze(1) * num_tokens
            token_offsets = torch.arange(T, dtype=torch.int64).unsqueeze(0)
            return rank_starts + token_offsets
        # Distinct per-rank token streams.
        return torch.randint(0, VOCAB, (N_RANKS, T), dtype=torch.int64)

    if balanced_routing:
        assert layer_id < M.num_hash_layers, "balanced routing requires a hash-routing layer"
        active_routes = N_RANKS * max(0, min(T, num_tokens)) * TOPK
        assert active_routes % N_EXPERTS_GLOBAL == 0, \
            "balanced routing requires the active route count to divide evenly across experts"

    # Per-rank routed expert weights (different shards).
    routed_w1_i8_list = []
    routed_w1_s_list = []
    routed_w3_i8_list = []
    routed_w3_s_list = []
    routed_w2_i8_list = []
    routed_w2_s_list = []
    for _ in range(N_RANKS):
        w1_i8, w1_s = gen_routed_weight((N_LOCAL, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
        w3_i8, w3_s = gen_routed_weight((N_LOCAL, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
        w2_i8, w2_s = gen_routed_weight((N_LOCAL, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])
        routed_w1_i8_list.append(w1_i8)
        routed_w1_s_list.append(w1_s)
        routed_w3_i8_list.append(w3_i8)
        routed_w3_s_list.append(w3_s)
        routed_w2_i8_list.append(w2_i8)
        routed_w2_s_list.append(w2_s)

    rw1_i8 = torch.stack(routed_w1_i8_list)
    rw1_s = torch.stack(routed_w1_s_list)
    rw3_i8 = torch.stack(routed_w3_i8_list)
    rw3_s = torch.stack(routed_w3_s_list)
    rw2_i8 = torch.stack(routed_w2_i8_list)
    rw2_s = torch.stack(routed_w2_s_list)

    # Shared expert weights — replicated across ranks.
    sw1_i8, sw1_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3_i8, sw3_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2_i8, sw2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)
    sw1_i8 = sw1_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw1_s = sw1_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    sw3_i8 = sw3_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw3_s = sw3_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    sw2_i8 = sw2_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw2_s = sw2_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    specs = [
        TensorSpec("x_hc",          [N_RANKS, T, HC_MULT, D],     torch.float32, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",     [N_RANKS, MIX_HC, HC_DIM],       torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale",  [N_RANKS, 3],                    torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",   [N_RANKS, MIX_HC],               torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",        [N_RANKS, D],                    torch.bfloat16,  init_value=init_norm_w),
        TensorSpec("gate_w",        [N_RANKS, N_EXPERTS_GLOBAL, D],  torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",     [N_RANKS, N_EXPERTS_GLOBAL],     torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",       [N_RANKS, VOCAB, TOPK],          torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids",     [N_RANKS, T],                 torch.int64,    init_value=init_input_ids),
        TensorSpec("routed_w1",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: rw1_i8),
        TensorSpec("routed_w1_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: rw1_s),
        TensorSpec("routed_w3",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: rw3_i8),
        TensorSpec("routed_w3_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: rw3_s),
        TensorSpec("routed_w2",        [N_RANKS, N_LOCAL, D, MOE_INTER], torch.int8,    init_value=lambda: rw2_i8),
        TensorSpec("routed_w2_scale",  [N_RANKS, N_LOCAL, D],            torch.float32, init_value=lambda: rw2_s),
        TensorSpec("shared_w1",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2",        [N_RANKS, D, MOE_INTER],          torch.int8,    init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale",  [N_RANKS, D],                     torch.float32, init_value=lambda: sw2_s),
        TensorSpec("x_next",           [N_RANKS, T, HC_MULT, D],      torch.float32),
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
    # All resident names are inputs (is_output=False), so the flag is always valid.
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


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=_EP_DEFAULT, choices=list(_EP_CHOICES),
                        help="EP world size / rank count")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T,
                        help=f"active token count for MoE dispatch/combine (0..{T})")
    parser.add_argument("--balanced-routing", action="store_true", default=False,
                        help="use deterministic hash routes balanced evenly across all experts")
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

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    golden_data = args.golden_data

    result = run(
        fn=l3_moe,
        specs=build_tensor_specs(
            layer_id=args.layer_id,
            num_tokens=args.num_tokens,
            balanced_routing=args.balanced_routing,
        ),
        golden_fn=golden_moe,
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
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            log_level=args.log_level,
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
