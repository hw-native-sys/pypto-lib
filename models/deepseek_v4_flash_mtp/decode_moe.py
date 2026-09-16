# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4-Flash decode MoE layer: hc_pre, gate, shared expert, EP dispatch, routed experts, combine, hc_post."""


# Sub-kernels freeze EP_WORLD_SIZE / n_routed_experts into their shapes at import
# time: read --ep from argv and override config before importing them below.
import dataclasses
import sys

import config

_EP_CHOICES = (2, 4, 8, 16)
_EP_DEFAULT = 2


def _parse_ep_argv():
    for i, tok in enumerate(sys.argv):
        if tok == "--ep" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith("--ep="):
            return int(tok.split("=", 1)[1])
    return _EP_DEFAULT


EP = _parse_ep_argv()

_n_routed_experts = config.FLASH.n_routed_experts // config.EP_WORLD_SIZE * EP
config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=_n_routed_experts)
config.EP_WORLD_SIZE = EP
config.RECV_MAX = EP * config.MOE_TOKENS

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

from config import FLASH as M, EP_WORLD_SIZE, MOE_TOKENS, RECV_MAX
from hc_pre import hc_pre
from hc_post import hc_post
from gate import gate
from expert_shared import expert_shared
from expert_routed import RECV_TILE, expert_routed_tile


# model config
T = MOE_TOKENS
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MOE_INTER = M.moe_intermediate_size
N_RANKS = EP_WORLD_SIZE
N_EXPERTS_GLOBAL = M.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
N_ROUTES = T * TOPK

# dispatch lanes: recv_x / recv_aux rows are [expert, source, slot], flat row
# e * RECV_MAX + src * MAX_PER_SRC + slot
MAX_PER_SRC = T
AUX_PAD = 8  # recv_aux FP32 row width
AUX_SCALE = 0
AUX_W = 1
AUX_ROUTE = 2  # route id t * TOPK + k, exact in FP32
IDX_PAD = 8  # INT32 index row width
PUSH_GROUPS = max(1, N_LOCAL // N_RANKS)  # dispatch_push blocks per destination rank

assert N_RANKS in _EP_CHOICES, f"--ep must be one of {_EP_CHOICES} (got {N_RANKS})"
# Overlapping source lanes would silently overwrite each other's rows.
assert RECV_MAX == N_RANKS * MAX_PER_SRC


# === Dispatch ================================================================
# Route counts, lane push, arrival wait, and gather into per-expert receive buffers.
@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    # per-expert outputs consumed by expert_routed_scatter
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
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id; `arrived`/`data_arrived` are monotonic so waits use `>= moe_epoch`.
    moe_epoch: pl.Scalar[pl.INT32],
):
    # Flat 2-D view kept outside the scope so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])

    # Count routes, publish counts, wait on `arrived`, cumsum -> recv_count_out.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta", allow_early_resolve=True) as _meta_tid:
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
        meta_indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(meta_indices_tile, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        # One meta row per dst (all N_LOCAL counts, zeros included), then AtomicAdd
        # the per-source arrival counter.
        meta_tile = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            pld.tile.remote_store(meta_tile, target=recv_meta, peer=dst, offsets=[my_rank, 0])
            if dst != my_rank:
                pld.system.notify(target=arrived, peer=dst, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)

        # Wait for every source's meta flag.
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(signal=arrived, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)

        # Cumsum recv_meta over sources -> per-expert receive count (sizes the routed-expert tile loop).
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Push x / aux to each destination lane, one block per (destination rank, local-expert
    # group); each (dst, loc_e) lane belongs to one block, filled in token-major order.
    with pl.spmd(N_RANKS * PUSH_GROUPS, name_hint="dispatch_push", allow_early_resolve=True) as _push_tid:
        push_block = pl.tile.get_block_idx()
        push_dst = push_block // PUSH_GROUPS
        push_group = push_block - push_dst * PUSH_GROUPS
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        slot_ctr = pl.array.create(N_LOCAL, pl.INT32)
        for e in pl.range(N_LOCAL):
            slot_ctr[e] = 0

        indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        weights_tile = pl.tile.load(weights, [0, 0], [T, AUX_PAD], valid_shape=[T, TOPK])
        # Pad tile zeroed once; used cols overwritten per push, then remote_store.
        aux_tile = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                dst = eid // N_LOCAL
                le = eid - dst * N_LOCAL
                le_group = le - (le // PUSH_GROUPS) * PUSH_GROUPS
                if dst == push_dst:
                    if le_group == push_group:
                        slot = slot_ctr[le]
                        slot_ctr[le] = slot + 1
                        # lane (le, my_rank, slot) on peer=dst
                        push_lane_base = pl.cast(le, pl.INDEX) * RECV_MAX + my_rank * MAX_PER_SRC
                        row = push_lane_base + pl.cast(slot, pl.INDEX)
                        pld.tensor.put(
                            dst=recv_x, peer=push_dst, src=x_norm_i8,
                            dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, D],
                        )
                        pl.tile.write(aux_tile, [0, AUX_SCALE], pl.read(x_norm_scale, [t, 0]))
                        pl.tile.write(aux_tile, [0, AUX_W], pl.tile.read(weights_tile, [t, k]))
                        push_route_i32 = pl.cast(t * TOPK + k, pl.INT32)
                        pl.tile.write(aux_tile, [0, AUX_ROUTE], pl.cast(push_route_i32, pl.FP32))
                        pld.tile.remote_store(aux_tile, target=recv_aux, peer=push_dst, offsets=[row, 0])

        # Notify this block's destination after its puts: a peer expects PUSH_GROUPS
        # notifies per source per epoch. Must follow the puts in program order --
        # recv_aux rides a non-draining remote_store (PTOAS#872).
        if push_dst != my_rank:
            pld.system.notify(
                target=data_arrived, peer=push_dst, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="dispatch_wait",
        deps=[_push_tid], allow_early_resolve=True,
    ) as _wait_tid:
        # Anchor the blocking wait to the local routing producer.
        _idx_anchor = pl.read(indices, [0, 0])
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=data_arrived, offsets=[src, 0],
                    expected=pl.cast(moe_epoch * PUSH_GROUPS, pl.INT32), cmp=pld.WaitCmp.Ge,
                )

    # Gather lanes into per-expert buffers, one block per local expert. deps: _wait_tid for
    # the peers' payload, _meta_tid for manual_dep recv_meta_local. No early resolve: routed
    # expert tasks stay off the cores until the gather retires.
    with pl.spmd(
        N_LOCAL, name_hint="dispatch_gather",
        deps=[_wait_tid, _meta_tid], allow_early_resolve=False,
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
                gather_route = pl.read(recv_aux, [in_row, AUX_ROUTE])
                pl.write(recv_r_route_out, [e, out_col], pl.cast(gather_route, pl.INT32))
            b = b + n

    return _push_tid


@pl.jit.inline(auto_scope=False)
def expert_routed_scatter(
    recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8],
    recv_scale_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_w_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
) -> pl.Scalar[pl.TASK_ID]:
    # Per receive tile: run the routed expert, then put each row back to its source rank at
    # its route offset. Rows are src-major; the running recv_meta_local prefix splits a tile
    # across sources.
    expert_completion_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for local_e in pl.parallel(N_LOCAL):
        tile_completion_tids = pl.array.create(RECV_MAX // RECV_TILE, pl.TASK_ID)
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
                local_e, tile_row, valid_rows,
            )
            # routed_y_buf rows are unique per route within an epoch; combine_wait orders the
            # reduction after every scatter through scatter_done.
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="expert_scatter",
                deps=[tile_ready], no_dep_args=[routed_y_buf],
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
                            dst=routed_y_buf, peer=src, src=recv_y_tile,
                            dst_offsets=[route, 0], src_offsets=[compact_row - tile_row, 0], shape=[1, D],
                        )
                    source_begin = source_end
            tile_completion_tids[tile] = scatter_tid
        expert_tiles_done = pl.system.task_dummy(deps=[tile_completion_tids])
        expert_completion_tids[local_e] = expert_tiles_done
    scatter_done = pl.system.task_dummy(deps=[expert_completion_tids[local_e] for local_e in range(N_LOCAL)])
    return scatter_done


@pl.jit.inline
def combine(
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    scatter_done: pl.Scalar[pl.TASK_ID],
    dispatch_push_tid: pl.Scalar[pl.TASK_ID],
):
    # Publish one completion after every local tile scatter retires.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="combine_wait",
        deps=[scatter_done, dispatch_push_tid],
    ) as _cwait_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_arrived, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(signal=combine_arrived, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)

    # ffn_out[t] = sh[t] + Sigma_k routed_y_buf[t*TOPK+k], after combine_wait.
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    with pl.spmd(T, name_hint="shared_routed", deps=[_cwait_tid]) as _reduce_tid:
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
        recv_meta, recv_x, recv_aux, arrived, data_arrived,
        num_tokens, my_rank, moe_epoch,
    )

    with pl.scope():
        scatter_done = expert_routed_scatter(
            recv_x_out, recv_scale_out, recv_w_out, recv_count_out,
            recv_r_route_out, recv_meta_local,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            routed_y_buf,
        )

        ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
        combine(
            sh,
            ffn_out,
            routed_y_buf, combine_arrived,
            num_tokens, my_rank, moe_epoch,
            scatter_done, dispatch_push_tid,
        )

        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next


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


@pl.jit
def l2_moe(
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
        recv_meta, recv_x, recv_aux, arrived, data_arrived,
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
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        l2_moe(
            x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            x_next[r],
            recv_meta, recv_x, recv_aux, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            layer_id, num_tokens, r, pl.const(1, pl.INT32),
            device=r,
        )


# === Golden + test ==========================================================
def golden_moe(tensors):
    """Per-rank torch reference: hc_pre, gate, src-major dispatch replay, routed and shared
    experts, r_route-keyed combine, hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from hc_post import golden_hc_post
    from gate import golden_gate_core
    from expert_shared import golden_expert_shared
    from expert_routed import golden_expert_routed

    T = tensors["x_hc"].shape[1]
    RECV_MAX = N_RANKS * T
    N_ROUTES = T * TOPK
    x_next_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # Stages 1-2: hc_pre + gate per rank.
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

    # Stages 4-5: dispatch replay + routed expert per dst.
    dst_recv_y = {}
    for dst in range(N_RANKS):
        # Pack onto rank dst in src-major order within each local expert.
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


def build_tensor_specs(layer_id=0, num_tokens=None, balanced_routing=False):
    if num_tokens is None:
        num_tokens = T
    import torch
    from golden import ScalarSpec, TensorSpec
    from expert_routed import gen_routed_weight
    from expert_shared import gen_shared_weight

    # Routed = MXFP4 (gen_routed_weight), shared = MXFP8 (gen_shared_weight) grids at
    # behaviorally-calibrated magnitudes, below the real ~2.5e-2.
    ROUTED_DEQUANT_STD = {"w1": 1.08e-2, "w2": 2.54e-2, "w3": 1.10e-2}
    SHARED_DEQUANT_STD = {"w1": 7.65e-3, "w2": 2.39e-2, "w3": 7.39e-3}

    # Shared (replicated) weights are broadcast across ranks; the routed
    # weights are per-rank shards.
    def init_x_hc():
        return torch.randn(N_RANKS, T, HC_MULT, D)

    # Real layer-0 hc_ffn scale/base (fn synthetic at real magnitude).
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
        # Distinct experts per token (sample without replacement).
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
        assert active_routes % N_EXPERTS_GLOBAL == 0, "balanced routing requires active routes divisible by experts"

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

    # Device-resident static weights, stacked per rank ([N_RANKS, *tail], shard r on card r).
    # x_hc, input_ids, and x_next stay per-dispatch.
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
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument(
        "--ep", type=int, default=_EP_DEFAULT, choices=list(_EP_CHOICES),
        help="EP world size / rank count",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
        help=f"comma-separated device ids (need {N_RANKS})",
    )
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument(
        "--num-tokens", type=int, default=T,
        help=f"active token count for MoE dispatch/combine (0..{T})",
    )
    parser.add_argument(
        "--balanced-routing", action="store_true", default=False,
        help="use deterministic hash routes balanced evenly across all experts",
    )
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None, help="dir with cached in/ and out/ tensors to replay")
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="runtime log threshold: debug, v0..v9, info, warn, error, null",
    )
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
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
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
