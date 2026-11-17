# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expert-parallel dispatch/combine shared by the prefill and decode MoE phases.

Each source rank routes its ``tokens x topk`` routes by destination rank and
destination-local expert.  A destination reserves one lane per
``(local expert, source)`` pair with the shared compile-time ``capacity`` — a source sends at
most one route per active token to any one expert, so ``N_RANKS * capacity`` rows is the worst-case
receive capacity for one local expert, derived from the reserved token tile
and full expert skew.  Counts travel on a small metadata window; payloads ride
``put`` so the completion notify covers their visibility.  Expert rows return
to the source's ``token * topk + k`` slot and each token sums its ``topk``
slots in FP32 before the shared expert is added.

Signal protocol (onboard backend): every phase handshake is a monotonic
counter bumped once per rank per epoch with ``AtomicAdd(1)`` and awaited with
``Ge(epoch)``.  ``NotifyOp.Set`` must not be used here: it lowers to a plain
scalar store to the peer's window (``TNOTIFY_IMPL``), and a scalar store
reaches DDR a whole 64-byte line at a time.  The ``[N_RANKS, 1]`` INT32 signal
window can share one such line, so concurrent ``Set`` notifies from the other
peers carry stale neighbours back and silently drop each other's slots, which
leaves the matching ``Ge`` wait spinning forever.  ``AtomicAdd`` is an
element-atomic remote add and is safe in a shared line.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import EP_SIZE, N_LOCAL_EXPERTS, FLASH as M, T_DYN
from hc import hc_post, hc_pre
from moe_experts import _expert_golden, expert_routed, expert_shared
from moe_gate import moe_gate
from rmsnorm import rms_norm


TOPK = M.num_experts_per_tok
N_RANKS = EP_SIZE
N_LOCAL = N_LOCAL_EXPERTS
D = M.hidden_size
INTER = M.moe_intermediate_size
N_EXPERTS = M.n_routed_experts
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
MIX_PAD = 32

# Metadata rows are padded to a 128-byte line so concurrent source writers are line-disjoint.
COUNT_PAD = ((N_LOCAL + 31) // 32) * 32
AUX_WEIGHT = 1
AUX_ROUTE = 0
EXPERT_TILE = 16


@pl.jit.inline(auto_scope=False)
def moe_ep(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    indices: pl.Tensor[[T_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[T_DYN, TOPK], pl.FP32],
    w1: pl.Tensor[[N_LOCAL, D, INTER], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[N_LOCAL, INTER], pl.FP32],
    w3: pl.Tensor[[N_LOCAL, D, INTER], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[N_LOCAL, INTER], pl.FP32],
    w2: pl.Tensor[[N_LOCAL, INTER, D], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[INTER], pl.FP32],
    shared_w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[INTER], pl.FP32],
    shared_w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_x: pld.DistributedTensor,
    recv_aux: pld.DistributedTensor,
    recv_meta: pld.DistributedTensor,
    meta_ready: pld.DistributedTensor,
    data_ready: pld.DistributedTensor,
    return_slots: pld.DistributedTensor,
    return_ready: pld.DistributedTensor,
    consumed: pld.DistributedTensor,
    return_consumed: pld.DistributedTensor,
    routed: pl.Tensor[[T_DYN, D], pl.BF16],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    capacity: pl.constexpr,
    rows: pl.constexpr = 0,
) -> tuple[pl.Tensor[[T_DYN, D], pl.BF16], pl.Scalar[pl.TASK_ID]]:
    """Dispatch this rank's routes, run its local experts, and combine the results.

    Windows are reused by every layer of the forward: the ``ready`` signals
    (``AtomicAdd(1)`` per epoch, monotonic) say the payload is visible, and the
    monotonic ``consumed`` counters say the previous epoch's region was fully
    read before it is overwritten (``AtomicAdd(1)`` per epoch, threshold
    ``epoch - 1``).
    """
    # ``rows`` pins the token count for programs that must stay fully static
    # (a dynamic token symbol in a combined program is not provably stable).
    if rows > 0:
        tokens = rows
    else:
        tokens = pl.tensor.dim(x, 0)
    recv_max = N_RANKS * capacity
    # Per-expert compute slabs include complete hardware tiles.
    chunks = (recv_max + EXPERT_TILE - 1) // EXPERT_TILE
    slab_stride = chunks * EXPERT_TILE
    recv_rows = N_LOCAL * slab_stride
    expert_counts = pl.create_tensor([N_LOCAL], dtype=pl.INT32)
    aux_stage = pl.create_tensor([N_RANKS * N_LOCAL * capacity, 8], dtype=pl.FP32)
    meta_local = pl.create_tensor([N_RANKS, COUNT_PAD], dtype=pl.INT32)
    slab_x = pl.create_tensor([N_LOCAL * chunks, EXPERT_TILE, D], dtype=pl.BF16)
    slab_w = pl.create_tensor([N_LOCAL * chunks, EXPERT_TILE, 16], dtype=pl.FP32)
    slab_route = pl.create_tensor([N_LOCAL * chunks, EXPERT_TILE, 16], dtype=pl.INT32)
    expert_y = pl.create_tensor([N_LOCAL * chunks, EXPERT_TILE, D], dtype=pl.BF16)
    return_stage = pl.create_tensor([N_LOCAL * N_RANKS * capacity, D], dtype=pl.BF16)
    slab_x_flat = pl.reshape(slab_x, [recv_rows, D])
    slab_w_flat = pl.reshape(slab_w, [recv_rows, 16])
    slab_route_flat = pl.reshape(slab_route, [recv_rows, 16])
    expert_y_flat = pl.reshape(expert_y, [recv_rows, D])

    # Every peer must have finished reading the previous epoch before this rank
    # overwrites the shared metadata and payload lanes.  The read anchor keeps
    # this spinning scope off the early-dispatch chain; it must not write the
    # initialized protocol window (never a zero-length active view).
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait", allow_early_resolve=False) as reuse_tid:
        _reuse_anchor = pl.read(consumed, [my_rank, 0])
        for reuse_peer in pl.range(N_RANKS):
            if reuse_peer != my_rank:
                pld.system.wait(
                    signal=consumed, offsets=[reuse_peer, 0],
                    expected=epoch - 1, cmp=pld.WaitCmp.Ge,
                )

    # Publish per-destination route counts and wait for every source's counts.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_dispatch_count", deps=[reuse_tid]) as count_tid:
        counts = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for count_lane in pl.range(N_RANKS * N_LOCAL):
            counts[count_lane] = 0
        for count_token in pl.range(tokens):
            for count_k in pl.range(TOPK):
                count_expert = pl.read(indices, [count_token, count_k])
                counts[count_expert] = counts[count_expert] + 1
        for count_dst in pl.range(N_RANKS):
            meta_row = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
            for count_e in pl.range(N_LOCAL):
                pl.tile.write(meta_row, [0, count_e], counts[count_dst * N_LOCAL + count_e])
            pl.store(meta_row, [count_dst, 0], meta_local)
            pld.tensor.put(
                dst=recv_meta, peer=count_dst, src=meta_local,
                dst_offsets=[my_rank, 0], src_offsets=[count_dst, 0], shape=[1, N_LOCAL],
            )
            if count_dst != my_rank:
                pld.system.notify(
                    target=meta_ready, peer=count_dst, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
        for count_src in pl.range(N_RANKS):
            if count_src != my_rank:
                pld.system.wait(signal=meta_ready, offsets=[count_src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        for count_e in pl.range(N_LOCAL):
            count_total = pl.const(0, pl.INT32)
            for count_src in pl.range(N_RANKS):
                count_total = count_total + pl.read(recv_meta, [count_src, count_e])
            pl.write(expert_counts, [count_e], count_total)

    # Push every route row to its destination's (expert, source) lane, plus its route metadata.
    with pl.spmd(N_RANKS * N_LOCAL, name_hint="moe_dispatch_push", deps=[count_tid], allow_early_resolve=False) as push_tid:
        push_block = pl.tile.get_block_idx()
        push_dst = push_block // N_LOCAL
        push_e = push_block % N_LOCAL
        push_lane_base = (push_e * N_RANKS + my_rank) * capacity
        push_stage_base = (push_dst * N_LOCAL + push_e) * capacity
        push_aux = pl.tile.full([capacity, 8], dtype=pl.FP32, value=0.0)
        push_slot_ctr = pl.array.create(1, pl.INT32)
        push_slot_ctr[0] = 0
        for push_token in pl.range(tokens):
            for push_k in pl.range(TOPK):
                push_expert = pl.read(indices, [push_token, push_k])
                if push_expert // N_LOCAL == push_dst:
                    if push_expert % N_LOCAL == push_e:
                        push_slot = push_slot_ctr[0]
                        push_slot_ctr[0] = push_slot + 1
                        push_row = push_lane_base + push_slot
                        pld.tensor.put(
                            dst=recv_x, peer=push_dst, src=x,
                            dst_offsets=[push_row, 0], src_offsets=[push_token, 0], shape=[1, D],
                        )
                        pl.tile.write(push_aux, [push_slot, AUX_ROUTE], pl.cast(pl.cast(push_token * TOPK + push_k, pl.INT32), pl.FP32))
                        pl.tile.write(push_aux, [push_slot, AUX_WEIGHT], pl.read(weights, [push_token, push_k]))
        pl.store(push_aux, [push_stage_base, 0], aux_stage)
        pld.tensor.put(
            dst=recv_aux, peer=push_dst, src=aux_stage,
            dst_offsets=[push_lane_base, 0], src_offsets=[push_stage_base, 0], shape=[capacity, 8],
        )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_dispatch_signal", deps=[push_tid], allow_early_resolve=False) as data_signal_tid:
        data_anchor = pl.read(aux_stage, [0, 0])
        for signal_dst in pl.range(N_RANKS):
            if signal_dst != my_rank:
                pld.system.notify(
                    target=data_ready, peer=signal_dst, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
        for signal_src in pl.range(N_RANKS):
            if signal_src != my_rank:
                pld.system.wait(signal=data_ready, offsets=[signal_src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    # Regroup received lanes into expert-major slabs; padding rows stay zero with zero weight.
    # The clear owns the whole slab, so ``moe_compact`` must be ordered after it; both are
    # writers of the same tensors and nothing else orders them.
    with pl.spmd(recv_rows // 16, name_hint="moe_compact_clear") as clear_tid:
        clear_block = pl.tile.get_block_idx()
        clear_row = clear_block * 16
        pl.store(pl.tile.full([16, D], dtype=pl.BF16, value=0.0), [clear_row, 0], slab_x_flat)
        pl.store(pl.tile.full([16, 16], dtype=pl.FP32, value=0.0), [clear_row, 0], slab_w_flat)
        pl.store(pl.tile.full([16, 16], dtype=pl.INT32, value=0), [clear_row, 0], slab_route_flat)
    with pl.spmd(N_LOCAL, name_hint="moe_compact", deps=[data_signal_tid, count_tid, clear_tid],
                 allow_early_resolve=False) as compact_tid:
        compact_e = pl.tile.get_block_idx()
        compact_running = pl.array.create(1, pl.INT32)
        compact_running[0] = 0
        for compact_src in pl.range(N_RANKS):
            compact_count = pl.read(recv_meta, [compact_src, compact_e])
            compact_lane_base = (compact_e * N_RANKS + compact_src) * capacity
            for compact_row in pl.range(compact_count):
                # Rows are 64-byte aligned, so the scalar aux writes below own whole
                # cache lines even though the compiler cannot prove the runtime index.
                compact_flat_row = compact_e * slab_stride + compact_running[0] + compact_row
                compact_value = pl.tile.load(recv_x, [compact_lane_base + compact_row, 0], [1, D])
                pl.store(compact_value, [compact_flat_row, 0], slab_x_flat)
                pl.write(
                    slab_w_flat, [compact_flat_row, 0],
                    pl.read(recv_aux, [compact_lane_base + compact_row, AUX_WEIGHT]),
                )
                pl.write(
                    slab_route_flat, [compact_flat_row, 0],
                    pl.cast(pl.read(recv_aux, [compact_lane_base + compact_row, AUX_ROUTE]), pl.INT32),
                )
            compact_running[0] = compact_running[0] + compact_count
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_consumed_signal", deps=[compact_tid],
               allow_early_resolve=False):
        # Read anchor only: the compacted slab is the data just consumed, and the
        # notify must not touch the slab (the expert tasks are reading it).
        _consumed_anchor = pl.read(slab_x_flat, [0, 0])
        for consumed_src in pl.range(N_RANKS):
            if consumed_src != my_rank:
                pld.system.notify(
                    target=consumed, peer=consumed_src, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    # Run every local expert that received rows over the worst-case receive slab.
    for expert_id in pl.parallel(N_LOCAL):
        expert_count = pl.read(expert_counts, [expert_id])
        for chunk in pl.range(chunks):
            if chunk * EXPERT_TILE < expert_count:
                expert_routed(
                    slab_x[expert_id * chunks + chunk], slab_w[expert_id * chunks + chunk],
                    w1[expert_id], w1_scale[expert_id],
                    w3[expert_id], w3_scale[expert_id],
                    w2[expert_id], w2_scale[expert_id],
                    expert_y[expert_id * chunks + chunk], rows=EXPERT_TILE,
                )

    # A source must have combined the previous epoch before its return slots are reused.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_return_reuse", allow_early_resolve=False) as return_reuse_tid:
        # Read anchor only; the expert counts are still being read by the
        # expert loop, so this scope must not write them.
        _return_anchor = pl.read(expert_counts, [0])
        for return_peer in pl.range(N_RANKS):
            if return_peer != my_rank:
                pld.system.wait(
                    signal=return_consumed, offsets=[return_peer, 0],
                    expected=epoch - 1, cmp=pld.WaitCmp.Ge,
                )

    # Return each expert row to its source's route slot.
    with pl.spmd(N_LOCAL * N_RANKS, name_hint="moe_return", deps=[return_reuse_tid],
                 allow_early_resolve=False) as return_tid:
        return_block = pl.tile.get_block_idx()
        return_e = return_block // N_RANKS
        return_src = return_block % N_RANKS
        return_prefix = pl.array.create(1, pl.INT32)
        return_prefix[0] = 0
        for return_prior in pl.range(return_src):
            return_prefix[0] = return_prefix[0] + pl.read(recv_meta, [return_prior, return_e])
        return_count = pl.read(recv_meta, [return_src, return_e])
        return_base = return_e * slab_stride + return_prefix[0]
        for return_row in pl.range(return_count):
            return_value = pl.tile.load(expert_y_flat, [return_base + return_row, 0], [1, D])
            return_stage_row = return_block * capacity + return_row
            pl.store(return_value, [return_stage_row, 0], return_stage)
            return_route = pl.read(slab_route_flat, [return_base + return_row, 0])
            pld.tensor.put(
                dst=return_slots, peer=return_src, src=return_stage,
                dst_offsets=[return_route, 0],
                src_offsets=[return_stage_row, 0], shape=[1, D],
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_return_signal", deps=[return_tid], allow_early_resolve=False) as ready_tid:
        return_anchor = pl.read(expert_counts, [0])
        for ready_src in pl.range(N_RANKS):
            if ready_src != my_rank:
                pld.system.notify(
                    target=return_ready, peer=ready_src, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
        for ready_src in pl.range(N_RANKS):
            if ready_src != my_rank:
                pld.system.wait(signal=return_ready, offsets=[ready_src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    # Sum the top-k route slots per token and add the replicated shared expert.
    shared_input = pl.create_tensor([EXPERT_TILE, D], dtype=pl.BF16)
    shared_y = pl.create_tensor([capacity, D], dtype=pl.BF16)
    for shared_chunk in pl.range((tokens + EXPERT_TILE - 1) // EXPERT_TILE):
        shared_begin = shared_chunk * EXPERT_TILE
        shared_active = pl.min(EXPERT_TILE, tokens - shared_begin)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_shared_stage"):
            pl.store(pl.tile.full([EXPERT_TILE, D], dtype=pl.BF16, value=0.0), [0, 0], shared_input)
            for shared_token in pl.range(shared_active):
                shared_row = pl.tile.load(x, [shared_begin + shared_token, 0], [1, D])
                pl.store(shared_row, [shared_token, 0], shared_input)
        shared_out = pl.create_tensor([EXPERT_TILE, D], dtype=pl.BF16)
        expert_shared(
            shared_input, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale, shared_out, rows=EXPERT_TILE,
        )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_shared_store"):
            for shared_row in pl.range(shared_active):
                shared_value = pl.tile.load(shared_out, [shared_row, 0], [1, D])
                pl.store(shared_value, [shared_begin + shared_row, 0], shared_y)
    with pl.spmd(capacity, name_hint="moe_combine", deps=[ready_tid]) as combine_tid:
        combine_token = pl.tile.get_block_idx()
        if combine_token < tokens:
            combine_acc = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
            for combine_k in pl.range(TOPK):
                combine_slot = combine_token * TOPK + combine_k
                combine_route = pl.tile.load(return_slots, [combine_slot, 0], [1, D])
                combine_acc = pl.add(combine_acc, pl.cast(combine_route, target_type=pl.FP32))
            combine_shared = pl.tile.load(shared_y, [combine_token, 0], [1, D])
            combine_value = pl.add(combine_acc, pl.cast(combine_shared, target_type=pl.FP32))
            pl.store(pl.cast(combine_value, target_type=pl.BF16, mode="rint"), [combine_token, 0], routed)
    # Release the return slots for the next epoch only after the combine read them.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_return_consumed", deps=[combine_tid],
               allow_early_resolve=False) as return_consumed_tid:
        # Protocol storage is initialized even when the active query view is empty.
        _return_consumed_anchor = pl.read(return_consumed, [my_rank, 0])
        for return_src in pl.range(N_RANKS):
            if return_src != my_rank:
                pld.system.notify(
                    target=return_consumed, peer=return_src, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
    return routed, return_consumed_tid


@pl.jit.inline(auto_scope=False)
def moe_ffn(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    w1: pl.Tensor[[N_LOCAL, D, INTER], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[N_LOCAL, INTER], pl.FP32],
    w3: pl.Tensor[[N_LOCAL, D, INTER], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[N_LOCAL, INTER], pl.FP32],
    w2: pl.Tensor[[N_LOCAL, INTER, D], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[INTER], pl.FP32],
    shared_w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[INTER], pl.FP32],
    shared_w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_x: pld.DistributedTensor,
    recv_aux: pld.DistributedTensor,
    recv_meta: pld.DistributedTensor,
    meta_ready: pld.DistributedTensor,
    data_ready: pld.DistributedTensor,
    return_slots: pld.DistributedTensor,
    return_ready: pld.DistributedTensor,
    consumed: pld.DistributedTensor,
    return_consumed: pld.DistributedTensor,
    x_hc_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rows: pl.constexpr,
) -> tuple[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32], pl.Scalar[pl.TASK_ID]]:
    """FFN on the runtime active prefix with full-capacity EP buffers."""
    tokens = pl.tensor.dim(x_hc, 0)
    ffn_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    ffn_normed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    ffn_post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    ffn_residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    route_indices = pl.create_tensor([tokens, TOPK], dtype=pl.INT32)
    route_weights = pl.create_tensor([tokens, TOPK], dtype=pl.FP32)
    routed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    active = pl.cast(num_tokens, pl.INDEX)
    x_active = pl.slice(x_hc, [active, HC_MULT, D], [0, 0, 0])
    pre_active = pl.slice(pre_mix, [active, HC_MULT], [0, 0])
    input_active = pl.slice(ffn_input, [active, D], [0, 0])
    normed_active = pl.slice(ffn_normed, [active, D], [0, 0])
    post_active = pl.slice(ffn_post_mix, [active, HC_MULT], [0, 0])
    residual_active = pl.slice(ffn_residual_mix, [active, HC_MULT, HC_MULT], [0, 0, 0])
    indices_active = pl.slice(route_indices, [active, TOPK], [0, 0])
    weights_active = pl.slice(route_weights, [active, TOPK], [0, 0])
    routed_active = pl.slice(routed, [active, D], [0, 0])
    output_active = pl.slice(x_hc_out, [active, HC_MULT, D], [0, 0, 0])
    next_pre_active = pl.slice(next_pre_mix, [active, HC_MULT], [0, 0])
    input_active = hc_pre(
        x_active, pre_active, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        input_active, next_pre_active, post_active, residual_active,
    )
    normed_active = rms_norm(
        input_active, norm_weight, normed_active, width=D, output_fp32=False, column_tile=256,
    )
    weights_active = moe_gate(
        normed_active, gate_weight, gate_bias, indices_active, weights_active,
        topk=TOPK, topk_pad=8,
    )
    routed_active, ep_done = moe_ep(
        normed_active, indices_active, weights_active,
        w1, w1_scale, w3, w3_scale, w2, w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        recv_x, recv_aux, recv_meta, meta_ready, data_ready,
        return_slots, return_ready, consumed, return_consumed,
        routed_active, my_rank, epoch, capacity=rows, rows=0,
    )
    x_hc_done = hc_post(routed_active, x_active, post_active, residual_active, output_active)
    return x_hc_out, ep_done


def golden_moe_ep(tensors):
    """Torch reference: emulate the expert-parallel dispatch, expert, return, and combine."""
    x = tensors["x"]
    indices = tensors["indices"]
    weights = tensors["weights"]
    ranks = x.shape[0]
    tokens = x.shape[1]
    width = x.shape[2]

    def owned_weight(name, rank, local=None):
        value = tensors[name]
        local_dims = 2 if name.endswith("_scale") else 3
        if name.startswith("shared_"):
            local_dims -= 1
        if value.ndim == local_dims + 1:
            if value.shape[0] != ranks:
                raise ValueError("rank-leading expert weights must match the EP world")
            value = value[rank]
        elif value.ndim != local_dims:
            raise ValueError("invalid routed/shared expert weight rank")
        return value if local is None else value[local]

    active_counts = tensors.get("active_counts")
    if active_counts is None:
        counts = [tokens] * ranks
    else:
        if not isinstance(active_counts, torch.Tensor) or active_counts.shape != (ranks,):
            raise ValueError("active_counts must have one integer count per source rank")
        if active_counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("active_counts must use an integer dtype")
        counts = active_counts.tolist()
        if any(count < 0 or count > tokens for count in counts):
            raise ValueError("active_counts exceed the physical token capacity")

    # Every route of every rank, grouped by the rank and local expert that owns it.
    received = {}
    for src in range(ranks):
        for token in range(counts[src]):
            for k in range(TOPK):
                expert = int(indices[src, token, k])
                key = (expert // N_LOCAL, expert % N_LOCAL)
                received.setdefault(key, []).append(
                    (src, token * TOPK + k, float(weights[src, token, k]), x[src, token])
                )
    slots = torch.zeros(ranks, tokens * TOPK, width, dtype=torch.bfloat16)
    for (dst, local_e), rows in received.items():
        slab = torch.stack([item[3] for item in rows])
        route_weight = torch.tensor([item[2] for item in rows], dtype=torch.float32).reshape(-1, 1)
        out = torch.zeros_like(slab)
        _expert_golden(
            {
                "recv_x": slab, "routed_weight": route_weight,
                "w1": owned_weight("w1", dst, local_e), "w1_scale": owned_weight("w1_scale", dst, local_e),
                "w3": owned_weight("w3", dst, local_e), "w3_scale": owned_weight("w3_scale", dst, local_e),
                "w2": owned_weight("w2", dst, local_e), "w2_scale": owned_weight("w2_scale", dst, local_e),
                "output": out,
            },
            routed=True,
        )
        for index, item in enumerate(rows):
            slots[item[0], item[1]] = out[index]
    shared_out = torch.zeros(ranks, tokens, width, dtype=torch.bfloat16)
    for rank in range(ranks):
        if counts[rank] == 0:
            continue
        _expert_golden(
            {
                "recv_x": x[rank, :counts[rank]],
                "w1": owned_weight("shared_w1", rank), "w1_scale": owned_weight("shared_w1_scale", rank),
                "w3": owned_weight("shared_w3", rank), "w3_scale": owned_weight("shared_w3_scale", rank),
                "w2": owned_weight("shared_w2", rank), "w2_scale": owned_weight("shared_w2_scale", rank),
                "output": shared_out[rank, :counts[rank]],
            },
            routed=False,
        )
    for rank in range(ranks):
        for token in range(counts[rank]):
            accumulator = slots[rank, token * TOPK:(token + 1) * TOPK].float().sum(dim=0)
            tensors["routed"][rank, token] = (accumulator + shared_out[rank, token].float()).to(torch.bfloat16)
