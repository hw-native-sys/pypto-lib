# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=8  # CI: 8-card TP run; the deployment world size, borrowed via task-submit --device-num
"""DeepSeek-V4 Flash MoE layer, tensor-parallel over the intermediate dim.

Every rank holds every expert and a 1/TP slice of the intermediate dim, and every
rank routes the same tokens, so grouping is a local permutation and the only
exchange is one all-reduce of the FFN output before ``hc_post``.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import FLASH as M, MOE_TOKENS, TP
from expert_routed import IDX_PAD, N_SLOTS, RECV_MAX
from expert_routed_persistent_balanced import expert_routed_persistent_balanced
from expert_shared import expert_shared
from gate import gate
from hc_post import hc_post
from hc_pre import hc_pre


# model config
T = MOE_TOKENS
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MOE_INTER = M.moe_intermediate_size // TP  # this rank's slice of the intermediate dim
N_EXPERTS = M.n_routed_experts
N_ROUTES = T * TOPK
N_RANKS = TP

# All-reduce window. Two lanes alternate by epoch so a rank can publish call e
# while a peer still reads call e-1, which keeps one barrier per call enough:
# reaching call e means every peer published e-1, hence finished reading e-2.
T_PAD = ((T + 15) // 16) * 16
REDUCE_LANE_ROWS = N_RANKS * T_PAD
REDUCE_WINDOW_ROWS = 2 * REDUCE_LANE_ROWS

# tiling
REDUCE_D_TILE = 512


@pl.jit.inline
def clear_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's all-reduce counters after its final MoE completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear", allow_early_resolve=True):
        # The final MoE output depends on this rank observing every peer's final
        # publish notify, so no peer can issue another one in this forward.
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(reduce_signal, [src, 0], zero)


# === Routing ================================================================
@pl.jit.inline
def route_group(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    # compact per-slot outputs consumed by the routed expert / combine_local
    recv_x: pl.Tensor[[N_SLOTS, RECV_MAX, D], pl.INT8],
    recv_scale: pl.Tensor[[N_SLOTS, RECV_MAX], pl.FP32],
    recv_w: pl.Tensor[[N_SLOTS, RECV_MAX], pl.FP32],
    recv_count: pl.Tensor[[N_SLOTS, IDX_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS, IDX_PAD], pl.INT32],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Group the T * TOPK routes by expert into one padded row tile per expert."""
    recv_x_flat = pl.reshape(recv_x, [N_SLOTS * RECV_MAX, D])
    expert_slot = pl.create_tensor([1, N_EXPERTS], dtype=pl.INT32, manual_dep=True)
    route_token = pl.create_tensor([N_ROUTES, IDX_PAD], dtype=pl.INT32, manual_dep=True)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_table_init", allow_early_resolve=True) as init_tid:
        expert_slot[:, :] = pl.full([1, N_EXPERTS], dtype=pl.INT32, value=-1)
        recv_count[:, :] = pl.full([N_SLOTS, IDX_PAD], dtype=pl.INT32, value=0)
        slot_expert[:, :] = pl.full([N_SLOTS, IDX_PAD], dtype=pl.INT32, value=0)
        recv_scale[:, :] = pl.full([N_SLOTS, RECV_MAX], dtype=pl.FP32, value=0.0)
        recv_w[:, :] = pl.full([N_SLOTS, RECV_MAX], dtype=pl.FP32, value=0.0)
        route_slot_row[:, :] = pl.full([N_ROUTES, IDX_PAD], dtype=pl.INT32, value=0)
        route_token[:, :] = pl.full([N_ROUTES, IDX_PAD], dtype=pl.INT32, value=0)

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    # One core owns the slot table: adjacent scalar writes from several SPMD
    # cores can race through overlapping DMA units.
    with pl.spmd(1, name_hint="route_group_scalar", allow_early_resolve=True, deps=[init_tid]) as scalar_tid:
        scalar_core = pl.tile.get_block_idx()
        next_slot = pl.cast(0, pl.INDEX)
        for token in pl.range(scalar_core, active_tokens):
            token_scale = pl.read(x_norm_scale, [token, 0])
            for k in pl.range(TOPK):
                expert_id = pl.read(indices, [token, k])
                expert_col = pl.cast(expert_id, pl.INDEX)
                slot = pl.read(expert_slot, [0, expert_col])
                if slot < 0:
                    slot = pl.cast(next_slot, pl.INT32)
                    pl.write(expert_slot, [0, expert_col], slot)
                    pl.write(slot_expert, [next_slot, 0], expert_id)
                    next_slot = next_slot + 1
                slot_row = pl.cast(slot, pl.INDEX)
                count = pl.read(recv_count, [slot_row, 0])
                count_col = pl.cast(count, pl.INDEX)
                pl.write(recv_count, [slot_row, 0], pl.cast(count + 1, pl.INT32))
                pl.write(recv_scale, [slot_row, count_col], token_scale)
                pl.write(recv_w, [slot_row, count_col], pl.read(weights, [token, k]))
                route = token * TOPK + k
                pl.write(route_slot_row, [route, 0], pl.cast(slot * RECV_MAX + count, pl.INT32))
                pl.write(route_token, [route, 0], pl.cast(token, pl.INT32))

    active_routes = active_tokens * TOPK
    with pl.spmd(N_ROUTES, name_hint="route_gather", allow_early_resolve=True, deps=[scalar_tid]) as _gather_tid:
        route = pl.tile.get_block_idx()
        if route < active_routes:
            dst_row = pl.cast(pl.read(route_slot_row, [route, 0]), pl.INDEX)
            src_row = pl.cast(pl.read(route_token, [route, 0]), pl.INDEX)
            recv_x_flat[dst_row : dst_row + 1, :] = x_norm_i8[src_row : src_row + 1, :]


# === Combine ================================================================
@pl.jit.inline
def combine_local(
    recv_y: pl.Tensor[[N_SLOTS, RECV_MAX, D], pl.BF16],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """ffn_partial[t] = sh[t] + Sigma_k recv_y[route_slot_row[t * TOPK + k]]."""
    recv_y_flat = pl.reshape(recv_y, [N_SLOTS * RECV_MAX, D])

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    with pl.spmd(T, name_hint="shared_routed", allow_early_resolve=True):
        t = pl.tile.get_block_idx()
        if t < active_tokens:
            acc = pl.cast(sh[t : t + 1, :], target_type=pl.FP32)
            for k in pl.range(TOPK):
                src_row = pl.cast(pl.read(route_slot_row, [t * TOPK + k, 0]), pl.INDEX)
                acc = pl.add(acc, pl.cast(recv_y_flat[src_row : src_row + 1, :], target_type=pl.FP32))
            ffn_partial[t : t + 1, :] = acc
        else:
            ffn_partial[t : t + 1, :] = pl.cast(sh[t : t + 1, :], target_type=pl.FP32)


# === All-reduce =============================================================
@pl.jit.inline
def all_reduce_ffn(
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id; reduce_signal is monotonic so waits use `>= moe_epoch`.
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Sum every rank's intermediate-slice partial into a full-precision ffn_out."""
    lane_base = pl.cast(((moe_epoch - 1) % 2) * REDUCE_LANE_ROWS, pl.INDEX)
    my_row = lane_base + pl.cast(my_rank, pl.INDEX) * T_PAD

    with pl.spmd(N_RANKS, name_hint="moe_reduce_publish", allow_early_resolve=True) as publish_tid:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=reduce_window,
            peer=peer,
            src=ffn_partial,
            dst_offsets=[my_row, 0],
            src_offsets=[0, 0],
            shape=[T, D],
            chunk_rows=T,
            chunk_cols=D,
        )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reduce_barrier", allow_early_resolve=True, deps=[publish_tid]) as barrier_tid:
        for peer in pl.range(N_RANKS):
            pld.system.notify(
                target=reduce_signal,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
        for src in pl.range(N_RANKS):
            pld.system.wait(
                signal=reduce_signal,
                offsets=[src, 0],
                expected=moe_epoch,
                cmp=pld.WaitCmp.Ge,
            )

    with pl.spmd(T * (D // REDUCE_D_TILE), name_hint="moe_reduce", allow_early_resolve=True, deps=[barrier_tid]) as _reduce_tid:
        block = pl.tile.get_block_idx()
        row = block // (D // REDUCE_D_TILE)
        d0 = (block % (D // REDUCE_D_TILE)) * REDUCE_D_TILE
        acc = pl.load(reduce_window, [lane_base + row, d0], [1, REDUCE_D_TILE])
        for src_rank in pl.range(1, N_RANKS):
            src_row = lane_base + src_rank * T_PAD + row
            acc = pl.add(acc, pl.load(reduce_window, [src_row, d0], [1, REDUCE_D_TILE]))
        reduced = pl.cast(acc, target_type=pl.BF16, mode="rint")
        pl.store(reduced, [row, d0], ffn_out)


@pl.jit.inline
def reduce_ffn_tp1(
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Single-rank path: the partial is already the full sum."""
    with pl.spmd(T * (D // REDUCE_D_TILE), name_hint="moe_reduce", allow_early_resolve=True):
        block = pl.tile.get_block_idx()
        row = block // (D // REDUCE_D_TILE)
        d0 = (block % (D // REDUCE_D_TILE)) * REDUCE_D_TILE
        partial = ffn_partial[row : row + 1, d0 : d0 + REDUCE_D_TILE]
        ffn_out[row : row + 1, d0 : d0 + REDUCE_D_TILE] = pl.cast(partial, target_type=pl.BF16, mode="rint")


@pl.jit.inline(auto_scope=False)
def moe(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # final output
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    # windows
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id for the shared reduce window (distinct from layer_id).
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

    recv_x = pl.create_tensor([N_SLOTS, RECV_MAX, D], dtype=pl.INT8)
    recv_scale = pl.create_tensor([N_SLOTS, RECV_MAX], dtype=pl.FP32)
    recv_w = pl.create_tensor([N_SLOTS, RECV_MAX], dtype=pl.FP32)
    recv_count = pl.create_tensor([N_SLOTS, IDX_PAD], dtype=pl.INT32)
    slot_expert = pl.create_tensor([N_SLOTS, IDX_PAD], dtype=pl.INT32)
    route_slot_row = pl.create_tensor([N_ROUTES, IDX_PAD], dtype=pl.INT32)
    route_group(
        indices, weights, x_norm_i8, x_norm_scale,
        recv_x, recv_scale, recv_w, recv_count, slot_expert, route_slot_row,
        num_tokens,
    )

    with pl.scope():
        recv_y = pl.create_tensor([N_SLOTS, RECV_MAX, D], dtype=pl.BF16)
        expert_routed_persistent_balanced(
            recv_x, recv_scale, recv_w, recv_count, slot_expert,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            recv_y,
        )

        ffn_partial = pl.create_tensor([T, D], dtype=pl.FP32)
        combine_local(recv_y, route_slot_row, sh, ffn_partial, num_tokens)

        ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
        if N_RANKS == 1:
            reduce_ffn_tp1(ffn_partial, ffn_out)
        else:
            all_reduce_ffn(ffn_partial, ffn_out, reduce_window, reduce_signal, my_rank, moe_epoch)

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
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # final output
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    # windows
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
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
        reduce_window, reduce_signal,
        layer_id, num_tokens, my_rank, moe_epoch,
    )
    clear_moe_signals(x_next, reduce_signal)
    return x_next


@pl.jit.host
def l3_moe(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_EXPERTS, D], pl.FP32],
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
    reduce_window_buf = pld.alloc_window_buffer([REDUCE_WINDOW_ROWS, D], dtype=pl.FP32)
    reduce_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        reduce_window = pld.window(reduce_window_buf, [REDUCE_WINDOW_ROWS, D], dtype=pl.FP32)
        reduce_signal = pld.window(reduce_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        moe_test(
            x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            x_next[r],
            reduce_window, reduce_signal,
            layer_id, num_tokens, r, pl.const(1, pl.INT32),
            device=r,
        )


# === Golden + test ==========================================================
def golden_moe(tensors):
    """Per-rank torch reference.

    Every rank sees the same tokens, so hc_pre / gate run once; each rank then
    computes its intermediate-slice partial and the partials are summed before
    hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from hc_post import golden_hc_post
    from gate import golden_gate_core
    from expert_shared import golden_expert_shared
    from expert_routed import golden_expert_routed

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # Stages 1-2: hc_pre + gate. Replicated inputs, so rank 0's result is every
    # rank's result.
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_ffn = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb_ffn = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x":        tensors["x_hc"][0],
        "hc_fn":    tensors["hc_ffn_fn"][0],
        "hc_scale": tensors["hc_ffn_scale"][0],
        "hc_base":  tensors["hc_ffn_base"][0],
        "x_mixed":  x_mixed,
        "post":     post_ffn,
        "comb":     comb_ffn,
    })
    x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
    x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
    indices = torch.zeros(T, TOPK, dtype=torch.int32)
    weights = torch.zeros(T, TOPK, dtype=torch.float32)
    golden_gate_core({
        "x_mixed":      x_mixed,
        "norm_w":       tensors["norm_w"][0],
        "gate_w":       tensors["gate_w"][0],
        "gate_bias":    tensors["gate_bias"][0],
        "layer_id":     tensors["layer_id"],
        "num_tokens":   tensors["num_tokens"],
        "tid2eid":      tensors["tid2eid"][0],
        "input_ids":    tensors["input_ids"][0],
        "x_norm_i8":    x_norm_i8,
        "x_norm_scale": x_norm_scale,
        "indices":      indices,
        "weights":      weights,
    })

    # Stage 3: the same local grouping route_group builds on device.
    slot_of_expert = {}
    slot_expert = torch.zeros(N_SLOTS, IDX_PAD, dtype=torch.int32)
    recv_count = torch.zeros(N_SLOTS, IDX_PAD, dtype=torch.int32)
    recv_scale = torch.zeros(N_SLOTS, RECV_MAX, dtype=torch.float32)
    recv_w = torch.zeros(N_SLOTS, RECV_MAX, dtype=torch.float32)
    recv_x = torch.zeros(N_SLOTS, RECV_MAX, D, dtype=torch.int8)
    route_slot_row = torch.zeros(N_ROUTES, dtype=torch.int64)
    for t in range(num_tokens):
        for k in range(TOPK):
            eid = int(indices[t, k].item())
            if eid not in slot_of_expert:
                slot = len(slot_of_expert)
                slot_of_expert[eid] = slot
                slot_expert[slot, 0] = eid
            slot = slot_of_expert[eid]
            count = int(recv_count[slot, 0].item())
            recv_count[slot, 0] = count + 1
            recv_scale[slot, count] = float(x_norm_scale[t, 0].item())
            recv_w[slot, count] = float(weights[t, k].item())
            recv_x[slot, count, :] = x_norm_i8[t, :]
            route_slot_row[t * TOPK + k] = slot * RECV_MAX + count

    # Stages 4-6: per-rank slice compute, then sum the partials.
    ffn_sum = torch.zeros(T, D, dtype=torch.float32)
    for r in range(N_RANKS):
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
        recv_y = torch.zeros(N_SLOTS, RECV_MAX, D, dtype=torch.bfloat16)
        golden_expert_routed({
            "recv_x":            recv_x,
            "recv_scale_dq":     recv_scale,
            "recv_weights":      recv_w,
            "recv_expert_count": recv_count,
            "slot_expert":       slot_expert,
            "routed_w1":         tensors["routed_w1"][r],
            "routed_w1_scale":   tensors["routed_w1_scale"][r],
            "routed_w3":         tensors["routed_w3"][r],
            "routed_w3_scale":   tensors["routed_w3_scale"][r],
            "routed_w2":         tensors["routed_w2"][r],
            "routed_w2_scale":   tensors["routed_w2_scale"][r],
            "recv_y":            recv_y,
        })
        recv_y_flat = recv_y.reshape(N_SLOTS * RECV_MAX, D).float()
        partial = sh.float().clone()
        for t in range(num_tokens):
            for k in range(TOPK):
                partial[t, :] += recv_y_flat[int(route_slot_row[t * TOPK + k].item()), :]
        ffn_sum += partial

    ffn_out = ffn_sum.to(torch.bfloat16)
    x_next_r = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x":        ffn_out,
        "residual": tensors["x_hc"][0],
        "post":     post_ffn,
        "comb":     comb_ffn,
        "y":        x_next_r,
    })
    tensors["x_next"][:] = x_next_r.unsqueeze(0).expand(N_RANKS, -1, -1, -1)


def build_tensor_specs(layer_id=0, num_tokens=T):
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
    MOE_INTER_FULL = M.moe_intermediate_size

    # Every rank carries the same activation: attention is replicated.
    def replicate(tensor):
        return tensor.unsqueeze(0).expand(N_RANKS, *([-1] * tensor.dim())).contiguous()

    x_hc_one = torch.randn(T, HC_MULT, D)

    def init_x_hc():
        return replicate(x_hc_one)

    # Real layer-0 hc_ffn scale/base (fn synthetic at real magnitude). A synthetic
    # scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling the FFN output and
    # hc residual to near-zero in x_next where W8A8 noise blows up the relative tail.
    def init_hc_ffn_fn():
        return replicate(torch.randn(MIX_HC, HC_DIM) * 0.0635)

    def init_hc_ffn_scale():
        return replicate(torch.tensor([0.11334, 0.035901, 0.058183]))

    def init_hc_ffn_base():
        base = torch.tensor([
            2.4153, -2.0252, -2.0019, -2.1947,
            -1.5430, -3.0228, -6.8248, 0.5894,
            2.1916, -7.2132, -3.0938, -2.1119,
            -3.0161, 3.3293, -3.2224, -4.0226,
            -2.0428, -3.3478, 3.0893, -3.4166,
            -1.8144, -3.8147, -3.1307, 1.7862,
        ])
        return replicate(base)

    def init_norm_w():
        return replicate(torch.ones(D))

    def init_gate_w():
        return replicate(torch.randn(N_EXPERTS, D) / D ** 0.5)

    def init_gate_bias():
        return replicate(torch.zeros(N_EXPERTS))

    # Distinct experts per token (sample without replacement) like real top-k.
    tid2eid_one = torch.argsort(torch.rand(VOCAB, N_EXPERTS), dim=1)[:, :TOPK].to(torch.int32)
    input_ids_one = torch.randint(0, VOCAB, (T,), dtype=torch.int64)

    def init_tid2eid():
        return replicate(tid2eid_one)

    def init_input_ids():
        return replicate(input_ids_one)

    # A step activates at most T * TOPK experts, so synthesizing the whole 256-expert
    # bank through the FP4 grid is wasted work: draw EXPERT_POOL distinct sets and tile
    # them, which keeps every expert non-zero whichever way the gate routes.
    EXPERT_POOL = min(N_EXPERTS, T * TOPK)
    pool_reps = (N_EXPERTS + EXPERT_POOL - 1) // EXPERT_POOL

    def routed_bank(shape_tail, dequant_std):
        w_i8, w_s = gen_routed_weight((EXPERT_POOL, *shape_tail), dequant_std)
        bank_i8 = w_i8.repeat(pool_reps, *([1] * len(shape_tail)))[:N_EXPERTS].contiguous()
        bank_s = w_s.repeat(pool_reps, 1)[:N_EXPERTS].contiguous()
        return bank_i8, bank_s

    # w1 / w3 slice the output-channel axis; w2 slices its reduction axis.
    rw1_i8_full, rw1_s_full = routed_bank((MOE_INTER_FULL, D), ROUTED_DEQUANT_STD["w1"])
    rw3_i8_full, rw3_s_full = routed_bank((MOE_INTER_FULL, D), ROUTED_DEQUANT_STD["w3"])
    rw2_i8_bank, rw2_s_bank = routed_bank((D, MOE_INTER_FULL), ROUTED_DEQUANT_STD["w2"])

    rw1_i8 = torch.stack([rw1_i8_full[:, r * MOE_INTER : (r + 1) * MOE_INTER, :] for r in range(N_RANKS)])
    rw1_s = torch.stack([rw1_s_full[:, r * MOE_INTER : (r + 1) * MOE_INTER] for r in range(N_RANKS)])
    rw3_i8 = torch.stack([rw3_i8_full[:, r * MOE_INTER : (r + 1) * MOE_INTER, :] for r in range(N_RANKS)])
    rw3_s = torch.stack([rw3_s_full[:, r * MOE_INTER : (r + 1) * MOE_INTER] for r in range(N_RANKS)])
    rw2_i8 = torch.stack([rw2_i8_bank[:, :, r * MOE_INTER : (r + 1) * MOE_INTER] for r in range(N_RANKS)])
    rw2_s = torch.stack([rw2_s_bank for _ in range(N_RANKS)])

    sw1_i8_full, sw1_s_full = gen_shared_weight((MOE_INTER_FULL, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3_i8_full, sw3_s_full = gen_shared_weight((MOE_INTER_FULL, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2_i8_full, sw2_s_full = gen_shared_weight((D, MOE_INTER_FULL), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)
    sw1_i8 = torch.stack([sw1_i8_full[r * MOE_INTER : (r + 1) * MOE_INTER, :] for r in range(N_RANKS)])
    sw1_s = torch.stack([sw1_s_full[r * MOE_INTER : (r + 1) * MOE_INTER] for r in range(N_RANKS)])
    sw3_i8 = torch.stack([sw3_i8_full[r * MOE_INTER : (r + 1) * MOE_INTER, :] for r in range(N_RANKS)])
    sw3_s = torch.stack([sw3_s_full[r * MOE_INTER : (r + 1) * MOE_INTER] for r in range(N_RANKS)])
    sw2_i8 = torch.stack([sw2_i8_full[:, r * MOE_INTER : (r + 1) * MOE_INTER] for r in range(N_RANKS)])
    sw2_s = torch.stack([sw2_s_full for _ in range(N_RANKS)])

    specs = [
        TensorSpec("x_hc",          [N_RANKS, T, HC_MULT, D],     torch.float32, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",     [N_RANKS, MIX_HC, HC_DIM],       torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale",  [N_RANKS, 3],                    torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",   [N_RANKS, MIX_HC],               torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",        [N_RANKS, D],                    torch.bfloat16,  init_value=init_norm_w),
        TensorSpec("gate_w",        [N_RANKS, N_EXPERTS, D],         torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",     [N_RANKS, N_EXPERTS],            torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",       [N_RANKS, VOCAB, TOPK],          torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids",     [N_RANKS, T],                 torch.int64,    init_value=init_input_ids),
        TensorSpec("routed_w1",        [N_RANKS, N_EXPERTS, MOE_INTER, D], torch.int8,    init_value=lambda: rw1_i8),
        TensorSpec("routed_w1_scale",  [N_RANKS, N_EXPERTS, MOE_INTER],    torch.float32, init_value=lambda: rw1_s),
        TensorSpec("routed_w3",        [N_RANKS, N_EXPERTS, MOE_INTER, D], torch.int8,    init_value=lambda: rw3_i8),
        TensorSpec("routed_w3_scale",  [N_RANKS, N_EXPERTS, MOE_INTER],    torch.float32, init_value=lambda: rw3_s),
        TensorSpec("routed_w2",        [N_RANKS, N_EXPERTS, D, MOE_INTER], torch.int8,    init_value=lambda: rw2_i8),
        TensorSpec("routed_w2_scale",  [N_RANKS, N_EXPERTS, D],            torch.float32, init_value=lambda: rw2_s),
        TensorSpec("shared_w1",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2",        [N_RANKS, D, MOE_INTER],          torch.int8,    init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale",  [N_RANKS, D],                     torch.float32, init_value=lambda: sw2_s),
        TensorSpec("x_next",           [N_RANKS, T, HC_MULT, D],      torch.float32, is_output=True),
        ScalarSpec("layer_id",         torch.int32,                      layer_id),
        ScalarSpec("num_tokens",       torch.int32,                      num_tokens),
    ]

    # Keep the static weight parameters device-resident (child_memory), sharded
    # per rank: each shard is a leading-dim-stacked [N_RANKS, *tail] tensor sliced
    # as weight[r] and dispatched to device=r; resident="stacked" uploads shard r
    # to card r once and reuses it across dispatches, skipping the per-dispatch
    # H2D/D2H.
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

    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP, choices=[1, 2, 4, 8],
                        help="tensor-parallel degree / rank count; config freezes it at import")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T,
                        help=f"active token count for MoE routing/combine (0..{T})")
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

    result = run_jit(
        fn=l3_moe,
        specs=build_tensor_specs(layer_id=args.layer_id, num_tokens=args.num_tokens),
        golden_fn=golden_moe,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            log_level=args.log_level,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 x_next. No max_diff_hd: near-zero residual/FFN cancellations
            # blow up relatively.
            "x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
