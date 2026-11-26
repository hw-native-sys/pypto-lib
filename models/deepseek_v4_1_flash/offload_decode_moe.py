# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Split decode MoE for the mapped expert bank."""

import sys
import pypto.language as pl
import pypto.language.distributed as pld
from expert_slots import combine_single_token_slots, compute_single_token_slots, gather_single_token_inputs
SCALE_BANK_LAYOUT = pl.ND
from config import FLASH as M, EP_SIZE, MOE_TOKENS, MOE_RECV_MAX, N_LOCAL_EXPERTS, T_DYN
from expert_supply import publish_routed_counts, supply_experts
from gate import gate
from expert_shared import SWIGLU_WORKER_TILE, W13_WORKER_TILE, W2_WORKER_TILE, expert_shared
from offload_expert_routed import dispatch, expert_routed_scatter
from expert_routed import (
    ROUTED_CUBE_WORKER_TILE, ROUTED_VECTOR_WORKER_TILE,
)
from hc import hc_head_norm_store, hc_project, hc_sinkhorn, hc_post


SLOT_DECODE = EP_SIZE == 1

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size
MX_GROUP = M.weight_block_size[0]


BALANCED_ROUTING = "--balanced-routing" in sys.argv

# EP transport layout
RECV_MAX = MOE_RECV_MAX
ROUTE_WIDTH = 8
T = MOE_TOKENS
N_RANKS = EP_SIZE
N_LOCAL = N_LOCAL_EXPERTS
K_SCALE = D // MX_GROUP
MAX_PER_SRC = T
IDX_PAD = ROUTE_WIDTH
SIGNAL_PAD = 128  # 512-byte padded monotonic counter lane

# tiling
COMBINE_T_TILE = 2

# Main decode worker budgets, ordered as W13 Cube, W2 Cube, SwiGLU Vector.
FULL_ROUTED_CUBES, FULL_ROUTED_VECTORS = 32, 64
SHORT_SHARED_W13, SHORT_SHARED_W2, SHORT_SHARED_ACT = 18, 10, 32
SHORT_TOKEN_LIMIT = 80
LONG_TOKEN_MIN = 256
SHORT_EXPERT_ROWS_MIN = 32
LONG_EXPERT_ROWS_MIN = 64
DENSE_EXPERT_ROWS = 128


@pl.jit.inline(auto_scope=False)
def initialize_balanced_routes(
    routes: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    local_experts: pl.constexpr,
    topk: pl.constexpr,
):
    """Write the equal-token balanced-routing fixture used by the MoE entry."""
    capacity = pl.tensor.dim(routes, 0)
    with pl.spmd(32, name_hint="initialize_balanced_routes"):
        core = pl.tile.get_block_idx()
        expanded = num_tokens * topk
        rank_offset = rank * expanded + rank
        for row in pl.range(core, capacity, 32):
            values = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
            for lane in pl.unroll(topk):
                offset = row * topk + lane + rank_offset
                peer = offset % N_RANKS
                expert = offset // N_RANKS % local_experts
                expert_id = pl.cast(peer * local_experts + expert, pl.INT32)
                pl.tile.write(values, [0, lane], expert_id)
            valid_values = pl.set_validshape(values, 1, topk)
            routes = pl.store(valid_values, [row, 0], routes)
    return routes


@pl.jit.inline(auto_scope=False)
def clear_moe_signals(
    completion_anchor: pl.Tensor,
    data_arrived: pld.DistributedTensor,
    combine_arrived: pld.DistributedTensor,
    anchor_indices: pl.constexpr,
):
    """Clear a completed entry's MoE arrival signals."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, anchor_indices)
        zero = pl.cast(0, pl.INT32)
        for peer in pl.range(EP_SIZE):
            pl.write(data_arrived, [peer, 0, 0], zero)
            pl.write(data_arrived, [peer, 0, 16], zero)
            pl.write(combine_arrived, [peer, 0, 0], zero)
        pl.write(completion_anchor, anchor_indices, anchor)
    return completion_anchor


@pl.jit.inline
def combine_scattered(
    shared_output: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    routed_output: pld.DistributedTensor,
    combine_arrived: pld.DistributedTensor,
    scatter_done: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    topk: pl.constexpr,
):
    arrivals = pl.array.create(1, pl.TASK_ID)
    arrivals[0] = scatter_done
    if N_RANKS > 1:
        with pl.spmd(N_RANKS, name_hint="combine_notify", deps=[scatter_done], allow_early_resolve=False) as notify_tid:
            peer = pl.tile.get_block_idx()
            if peer != my_rank:
                pld.system.notify(target=combine_arrived, peer=peer, offsets=[my_rank, 0, 0], value=moe_epoch, op=pld.NotifyOp.Set)

        peer_waits = pl.array.create(N_RANKS, pl.TASK_ID)
        for src in pl.range(N_RANKS):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_wait", deps=[scatter_done], no_dep_args=[combine_arrived]) as peer_wait:
                if src != my_rank:
                    pld.system.defer_wait(signal=combine_arrived, offsets=[src, 0, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)
            peer_waits[src] = peer_wait
        arrivals[0] = pl.system.task_dummy(deps=[peer_waits, notify_tid])
    _cwait_tid = arrivals[0]

    # Wait for local and peer routed-output stores.
    routed_rows = pl.tensor.view(routed_output, [T, topk * D])
    with pl.spmd(pl.min((T + COMBINE_T_TILE - 1) // COMBINE_T_TILE, 64), name_hint="combine_reduce", deps=[_cwait_tid]) as _reduce_tid:
        core = pl.tile.get_block_idx()
        for t in pl.range(core * COMBINE_T_TILE, T, 64 * COMBINE_T_TILE):
            rows = pl.min(COMBINE_T_TILE, T - t)
            active_rows = pl.min(rows, pl.max(num_tokens - t, 0))
            result = pl.tile.full([COMBINE_T_TILE, D], dtype=pl.BF16, value=0.0)
            if active_rows > 0:
                shared_row = pl.load(shared_output, [t, 0], [COMBINE_T_TILE, D], valid_shape=[active_rows, D])
                acc = pl.cast(shared_row, target_type=pl.FP32)
                for k in pl.pipeline(topk, stage=2):
                    routed_row = pl.load(routed_rows, [t, k * D], [COMBINE_T_TILE, D], valid_shape=[active_rows, D])
                    routed_fp32 = pl.cast(routed_row, target_type=pl.FP32)
                    acc = pl.add(acc, routed_fp32)
                ffn_row = pl.cast(acc, target_type=pl.BF16, mode="rint")
                result = pl.tile.assemble(result, ffn_row, [0, 0])
            result_valid = pl.set_validshape(result, rows, D)
            ffn_out = pl.store(result_valid, [t, 0], ffn_out)
    return _reduce_tid


@pl.jit.inline(auto_scope=False)
def _decode_ffn_workers(num_tokens: pl.Scalar[pl.INT32], local_experts: pl.constexpr, topk: pl.constexpr):
    """Select launch sizes from the active rows on this rank and the balanced expert workload."""
    rw13 = pl.cast(ROUTED_CUBE_WORKER_TILE, pl.INT32)
    rw2 = pl.cast(ROUTED_CUBE_WORKER_TILE, pl.INT32)
    ract = pl.cast(ROUTED_VECTOR_WORKER_TILE, pl.INT32)
    sw13 = pl.cast(W13_WORKER_TILE, pl.INT32)
    sw2 = pl.cast(W2_WORKER_TILE, pl.INT32)
    sact = pl.cast(SWIGLU_WORKER_TILE, pl.INT32)
    if topk == 6:
        routed_rows = num_tokens * topk
        if num_tokens <= SHORT_TOKEN_LIMIT:
            if routed_rows >= local_experts * SHORT_EXPERT_ROWS_MIN:
                rw13 = pl.cast(FULL_ROUTED_CUBES, pl.INT32)
                rw2 = pl.cast(FULL_ROUTED_CUBES, pl.INT32)
                ract = pl.cast(FULL_ROUTED_VECTORS, pl.INT32)
                sw13 = pl.cast(SHORT_SHARED_W13, pl.INT32)
                sw2 = pl.cast(SHORT_SHARED_W2, pl.INT32)
                sact = pl.cast(SHORT_SHARED_ACT, pl.INT32)
        elif (num_tokens >= LONG_TOKEN_MIN) and (routed_rows >= local_experts * LONG_EXPERT_ROWS_MIN):
            rw13 = pl.cast(FULL_ROUTED_CUBES, pl.INT32)
            rw2 = pl.cast(FULL_ROUTED_CUBES, pl.INT32)
            ract = pl.cast(FULL_ROUTED_VECTORS, pl.INT32)
            sw13 = pl.cast(18, pl.INT32)
            sw2 = pl.cast(20, pl.INT32)
            sact = pl.cast(8, pl.INT32)
            if routed_rows >= local_experts * DENSE_EXPERT_ROWS:
                sw13 = pl.cast(16, pl.INT32)
                sw2 = pl.cast(16, pl.INT32)
    return rw13, rw2, ract, sw13, sw2, sact


@pl.jit.inline(auto_scope=False)
def moe(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor,
    correction_bias: pl.Tensor,
    routed_w1: pl.Tensor,
    routed_w1_scale: pl.Tensor[None, pl.FP8E8M0, SCALE_BANK_LAYOUT],
    routed_w2: pl.Tensor,
    routed_w2_scale: pl.Tensor[None, pl.FP8E8M0, SCALE_BANK_LAYOUT],
    routed_w3: pl.Tensor,
    routed_w3_scale: pl.Tensor[None, pl.FP8E8M0, SCALE_BANK_LAYOUT],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor,
    recv_x: pld.DistributedTensor,
    recv_routes: pld.DistributedTensor,
    data_arrived: pld.DistributedTensor,
    routed_output: pld.DistributedTensor,
    combine_arrived: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    force_topk: pl.Tensor,
    recv_x_local: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_x_last_use: pl.Array[1, pl.TASK_ID],
    local_experts: pl.constexpr,
    topk: pl.constexpr,
    dispatch_worker_limit: pl.constexpr,
    expert_state: pl.Tensor,
    expert_descriptors: pl.Tensor,
    expert_layer: pl.Scalar[pl.INDEX],
) -> pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]:
    """Run delayed HC pre-mix, MoE, and residual expansion."""
    t = MOE_TOKENS
    rw13, rw2, ract, sw13, sw2, sact = _decode_ffn_workers(num_tokens, local_experts, topk)
    post_mix = pl.create_tensor([t, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([t, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_norm = pl.create_tensor([t, D], dtype=pl.BF16)
    hc_head_norm_store(x_hc, pre_mix, norm_weight, x_norm, x_mixed, num_tokens, worker_limit=64)
    inv_rms, partials = hc_project(x_hc, hc_ffn_fn, rms_worker_limit=64)
    mixes_ready = hc_sinkhorn(inv_rms, partials, hc_ffn_scale, hc_ffn_base, next_pre_mix, post_mix, residual_mix)

    # MoE transport and routed-result scope.
    with pl.scope():
        x_norm_mx = pl.create_tensor([t, D], dtype=pl.FP8E4M3FN)
        x_norm_scale = pl.create_tensor([1, t * (D // MX_GROUP)], dtype=pl.FP8E8M0)
        indices = pl.create_tensor([MOE_TOKENS, topk], dtype=pl.INT32)
        weights = pl.create_tensor([t, topk], dtype=pl.FP32)
        gate(
            x_norm, gate_weight, correction_bias, num_tokens,
            x_norm_mx, x_norm_scale, indices, weights, n_experts=EP_SIZE * local_experts, topk=topk,
        )
        if BALANCED_ROUTING:
            dispatch_indices = force_topk
        else:
            dispatch_indices = indices

        shared_output = pl.create_tensor([t, D], dtype=pl.BF16)
        shared_result = expert_shared(
            x_norm_mx, x_norm_scale,
            shared_w1, shared_w1_scale,
            shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            shared_output,
        )

        # Contiguous ND backing for the routed-expert MX_A_ZZ view.
        recv_scale_local_backing = pl.create_tensor([1, local_experts * RECV_MAX * (D // MX_GROUP)], dtype=pl.FP8E8M0)
        recv_weight_local = pl.create_tensor([local_experts, RECV_MAX], dtype=pl.FP32)
        recv_route_local = pl.create_tensor([local_experts, RECV_MAX], dtype=pl.INT32)
        recv_count_local = pl.create_tensor([local_experts, 1], dtype=pl.INT32)
        recv_meta_local = pl.create_tensor([EP_SIZE, local_experts], dtype=pl.INT32)
        previous_use = recv_x_last_use[0]
        dispatch_deps = pl.array.create(2, pl.TASK_ID)
        if SLOT_DECODE and num_tokens == 1:
            dispatch_deps[0] = publish_routed_counts(
                dispatch_indices, recv_count_local, recv_meta_local, recv_meta, data_arrived,
                num_tokens, ep_rank, moe_epoch, local_experts, topk)
            dispatch_deps[1] = gather_single_token_inputs(
                dispatch_indices, x_norm_mx, x_norm_scale, weights,
                recv_x_local, recv_scale_local_backing, recv_weight_local, previous_use, topk)
        else:
            generic_counts_ready, generic_input_ready = dispatch(
                dispatch_indices, x_norm_mx, x_norm_scale, weights,
                recv_x_local, recv_scale_local_backing,
                recv_weight_local, recv_route_local, recv_count_local, recv_meta_local,
                recv_meta, recv_x, recv_routes,
                data_arrived, num_tokens, ep_rank, moe_epoch, previous_use,
                local_experts=local_experts, topk=topk, dispatch_worker_limit=dispatch_worker_limit, early_counts=True,
            )
            dispatch_deps[0] = generic_counts_ready
            dispatch_deps[1] = generic_input_ready
        counts_ready = dispatch_deps[0]
        input_ready = dispatch_deps[1]

        sublayer = pl.create_tensor([t, D], dtype=pl.BF16)
        output_deps = pl.array.create(1, pl.TASK_ID)
        if SLOT_DECODE and num_tokens == 1:
            routed_rows = pl.create_tensor([topk * 16, D], dtype=pl.BF16)
            routed_rows, routed_ready = compute_single_token_slots(
                recv_x_local, recv_scale_local_backing, recv_weight_local, recv_count_local,
                dispatch_indices, expert_state, expert_descriptors,
                routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
                routed_rows, expert_layer, counts_ready, input_ready, topk)
            recv_x_last_use[0] = routed_ready
            output_deps[0] = combine_single_token_slots(
                shared_result, routed_rows, sublayer, routed_ready, topk)
        else:
            slot_map, bank_w1, scale_w1, bank_w2, scale_w2, bank_w3, scale_w3, weights_ready = supply_experts(
                recv_count_local, expert_state, expert_descriptors, routed_w1, routed_w1_scale,
                routed_w2, routed_w2_scale, routed_w3, routed_w3_scale, expert_layer, counts_ready,
                experts=local_experts)

            compute_ready = pl.system.task_dummy(deps=[weights_ready, input_ready])
            scatter_done = expert_routed_scatter(
                recv_x_local, recv_scale_local_backing,
                recv_weight_local, recv_count_local,
                bank_w1, scale_w1, bank_w3, scale_w3,
                bank_w2, scale_w2,
                recv_route_local, recv_meta_local,
                routed_output, compute_ready, slot_map, by_slot=True, local_experts=local_experts, topk=topk,
                w13_workers=rw13, w2_workers=rw2, swiglu_workers=ract,
            )
            recv_x_last_use[0] = scatter_done
            output_deps[0] = combine_scattered(
                shared_result, sublayer, routed_output, combine_arrived,
                scatter_done, num_tokens, ep_rank, moe_epoch, topk=topk)
        hc_post_deps = pl.array.create(2, pl.TASK_ID)
        hc_post_deps[0] = output_deps[0]
        hc_post_deps[1] = mixes_ready
        hc_post(sublayer, x_hc, post_mix, residual_mix, x_next, deps=hc_post_deps, dep_count=2)
    return x_next
