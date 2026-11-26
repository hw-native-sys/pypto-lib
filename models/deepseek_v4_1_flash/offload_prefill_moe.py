# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Split prefill MoE for the mapped expert bank."""

import pypto.language as pl
import pypto.language.distributed as pld
SCALE_BANK_LAYOUT = pl.ND
from config import FLASH as M
from config import EP_SIZE
from config import MOE_TOKENS, MOE_RECV_MAX, N_LOCAL_EXPERTS, LOCAL_T_DYN, ROUTE_T_DYN
from config import T_DYN
from expert_supply import supply_experts
from gate import gate, gather_routing_bias
from expert_shared import expert_shared

BIAS_DYN = pl.dynamic("PREFILL_MOE_BIAS_DYN")
from offload_expert_routed import expert_routed_scatter
from expert_routed import (
    ROUTED_CUBE_WORKER_TILE, ROUTED_VECTOR_WORKER_TILE,
)
from hc import hc_head_norm_store, hc_project, hc_sinkhorn, hc_post
from input_pack import gather_rows


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
RECV_MAX = MOE_RECV_MAX
AUX_WIDTH = 8
ROUTE_WIDTH = 8
T = MOE_TOKENS
TOKENS = MOE_TOKENS
N_RANKS = EP_SIZE
N_LOCAL = N_LOCAL_EXPERTS
K_SCALE = D // MX_GROUP
MAX_PER_SRC = T
AUX_W = 0
AUX_PAD = AUX_WIDTH
IDX_PAD = ROUTE_WIDTH
SIGNAL_PAD = 128  # 512-byte padded monotonic counter lane

# tiling
SCALE_PACK_TMP_TILE = 2 * ((K_SCALE + 63) // 32) * 32
RECV_GATHER_TILE = 32


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
    scale_src = pl.create_tensor([T, K_SCALE], dtype=pl.UINT8, manual_dep=True)
    with pl.spmd(32, name_hint="dispatch_scale_unpack") as _stage_tid:
        worker = pl.tile.get_block_idx()
        for first in pl.range(worker * 16, num_tokens, 32 * 16):
            packed = pl.load(x_norm_scale, [0, first * K_SCALE], [1, 16 * K_SCALE])
            scale_bytes = pl.reinterpret_view(packed, pl.UINT8)
            pairs = pl.reinterpret_view(scale_bytes, pl.UINT16, shape=[K_SCALE // 2, 16])
            pairs_nd = pl.transpose(pairs, 0, 1)
            bytes_nd = pl.reinterpret_view(pairs_nd, pl.UINT8, shape=[16, K_SCALE])
            pl.store(bytes_nd, [first, 0], scale_src)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait") as _reuse_tid:
        _indices_anchor = pl.read(indices, [0, 0])

    route_positions = pl.create_tensor([N_RANKS * N_LOCAL, MAX_PER_SRC], dtype=pl.INT32, manual_dep=True)
    send_counts = pl.create_tensor([N_RANKS * N_LOCAL], dtype=pl.INT32, manual_dep=True)
    # Publish per-source route counts.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_publish", deps=[_reuse_tid], allow_early_resolve=True) as _meta_push_tid:
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
        indices_tile = pl.tile.load(indices, [0, 0], [T, IDX_PAD], valid_shape=[T, TOPK])
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                lane = dst * N_LOCAL + loc_e
                slot = cursor[lane]
                pl.write(route_positions, [lane, slot], pl.cast(t * TOPK + k, pl.INT32))
                cursor[lane] = slot + 1

        meta_tile = pl.tile.full([1, (N_LOCAL + 7) // 8 * 8], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
                pl.write(send_counts, [dst * N_LOCAL + e], cursor[dst * N_LOCAL + e])
            meta_valid = pl.tile.set_validshape(meta_tile, 1, N_LOCAL)
            pld.tile.remote_store(meta_valid, target=recv_meta, peer=dst, offsets=[my_rank, 0])
            if dst != my_rank:
                pld.system.notify(
                    target=arrived, peer=dst, offsets=[my_rank, 0],
                    value=moe_epoch, op=pld.NotifyOp.Set,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait") as _meta_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=arrived, offsets=[src, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_finalize", deps=[_meta_push_tid, _meta_wait_tid], allow_early_resolve=True) as _meta_tid:
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Publish one payload lane per destination and local expert.
    push_workers = pl.min(N_RANKS * N_LOCAL, 56)
    with pl.spmd(push_workers, name_hint="dispatch_push", deps=[_reuse_tid, _stage_tid, _meta_push_tid], allow_early_resolve=False) as _push_tid:
        worker = pl.tile.get_block_idx()
        for push_block in pl.range(worker, N_RANKS * N_LOCAL, push_workers):
            dst = push_block // N_LOCAL
            loc_e = push_block - dst * N_LOCAL
            active_tokens = pl.cast(num_tokens, pl.INDEX)
            if active_tokens < 0:
                active_tokens = pl.cast(0, pl.INDEX)
            if active_tokens > T:
                active_tokens = pl.cast(T, pl.INDEX)

            e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC
            route_count = pl.cast(pl.read(send_counts, [push_block]), pl.INDEX)
            if route_count > 0:
                positions = pl.load(route_positions, [push_block, 0], [1, MAX_PER_SRC])
                weights_tile = pl.tile.load(weights, [0, 0], [T, AUX_PAD], valid_shape=[T, TOPK])
                aux_lane = pl.tile.full([MAX_PER_SRC, AUX_PAD], dtype=pl.FP32, value=0.0)
                route_lane = pl.tile.full([MAX_PER_SRC, IDX_PAD], dtype=pl.INT32, value=0)
                for slot in pl.range(route_count):
                    route_index = pl.tile.read(positions, [0, slot])
                    t = pl.cast(route_index // TOPK, pl.INDEX)
                    k = pl.cast(route_index % TOPK, pl.INDEX)
                    row = e_lane_base + slot
                    payload = pl.load(x_norm_mx, [t, 0], [1, D])
                    payload_bytes = pl.reinterpret_view(payload, pl.INT8)
                    pld.tile.remote_store(payload_bytes, recv_x, peer=dst, offsets=[row, 0])
                    pld.tensor.put(
                        dst=recv_scale, peer=dst, src=scale_src,
                        dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, K_SCALE],
                    )
                    route_weight = pl.tile.read(weights_tile, [t, k])
                    pl.tile.write(aux_lane, [slot, AUX_W], route_weight)
                    pl.tile.write(route_lane, [slot, 0], route_index)
                pld.tile.remote_store(aux_lane, target=recv_weights, peer=dst, offsets=[e_lane_base, 0])
                pld.tile.remote_store(route_lane, target=recv_routes, peer=dst, offsets=[e_lane_base, 0])
    # Publish one epoch only after every local payload store completes.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_publish_epoch", deps=[_push_tid], allow_early_resolve=False) as _publish_tid:
        for dst in pl.range(N_RANKS):
            if dst != my_rank:
                pld.system.notify(
                    target=data_arrived, peer=dst, offsets=[my_rank, 0, 0],
                    value=moe_epoch, op=pld.NotifyOp.Set,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[_publish_tid]) as _wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(
                    signal=data_arrived, offsets=[src, 0, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

    # Gather compact expert rows and pack their MX scales.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])
    with pl.spmd(32, name_hint="dispatch_gather_pack", deps=[_wait_tid, _push_tid, _meta_tid]) as gathered:
        worker = pl.tile.get_block_idx()
        for e in pl.range(worker, N_LOCAL, 32):
            expert_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
            for tile_row in pl.range(0, expert_rows, RECV_GATHER_TILE):
                tile_end = pl.min(tile_row + RECV_GATHER_TILE, expert_rows)
                e_base_row = e * RECV_MAX
                gather_scale_tile = pl.tile.full([RECV_GATHER_TILE, K_SCALE], dtype=pl.UINT8, value=127)
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    source_count = pl.read(recv_meta_local, [src, e])
                    source_rows = pl.cast(source_count, pl.INDEX)
                    source_end = source_begin + source_rows
                    src_base_row = e_base_row + src * MAX_PER_SRC
                    gather_begin = pl.max(tile_row, source_begin)
                    gather_end = pl.min(tile_end, source_end)
                    if gather_begin < gather_end:
                        in_row = src_base_row + gather_begin - source_begin
                        out_row = e_base_row + gather_begin
                        gather_rows = gather_end - gather_begin
                        recv_x_raw = pl.load(recv_x, [in_row, 0], [RECV_GATHER_TILE, D], valid_shape=[gather_rows, D])
                        recv_x_mx = pl.reinterpret_view(recv_x_raw, pl.FP8E4M3FN)
                        recv_x_out_flat = pl.store(recv_x_mx, [out_row, 0], recv_x_out_flat)
                        recv_scale_raw = pl.load(recv_scale, [in_row, 0], [RECV_GATHER_TILE, K_SCALE], valid_shape=[gather_rows, K_SCALE])
                        gather_scale_tile = pl.tile.assemble(gather_scale_tile, recv_scale_raw, [gather_begin - tile_row, 0])
                    for out_col in pl.range(gather_begin, gather_end):
                        in_row = src_base_row + out_col - source_begin
                        recv_weight = pl.read(recv_weights, [in_row, AUX_W])
                        pl.write(recv_weight_out, [e, out_col], recv_weight)
                        recv_route = pl.read(recv_routes, [in_row, 0])
                        pl.write(recv_route_out, [e, out_col], recv_route)
                    source_begin = source_end
                gather_scale_tmp = pl.create_tile([1, SCALE_PACK_TMP_TILE], dtype=pl.UINT8)
                gather_packed = pl.tmov_x2zz(gather_scale_tile, gather_scale_tmp, group_axis=1, dst_rows=RECV_GATHER_TILE, dst_cols=K_SCALE)
                gather_packed_mx = pl.reinterpret_view(gather_packed, pl.FP8E8M0)
                gather_packed_flat = pl.reshape(gather_packed_mx, [1, RECV_GATHER_TILE * K_SCALE])
                packed_offset = (e_base_row + tile_row) * K_SCALE
                recv_scale_out = pl.store(gather_packed_flat, [0, packed_offset], recv_scale_out)
    return _meta_tid, gathered


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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_notify", deps=[scatter_done], allow_early_resolve=False) as notify_tid:
        for peer in pl.range(N_RANKS):
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

    # Wait for local and peer routed-output stores.
    with pl.spmd(32, name_hint="combine_reduce", deps=[_cwait_tid, output_ready]) as _reduce_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, 32):
            shared_row = pl.load(shared_output, [t, 0], [1, D])
            acc = pl.cast(shared_row, target_type=pl.FP32)
            for k in pl.range(TOPK):
                routed_row = pl.load(routed_output, [t * TOPK + k, 0], [1, D])
                routed_fp32 = pl.cast(routed_row, target_type=pl.FP32)
                acc = pl.add(acc, routed_fp32)
            ffn_row = pl.cast(acc, target_type=pl.BF16, mode="rint")
            ffn_out = pl.store(ffn_row, [t, 0], ffn_out)
    return _reduce_tid


@pl.jit.inline(auto_scope=False)
def prefill_moe(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
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
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    expert_state: pl.Tensor,
    expert_descriptors: pl.Tensor,
    expert_layer: pl.Scalar[pl.INDEX],
) -> pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]:
    """Run delayed HC pre-mix, MoE, and residual expansion."""
    t = MOE_TOKENS
    post_mix = pl.create_tensor([t, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([t, HC_MULT, HC_MULT], dtype=pl.FP32)
    ffn_input = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    hc_head_norm_store(x_hc, pre_mix, norm_weight, ffn_input, x_mixed, num_tokens, worker_limit=64)
    inv_rms, partials = hc_project(x_hc, hc_ffn_fn, rms_worker_limit=64)
    mixes_ready = hc_sinkhorn(inv_rms, partials, hc_ffn_scale, hc_ffn_base, next_pre_mix, post_mix, residual_mix)

    # MoE transport and routed-result scope.
    with pl.scope():
        sublayer = pl.create_tensor([t, D], dtype=pl.BF16)
        with pl.spmd(8, name_hint="moe_output_zero") as _output_zero_tid:
            worker = pl.tile.get_block_idx()
            zero_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
            for zero_t in pl.range(worker, t, 8):
                sublayer = pl.store(zero_row, [zero_t, 0], sublayer)

        x_norm_mx = pl.create_tensor([t, D], dtype=pl.FP8E4M3FN)
        x_norm_scale = pl.create_tensor([1, t * (D // MX_GROUP)], dtype=pl.FP8E8M0)
        indices = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
        weights = pl.create_tensor([t, TOPK], dtype=pl.FP32)
        gate(
            ffn_input, gate_weight, correction_bias, num_tokens,
            x_norm_mx, x_norm_scale, indices, weights, n_experts=N_EXPERTS, topk=TOPK,
        )
        dispatch_indices = indices

        shared_output = pl.create_tensor([t, D], dtype=pl.BF16)
        shared_result = expert_shared(
            x_norm_mx, x_norm_scale,
            shared_w1, shared_w1_scale,
            shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            shared_output,
        )

        recv_x_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.FP8E4M3FN)
        # Contiguous ND backing for the routed-expert MX_A_ZZ view.
        recv_scale_local_backing = pl.create_tensor([1, N_LOCAL_EXPERTS * RECV_MAX * (D // MX_GROUP)], dtype=pl.FP8E8M0)
        recv_weight_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
        recv_route_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
        recv_count_local = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
        recv_meta_local = pl.create_tensor([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        _counts_ready, input_ready = dispatch(
            dispatch_indices, x_norm_mx, x_norm_scale, weights,
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_route_local, recv_count_local, recv_meta_local,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, num_tokens, ep_rank, moe_epoch,
        )

        slot_map, bank_w1, scale_w1, bank_w2, scale_w2, bank_w3, scale_w3, weights_ready = supply_experts(
            recv_count_local, expert_state, expert_descriptors, routed_w1, routed_w1_scale,
            routed_w2, routed_w2_scale, routed_w3, routed_w3_scale, expert_layer, input_ready,
            experts=N_LOCAL_EXPERTS)

        compute_ready = pl.system.task_dummy(deps=[weights_ready, input_ready])
        scatter_done = expert_routed_scatter(
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_count_local,
            bank_w1, scale_w1, bank_w3, scale_w3,
            bank_w2, scale_w2,
            recv_route_local, recv_meta_local,
            routed_output, compute_ready, slot_map, by_slot=True, local_experts=N_LOCAL_EXPERTS, topk=TOPK,
            w13_workers=ROUTED_CUBE_WORKER_TILE, w2_workers=ROUTED_CUBE_WORKER_TILE,
            swiglu_workers=ROUTED_VECTOR_WORKER_TILE,
        )
        output_ready = combine_scattered(
            shared_result, sublayer,
            routed_output, combine_arrived,
            _output_zero_tid, scatter_done,
            num_tokens, ep_rank, moe_epoch,
        )
        # Complete all peer readers before reusing the windows.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_round_release", deps=[output_ready]) as released:
            for peer in pl.range(EP_SIZE):
                if peer != ep_rank:
                    pld.system.notify(
                        target=combine_arrived, peer=peer, offsets=[ep_rank, 1, 0],
                        value=moe_epoch, op=pld.NotifyOp.Set,
                    )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_round_released", deps=[released]) as round_ready:
            for peer in pl.range(EP_SIZE):
                if peer != ep_rank:
                    pld.system.defer_wait(
                        signal=combine_arrived, offsets=[peer, 1, 0],
                        expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                    )
        hc_post_deps = pl.array.create(2, pl.TASK_ID)
        hc_post_deps[0] = round_ready
        hc_post_deps[1] = mixes_ready
        hc_post(sublayer, x_hc, post_mix, residual_mix, x_next, deps=hc_post_deps, dep_count=2)
    return x_next


@pl.jit.inline(auto_scope=False)
def prefill_moe_slabs(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor,
    routed_w1: pl.Tensor,
    routed_w1_scale: pl.Tensor[None, pl.FP8E8M0, SCALE_BANK_LAYOUT],
    routed_w2: pl.Tensor,
    routed_w2_scale: pl.Tensor[None, pl.FP8E8M0, SCALE_BANK_LAYOUT],
    routed_w3: pl.Tensor,
    routed_w3_scale: pl.Tensor[None, pl.FP8E8M0, SCALE_BANK_LAYOUT],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[
        [D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[
        [MOE_INTER // 32, D], pl.FP8E8M0, pl.MX_B_NN
    ],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[
        [D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    x_next: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, D // 32], pl.UINT8
    ],
    recv_weights: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32
    ],
    recv_routes: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32
    ],
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    slab_hidden: pl.Tensor[[2 * MOE_TOKENS, HC_MULT, D], pl.FP32],
    slab_pre: pl.Tensor[[2 * MOE_TOKENS, HC_MULT], pl.FP32],
    slab_mixed: pl.Tensor[[MOE_TOKENS, D], pl.BF16],
    slab_rows: pl.Tensor[[MOE_TOKENS], pl.INT32],
    reuse_ready: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    capacity: pl.constexpr,
    expert_state: pl.Tensor,
    expert_descriptors: pl.Tensor,
    expert_layer: pl.Scalar[pl.INDEX],
):
    """Run bounded MoE slabs and return completion of every scratch reader."""
    source_hidden = pl.reshape(x_hc, [capacity, HC_DIM])
    round_completion = pl.array.create(1, pl.TASK_ID)
    round_completion[0] = reuse_ready
    owner_capacity = capacity
    owner_begin = 0
    rounds = (owner_capacity + MOE_TOKENS - 1) // MOE_TOKENS
    for round_id in pl.range(rounds):
        begin = owner_begin + round_id * MOE_TOKENS
        rows = pl.min(MOE_TOKENS, owner_begin + owner_capacity - begin)
        # Keep caller-owned destinations outside the slab allocation scope.
        hidden_destination = pl.slice(x_next, [rows, HC_MULT, D], [begin, 0, 0])
        mixed_destination = pl.slice(x_mixed, [rows, D], [begin, 0])
        pre_destination = pl.slice(next_pre_mix, [rows, HC_MULT], [begin, 0])
        # Carry completion explicitly before overwriting shared scratch.
        row_ids = pl.slice(slab_rows, [MOE_TOKENS], [0])
        round_hidden = pl.slice(slab_hidden, [MOE_TOKENS, HC_MULT, D], [0, 0, 0])
        round_output = pl.slice(slab_hidden, [MOE_TOKENS, HC_MULT, D], [MOE_TOKENS, 0, 0])
        round_pre = pl.slice(slab_pre, [MOE_TOKENS, HC_MULT], [0, 0])
        round_next = pl.slice(slab_pre, [MOE_TOKENS, HC_MULT], [MOE_TOKENS, 0])
        round_mixed = pl.slice(slab_mixed, [MOE_TOKENS, D], [0, 0])
        with pl.scope():
            active = pl.max(0, pl.min(rows, num_tokens - begin))
            with pl.spmd(1, name_hint="prefill_moe_rows", deps=[round_completion[0]]) as rows_ready:
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, MOE_TOKENS):
                    source_row = pl.cast(-1, pl.INT32)
                    if row < active:
                        source_row = pl.cast(begin + row, pl.INT32)
                    pl.write(row_ids, [row], source_row)
            packed_hidden = pl.reshape(round_hidden, [MOE_TOKENS, HC_DIM])
            gather_rows(source_hidden, row_ids, packed_hidden, rows_ready, column_tile=D)
            gather_rows(pre_mix, row_ids, round_pre, rows_ready, column_tile=512)
            bias_rows = pl.min(pl.tensor.dim(correction_bias, 0) // N_EXPERTS, MOE_TOKENS)
            bias_tile_rows = bias_rows
            if bias_rows > 1:
                bias_tile_rows = (bias_rows + 15) // 16 * 16
            bias_elements = bias_tile_rows * N_EXPERTS
            round_bias = pl.create_tensor([bias_elements], dtype=pl.FP32)
            gather_routing_bias(correction_bias, row_ids, round_bias, rows_ready, n_experts=N_EXPERTS)
            epoch = (moe_epoch - 1) * rounds + round_id + 1
            prefill_moe(
                round_hidden, round_pre, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_weight,
                gate_weight, round_bias,
                routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
                shared_w1, shared_w1_scale, shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
                round_next, round_mixed, round_output,
                recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived,
                routed_output, combine_arrived, active, ep_rank, epoch,
                expert_state=expert_state, expert_descriptors=expert_descriptors, expert_layer=expert_layer,
            )
            with pl.spmd(32, name_hint="prefill_moe_results") as hidden_copied:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, MOE_TOKENS, 32):
                    if row < rows:
                        for lane in pl.range(HC_MULT):
                            for column in pl.range(0, D, D):
                                hidden = round_output[row:row + 1, lane:lane + 1, column:column + D]
                                hidden_destination[row:row + 1, lane:lane + 1, column:column + D] = hidden
                        mixed = round_mixed[row:row + 1, :]
                        mixed_destination[row:row + 1, :] = mixed
            with pl.spmd(1, name_hint="prefill_moe_pre_mix") as pre_copied:
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, MOE_TOKENS):
                    if row < rows:
                        for lane in pl.range(HC_MULT):
                            value = pl.read(round_next, [row, lane])
                            pl.write(pre_destination, [row, lane], value)
            round_completion[0] = pl.system.task_dummy(deps=[hidden_copied, pre_copied])
    return round_completion[0]
