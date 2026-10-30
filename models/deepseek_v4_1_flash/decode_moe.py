# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4.1 decode MoE with mHC and expert parallelism."""

from dataclasses import replace
import sys
import pypto.language as pl
import pypto.language.distributed as pld
import config


def _parse_experts_per_rank():
    for index, argument in enumerate(sys.argv):
        if argument == "--experts-per-rank" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith("--experts-per-rank="):
            return int(argument.split("=", 1)[1])
    return config.N_LOCAL_EXPERTS


# Resolve test expert dimensions before importing kernels.
_experts_per_rank = _parse_experts_per_rank()
_test_experts = config.EP_SIZE * _experts_per_rank
if _experts_per_rank <= 0 or not config.FLASH.num_experts_per_tok <= _test_experts <= 384:
    raise ValueError("--experts-per-rank must give between TopK and 384 total routed experts")
config.FLASH = replace(config.FLASH, n_routed_experts=_test_experts)
config.N_LOCAL_EXPERTS = _experts_per_rank


from config import FLASH as M, EP_SIZE, MOE_TOKENS, MOE_RECV_MAX, N_LOCAL_EXPERTS, LOCAL_T_DYN, ROUTE_T_DYN
from gate import gate
from expert_shared import expert_shared
from expert_routed import (
    MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS, expert_routed_scatter,
)
from hc import hc_pre, hc_post


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
AUX_WIDTH = 8
ROUTE_WIDTH = 8
T = MOE_TOKENS
N_RANKS = EP_SIZE
N_LOCAL = N_LOCAL_EXPERTS
K_SCALE = D // MX_GROUP
MAX_PER_SRC = T
AUX_W = 0
AUX_PAD = AUX_WIDTH
IDX_PAD = ROUTE_WIDTH
SIGNAL_PAD = 128  # 512-byte padded monotonic counter lane

# tiling
DISPATCH_STAGE_TILE = 8
COMBINE_T_TILE = 2
SCALE_PACK_TMP_TILE = ((64 + K_SCALE + 31) // 32) * 32
RECV_TILE = 16


@pl.jit.inline(auto_scope=False)
def clear_moe_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    anchor_indices: pl.constexpr,
):
    """Clear a completed entry's MoE arrival signals."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, anchor_indices)
        zero = pl.cast(0, pl.INT32)
        for peer in pl.range(EP_SIZE):
            pl.write(arrived, [peer, 0], zero)
            pl.write(data_arrived, [peer, 0, 0], zero)
            pl.write(combine_arrived, [peer, 0, 0], zero)
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
    # Raw payload and logical MX scale rows in ND order.
    x_norm_mx_raw = pl.create_tensor([T, D], dtype=pl.INT8)
    scale_src = pl.create_tensor([T, K_SCALE], dtype=pl.UINT8, manual_dep=True)
    with pl.spmd(T // DISPATCH_STAGE_TILE, name_hint="dispatch_stage") as _stage_tid:
        t_begin = pl.tile.get_block_idx() * DISPATCH_STAGE_TILE
        raw_row = pl.load(x_norm_mx, [t_begin, 0], [DISPATCH_STAGE_TILE, D])
        raw_row_i8 = pl.reinterpret_view(raw_row, pl.INT8)
        x_norm_mx_raw = pl.store(raw_row_i8, [t_begin, 0], x_norm_mx_raw)
        scale_offset = t_begin // 16 * 16 * K_SCALE
        raw_scale = pl.load(x_norm_scale, [0, scale_offset], [1, 16 * K_SCALE])
        raw_scale_u8 = pl.reinterpret_view(raw_scale, pl.UINT8)
        raw_scale_i8 = pl.reinterpret_view(raw_scale_u8, pl.INT8)
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        # Each block owns whole 64-byte lines in the UINT8 scale buffer.
        t_end = pl.min(t_begin + DISPATCH_STAGE_TILE, active_tokens)
        for t in pl.range(t_begin, t_end):
            for group in pl.range(K_SCALE):
                scale_index = group // 2 * 32 + t % 16 * 2 + group % 2
                scale_i8 = pl.tile.read(raw_scale_i8, [0, scale_index])
                scale = pl.cast(scale_i8, pl.UINT8)
                pl.write(scale_src, [t, group], scale)

    # Publish per-source route counts.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_publish", allow_early_resolve=True) as _meta_push_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        # Route counts for each destination and local expert.
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
                cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        meta_tile = pl.tile.full([1, (N_LOCAL + 7) // 8 * 8], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            meta_valid = pl.tile.set_validshape(meta_tile, 1, N_LOCAL)
            pld.tile.remote_store(meta_valid, target=recv_meta, peer=dst, offsets=[my_rank, 0])
            if dst != my_rank:
                pld.system.notify(target=arrived, peer=dst, offsets=[my_rank, 0], value=moe_epoch, op=pld.NotifyOp.Set)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_wait") as _meta_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=arrived, offsets=[src, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_meta_finalize", deps=[_meta_push_tid, _meta_wait_tid], allow_early_resolve=True) as _meta_tid:
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Publish one payload lane per destination and local expert.
    with pl.spmd(N_RANKS * N_LOCAL, name_hint="dispatch_push", deps=[_stage_tid], allow_early_resolve=False) as _push_tid:
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
        aux_lane = pl.tile.full([MAX_PER_SRC, AUX_PAD], dtype=pl.FP32, value=0.0)
        route_lane = pl.tile.full([MAX_PER_SRC, IDX_PAD], dtype=pl.INT32, value=0)
        slot_ctr = pl.array.create(1, pl.INT32)
        slot_ctr[0] = 0

        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.tile.read(indices_tile, [t, k])
                route_dst = eid // N_LOCAL
                route_local_e = eid - route_dst * N_LOCAL
                if route_local_e == loc_e:
                    if route_dst == dst:
                        slot = slot_ctr[0]
                        slot_ctr[0] = slot + 1
                        row = e_lane_base + slot
                        pld.tensor.put(dst=recv_x, peer=dst, src=x_norm_mx_raw, dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, D])
                        pld.tensor.put(dst=recv_scale, peer=dst, src=scale_src, dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, K_SCALE])
                        route_index = pl.cast(t * TOPK + k, pl.INT32)
                        route_weight = pl.tile.read(weights_tile, [t, k])
                        pl.tile.write(aux_lane, [slot, AUX_W], route_weight)
                        pl.tile.write(route_lane, [slot, 0], route_index)

        pld.tile.remote_store(aux_lane, target=recv_weights, peer=dst, offsets=[e_lane_base, 0])
        pld.tile.remote_store(route_lane, target=recv_routes, peer=dst, offsets=[e_lane_base, 0])
    # Publish one epoch only after every local payload store completes.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_publish_epoch", deps=[_push_tid], allow_early_resolve=False) as _publish_tid:
        for dst in pl.range(N_RANKS):
            if dst != my_rank:
                pld.system.notify(target=data_arrived, peer=dst, offsets=[my_rank, 0, 0], value=moe_epoch, op=pld.NotifyOp.Set)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait", deps=[_publish_tid]) as _wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=data_arrived, offsets=[src, 0, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)

    # Gather aligned row tiles into compact expert buffers.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])
    recv_scale_nd = pl.create_tensor([N_LOCAL * RECV_MAX, K_SCALE], dtype=pl.FP8E8M0, manual_dep=True)
    gather_tids = pl.array.create(N_LOCAL, pl.TASK_ID)
    for e in pl.parallel(N_LOCAL):
        gather_tile_tids = pl.array.create(((RECV_MAX + RECV_TILE - 1) // RECV_TILE), pl.TASK_ID)
        for tile in pl.range(((RECV_MAX + RECV_TILE - 1) // RECV_TILE)):
            gather_tile_tids[tile] = pl.system.task_dummy(deps=[])
        # recv_count_out is produced by a device task.  Keep graph
        # construction independent of that runtime value; source metadata
        # below still bounds the actual rows copied into the local buffer.
        expert_rows = pl.cast(RECV_MAX, pl.INDEX)
        expert_tiles = (RECV_MAX + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(expert_tiles):
            tile_row = tile * RECV_TILE
            tile_end = pl.min(tile_row + RECV_TILE, expert_rows)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_gather", deps=[_wait_tid, _push_tid, _meta_tid], allow_early_resolve=False) as gather_tile_tid:
                e_base_row = e * RECV_MAX
                source_begin = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    source_count = pl.read(recv_meta_local, [src, e])
                    source_rows = pl.cast(source_count, pl.INDEX)
                    source_end = source_begin + source_rows
                    src_base_row = e_base_row + src * MAX_PER_SRC
                    gather_begin = pl.max(tile_row, source_begin)
                    gather_end = pl.min(tile_end, source_end)
                    for out_col in pl.range(gather_begin, gather_end):
                        in_row = src_base_row + out_col - source_begin
                        out_row = e_base_row + out_col
                        recv_x_raw = pl.load(recv_x, [in_row, 0], [1, D])
                        recv_x_mx = pl.reinterpret_view(recv_x_raw, pl.FP8E4M3FN)
                        recv_x_out_flat = pl.store(recv_x_mx, [out_row, 0], recv_x_out_flat)
                        recv_scale_raw = pl.load(recv_scale, [in_row, 0], [1, K_SCALE])
                        recv_scale_mx = pl.reinterpret_view(recv_scale_raw, pl.FP8E8M0)
                        recv_scale_nd = pl.store(recv_scale_mx, [out_row, 0], recv_scale_nd)
                        recv_weight = pl.read(recv_weights, [in_row, AUX_W])
                        pl.write(recv_weight_out, [e, out_col], recv_weight)
                        recv_route = pl.read(recv_routes, [in_row, 0])
                        pl.write(recv_route_out, [e, out_col], recv_route)
                    source_begin = source_end
                flat_t0 = e_base_row + tile_row
                scale_nd = pl.load(recv_scale_nd, [flat_t0, 0], [RECV_TILE, K_SCALE])
                scale_raw = pl.reinterpret_view(scale_nd, pl.UINT8)
                tmp = pl.create_tile([1, SCALE_PACK_TMP_TILE], dtype=pl.UINT8)
                scale_zz_raw = pl.tmov_x2zz(scale_raw, tmp, group_axis=1, dst_rows=RECV_TILE, dst_cols=K_SCALE)
                scale_zz = pl.reinterpret_view(scale_zz_raw, pl.FP8E8M0)
                scale_zz_flat = pl.reshape(scale_zz, [1, RECV_TILE * K_SCALE])
                recv_scale_out = pl.store(scale_zz_flat, [0, flat_t0 * K_SCALE], recv_scale_out)
            gather_tile_tids[tile] = gather_tile_tid
        gather_tids[e] = pl.system.task_dummy(deps=[gather_tile_tids])
    _gather_tid = pl.system.task_dummy(deps=[gather_tids[e] for e in range(N_LOCAL)])

    return _gather_tid


@pl.jit.inline
def combine_scattered(
    shared_output: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    routed_output: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    scatter_done: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_notify", deps=[scatter_done], allow_early_resolve=False) as notify_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(target=combine_arrived, peer=peer, offsets=[my_rank, 0, 0], value=moe_epoch, op=pld.NotifyOp.Set)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_wait", deps=[notify_tid]) as _cwait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=combine_arrived, offsets=[src, 0, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge)

    # Wait for local and peer routed-output stores.
    with pl.spmd((T + COMBINE_T_TILE - 1) // COMBINE_T_TILE, name_hint="combine_reduce", deps=[_cwait_tid]) as _reduce_tid:
        t_begin = pl.tile.get_block_idx() * COMBINE_T_TILE
        for t in pl.range(t_begin, pl.min(t_begin + COMBINE_T_TILE, T)):
            if t < num_tokens:
                shared_row = pl.load(shared_output, [t, 0], [1, D])
                acc = pl.cast(shared_row, target_type=pl.FP32)
                for k in pl.range(TOPK):
                    routed_row = pl.load(routed_output, [t * TOPK + k, 0], [1, D])
                    routed_fp32 = pl.cast(routed_row, target_type=pl.FP32)
                    acc = pl.add(acc, routed_fp32)
                ffn_row = pl.cast(acc, target_type=pl.BF16, mode="rint")
                ffn_out = pl.store(ffn_row, [t, 0], ffn_out)
            else:
                zero_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
                ffn_out = pl.store(zero_row, [t, 0], ffn_out)
    return _reduce_tid


@pl.jit.inline(auto_scope=False)
def moe(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
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
    force_topk: pl.Tensor[[MOE_TOKENS, TOPK], pl.INT32],
) -> pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]:
    """Run delayed mHC pre-mix, MoE, and residual expansion."""
    t = MOE_TOKENS
    post_mix = pl.create_tensor([t, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([t, HC_MULT, HC_MULT], dtype=pl.FP32)
    mixes_ready = hc_pre(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, next_pre_mix, post_mix, residual_mix,
    )

    # MoE transport and routed-result scope.
    with pl.scope():
        x_norm_mx = pl.create_tensor([t, D], dtype=pl.FP8E4M3FN)
        x_norm_scale = pl.create_tensor([1, t * (D // MX_GROUP)], dtype=pl.FP8E8M0)
        indices = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
        weights = pl.create_tensor([t, TOPK], dtype=pl.FP32)
        gate(
            x_mixed, norm_weight, gate_weight, correction_bias, num_tokens,
            x_norm_mx, x_norm_scale, indices, weights,
        )
        if BALANCED_ROUTING:
            dispatch_indices = force_topk
        else:
            dispatch_indices = indices

        shared_output = pl.create_tensor([t, D], dtype=pl.BF16)
        expert_shared(
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
        input_ready = dispatch(
            dispatch_indices, x_norm_mx, x_norm_scale, weights,
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_route_local, recv_count_local, recv_meta_local,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, num_tokens, ep_rank, moe_epoch,
        )

        scatter_done = expert_routed_scatter(
            recv_x_local, recv_scale_local_backing,
            recv_weight_local, recv_count_local,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale, mxfp4_pair_lut,
            recv_route_local, recv_meta_local,
            routed_output, input_ready,
        )
        sublayer = pl.create_tensor([t, D], dtype=pl.BF16)
        output_ready = combine_scattered(
            shared_output, sublayer,
            routed_output, combine_arrived,
            scatter_done,
            num_tokens, ep_rank, moe_epoch,
        )
        hc_post_deps = pl.array.create(2, pl.TASK_ID)
        hc_post_deps[0] = output_ready
        hc_post_deps[1] = mixes_ready
        hc_post(sublayer, x_hc, post_mix, residual_mix, x_next, deps=hc_post_deps)
    return x_next


@pl.jit
def l2_moe(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
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
    num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    force_topk: pl.Tensor[[MOE_TOKENS, TOPK], pl.INT32],
) -> pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]:
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    next_pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    x_mixed.bind_dynamic(0, LOCAL_T_DYN)
    x_next.bind_dynamic(0, LOCAL_T_DYN)
    local_num_tokens = pl.read(num_tokens, [ep_rank])
    moe(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_weight, gate_weight, correction_bias,
        routed_w1, routed_w1_scale, routed_w2, routed_w2_scale,
        routed_w3, routed_w3_scale, mxfp4_pair_lut, shared_w1, shared_w1_scale,
        shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
        next_pre_mix, x_mixed, x_next,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived, routed_output, combine_arrived,
        local_num_tokens, ep_rank, moe_epoch, force_topk,
    )
    clear_moe_signals(x_next, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    return x_next


@pl.jit.host
def l3_moe(
    x_hc: pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    mxfp4_pair_lut: pl.Tensor[[EP_SIZE, 2, 256], pl.INT16],
    shared_w1: pl.Tensor[[EP_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    next_pre_mix: pl.Out[pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[EP_SIZE, MOE_TOKENS, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    force_topk: pl.Tensor[[EP_SIZE, MOE_TOKENS, TOPK], pl.INT32],
):
    """EP host driver for the mHC-wrapped Flash MoE block."""
    recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
    recv_weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
    routed_output_buf = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)

    rank_count = pld.world_size()
    for r in pl.range(rank_count):
        recv_meta = pld.window(recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buf, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
        # Per-rank MX_B_NN scale views.
        routed_w1_scale_r: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w1_scale[r]
        routed_w2_scale_r: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN] = routed_w2_scale[r]
        routed_w3_scale_r: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w3_scale[r]
        shared_w1_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[r]
        shared_w2_scale_r: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[r]
        shared_w3_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[r]
        l2_moe(
            x_hc[r], pre_mix[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_weight[r], gate_weight[r], correction_bias[r],
            routed_w1[r], routed_w1_scale_r, routed_w2[r], routed_w2_scale_r,
            routed_w3[r], routed_w3_scale_r, mxfp4_pair_lut[r], shared_w1[r], shared_w1_scale_r,
            shared_w2[r], shared_w2_scale_r, shared_w3[r], shared_w3_scale_r,
            next_pre_mix[r], x_mixed[r], x_next[r],
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            num_tokens, r, moe_epoch, force_topk[r], device=r,
        )


def _normalise_num_tokens(value):
    """Return validated local-token counts for every EP source rank."""
    import torch

    counts = torch.as_tensor(value, dtype=torch.int32).reshape(-1)
    if counts.numel() == 1:
        counts = counts.repeat(EP_SIZE)
    if counts.numel() != EP_SIZE:
        raise ValueError(f'num_tokens must be a scalar or contain {EP_SIZE} counts, got shape {tuple(counts.shape)}')
    if bool((counts < 0).any()) or bool((counts > MOE_TOKENS).any()):
        raise ValueError(f'num_tokens values must be in [0, {MOE_TOKENS}], got {counts.tolist()}')
    if BALANCED_ROUTING:
        if not bool((counts == counts[0]).all()) or int(counts[0]) * TOPK % EP_SIZE:
            raise ValueError("balanced routing requires equal token counts and per-rank routes divisible by EP")
    return counts.contiguous()


def _route_bias():
    """Build correction biases for TopK expert ids spread across ranks."""
    import torch

    selected = (torch.arange(TOPK, dtype=torch.int64) * max(1, N_EXPERTS // TOPK))
    selected = torch.remainder(selected, N_EXPERTS)
    bias = torch.full((N_EXPERTS,), -1.0, dtype=torch.float32)
    bias[selected] = 4.0
    return bias


def build_moe_specs(num_tokens=MOE_TOKENS, *, initialize=True):
    """Build mHC, expert weights, EP outputs, and runtime specifications."""
    import math
    import torch
    from functools import partial
    from golden import ScalarSpec, TensorSpec
    from expert_routed import ROUTED_DEQUANT_STD, gen_routed_mxfp4_weights, build_mxfp4_pair_lut
    from quant import generate_mxfp8_weight_kn

    counts = _normalise_num_tokens(num_tokens)
    torch.manual_seed(41)
    x = (torch.randn(EP_SIZE, MOE_TOKENS, D) * 0.25).to(torch.bfloat16)
    norm_weight = torch.ones(EP_SIZE, D, dtype=torch.bfloat16)
    gate_weight = torch.randn(N_EXPERTS, D) / D ** 0.5
    gate_weight = gate_weight.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    correction_bias = _route_bias().unsqueeze(0).expand(EP_SIZE, -1).contiguous()

    shapes = {
        "norm_weight": [EP_SIZE, D],
        "gate_weight": [EP_SIZE, N_EXPERTS, D],
        "correction_bias": [EP_SIZE, N_EXPERTS],
        "routed_w1": [EP_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        "routed_w1_scale": [EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        "routed_w2": [EP_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        "routed_w2_scale": [EP_SIZE, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
        "routed_w3": [EP_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        "routed_w3_scale": [EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        "mxfp4_pair_lut": [EP_SIZE, 2, 256],
        "shared_w1": [EP_SIZE, D, MOE_INTER],
        "shared_w1_scale": [EP_SIZE, D // MX_GROUP, MOE_INTER],
        "shared_w2": [EP_SIZE, MOE_INTER, D],
        "shared_w2_scale": [EP_SIZE, MOE_INTER // MX_GROUP, D],
        "shared_w3": [EP_SIZE, D, MOE_INTER],
        "shared_w3_scale": [EP_SIZE, D // MX_GROUP, MOE_INTER],
        "x_hc": [EP_SIZE, MOE_TOKENS, HC_MULT, D],
        "pre_mix": [EP_SIZE, MOE_TOKENS, HC_MULT],
        "hc_ffn_fn": [EP_SIZE, MIX_HC, HC_DIM],
        "hc_ffn_scale": [EP_SIZE, 3],
        "hc_ffn_base": [EP_SIZE, MIX_HC],
        "next_pre_mix": [EP_SIZE, MOE_TOKENS, HC_MULT],
        "x_mixed": [EP_SIZE, MOE_TOKENS, D],
        "x_next": [EP_SIZE, MOE_TOKENS, HC_MULT, D],
        "num_tokens": [EP_SIZE],
        "force_topk": [EP_SIZE, MOE_TOKENS, TOPK],
    }
    values = {"norm_weight": norm_weight, "gate_weight": gate_weight, "correction_bias": correction_bias}
    values["mxfp4_pair_lut"] = build_mxfp4_pair_lut().unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    if initialize:
        shards = [
            gen_routed_mxfp4_weights(N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=rank * N_LOCAL_EXPERTS * 3)
            for rank in range(EP_SIZE)
        ]
        routed_names = ("routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2", "routed_w2_scale")
        for index, name in enumerate(routed_names):
            values[name] = torch.stack([shard[index] for shard in shards])
        for name, out_dim, in_dim, std, chan_cv, seed in (
            ("shared_w1", MOE_INTER, D, 1.71e-2, 0.50, 101),
            ("shared_w3", MOE_INTER, D, 1.70e-2, 0.50, 102),
            ("shared_w2", D, MOE_INTER, 1.68e-2, 0.33, 103),
        ):
            weight, scale = generate_mxfp8_weight_kn(out_dim, in_dim, std, chan_cv=chan_cv, seed=seed)
            values[name] = weight.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
            values[name + "_scale"] = scale.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()

    generator = torch.Generator().manual_seed(3)
    magnitudes = (0.5 + torch.arange(MOE_TOKENS) % 4).reshape(1, -1, 1, 1)
    x_hc = torch.randn(EP_SIZE, MOE_TOKENS, HC_MULT, D, generator=generator) * magnitudes
    values["x_hc"] = x_hc.bfloat16().float()
    values["pre_mix"] = torch.sigmoid(torch.randn(EP_SIZE, MOE_TOKENS, HC_MULT, generator=generator)) + M.hc_eps
    hc_ffn_fn = torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)
    hc_ffn_scale = torch.randn(3, generator=generator)
    hc_ffn_base = torch.randn(MIX_HC, generator=generator)
    values["hc_ffn_fn"] = hc_ffn_fn.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    values["hc_ffn_scale"] = hc_ffn_scale.unsqueeze(0).expand(EP_SIZE, -1).contiguous()
    values["hc_ffn_base"] = hc_ffn_base.unsqueeze(0).expand(EP_SIZE, -1).contiguous()
    values["num_tokens"] = counts
    force_topk = torch.zeros(EP_SIZE, MOE_TOKENS, TOPK, dtype=torch.int32)
    for rank in range(EP_SIZE):
        active = int(counts[rank])
        expanded = active * TOPK
        slots = torch.arange(expanded, dtype=torch.int64)
        offsets = slots + expanded * rank + rank
        dst = offsets % EP_SIZE
        local_expert = offsets // EP_SIZE % N_LOCAL_EXPERTS
        force_topk[rank, :active] = (dst * N_LOCAL_EXPERTS + local_expert).reshape(active, TOPK)
    values["force_topk"] = force_topk

    fp8 = torch.float8_e4m3fn
    e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    nonresident = {"x_hc", "pre_mix", "next_pre_mix", "x_mixed", "x_next", "num_tokens"}
    specs = []
    for name in l3_moe.param_names:
        if name == "moe_epoch":
            specs.append(ScalarSpec(name, torch.int32, 1, compile_runtime=True))
            continue
        dtype = torch.float32
        if name == "norm_weight" or name == "x_mixed":
            dtype = torch.bfloat16
        elif name == "num_tokens" or name == "force_topk":
            dtype = torch.int32
        elif name == "mxfp4_pair_lut":
            dtype = torch.int16
        elif name.startswith("routed_"):
            dtype = e8m0 if name.endswith("_scale") else torch.uint8
        elif name.startswith("shared_"):
            dtype = e8m0 if name.endswith("_scale") else fp8
        value = values.get(name)
        init = None if value is None else partial(torch.clone, value)
        resident = None if name in nonresident else "stacked"
        specs.append(TensorSpec(name, shapes[name], dtype, init_value=init, resident=resident))
    return specs


def golden_moe(tensors):
    """Compose the validated mHC references around the MoE reference."""
    import torch
    from hc import golden_mhc_mixes, golden_hc_collapse, golden_hc_post
    from rmsnorm import golden_rms_norm_tiled
    from gate import golden_gate_normalized_core
    from expert_shared import golden_expert_shared
    from expert_routed import golden_expert_routed
    from quant import pack_mx_a_scale, unpack_mx_a_scale

    x_hc = tensors["x_hc"]
    next_pre_mix = torch.zeros(EP_SIZE, MOE_TOKENS, HC_MULT, dtype=torch.float32)
    post_mix = torch.zeros_like(next_pre_mix)
    residual_mix = torch.zeros(EP_SIZE, MOE_TOKENS, HC_MULT, HC_MULT, dtype=torch.float32)
    x_mixed = torch.zeros(EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16)
    for rank in range(EP_SIZE):
        pre, post, residual = golden_mhc_mixes(
            x_hc[rank],
            tensors['hc_ffn_fn'][rank], tensors['hc_ffn_scale'][rank], tensors['hc_ffn_base'][rank],
        )
        next_pre_mix[rank] = pre
        post_mix[rank] = post
        residual_mix[rank] = residual
        x_mixed[rank] = golden_hc_collapse(x_hc[rank], tensors['pre_mix'][rank]).to(torch.bfloat16)

    tensors["next_pre_mix"][:] = next_pre_mix
    tensors["x_mixed"][:] = x_mixed

    counts = _normalise_num_tokens(tensors.get("num_tokens", MOE_TOKENS))
    fp8 = torch.float8_e4m3fn
    e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

    all_indices = []
    all_weights = []
    all_x_mx = []
    all_scale_packed = []
    all_scale_logical = []
    all_shared = []

    dummy_tid2eid = torch.zeros(M.vocab_size, TOPK, dtype=torch.int32)
    dummy_input_ids = torch.zeros(MOE_TOKENS, dtype=torch.int64)

    for src in range(EP_SIZE):
        active = int(counts[src])
        x_norm_mx = torch.zeros(MOE_TOKENS, D, dtype=torch.uint8).view(fp8)
        x_norm_scale = torch.zeros(1, MOE_TOKENS * (D // MX_GROUP), dtype=torch.uint8).view(e8m0)
        indices = torch.zeros(MOE_TOKENS, TOPK, dtype=torch.int32)
        weights = torch.zeros(MOE_TOKENS, TOPK, dtype=torch.float32)
        golden_gate_normalized_core({
            "x_normed": golden_rms_norm_tiled(x_mixed[src], tensors['norm_weight'][src]),
            "gate_w": tensors["gate_weight"][src],
            "gate_bias": tensors["correction_bias"][src],
            "layer_id": 0,
            "num_tokens": active,
            "tid2eid": dummy_tid2eid,
            "input_ids": dummy_input_ids,
            "x_norm_mx": x_norm_mx,
            "x_norm_scale": x_norm_scale,
            "indices": indices,
            "weights": weights,
        })
        if BALANCED_ROUTING:
            indices[:active] = tensors["force_topk"][src, :active]
        shared = torch.zeros(MOE_TOKENS, D, dtype=torch.bfloat16)
        golden_expert_shared({
            "x_local": x_norm_mx,
            "x_local_scale": x_norm_scale,
            "shared_w1": tensors["shared_w1"][src],
            "shared_w1_scale": tensors["shared_w1_scale"][src],
            "shared_w3": tensors["shared_w3"][src],
            "shared_w3_scale": tensors["shared_w3_scale"][src],
            "shared_w2": tensors["shared_w2"][src],
            "shared_w2_scale": tensors["shared_w2_scale"][src],
            "sh": shared,
        })
        all_indices.append(indices)
        all_weights.append(weights)
        all_x_mx.append(x_norm_mx)
        scale_bytes = x_norm_scale.view(torch.uint8).reshape(MOE_TOKENS, D // MX_GROUP)
        all_scale_logical.append(unpack_mx_a_scale(scale_bytes))
        all_scale_packed.append(x_norm_scale)
        all_shared.append(shared)

    send_counts = torch.zeros(EP_SIZE, EP_SIZE, N_LOCAL_EXPERTS, dtype=torch.int32)
    for src in range(EP_SIZE):
        for t in range(int(counts[src])):
            for k in range(TOPK):
                eid = int(all_indices[src][t, k].item())
                dst, local_e = divmod(eid, N_LOCAL_EXPERTS)
                send_counts[src, dst, local_e] += 1

    dst_recv_y = []
    for dst in range(EP_SIZE):
        recv_x = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.uint8).view(fp8)
        recv_scale_logical = torch.zeros(N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP, dtype=torch.uint8)
        recv_weights = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
        recv_count = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
        slot_offsets = torch.zeros(EP_SIZE, N_LOCAL_EXPERTS, dtype=torch.int32)
        running = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
        for src in range(EP_SIZE):
            slot_offsets[src] = running.clone()
            running = running + send_counts[src, dst]
        recv_count[:, 0] = running

        for src in range(EP_SIZE):
            cursors = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
            for t in range(int(counts[src])):
                for k in range(TOPK):
                    eid = int(all_indices[src][t, k].item())
                    route_dst, local_e = divmod(eid, N_LOCAL_EXPERTS)
                    if route_dst != dst:
                        continue
                    slot = int(slot_offsets[src, local_e].item() + cursors[local_e].item())
                    cursors[local_e] += 1
                    recv_x[local_e, slot] = all_x_mx[src][t]
                    recv_scale_logical[local_e * RECV_MAX + slot] = all_scale_logical[src][t]
                    recv_weights[local_e, slot] = all_weights[src][t, k]

        recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
        golden_expert_routed({
            "recv_x": recv_x,
            "recv_mx_scale": pack_mx_a_scale(recv_scale_logical).view(e8m0),
            "recv_weights": recv_weights,
            "recv_expert_count": recv_count,
            "routed_w1_packed": tensors["routed_w1"][dst],
            "routed_w1_scale": tensors["routed_w1_scale"][dst],
            "routed_w3_packed": tensors["routed_w3"][dst],
            "routed_w3_scale": tensors["routed_w3_scale"][dst],
            "routed_w2_packed": tensors["routed_w2"][dst],
            "routed_w2_scale": tensors["routed_w2_scale"][dst],
            "mxfp4_pair_lut": tensors["mxfp4_pair_lut"][dst],
            "recv_y": recv_y,
        })
        dst_recv_y.append(recv_y)

    output = torch.zeros(EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16)
    for src in range(EP_SIZE):
        routed = torch.zeros(MOE_TOKENS * TOPK, D, dtype=torch.bfloat16)
        cursors = {}
        for t in range(int(counts[src])):
            for k in range(TOPK):
                eid = int(all_indices[src][t, k].item())
                dst, local_e = divmod(eid, N_LOCAL_EXPERTS)
                src_off = int(send_counts[:src, dst, local_e].sum().item())
                cursor = cursors.get((dst, local_e), 0)
                cursors[(dst, local_e)] = cursor + 1
                routed[t * TOPK + k] = dst_recv_y[dst][local_e, src_off + cursor]
        for t in range(int(counts[src])):
            acc = all_shared[src][t].float()
            for k in range(TOPK):
                acc = acc + routed[t * TOPK + k].float()
            output[src, t] = acc.to(torch.bfloat16)

    sublayer = output

    x_next = torch.zeros(EP_SIZE, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    for rank in range(EP_SIZE):
        x_next[rank] = golden_hc_post(sublayer[rank], x_hc[rank], post_mix[rank], residual_mix[rank])
    tensors["x_next"][:] = x_next


def _local_mhc_compare(num_tokens):
    """Compare the complete local-token output for every EP rank."""
    import torch
    from golden.validation import ratio_reldiff

    counts = _normalise_num_tokens(num_tokens)
    base_compare = ratio_reldiff(diff_thd=3e-3, pct_thd=0.02)
    max_abs_diff = 0.25

    def compare_region(actual, expected, **kwargs):
        ok, message = base_compare(actual, expected, **kwargs)
        if not ok:
            return False, message
        actual_f = actual.float()
        expected_f = expected.float()
        abs_diff = (actual_f - expected_f).abs()
        worst_flat = int(abs_diff.flatten().argmax().item())
        worst_abs = float(abs_diff.flatten()[worst_flat].item())
        if worst_abs > max_abs_diff:
            actual_value = float(actual_f.flatten()[worst_flat].item())
            expected_value = float(expected_f.flatten()[worst_flat].item())
            return False, (
                f"    worst absolute diff={worst_abs:.6g} at flat index "
                f"{worst_flat}: actual={actual_value:.8g}, "
                f"expected={expected_value:.8g}; exceeds "
                f"max_abs_diff={max_abs_diff:.6g}"
            )
        return True, ""

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, (
                f"    output shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        # Validate the active prefixes against the fixture counts.
        inputs = kwargs.get("inputs", {})
        if "num_tokens" in inputs:
            try:
                runtime_counts = _normalise_num_tokens(inputs["num_tokens"])
            except (TypeError, ValueError) as exc:
                return False, f"    invalid num_tokens: {exc}"
            if not torch.equal(runtime_counts, counts):
                return False, (
                    "    num_tokens changed between fixture and comparator: "
                    f"expected={counts.tolist()} actual={runtime_counts.tolist()}"
                )
        for rank, active in enumerate(counts.tolist()):
            if active:
                ok, message = compare_region(actual[rank:rank + 1, :active], expected[rank:rank + 1, :active], **kwargs)
                if not ok:
                    return False, (
                        f"    rank {rank} active rows [:{active}] left the "
                        f"MoE budget\n{message}"
                    )
            for row in range(active, MOE_TOKENS):
                ok, message = compare_region(
                    actual[rank:rank + 1, row:row + 1],
                    expected[rank:rank + 1, row:row + 1],
                    **kwargs,
                )
                if not ok:
                    return False, (
                        f"    rank {rank} padded row {row} left the "
                        f"padding budget\n{message}"
                    )
        return True, ""

    compare.__name__ = f"local_mhc_compare(num_tokens={counts.tolist()})"
    return compare


if __name__ == "__main__":
    import torch
    from pypto.ir import DistributedConfig
    from config import SUPPORTED_EP_SIZES

    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=list(SUPPORTED_EP_SIZES))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(EP_SIZE)))
    parser.add_argument('--experts-per-rank', type=int, default=N_LOCAL_EXPERTS, help='local routed experts; test model has EP times this many total experts')
    parser.add_argument("--balanced-routing", action="store_true", help="force balanced expert ids, retaining gate weights")
    parser.add_argument("--num-tokens", type=int, default=MOE_TOKENS)
    parser.add_argument('--num-tokens-per-rank', type=str, default=None, help='comma-separated local counts (overrides --num-tokens)')
    parser.add_argument("--moe-epoch", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device_ids = [int(d) for d in args.device.split(",") if d != ""]
    if len(device_ids) != EP_SIZE:
        raise SystemExit(f"need exactly {EP_SIZE} device ids for EP{EP_SIZE}, got {device_ids}")
    if args.ep != EP_SIZE:
        raise SystemExit(
            "--ep is parsed by config.py before argparse; run the module "
            "with the desired values, for example `python models/deepseek_v4_1_flash/decode_moe.py --ep 8 ...`"
        )

    entry = l3_moe
    if args.num_tokens_per_rank is None:
        count_value = args.num_tokens
    else:
        try:
            count_value = [int(value) for value in args.num_tokens_per_rank.split(",")]
        except ValueError as exc:
            raise SystemExit('--num-tokens-per-rank must be comma-separated integers') from exc
    try:
        counts = _normalise_num_tokens(count_value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    specs = build_moe_specs(counts, initialize=not (args.compile_only or args.golden_data))
    golden_fn = golden_moe
    compare_fn = {
        "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
        "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "x_next": _local_mhc_compare(counts),
    }

    result = run(
        fn=entry,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=args.golden_data,
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
            enable_dep_gen=args.enable_dep_gen,
            enable_chip_swimlane=args.enable_chip_swimlane,
            # Resident MoE heap capacity.
            ring_heap=2 * 1024**3,
            log_level=args.log_level,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
