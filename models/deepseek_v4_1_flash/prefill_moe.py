# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""DeepSeek-V4.1 prefill MoE with local token rows and HC."""

import argparse
from dataclasses import replace
import math
import sys
import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

import config


def _parse_int_option(flag, default):
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(flag + "="):
            return int(argument.split("=", 1)[1])
    return default


if __name__ == "__main__":
    config.set_moe_tokens(_parse_int_option("--tokens", 48))
    config.N_LOCAL_EXPERTS = _parse_int_option("--experts-per-rank", 12)
    total_experts = config.EP_SIZE * config.N_LOCAL_EXPERTS
    if config.N_LOCAL_EXPERTS <= 0 or not config.FLASH.num_experts_per_tok <= total_experts <= 384:
        raise ValueError("--experts-per-rank must give between TopK and 384 total routed experts")
    config.FLASH = replace(config.FLASH, n_routed_experts=total_experts)


from config import FLASH as M
from config import EP_SIZE, SUPPORTED_EP_SIZES
from config import MOE_TOKENS, MOE_RECV_MAX, N_LOCAL_EXPERTS, LOCAL_T_DYN, ROUTE_T_DYN
from config import T_DYN
from gate import gate_from_bf16, gather_routing_bias, prepare_routing_bias
from expert_shared import expert_shared

BIAS_DYN = pl.dynamic("PREFILL_MOE_BIAS_DYN")
from expert_routed import (
    MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS,
    expert_routed_scatter,
)
from hc import golden_hc_mixes, golden_hc_collapse, golden_hc_norm, golden_hc_post
from hc import hc_pre, hc_normalize, hc_post
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


# Standalone validation shapes.
_entry_parser = argparse.ArgumentParser(add_help=False)
_entry_parser.add_argument("--entry", choices=("local", "slabs"), default="slabs")
_entry_parser.add_argument("--slab-capacity", type=int, default=(2 * MOE_TOKENS + 16))
_entry_options = _entry_parser.parse_known_args()[0]
VALIDATE_SLABS = _entry_options.entry == "slabs"
SLAB_CAPACITY = _entry_options.slab_capacity
VALIDATION_TOKENS = SLAB_CAPACITY if VALIDATE_SLABS else MOE_TOKENS


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
    return gathered


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
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
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
) -> pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]:
    """Run delayed HC pre-mix, MoE, and residual expansion."""
    t = MOE_TOKENS
    post_mix = pl.create_tensor([t, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([t, HC_MULT, HC_MULT], dtype=pl.FP32)
    ffn_input = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    mixes_ready = hc_pre(
        x_hc, pre_mix,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, next_pre_mix, post_mix, residual_mix,
    )
    hc_normalize(x_mixed, norm_weight, ffn_input, num_tokens)

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
        gate_from_bf16(
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
            routed_w2, routed_w2_scale,
            recv_route_local, recv_meta_local,
            routed_output, input_ready, local_experts=N_LOCAL_EXPERTS, topk=TOPK,
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
        hc_post(sublayer, x_hc, post_mix, residual_mix, x_next, deps=hc_post_deps)
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
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // 32), MOE_INTER],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (MOE_INTER // 32), D],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // 32), MOE_INTER],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
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
            round_bias = pl.create_tensor([bias_rows * N_EXPERTS], dtype=pl.FP32)
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



@pl.jit
def l2_prefill_moe(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
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
) -> pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]:
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    next_pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    x_mixed.bind_dynamic(0, LOCAL_T_DYN)
    x_next.bind_dynamic(0, LOCAL_T_DYN)
    local_num_tokens = pl.read(num_tokens, [ep_rank])
    prefill_moe(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_weight, gate_weight, correction_bias,
        routed_w1, routed_w1_scale, routed_w2, routed_w2_scale,
        routed_w3, routed_w3_scale, shared_w1, shared_w1_scale,
        shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
        next_pre_mix, x_mixed, x_next,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived, routed_output, combine_arrived,
        local_num_tokens, ep_rank, moe_epoch,
    )
    clear_moe_signals(x_next, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    return x_next


@pl.jit
def l2_prefill_moe_slabs(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[BIAS_DYN], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
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
) -> pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]:
    """Validate bounded MoE slabs on each rank."""
    correction_bias.bind_dynamic(0, BIAS_DYN)
    capacity = pl.tensor.dim(x_hc, 0)
    slab_hidden = pl.create_tensor([2 * MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
    slab_pre = pl.create_tensor([2 * MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    slab_mixed = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    slab_rows = pl.create_tensor([MOE_TOKENS], dtype=pl.INT32)
    with pl.spmd(32, name_hint="slab_mixed_zero") as initialized:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, capacity, 32):
            x_mixed[row:row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    active = pl.read(num_tokens, [ep_rank])
    completed = prefill_moe_slabs(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_weight,
        gate_weight, correction_bias,
        routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
        shared_w1, shared_w1_scale, shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
        next_pre_mix, x_mixed, x_next,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived,
        routed_output, combine_arrived, slab_hidden, slab_pre, slab_mixed, slab_rows,
        initialized, active, ep_rank, moe_epoch, capacity=SLAB_CAPACITY,
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_slabs_complete", deps=[completed]):
        anchor = pl.read(x_next, [0, 0, 0])
        pl.write(x_next, [0, 0, 0], anchor)
    clear_moe_signals(x_next, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    return x_next


@pl.jit.host
def l3_prefill_moe(
    x_hc: pl.Tensor[[EP_SIZE, VALIDATION_TOKENS, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[EP_SIZE, VALIDATION_TOKENS, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[EP_SIZE, BIAS_DYN], pl.FP32],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[EP_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    next_pre_mix: pl.Out[pl.Tensor[[EP_SIZE, VALIDATION_TOKENS, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[EP_SIZE, VALIDATION_TOKENS, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[EP_SIZE, VALIDATION_TOKENS, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """EP host driver for the HC-wrapped Flash MoE block."""
    correction_bias.bind_dynamic(1, BIAS_DYN)
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
    if VALIDATE_SLABS:
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
            routed_w1_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w1_scale[r]
            routed_w2_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w2_scale[r]
            routed_w3_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w3_scale[r]
            shared_w1_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[r]
            shared_w2_scale_r: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[r]
            shared_w3_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[r]
            l2_prefill_moe_slabs(
                x_hc[r], pre_mix[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
                norm_weight[r], gate_weight[r], correction_bias[r],
                routed_w1[r], routed_w1_scale_r, routed_w2[r], routed_w2_scale_r,
                routed_w3[r], routed_w3_scale_r, shared_w1[r], shared_w1_scale_r,
                shared_w2[r], shared_w2_scale_r, shared_w3[r], shared_w3_scale_r,
                next_pre_mix[r], x_mixed[r], x_next[r],
                recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
                arrived, data_arrived, routed_output, combine_arrived,
                num_tokens, r, moe_epoch, device=r,
            )
    else:
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
            routed_w1_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w1_scale[r]
            routed_w2_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w2_scale[r]
            routed_w3_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w3_scale[r]
            shared_w1_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[r]
            shared_w2_scale_r: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[r]
            shared_w3_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[r]
            l2_prefill_moe(
                x_hc[r], pre_mix[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
                norm_weight[r], gate_weight[r], correction_bias[r],
                routed_w1[r], routed_w1_scale_r, routed_w2[r], routed_w2_scale_r,
                routed_w3[r], routed_w3_scale_r, shared_w1[r], shared_w1_scale_r,
                shared_w2[r], shared_w2_scale_r, shared_w3[r], shared_w3_scale_r,
                next_pre_mix[r], x_mixed[r], x_next[r],
                recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
                arrived, data_arrived, routed_output, combine_arrived,
                num_tokens, r, moe_epoch, device=r,
            )


def _gen_routed_mx_weights_fixture(n_experts, dequant_std, seed_base=0):
    """Generate the packed FP4 device ABI used by the routed kernel."""
    from expert_routed import (
        gen_routed_mxfp4_weights,
    )
    return gen_routed_mxfp4_weights(n_experts, dequant_std, seed_base)


def _fp8_dtype():
    return torch.float8_e4m3fn


def _e8m0_dtype():
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


def _normalise_num_tokens(value):
    """Return validated local-token counts for every EP source rank."""
    counts = torch.as_tensor(value, dtype=torch.int32).reshape(-1)
    if counts.numel() == 1:
        counts = counts.repeat(EP_SIZE)
    if counts.numel() != EP_SIZE:
        raise ValueError(
            f"num_tokens must be a scalar or contain {EP_SIZE} counts, "
            f"got shape {tuple(counts.shape)}"
        )
    if bool((counts < 0).any()) or bool((counts > MOE_TOKENS).any()):
        raise ValueError(
            f"num_tokens values must be in [0, {MOE_TOKENS}], "
            f"got {counts.tolist()}"
        )
    return counts.contiguous()


def _route_bias() -> torch.Tensor:
    # Fixed TopK expert ids spread across ranks.
    selected = (torch.arange(TOPK, dtype=torch.int64) * max(1, N_EXPERTS // TOPK))
    selected = torch.remainder(selected, N_EXPERTS)
    bias = torch.full((N_EXPERTS,), -1.0, dtype=torch.float32)
    bias[selected] = 4.0
    return bias


def _build_moe_tensor_specs(num_tokens: int = MOE_TOKENS):
    from functools import cache

    import torch

    from golden.spec import ScalarSpec, TensorSpec
    from expert_routed import (
        ROUTED_DEQUANT_STD,
    )
    from quant import generate_mxfp8_weight_kn

    counts = _normalise_num_tokens(num_tokens)
    torch.manual_seed(41)

    x = (torch.randn(EP_SIZE, MOE_TOKENS, D) * 0.25).to(torch.bfloat16)
    norm_weight = torch.ones(EP_SIZE, D, dtype=torch.bfloat16)
    gate_weight = (torch.randn(N_EXPERTS, D) / D ** 0.5)
    gate_weight = gate_weight.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    correction_bias = _route_bias().unsqueeze(0).expand(EP_SIZE, -1).contiguous()

    routed_w1_shape = (EP_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS)
    routed_w1_scale_shape = (EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER)
    routed_w2_shape = (EP_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS)
    routed_w2_scale_shape = (EP_SIZE, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D)
    routed_w3_shape = routed_w1_shape
    routed_w3_scale_shape = routed_w1_scale_shape

    # Cache packed weights and matching scales.
    @cache
    def routed_weights():
        # Tuple order: w1, w1_scale, w3, w3_scale, w2, w2_scale.
        shards = [
            _gen_routed_mx_weights_fixture(
                N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=rank * N_LOCAL_EXPERTS * 3,
            )
            for rank in range(EP_SIZE)
        ]
        return tuple(torch.stack([shard[index] for shard in shards]) for index in range(6))

    @cache
    def shared_weights():
        shared_std = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
        sw1, sw1_s = generate_mxfp8_weight_kn(MOE_INTER, D, shared_std["w1"], chan_cv=0.50, seed=101)
        sw3, sw3_s = generate_mxfp8_weight_kn(MOE_INTER, D, shared_std["w3"], chan_cv=0.50, seed=102)
        sw2, sw2_s = generate_mxfp8_weight_kn(D, MOE_INTER, shared_std["w2"], chan_cv=0.33, seed=103)
        return tuple(
            weight.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
            for weight in (sw1, sw1_s, sw3, sw3_s, sw2, sw2_s)
        )

    fp8 = _fp8_dtype()
    e8m0 = _e8m0_dtype()
    specs = [
        TensorSpec("x", [EP_SIZE, MOE_TOKENS, D], torch.bfloat16, init_value=lambda: x),
        TensorSpec("norm_weight", [EP_SIZE, D], torch.bfloat16, init_value=lambda: norm_weight),
        TensorSpec("gate_weight", [EP_SIZE, N_EXPERTS, D], torch.bfloat16, init_value=lambda: gate_weight),
        TensorSpec("correction_bias", [EP_SIZE, N_EXPERTS], torch.float32, init_value=lambda: correction_bias),
        TensorSpec("routed_w1", list(routed_w1_shape), torch.uint8, init_value=lambda: routed_weights()[0]),
        TensorSpec("routed_w1_scale", list(routed_w1_scale_shape), e8m0, init_value=lambda: routed_weights()[1]),
        TensorSpec("routed_w2", list(routed_w2_shape), torch.uint8, init_value=lambda: routed_weights()[4]),
        TensorSpec("routed_w2_scale", list(routed_w2_scale_shape), e8m0, init_value=lambda: routed_weights()[5]),
        TensorSpec("routed_w3", list(routed_w3_shape), torch.uint8, init_value=lambda: routed_weights()[2]),
        TensorSpec("routed_w3_scale", list(routed_w3_scale_shape), e8m0, init_value=lambda: routed_weights()[3]),
        TensorSpec("shared_w1", [EP_SIZE, D, MOE_INTER], fp8, init_value=lambda: shared_weights()[0]),
        TensorSpec("shared_w1_scale", [EP_SIZE, D // MX_GROUP, MOE_INTER], e8m0, init_value=lambda: shared_weights()[1]),
        TensorSpec("shared_w2", [EP_SIZE, MOE_INTER, D], fp8, init_value=lambda: shared_weights()[4]),
        TensorSpec("shared_w2_scale", [EP_SIZE, MOE_INTER // MX_GROUP, D], e8m0, init_value=lambda: shared_weights()[5]),
        TensorSpec("shared_w3", [EP_SIZE, D, MOE_INTER], fp8, init_value=lambda: shared_weights()[2]),
        TensorSpec("shared_w3_scale", [EP_SIZE, D // MX_GROUP, MOE_INTER], e8m0, init_value=lambda: shared_weights()[3]),
        TensorSpec("output", [EP_SIZE, MOE_TOKENS, D], torch.bfloat16),
        TensorSpec("num_tokens", [EP_SIZE], torch.int32,
                   init_value=lambda: counts),
        ScalarSpec("moe_epoch", torch.int32, 1, compile_runtime=True),
    ]

    for spec in specs:
        if spec.name not in {"x", "output", "num_tokens", "moe_epoch"}:
            spec.resident = "stacked"
    return specs


def build_tensor_specs(num_tokens: int = MOE_TOKENS):
    """Build the default HC+MoE integration fixture."""
    import torch
    from golden.spec import TensorSpec

    base_specs = _build_moe_tensor_specs(num_tokens)
    common_specs = [
        spec for spec in base_specs
        if spec.name not in {"x", "output", "num_tokens", "moe_epoch"}
    ]
    runtime_specs = [
        spec for spec in base_specs
        if spec.name in {"num_tokens", "moe_epoch"}
    ]

    generator = torch.Generator().manual_seed(3)
    magnitudes = (0.5 + torch.arange(MOE_TOKENS) % 4).reshape(1, -1, 1, 1)
    # BF16 residual values stored in FP32 with per-token magnitudes.
    x_hc = (torch.randn(EP_SIZE, MOE_TOKENS, HC_MULT, D, generator=generator) * magnitudes).bfloat16().float()
    pre_mix = torch.sigmoid(torch.randn(
        EP_SIZE, MOE_TOKENS, HC_MULT, generator=generator
    )) + M.hc_eps
    hc_ffn_fn = torch.randn(
        MIX_HC, HC_DIM, generator=generator
    ) / math.sqrt(HC_DIM)
    hc_ffn_scale = torch.randn(3, generator=generator)
    hc_ffn_base = torch.randn(MIX_HC, generator=generator)
    hc_ffn_fn = hc_ffn_fn.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    hc_ffn_scale = hc_ffn_scale.unsqueeze(0).expand(EP_SIZE, -1).contiguous()
    hc_ffn_base = hc_ffn_base.unsqueeze(0).expand(EP_SIZE, -1).contiguous()

    hc_specs = [
        TensorSpec(
            "x_hc", [EP_SIZE, MOE_TOKENS, HC_MULT, D],
            torch.float32, init_value=lambda: x_hc,
        ),
        TensorSpec(
            "pre_mix", [EP_SIZE, MOE_TOKENS, HC_MULT],
            torch.float32, init_value=lambda: pre_mix,
        ),
        TensorSpec(
            "hc_ffn_fn", [EP_SIZE, MIX_HC, HC_DIM],
            torch.float32, init_value=lambda: hc_ffn_fn,
        ),
        TensorSpec(
            "hc_ffn_scale", [EP_SIZE, 3],
            torch.float32, init_value=lambda: hc_ffn_scale,
        ),
        TensorSpec(
            "hc_ffn_base", [EP_SIZE, MIX_HC],
            torch.float32, init_value=lambda: hc_ffn_base,
        ),
    ]
    next_pre_mix_spec = TensorSpec('next_pre_mix', [EP_SIZE, MOE_TOKENS, HC_MULT], torch.float32)
    x_mixed_spec = TensorSpec('x_mixed', [EP_SIZE, MOE_TOKENS, D], torch.bfloat16)
    x_next_spec = TensorSpec('x_next', [EP_SIZE, MOE_TOKENS, HC_MULT, D], torch.float32)
    # MoE ABI order: HC inputs, weights, outputs, runtime inputs.
    output_specs = [next_pre_mix_spec, x_mixed_spec, x_next_spec]
    specs = hc_specs + common_specs + output_specs + runtime_specs
    for spec in specs:
        if spec.name in {
            "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
        }:
            spec.resident = "stacked"
    return specs



def build_slab_specs(counts, *, token_bias=False):
    """Build independent nonzero prompt rows per rank with bounded MoE weights."""
    specs = build_tensor_specs(MOE_TOKENS)
    generator = torch.Generator().manual_seed(73)
    hidden = torch.randn(EP_SIZE, SLAB_CAPACITY, HC_MULT, D, generator=generator).bfloat16().float()
    pre = torch.sigmoid(torch.randn(EP_SIZE, SLAB_CAPACITY, HC_MULT, generator=generator)) + M.hc_eps
    token_names = {"x_hc", "pre_mix", "next_pre_mix", "x_mixed", "x_next"}
    for spec in specs:
        if spec.name in token_names:
            spec.shape[1] = SLAB_CAPACITY
        if spec.name == "x_hc":
            spec.init_value = hidden
        elif spec.name == "pre_mix":
            spec.init_value = pre
        elif spec.name == "num_tokens":
            spec.init_value = counts
        elif spec.name == "correction_bias" and token_bias:
            text_bias = _route_bias()
            vision_bias = text_bias.roll(TOPK)
            image_rows = torch.arange(SLAB_CAPACITY) % 3 == 1
            bias = prepare_routing_bias(text_bias, vision_bias, image_rows, SLAB_CAPACITY)
            spec.shape = [EP_SIZE, SLAB_CAPACITY * N_EXPERTS]
            spec.init_value = bias.repeat(EP_SIZE, 1)
    return specs


def golden_moe_slabs(tensors):
    """Evaluate each rank's prompt in bounded MoE slabs."""
    capacity = tensors["x_hc"].shape[1]
    output_names = ("next_pre_mix", "x_mixed", "x_next")
    for name in output_names:
        tensors[name].zero_()
    for offset in range(0, capacity, MOE_TOKENS):
        rows = min(MOE_TOKENS, capacity - offset)
        slab = dict(tensors)
        for name in ("x_hc", "pre_mix", *output_names):
            shape = list(tensors[name].shape)
            shape[1] = MOE_TOKENS
            slab[name] = torch.zeros(shape, dtype=tensors[name].dtype)
        counts = torch.zeros(EP_SIZE, dtype=torch.int32)
        bias_rows = tensors["correction_bias"].shape[1] // N_EXPERTS
        if bias_rows > 1:
            slab["correction_bias"] = torch.zeros(EP_SIZE, MOE_TOKENS * N_EXPERTS)
        for rank in range(EP_SIZE):
            begin = offset
            active = max(0, min(rows, int(tensors["num_tokens"][rank]) - begin))
            counts[rank] = active
            if bias_rows > 1:
                source_bias = tensors["correction_bias"][rank].view(capacity, N_EXPERTS)
                target_bias = slab["correction_bias"][rank].view(MOE_TOKENS, N_EXPERTS)
                target_bias[:active].copy_(source_bias[begin:begin + active])
            for name in ("x_hc", "pre_mix"):
                slab[name][rank, :active].copy_(tensors[name][rank, begin:begin + active])
        slab["num_tokens"] = counts
        golden_moe(slab)
        for rank in range(EP_SIZE):
            begin = offset
            for name in output_names:
                tensors[name][rank, begin:begin + rows].copy_(slab[name][rank, :rows])


def _golden_moe_core(tensors):
    import torch

    from gate import golden_gate_normalized_core
    from expert_shared import golden_expert_shared
    from expert_routed import golden_expert_routed
    from quant import pack_mx_a_scale, unpack_mx_a_scale

    counts = _normalise_num_tokens(tensors.get("num_tokens", MOE_TOKENS))
    fp8 = _fp8_dtype()
    e8m0 = _e8m0_dtype()

    all_indices = []
    all_weights = []
    all_x_mx = []
    all_scale_packed = []
    all_scale_logical = []
    all_shared = []


    for src in range(EP_SIZE):
        active = int(counts[src])
        x_norm_mx = torch.zeros(MOE_TOKENS, D, dtype=torch.uint8).view(fp8)
        x_norm_scale = torch.zeros(1, MOE_TOKENS * (D // MX_GROUP), dtype=torch.uint8).view(e8m0)
        indices = torch.zeros(MOE_TOKENS, TOPK, dtype=torch.int32)
        weights = torch.zeros(MOE_TOKENS, TOPK, dtype=torch.float32)
        golden_gate_normalized_core({
            "x_normed": golden_hc_norm(tensors["x"][src], tensors["norm_weight"][src]),
            "gate_w": tensors["gate_weight"][src],
            "gate_bias": tensors["correction_bias"][src],
            "num_tokens": active,
            "x_norm_mx": x_norm_mx,
            "x_norm_scale": x_norm_scale,
            "indices": indices,
            "weights": weights,
        })
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

    tensors["output"][:] = output


def golden_moe(tensors):
    """Compose the validated HC references around the MoE reference."""
    import torch

    x_hc = tensors["x_hc"]
    next_pre_mix = torch.zeros(EP_SIZE, MOE_TOKENS, HC_MULT, dtype=torch.float32)
    post_mix = torch.zeros_like(next_pre_mix)
    residual_mix = torch.zeros(EP_SIZE, MOE_TOKENS, HC_MULT, HC_MULT, dtype=torch.float32)
    x_mixed = torch.zeros(EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16)
    for rank in range(EP_SIZE):
        pre, post, residual = golden_hc_mixes(
            x_hc[rank],
            tensors['hc_ffn_fn'][rank], tensors['hc_ffn_scale'][rank], tensors['hc_ffn_base'][rank],
        )
        next_pre_mix[rank] = pre
        post_mix[rank] = post
        residual_mix[rank] = residual
        x_mixed[rank] = golden_hc_collapse(x_hc[rank], tensors['pre_mix'][rank]).to(torch.bfloat16)

    tensors["next_pre_mix"][:] = next_pre_mix
    tensors["x_mixed"][:] = x_mixed

    # Compose the MoE reference with HC inputs and outputs.
    moe_tensors = dict(tensors)
    moe_tensors["x"] = x_mixed
    sublayer = torch.zeros(EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16)
    moe_tensors["output"] = sublayer
    _golden_moe_core(moe_tensors)

    x_next = torch.zeros(EP_SIZE, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    for rank in range(EP_SIZE):
        x_next[rank] = golden_hc_post(sublayer[rank], x_hc[rank], post_mix[rank], residual_mix[rank])
    tensors["x_next"][:] = x_next


def _local_hc_compare(num_tokens):
    """Compare the complete local-token output for every EP rank."""
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

    compare.__name__ = f"local_hc_compare(num_tokens={counts.tolist()})"
    return compare


if __name__ == "__main__":
    from golden import ratio_allclose, ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=list(SUPPORTED_EP_SIZES))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(EP_SIZE)))
    parser.add_argument(
        "--experts-per-rank", type=int, default=N_LOCAL_EXPERTS,
        help="local routed experts; test model has EP times this many total experts",
    )
    parser.add_argument("--entry", choices=("local", "slabs"), default=_entry_options.entry)
    parser.add_argument("--slab-capacity", type=int, default=SLAB_CAPACITY)
    parser.add_argument("--tokens", type=int, default=MOE_TOKENS)
    parser.add_argument("--num-tokens", type=int, default=SLAB_CAPACITY - 1 if VALIDATE_SLABS else MOE_TOKENS)
    parser.add_argument(
        "--num-tokens-per-rank", type=str, default=None,
        help="comma-separated local counts (overrides --num-tokens)",
    )
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
    parser.add_argument("--token-bias", action="store_true", help="mixed text/vision routing in slab validation")
    args = parser.parse_args()
    if args.token_bias and not VALIDATE_SLABS:
        parser.error("--token-bias requires --entry slabs")

    torch.manual_seed(args.seed)
    device_ids = [int(d) for d in args.device.split(",") if d != ""]
    if len(device_ids) != EP_SIZE:
        raise SystemExit(f"need exactly {EP_SIZE} device ids for EP{EP_SIZE}, got {device_ids}")
    if args.ep != EP_SIZE:
        raise SystemExit(
            "--ep are parsed by config.py before argparse; run the module "
            "with the desired values, for example `python models/deepseek_v4_1_flash/prefill_moe.py --ep 8 ...`"
        )

    entry = l3_prefill_moe
    if args.num_tokens_per_rank is None:
        count_value = args.num_tokens
    else:
        try:
            count_value = [int(value) for value in args.num_tokens_per_rank.split(",")]
        except ValueError as exc:
            raise SystemExit(
                "--num-tokens-per-rank must be comma-separated integers"
            ) from exc
    if VALIDATE_SLABS:
        if SLAB_CAPACITY <= 0:
            parser.error("--slab-capacity must be positive")
        counts = torch.as_tensor(count_value, dtype=torch.int32).reshape(-1)
        if counts.numel() == 1:
            counts = counts.expand(EP_SIZE).clone()
        if counts.numel() != EP_SIZE or bool(((counts < 0) | (counts > SLAB_CAPACITY)).any()):
            parser.error("slab counts must provide an active prefix in [0, capacity] per rank")
        specs = build_slab_specs(counts, token_bias=args.token_bias)
        golden_fn = golden_moe_slabs
        compare_fn = {
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.02),
        }
    else:
        try:
            counts = _normalise_num_tokens(count_value)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        specs = build_tensor_specs(counts)
        golden_fn = golden_moe
        compare_fn = {
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "x_next": _local_hc_compare(counts),
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
            # Resident prefill heap capacity.
            ring_heap=2 * 1024**3,
            log_level=args.log_level,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
