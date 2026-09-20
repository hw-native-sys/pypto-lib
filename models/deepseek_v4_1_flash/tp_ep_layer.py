# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-sharded Attention/mHC/EP-MoE boundary and TP residual reconstruction."""

import pypto.language as pl
import pypto.language.distributed as pld

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_tp import OUTPUT_T_DYN
from models.deepseek_v4_1_flash.hc_mixes import mhc_mixes
from models.deepseek_v4_1_flash.hc_pre import mhc_pre
from models.deepseek_v4_1_flash.hc_post import mhc_post
from models.deepseek_v4_1_flash.moe import moe_core

D = C.D
HC = C.HC_MULT
HC_DIM = C.HC_DIM
T_DYN = C.T_DYN
TP_SIZE = C.TP_SIZE
EP_SIZE = C.EP_SIZE
N_LOCAL_EXPERTS = C.N_LOCAL_EXPERTS
MX_GROUP = C.MX_GROUP
MOE_INTER = C.MOE_INTER
RECV_MAX = C.EP_SIZE * C.MOE_TOKENS
AUX_WIDTH = C.AUX_WIDTH
ROUTE_WIDTH = C.ROUTE_WIDTH
BLOCK = C.MOE_TOKENS
TOPK = C.TOPK
SHARD_MAX = (C.PREFILL_MAX_TOKENS + TP_SIZE - 1) // TP_SIZE
SIGNAL_WINDOW_BYTES = 64


@pl.jit.inline
def pack_layer_shard(
    attention: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC], pl.FP32],
    mixes: pl.Tensor[[T_DYN, HC, HC], pl.FP32],
    attention_block: pl.Tensor[[BLOCK, D], pl.BF16],
    residual_block: pl.Tensor[[BLOCK, HC, D], pl.FP32],
    post_block: pl.Tensor[[BLOCK, HC], pl.FP32],
    mixes_block: pl.Tensor[[BLOCK, HC, HC], pl.FP32],
    shard_first: pl.Scalar[pl.INT32],
    block_first: pl.Scalar[pl.INT32],
    count: pl.Scalar[pl.INT32],
):
    """Select matching residual/mHC rows without reducing their TP replicas."""
    full_rows = pl.tensor.dim(residual, 0)
    full_flat = pl.reshape(residual, [full_rows, HC_DIM])
    local_flat = pl.reshape(residual_block, [BLOCK, HC_DIM])
    for t in pl.spmd(BLOCK, name_hint="tp_ep_pack"):
        source = shard_first + block_first + t
        for col in pl.range(0, D, 512):
            value = pl.tile.full([1, 512], dtype=pl.BF16, value=0.0)
            if t < count:
                value = pl.load(attention, [block_first + t, col], [1, 512])
            attention_block = pl.store(value, [t, col], attention_block)
        for col in pl.range(0, HC_DIM, 512):
            residual_value = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
            if t < count:
                residual_value = pl.load(full_flat, [source, col], [1, 512])
            local_flat = pl.store(residual_value, [t, col], local_flat)
    # Scalar metadata writes must not share cache lines across cores.
    for _ in pl.spmd(1, name_hint="tp_ep_pack_metadata"):
        for t in pl.range(BLOCK):
            source = shard_first + block_first + t
            for h in pl.unroll(HC):
                post_value = pl.cast(0.0, pl.FP32)
                if t < count:
                    post_value = pl.read(post, [source, h])
                pl.write(post_block, [t, h], post_value)
                for k in pl.unroll(HC):
                    mix_value = pl.cast(0.0, pl.FP32)
                    if t < count:
                        mix_value = pl.read(mixes, [source, h, k])
                    pl.write(mixes_block, [t, h, k], mix_value)
    return attention_block, residual_block, post_block, mixes_block


@pl.jit.inline
def unpack_layer_shard(
    block: pl.Tensor[[BLOCK, HC, D], pl.FP32],
    shard: pl.Tensor[[OUTPUT_T_DYN, HC, D], pl.FP32],
    first: pl.Scalar[pl.INT32],
    count: pl.Scalar[pl.INT32],
):
    block_flat = pl.reshape(block, [BLOCK, HC_DIM])
    shard_rows = pl.tensor.dim(shard, 0)
    shard_flat = pl.reshape(shard, [shard_rows, HC_DIM])
    for t in pl.spmd(BLOCK, name_hint="tp_ep_unpack"):
        if t < count:
            for col in pl.range(0, HC_DIM, 512):
                value = pl.load(block_flat, [t, col], [1, 512])
                shard_flat = pl.store(value, [first + t, col], shard_flat)
    return shard


@pl.jit.inline(auto_scope=False)
def tp_residual_all_gather(
    shard: pl.Tensor[[OUTPUT_T_DYN, HC, D], pl.FP32],
    window: pld.DistributedTensor[[SHARD_MAX, HC_DIM], pl.FP32],
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, HC, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    width = (num_tokens + TP_SIZE - 1) // TP_SIZE
    first = pl.min(tp_rank * width, num_tokens)
    count = pl.min(width, num_tokens - first)
    shard_rows = pl.tensor.dim(shard, 0)
    flat = pl.reshape(shard, [shard_rows, HC_DIM])
    output_rows = pl.tensor.dim(output, 0)
    output_flat = pl.reshape(output, [output_rows, HC_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_hc_reuse", allow_early_resolve=False) as reused:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=(epoch - 1) * 2, cmp=pld.WaitCmp.Ge)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_hc_publish", deps=[reused]) as ready:
        if count > 0:
            pld.tensor.put(dst=window, peer=group_base + tp_rank, src=flat,
                           dst_offsets=[0, 0], src_offsets=[0, 0], shape=[count, HC_DIM],
                           chunk_rows=1, chunk_cols=512, pipeline=True)
        for peer in pl.range(TP_SIZE):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_hc_wait", deps=[ready],
               allow_early_resolve=False) as waited:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="tp_hc_gather", deps=[waited]) as gathered:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, num_tokens * (HC_DIM // 512), 64):
            row = tile // (HC_DIM // 512)
            col = tile % (HC_DIM // 512) * 512
            peer = row // width
            value = pld.tile.remote_load(window, peer=group_base + peer,
                                         offsets=[row - peer * width, col], shape=[1, 512])
            output_flat = pl.store(value, [row, col], output_flat)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_hc_release", deps=[gathered]) as released:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_hc_consumed", deps=[released],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=epoch * 2, cmp=pld.WaitCmp.Ge)
    return output



@pl.jit
def tp_ep_layer_tail(
    attention_shard: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC, D], pl.FP32],
    attention_post: pl.Tensor[[T_DYN, HC], pl.FP32],
    attention_mix: pl.Tensor[[T_DYN, HC, HC], pl.FP32],
    hc_function: pl.Tensor[[C.MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, C.MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w1: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    residual_window: pld.DistributedTensor[[SHARD_MAX, HC_DIM], pl.FP32],
    residual_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, HC, D], pl.FP32]],
    token_count: pl.Tensor[[1], pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch_base: pl.Scalar[pl.INT32],
    residual_epoch: pl.Scalar[pl.INT32],
):
    """Complete a layer after Attention with reduce_scatter=True.

    All EP ranks use the same positive shard capacity and therefore the same
    number of MoE rounds, even when their DP token counts differ or are zero.
    `moe_epoch_base` advances by that round count on each invocation; the
    separate TP residual epoch advances by one. Both start at one.
    """
    attention_shard.bind_dynamic(0, OUTPUT_T_DYN)
    residual.bind_dynamic(0, T_DYN)
    num_tokens = pl.read(token_count, [0])
    capacity = pl.tensor.dim(attention_shard, 0)
    width = (num_tokens + TP_SIZE - 1) // TP_SIZE
    tp_rank = ep_rank % TP_SIZE
    group_base = ep_rank - tp_rank
    first = pl.min(tp_rank * width, num_tokens)
    local_count = pl.min(width, num_tokens - first)
    shard = pl.create_tensor([capacity, HC, D], dtype=pl.FP32)
    for block in pl.range((capacity + BLOCK - 1) // BLOCK):
        block_first = block * BLOCK
        count = pl.max(0, pl.min(BLOCK, local_count - block_first))
        attention_block = pl.create_tensor([BLOCK, D], dtype=pl.BF16)
        residual_block = pl.create_tensor([BLOCK, HC, D], dtype=pl.FP32)
        post_block = pl.create_tensor([BLOCK, HC], dtype=pl.FP32)
        mix_block = pl.create_tensor([BLOCK, HC, HC], dtype=pl.FP32)
        pack_layer_shard(attention_shard, residual, attention_post, attention_mix,
                         attention_block, residual_block, post_block, mix_block,
                         first, block_first, count)
        after_attention = pl.create_tensor([BLOCK, HC, D], dtype=pl.FP32)
        mhc_post(attention_block, residual_block, post_block, mix_block, after_attention)
        pre_mix = pl.create_tensor([BLOCK, HC], dtype=pl.FP32)
        post_mix = pl.create_tensor([BLOCK, HC], dtype=pl.FP32)
        residual_mix = pl.create_tensor([BLOCK, HC, HC], dtype=pl.FP32)
        mhc_mixes(after_attention, hc_function, hc_scale, hc_base, pre_mix, post_mix, residual_mix)
        moe_input = pl.create_tensor([BLOCK, D], dtype=pl.BF16)
        mhc_pre(after_attention, pre_mix, moe_input)
        moe_output = pl.create_tensor([BLOCK, D], dtype=pl.BF16)
        moe_core(
            moe_input, norm_weight, gate_weight, correction_bias,
            routed_w1, routed_w1_scale, routed_w2, routed_w2_scale,
            routed_w3, routed_w3_scale, shared_w1, shared_w1_scale,
            shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
            recv_meta, recv_x, recv_scale, recv_weights,
            recv_routes, arrived, data_arrived, routed_output,
            combine_arrived, moe_output, count, ep_rank,
            moe_epoch_base + block,
        )
        after_moe = pl.create_tensor([BLOCK, HC, D], dtype=pl.FP32)
        mhc_post(moe_output, after_attention, post_mix, residual_mix, after_moe)
        unpack_layer_shard(after_moe, shard, block_first, count)
    tp_residual_all_gather(shard, residual_window, residual_arrived, output,
                           group_base, tp_rank, num_tokens, residual_epoch)
    return output



@pl.jit.host
def l3_tp_ep_layer_tail(
    attention_shard: pl.Tensor[[EP_SIZE, OUTPUT_T_DYN, D], pl.BF16],
    residual: pl.Tensor[[EP_SIZE, T_DYN, HC, D], pl.FP32],
    attention_post: pl.Tensor[[EP_SIZE, T_DYN, HC], pl.FP32],
    attention_mix: pl.Tensor[[EP_SIZE, T_DYN, HC, HC], pl.FP32],
    hc_function: pl.Tensor[[EP_SIZE, C.MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    hc_base: pl.Tensor[[EP_SIZE, C.MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[EP_SIZE, C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, C.MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[EP_SIZE, D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0],
    token_counts: pl.Tensor[[EP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, HC, D], pl.FP32]],
    moe_epoch_base: pl.Scalar[pl.INT32],
    residual_epoch: pl.Scalar[pl.INT32],
):
    """Drive all EP ranks, including empty local shards, with persistent windows."""
    recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
    recv_weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    # The runtime carves buffers consecutively; isolate signal cache maintenance
    # from adjacent payloads while retaining the packed logical counter views.
    arrived_buf = pld.alloc_window_buffer(SIGNAL_WINDOW_BYTES)
    data_arrived_buf = pld.alloc_window_buffer(SIGNAL_WINDOW_BYTES)
    routed_output_buf = pld.alloc_window_buffer([BLOCK * TOPK, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer(SIGNAL_WINDOW_BYTES)

    residual_buf = pld.alloc_window_buffer([SHARD_MAX, HC_DIM], dtype=pl.FP32)
    residual_signal_buf = pld.alloc_window_buffer(SIGNAL_WINDOW_BYTES)
    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buf, [BLOCK * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
        residual_window = pld.window(residual_buf, [SHARD_MAX, HC_DIM], dtype=pl.FP32)
        residual_arrived = pld.window(residual_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
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
        tp_ep_layer_tail(
            attention_shard[r], residual[r], attention_post[r], attention_mix[r],
            hc_function[r], hc_scale[r], hc_base[r],
            norm_weight[r], gate_weight[r], correction_bias[r], routed_w1[r],
            routed_w1_scale_r, routed_w2[r], routed_w2_scale_r, routed_w3[r],
            routed_w3_scale_r, shared_w1[r], shared_w1_scale_r, shared_w2[r],
            shared_w2_scale_r, shared_w3[r], shared_w3_scale_r,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            residual_window, residual_arrived, output[r], token_counts[r], r,
            moe_epoch_base, residual_epoch, device=r,
        )
