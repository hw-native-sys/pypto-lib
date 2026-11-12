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
"""C2A reuse prefill and local EP MoE layer."""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch

import config as C

_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--tokens", type=int, default=512)
_shape_parser.add_argument("--moe-tokens", type=int, default=512)
_shape_options = _shape_parser.parse_known_args()[0]
if C.EP_SIZE % C.CP_SIZE:
    raise ValueError("prefill layer requires EP = DP * CP")
_chunk = (_shape_options.tokens + 2 * C.CP_SIZE - 1) // (2 * C.CP_SIZE)
LOCAL_CAPACITY = max(
    max(0, min(_chunk, _shape_options.tokens - rank * _chunk))
    + max(0, min(_chunk, _shape_options.tokens - (2 * C.CP_SIZE - 1 - rank) * _chunk))
    for rank in range(C.CP_SIZE)
)
C.set_moe_tokens(min((LOCAL_CAPACITY + 15) // 16 * 16, _shape_options.moe_tokens))

from config import (
    FLASH as M, CP_SIZE, EP_SIZE, RANKS_DYN, LOCAL_T_DYN, T_DYN, ORI_BLOCKS_DYN, CMP_BLOCKS_DYN,
    ROUTE_T_DYN, PREFILL_MAX_TOKENS, WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES, MOE_TOKENS,
)
from prefill_c1a_reuse import (
    l2_prefill_c1a_reuse, l2_prefill_c1a_reuse_cp, l3_prefill_c1a_reuse, l3_prefill_c1a_reuse_cp,
)
from prefill_c2a_reuse import build_c2a_reuse_specs
from prefill_c2a_full import golden_c2a_prefill
from prefill_moe import (
    AUX_WIDTH, ROUTE_WIDTH, SIGNAL_PAD, RECV_MAX, N_LOCAL_EXPERTS,
    prefill_moe_slabs, clear_moe_signals, build_tensor_specs as build_moe_specs, golden_moe_slabs,
)
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
Q_LORA = M.q_lora_rank
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
O_LORA = M.o_lora_rank
INDEX_TOPK = M.index_topk
MOE_INTER = M.moe_intermediate_size
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
MX_GROUP = M.weight_block_size[0]


@pl.jit
def l2_prefill_layer_ffn(
    hc_output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_ffn_norm: pl.Tensor[[D], pl.BF16],
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
    ffn_input: pl.Out[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]:
    """Run local HC and bounded EP MoE slabs."""
    capacity = pl.tensor.dim(hc_output, 0)
    slab_hidden = pl.create_tensor([2 * MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
    slab_pre = pl.create_tensor([2 * MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    slab_mixed = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    slab_rows = pl.create_tensor([MOE_TOKENS], dtype=pl.INT32)
    with pl.spmd(32, name_hint="slab_mixed_zero") as initialized:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, capacity, 32):
            ffn_input[row:row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    active = pl.read(num_tokens, [0])
    completed = prefill_moe_slabs(
        hc_output, hc_next_pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, w_ffn_norm,
        gate_weight, correction_bias,
        routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
        shared_w1, shared_w1_scale, shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
        next_pre_mix, ffn_input, output,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived,
        routed_output, combine_arrived, slab_hidden, slab_pre, slab_mixed, slab_rows,
        initialized, active, ep_rank, moe_epoch, capacity=LOCAL_CAPACITY,
    )
    clear_moe_signals(output, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    return output


@pl.jit.host
def l3_prefill_layer(
    # HC input and weights.
    hc_input: pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    # Q projection weights.
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    wq_norm: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    # KV projection weights and attention sink.
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    wkv_norm: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    # Output projection weights.
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    q_window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    q_compressed_indices: pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    w_ffn_norm: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    gate_weight: pl.Tensor[[RANKS_DYN, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[RANKS_DYN, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[RANKS_DYN, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[RANKS_DYN, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[RANKS_DYN, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[RANKS_DYN, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[RANKS_DYN, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[RANKS_DYN, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT], pl.FP32]],
    ffn_input: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT, D], pl.FP32]],
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
    recv_weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
    routed_output_buf = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)


    for rank in pl.range(pld.world_size()):
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        l2_prefill_c1a_reuse(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            q_window_indices[rank],
            q_compressed_indices[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank],
            window_cache[rank],
            compressed_cache[rank],
            kv_hidden[rank],
            hc_output[rank], hc_next_pre_mix[rank],
            device=rank,
        )
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
        ] = routed_w1_scale[rank]
        routed_w2_scale_r: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = routed_w2_scale[rank]
        routed_w3_scale_r: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = routed_w3_scale[rank]
        shared_w1_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[rank]
        shared_w2_scale_r: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[rank]
        shared_w3_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[rank]
        l2_prefill_layer_ffn(
            hc_output[rank], hc_next_pre_mix[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            w_ffn_norm[rank], gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale_r, routed_w2[rank], routed_w2_scale_r,
            routed_w3[rank], routed_w3_scale_r, shared_w1[rank], shared_w1_scale_r,
            shared_w2[rank], shared_w2_scale_r, shared_w3[rank], shared_w3_scale_r,
            next_pre_mix[rank], ffn_input[rank], output[rank],
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            kv_token_count[rank], rank, 1, device=rank,
        )


@pl.jit.host
def l3_prefill_layer_cp(
    # HC input and weights.
    hc_input: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    # Q projection weights.
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    wq_norm: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    # KV projection weights and attention sink.
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    wkv_norm: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    # Output projection weights.
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    # Local Q metadata.
    q_rope_cos: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_window_indices: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, 128], pl.INT32],
    q_compressed_indices: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, INDEX_TOPK], pl.INT32],
    q_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    kv_restore_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    w_ffn_norm: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    gate_weight: pl.Tensor[[RANKS_DYN, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[RANKS_DYN, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[RANKS_DYN, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[RANKS_DYN, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[RANKS_DYN, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[RANKS_DYN, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[RANKS_DYN, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[RANKS_DYN, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[RANKS_DYN, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    ffn_input: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, LOCAL_T_DYN)
    kv_rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    cp_hidden_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    cp_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
    recv_weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
    routed_output_buf = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)


    for rank in pl.range(pld.world_size()):
        cp_hidden_window = pld.window(cp_hidden_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        cp_hidden_arrived = pld.window(cp_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        l2_prefill_c1a_reuse_cp(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            q_rope_cos[rank], q_rope_sin[rank],
            q_window_indices[rank],
            q_compressed_indices[rank], q_token_count[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank],
            window_cache[rank],
            compressed_cache[rank],
            cp_hidden_window, cp_hidden_arrived, kv_restore_indices[rank], rank, attention_epoch,
            kv_hidden[rank],
            hc_output[rank], hc_next_pre_mix[rank],
            device=rank,
        )
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
        ] = routed_w1_scale[rank]
        routed_w2_scale_r: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = routed_w2_scale[rank]
        routed_w3_scale_r: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = routed_w3_scale[rank]
        shared_w1_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[rank]
        shared_w2_scale_r: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[rank]
        shared_w3_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[rank]
        l2_prefill_layer_ffn(
            hc_output[rank], hc_next_pre_mix[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            w_ffn_norm[rank], gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale_r, routed_w2[rank], routed_w2_scale_r,
            routed_w3[rank], routed_w3_scale_r, shared_w1[rank], shared_w1_scale_r,
            shared_w2[rank], shared_w2_scale_r, shared_w3[rank], shared_w3_scale_r,
            next_pre_mix[rank], ffn_input[rank], output[rank],
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            q_token_count[rank], rank, 1, device=rank,
        )

def build_layer_specs(args, entry):
    """Compose attention fixtures with the shared MoE weight fixture."""
    attention = l3_prefill_c1a_reuse if CP_SIZE == 1 else l3_prefill_c1a_reuse_cp
    specs = build_c2a_reuse_specs(args, attention)
    for spec in build_moe_specs(MOE_TOKENS):
        if spec.name in ("x_hc", "pre_mix", "num_tokens", "moe_epoch"):
            continue
        if spec.name == "norm_weight":
            spec.name = "w_ffn_norm"
        elif spec.name == "x_mixed":
            spec.name = "ffn_input"
        elif spec.name == "x_next":
            spec.name = "output"
        if spec.name in ("next_pre_mix", "ffn_input", "output"):
            spec.shape[1] = LOCAL_CAPACITY
        spec.resident = "stacked"
        specs.append(spec)
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in entry.param_names]


def golden_layer(tensors):
    golden_c2a_prefill(tensors, mode="reuse")
    moe = dict(tensors)
    moe.update(x_hc=tensors["hc_output"], pre_mix=tensors["hc_next_pre_mix"], norm_weight=tensors["w_ffn_norm"])
    count_name = "q_token_count" if CP_SIZE > 1 else "kv_token_count"
    moe.update(num_tokens=tensors[count_name].flatten(), x_mixed=tensors["ffn_input"], x_next=tensors["output"])
    golden_moe_slabs(moe)


class LayerMoeReference:
    """Cache the MoE reference evaluated on the device's attention outputs."""

    def __init__(self):
        self.source = None
        self.values = None

    def evaluate(self, inputs, actual_outputs):
        if self.source is not actual_outputs:
            moe = dict(inputs)
            count_name = "q_token_count" if CP_SIZE > 1 else "kv_token_count"
            moe.update(
                x_hc=actual_outputs["hc_output"], pre_mix=actual_outputs["hc_next_pre_mix"],
                norm_weight=inputs["w_ffn_norm"], num_tokens=inputs[count_name].flatten(),
                x_mixed=torch.zeros_like(actual_outputs["ffn_input"]),
                x_next=torch.zeros_like(actual_outputs["output"]),
                next_pre_mix=torch.zeros_like(actual_outputs["next_pre_mix"]),
            )
            golden_moe_slabs(moe)
            self.source, self.values = actual_outputs, moe
        return self.values

    def compare(self, name, check, *, end_to_end_bound=None):
        """Apply the supplied stage comparator and optional end-to-end relative L2 bound."""
        def compare(actual, expected, *, inputs, actual_outputs, **kwargs):
            forced = self.evaluate(inputs, actual_outputs)[name]
            valid, message = check(actual, forced, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
            if not valid or end_to_end_bound is None:
                return valid, message
            count_name = "q_token_count" if CP_SIZE > 1 else "kv_token_count"
            for rank in range(EP_SIZE):
                count = int(inputs[count_name][rank, 0])
                value, target = actual[rank, :count].float(), expected[rank, :count].float()
                difference = torch.linalg.vector_norm(value - target)
                relative = difference / torch.linalg.vector_norm(target).clamp_min(1e-12)
                if not bool(torch.isfinite(relative)) or float(relative) > end_to_end_bound:
                    return False, f"rank {rank}: end-to-end relative L2 {float(relative):.6g} > {end_to_end_bound}"
            return True, "stage and end-to-end precision checks passed"
        return compare


if __name__ == "__main__":
    from golden import mapped_pool_ratio_allclose, ratio_allclose, ratio_reldiff, run
    from utils import packed_cache_compare
    from pypto.ir import DistributedConfig

    from utils import parse_prefill_groups

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=[1, 2, 4, 8])
    parser.add_argument("--dp", type=int, default=EP_SIZE // CP_SIZE, choices=[1, 2, 4, 8])
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=[2, 4, 8])
    parser.add_argument("--tokens", type=int, default=_shape_options.tokens, help="token capacity per DP group")
    parser.add_argument("--moe-tokens", type=int, default=_shape_options.moe_tokens)
    parser.add_argument("--requests", type=int, default=1, help="request count per DP when query-lens is omitted")
    parser.add_argument("--query-lens", type=parse_prefill_groups, help="request lengths, e.g. '17,15;32'")
    parser.add_argument("--start-pos", type=parse_prefill_groups, help="request prefix lengths, e.g. '127,0;256'")
    parser.add_argument("--case", choices=["normal", "prefix"], default="normal")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.set_defaults(dp_tokens=None)
    args = parser.parse_args()
    if args.dp * CP_SIZE != EP_SIZE:
        parser.error("EP must equal DP * CP")
    if not 1 <= args.tokens <= PREFILL_MAX_TOKENS:
        parser.error(f"tokens must be in [1, {PREFILL_MAX_TOKENS}]")
    devices = list(range(EP_SIZE)) if args.device is None else [int(value) for value in args.device.split(",")]
    if len(devices) != EP_SIZE or len(set(devices)) != EP_SIZE:
        parser.error("provide one distinct device per EP rank")
    torch.set_num_threads(8)
    entry = l3_prefill_layer if CP_SIZE == 1 else l3_prefill_layer_cp
    moe_reference = LayerMoeReference()
    result = run(
        fn=entry, specs=build_layer_specs(args, entry), golden_fn=golden_layer,
        compare_fn={
            "kv_hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "hc_next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "hc_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": packed_cache_compare(
                mapped_pool_ratio_allclose(
                    "kv_write_slots", pool_name="window_cache",
                    mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.01,
                ),
                mapped_pool_ratio_allclose(
                    "kv_write_slots", pool_name="window_cache scale",
                    mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.0,
                ),
                fp4=False,
            ),
            "ffn_input": moe_reference.compare(
                "x_mixed", ratio_allclose(atol=1e-4, rtol=2**-8, max_error_ratio=0.002), end_to_end_bound=0.01,
            ),
            "next_pre_mix": moe_reference.compare("next_pre_mix", ratio_allclose(atol=2.5e-5, rtol=5e-3)),
            "output": moe_reference.compare("x_next", ratio_reldiff(diff_thd=3e-3, pct_thd=0.02), end_to_end_bound=0.15),
        },
        compile_only=args.compile_only, save_data=args.save_data, golden_data=args.golden_data,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=False,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            ring_heap=(4 * 1024**3,) * 4,
            enable_dep_gen=args.enable_dep_gen, enable_chip_swimlane=args.enable_chip_swimlane,
        ),
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
