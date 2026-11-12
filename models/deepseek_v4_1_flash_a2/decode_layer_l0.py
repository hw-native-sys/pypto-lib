# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete physical L0 Decode composition for the two-rank EP2 gate.

The attention and FFN residual streams stay rank-local.  The MoE child owns
the only cross-rank transport and receives the normalized FFN stream.  This
entry is a product path: all model computation is emitted through PyPTO.
"""

from __future__ import annotations

from dataclasses import asdict

import pypto.language as pl
import pypto.language.distributed as pld

from .attention_block import make_attention_block
from .config import FLASH
from .decode_moe import (
    AUX_PAD,
    COUNT_PAD,
    D,
    EP_SIZE,
    LANE_ROWS,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    T,
    TOPK,
    decode_ep2_rank,
)
from .hc import make_hc_post_inline, make_hc_pre_inline
from .rmsnorm import NORM_T_TILE, rms_norm_tiled_with_fp32


CONFIG = asdict(FLASH)
HC = FLASH.hc_mult
CHANNELS = FLASH.mix_hc
HIDDEN = FLASH.hidden_size
LATENT = FLASH.q_lora_rank
WIDTH = FLASH.num_attention_heads * FLASH.head_dim
ROPE_PAIRS = FLASH.qk_rope_head_dim // 2
GROUP_INPUT = WIDTH // FLASH.o_groups
O_RANK = FLASH.o_lora_rank
WINDOW = FLASH.sliding_window

ATTENTION = pl.jit.inline(make_attention_block(T, CONFIG)._func)
FFN_PRE = make_hc_pre_inline(T, CONFIG)
FFN_POST = make_hc_post_inline(T, CONFIG)
EP2_RANK = pl.jit.inline(decode_ep2_rank._func)


@pl.jit(auto_scope=False)
def decode_layer_ep2_rank(
    x_hc: pl.Tensor[[T, HC, HIDDEN], pl.BF16],
    attn_hc_function: pl.Tensor[[CHANNELS, HC * HIDDEN], pl.FP32],
    attn_hc_scale: pl.Tensor[[3], pl.FP32],
    attn_hc_base: pl.Tensor[[CHANNELS], pl.FP32],
    attn_incoming: pl.Tensor[[T, HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[HIDDEN], pl.BF16],
    wq_a: pl.Tensor[[HIDDEN, LATENT], pl.INT8],
    wq_a_scale: pl.Tensor[[LATENT], pl.FP32],
    q_norm_weight: pl.Tensor[[LATENT], pl.BF16],
    wq_b: pl.Tensor[[LATENT, WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[WIDTH], pl.FP32],
    wkv: pl.Tensor[[HIDDEN, FLASH.head_dim], pl.INT8],
    wkv_scale: pl.Tensor[[FLASH.head_dim], pl.FP32],
    kv_norm_weight: pl.Tensor[[FLASH.head_dim], pl.BF16],
    wo_a: pl.Tensor[[FLASH.o_groups, GROUP_INPUT, O_RANK], pl.BF16],
    wo_b: pl.Tensor[[FLASH.o_groups * O_RANK, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_PAIRS], pl.FP32],
    rope_sin: pl.Tensor[[T, ROPE_PAIRS], pl.FP32],
    slots: pl.Tensor[[T, 2], pl.INT64],
    indices: pl.Tensor[[T, WINDOW], pl.INT32],
    sink: pl.Tensor[[FLASH.num_attention_heads], pl.FP32],
    positions: pl.InOut[pl.Tensor[[T], pl.INT64]],
    query_latent: pl.InOut[pl.Tensor[[T, LATENT], pl.BF16]],
    query: pl.InOut[pl.Tensor[[T, WIDTH], pl.BF16]],
    window_kv: pl.InOut[pl.Tensor[[T, FLASH.head_dim], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[2, 128, 1, FLASH.head_dim], pl.BF16]],
    cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
    attn_output: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    attn_next_pre: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    attn_post: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    attn_comb: pl.InOut[pl.Tensor[[T, HC, HC], pl.FP32]],
    attn_collapsed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    attn_normed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    after_attention: pl.InOut[pl.Tensor[[T, HC, HIDDEN], pl.BF16]],
    ffn_hc_function: pl.Tensor[[CHANNELS, HC * HIDDEN], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[CHANNELS], pl.FP32],
    ffn_norm_weight: pl.Tensor[[HIDDEN], pl.BF16],
    ffn_next_pre: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    ffn_post: pl.InOut[pl.Tensor[[T, HC], pl.FP32]],
    ffn_comb: pl.InOut[pl.Tensor[[T, HC, HC], pl.FP32]],
    ffn_collapsed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    ffn_normed: pl.InOut[pl.Tensor[[T, HIDDEN], pl.BF16]],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[D], pl.FP32],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[T * TOPK, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    logits: pl.InOut[pl.Tensor[[T, N_EXPERTS], pl.FP32]],
    route_ids: pl.InOut[pl.Tensor[[T, TOPK], pl.INT32]],
    route_weights: pl.InOut[pl.Tensor[[T, TOPK], pl.FP32]],
    moe_output: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T, HC, HIDDEN], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Run one rank's attention, FFN residual, and EP2 MoE composition."""

    ATTENTION(
        x_hc,
        attn_hc_function,
        attn_hc_scale,
        attn_hc_base,
        attn_incoming,
        attn_norm_weight,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        wo_a,
        wo_b,
        rope_cos,
        rope_sin,
        slots,
        indices,
        sink,
        positions,
        query_latent,
        query,
        window_kv,
        cache,
        cache_lens,
        attn_output,
        attn_next_pre,
        attn_post,
        attn_comb,
        attn_collapsed,
        attn_normed,
        after_attention,
        active,
    )

    ffn_collapsed_fp32 = pl.create_tensor([T, HIDDEN], dtype=pl.FP32)
    FFN_PRE(
        after_attention,
        ffn_hc_function,
        ffn_hc_scale,
        ffn_hc_base,
        attn_next_pre,
        ffn_next_pre,
        ffn_post,
        ffn_comb,
        ffn_collapsed, ffn_collapsed_fp32,
        active,
    )
    ffn_router_normed = pl.create_tensor([T, HIDDEN], dtype=pl.FP32)
    for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="dsv41_decode_ffn_norm"):
        # The fused reference normalizes the BF16 HC collapse and exposes its
        # FP32 result to the router.  Keep the BF16 store boundary before both
        # outputs so expert input and router input share the same source.
        rms_norm_tiled_with_fp32(
            ffn_collapsed, ffn_norm_weight, ffn_normed, ffn_router_normed, block, active
        )

    EP2_RANK(
        ffn_normed,
        ffn_router_normed,
        gate_weight,
        text_bias,
        vision_bias,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_gate_up,
        shared_gate_up_scale,
        shared_down,
        shared_down_scale,
        lane_x,
        lane_aux,
        lane_count,
        lane_ready,
        slot_out,
        return_ready,
        logits,
        route_ids,
        route_weights,
        moe_output,
        active,
        rank,
        epoch,
    )
    FFN_POST(moe_output, after_attention, ffn_post, ffn_comb, output, active)
    return output


@pl.jit.host
def decode_layer_ep2(
    x_hc: pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16],
    attn_hc_function: pl.Tensor[[EP_SIZE, CHANNELS, HC * HIDDEN], pl.FP32],
    attn_hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    attn_hc_base: pl.Tensor[[EP_SIZE, CHANNELS], pl.FP32],
    attn_incoming: pl.Tensor[[EP_SIZE, T, HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, HIDDEN], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, HIDDEN, LATENT], pl.INT8],
    wq_a_scale: pl.Tensor[[EP_SIZE, LATENT], pl.FP32],
    q_norm_weight: pl.Tensor[[EP_SIZE, LATENT], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, LATENT, WIDTH], pl.INT8],
    wq_b_scale: pl.Tensor[[EP_SIZE, WIDTH], pl.FP32],
    wkv: pl.Tensor[[EP_SIZE, HIDDEN, FLASH.head_dim], pl.INT8],
    wkv_scale: pl.Tensor[[EP_SIZE, FLASH.head_dim], pl.FP32],
    kv_norm_weight: pl.Tensor[[EP_SIZE, FLASH.head_dim], pl.BF16],
    wo_a: pl.Tensor[[EP_SIZE, FLASH.o_groups, GROUP_INPUT, O_RANK], pl.BF16],
    wo_b: pl.Tensor[[EP_SIZE, FLASH.o_groups * O_RANK, HIDDEN], pl.BF16],
    rope_cos: pl.Tensor[[EP_SIZE, T, ROPE_PAIRS], pl.FP32],
    rope_sin: pl.Tensor[[EP_SIZE, T, ROPE_PAIRS], pl.FP32],
    slots: pl.Tensor[[EP_SIZE, T, 2], pl.INT64],
    indices: pl.Tensor[[EP_SIZE, T, WINDOW], pl.INT32],
    sink: pl.Tensor[[EP_SIZE, FLASH.num_attention_heads], pl.FP32],
    positions: pl.InOut[pl.Tensor[[EP_SIZE, T], pl.INT64]],
    query_latent: pl.InOut[pl.Tensor[[EP_SIZE, T, LATENT], pl.BF16]],
    query: pl.InOut[pl.Tensor[[EP_SIZE, T, WIDTH], pl.BF16]],
    window_kv: pl.InOut[pl.Tensor[[EP_SIZE, T, FLASH.head_dim], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[EP_SIZE, 2, 128, 1, FLASH.head_dim], pl.BF16]],
    cache_lens: pl.InOut[pl.Tensor[[EP_SIZE, 2], pl.INT32]],
    attn_output: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    attn_next_pre: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    attn_post: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    attn_comb: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HC], pl.FP32]],
    attn_collapsed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    attn_normed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    after_attention: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16]],
    ffn_hc_function: pl.Tensor[[EP_SIZE, CHANNELS, HC * HIDDEN], pl.FP32],
    ffn_hc_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    ffn_hc_base: pl.Tensor[[EP_SIZE, CHANNELS], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, HIDDEN], pl.BF16],
    ffn_next_pre: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    ffn_post: pl.InOut[pl.Tensor[[EP_SIZE, T, HC], pl.FP32]],
    ffn_comb: pl.InOut[pl.Tensor[[EP_SIZE, T, HC, HC], pl.FP32]],
    ffn_collapsed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    ffn_normed: pl.InOut[pl.Tensor[[EP_SIZE, T, HIDDEN], pl.BF16]],
    gate_weight: pl.Tensor[[EP_SIZE, N_EXPERTS, D], pl.FP32],
    text_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    vision_bias: pl.Tensor[[EP_SIZE, N_EXPERTS], pl.FP32],
    input_ids: pl.Tensor[[EP_SIZE, T], pl.INT64],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, 2560], pl.INT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[EP_SIZE, D, MOE_INTER * 2], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[EP_SIZE, MOE_INTER * 2], pl.FP32],
    shared_down: pl.Tensor[[EP_SIZE, MOE_INTER, D], pl.INT8],
    shared_down_scale: pl.Tensor[[EP_SIZE, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[EP_SIZE, T, N_EXPERTS], pl.FP32]],
    route_ids: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.INT32]],
    route_weights: pl.Out[pl.Tensor[[EP_SIZE, T, TOPK], pl.FP32]],
    moe_output: pl.Out[pl.Tensor[[EP_SIZE, T, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[EP_SIZE, T, HC, HIDDEN], pl.BF16]],
    active: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Launch the complete L0 Decode layer across the two EP ranks."""

    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([T * TOPK, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(EP_SIZE):
        lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buf, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        lane_ready = pld.window(lane_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buf, [T * TOPK, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        decode_layer_ep2_rank(
            x_hc[rank],
            attn_hc_function[rank],
            attn_hc_scale[rank],
            attn_hc_base[rank],
            attn_incoming[rank],
            attn_norm_weight[rank],
            wq_a[rank],
            wq_a_scale[rank],
            q_norm_weight[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            wkv_scale[rank],
            kv_norm_weight[rank],
            wo_a[rank],
            wo_b[rank],
            rope_cos[rank],
            rope_sin[rank],
            slots[rank],
            indices[rank],
            sink[rank],
            positions[rank],
            query_latent[rank],
            query[rank],
            window_kv[rank],
            cache[rank],
            cache_lens[rank],
            attn_output[rank],
            attn_next_pre[rank],
            attn_post[rank],
            attn_comb[rank],
            attn_collapsed[rank],
            attn_normed[rank],
            after_attention[rank],
            ffn_hc_function[rank],
            ffn_hc_scale[rank],
            ffn_hc_base[rank],
            ffn_norm_weight[rank],
            ffn_next_pre[rank],
            ffn_post[rank],
            ffn_comb[rank],
            ffn_collapsed[rank],
            ffn_normed[rank],
            gate_weight[rank],
            text_bias[rank],
            vision_bias[rank],
            input_ids[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            shared_gate_up[rank],
            shared_gate_up_scale[rank],
            shared_down[rank],
            shared_down_scale[rank],
            lane_x,
            lane_aux,
            lane_count,
            lane_ready,
            slot_out,
            return_ready,
            logits[rank],
            route_ids[rank],
            route_weights[rank],
            moe_output[rank],
            output[rank],
            active,
            rank,
            epoch,
            device=rank,
        )


def product_contract() -> dict[str, object]:
    """Return the complete L0 EP2 composition boundary."""

    return {
        "status": "implementation_ready_not_run",
        "layer_id": 0,
        "topology": {"dp": 2, "tp": 1, "ep": 2, "pp": 1, "pcp": 1, "dcp": 1},
        "stages": [
            "mHC attention pre",
            "RMSNorm/QKV/RoPE/SWA cache read-write",
            "attention mHC residual expansion",
            "mHC FFN pre and RMSNorm",
            "PyPTO routed/shared EP2 dispatch-compute-combine",
            "mHC FFN output restoration",
        ],
        "product_compute": "PyPTO only",
        "reference_process": "independent vLLM-Ascend process",
        "formal_accuracy": "not_run",
    }


__all__ = ["decode_layer_ep2", "decode_layer_ep2_rank", "product_contract"]
