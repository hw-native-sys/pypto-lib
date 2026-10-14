# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SWA decode Attention half-layer composition entry."""

import hc_post as _decode_hc_post
import hc_pre as _decode_hc_pre
import utils as _decode_utils

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    FLASH as M,
    DECODE_INPUT_MAX_TOKENS,
    DECODE_MAX_TOKENS,
    TP_SIZE,
    T_DYN,
    ORI_BLOCKS_DYN,
    AttentionMode,
    WINDOW_CACHE_GROUP,
)



import decode_attn_swa as swa
from sparse_attention import SWA_INPUT_NAMES, build_swa_leaf_specs, golden_swa_leaf
from golden import ratio_allclose

# PyPTO resolves inline calls and JIT-body constants only from direct names.
from config import LOCAL_T_DYN
from decode_cp_allgather import decode_tp_input_all_gather, slab_owner
from decode_attn_swa import decode_attn_swa, decode_attn_swa_sharded
from hc_pre import attention_pre
from hc_post import mhc_post


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
MX_GROUP = M.weight_block_size[0]

KIND = "SWA"
REPRESENTATIVE_LAYER_ID = 0
KERNEL_READY = True
LEAF_NAMES = tuple(name for name in SWA_INPUT_NAMES if name != "x")
ATTENTION_SPEC_NAMES = (
    *_decode_utils.BOUNDARY_PREFIX_NAMES,
    *LEAF_NAMES,
    *_decode_utils.BOUNDARY_OUTPUT_NAMES,
    *_decode_utils.SCALAR_NAMES,
)
SHARDED_ATTENTION_SPEC_NAMES = (
    *_decode_utils.BOUNDARY_PREFIX_NAMES,
    *LEAF_NAMES,
    *_decode_utils.BOUNDARY_OUTPUT_NAMES,
    "gathered",
    *_decode_utils.SCALAR_NAMES,
)


def skip_reason(layer_id=REPRESENTATIVE_LAYER_ID):
    layer = M.layer_config(layer_id)
    if layer.mode != AttentionMode.SWA:
        return f"layer {layer_id} resolves to {layer.mode.value}, expected {KIND}"
    return None if KERNEL_READY else "SWA attention half-layer composition is pending"


@pl.jit.inline(auto_scope=False)
def decode_swa(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    attention_output: pl.Tensor[[T_DYN, D], pl.BF16],
    attention_hidden: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    attention_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one SWA attention sublayer and emit the staggered mix for the next sublayer."""
    tokens = pl.tensor.dim(x_hc, 0)
    attention_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normalized_attention = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post_mix, residual_mix = attention_pre(
        x_hc,
        incoming_pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        attention_input,
        normalized_attention,
        attention_pre_mix,
        num_tokens,
    )
    group_base = rank // TP_SIZE * TP_SIZE
    tp_rank = rank % TP_SIZE
    decode_attn_swa(
        normalized_attention,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        output_window,
        output_arrived,
        attention_output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    mhc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden)
    return attention_hidden, attention_pre_mix


@pl.jit.inline(auto_scope=False)
def decode_swa_sharded(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    attention_output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
    attention_hidden: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    attention_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    gathered: pl.Tensor[[T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one SWA attention sublayer and emit the staggered mix for the next sublayer."""
    tokens = pl.tensor.dim(x_hc, 0)
    group_base = rank // TP_SIZE * TP_SIZE
    tp_rank = rank % TP_SIZE
    # Sequence parallel: the residual stream stays local, only the normalized
    # Attention input crosses the TP group ([T_local, D] -> [T, D]).
    _, local_count = slab_owner(tp_rank, tokens, num_tokens)
    attention_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normalized_attention = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post_mix, residual_mix = attention_pre(
        x_hc,
        incoming_pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        attention_input,
        normalized_attention,
        attention_pre_mix,
        local_count,
    )
    gathered = decode_tp_input_all_gather(
        normalized_attention, input_window, input_arrived, gathered, group_base, tp_rank,
        num_tokens, attention_epoch,
    )
    decode_attn_swa_sharded(
        gathered,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        output_window,
        output_arrived,
        attention_output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    mhc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden)
    return attention_hidden, attention_pre_mix


@pl.jit
def decode_swa_rank(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    attention_output: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Dispatch one production SWA composition to a device-selected JIT entry."""
    return decode_swa(
        x_hc,
        incoming_pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        attention_output,
        attention_hidden,
        attention_pre_mix,
        output_window,
        output_arrived,
        rank,
        num_tokens,
        attention_epoch,
    )


@pl.jit
def decode_swa_rank_sharded(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    attention_output: pl.InOut[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    gathered: pl.Tensor[[T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Dispatch one production SWA composition to a device-selected JIT entry."""
    return decode_swa_sharded(
        x_hc,
        incoming_pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        attention_output,
        attention_hidden,
        attention_pre_mix,
        gathered,
        input_window,
        input_arrived,
        output_window,
        output_arrived,
        rank,
        num_tokens,
        attention_epoch,
    )


def _make_program(world_size, epochs, specs):
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be in [1, 1000]")
    tensor_specs = _decode_utils.check_program_specs(world_size, specs, ATTENTION_SPEC_NAMES)

    @pl.jit.host
    def attention_group(
        x_hc: pl.Tensor[tensor_specs.x_hc.shape, tensor_specs.x_hc.dtype],
        incoming_pre_mix: pl.Tensor[tensor_specs.incoming_pre_mix.shape, tensor_specs.incoming_pre_mix.dtype],
        hc_attn_fn: pl.Tensor[tensor_specs.hc_attn_fn.shape, tensor_specs.hc_attn_fn.dtype],
        hc_attn_scale: pl.Tensor[tensor_specs.hc_attn_scale.shape, tensor_specs.hc_attn_scale.dtype],
        hc_attn_base: pl.Tensor[tensor_specs.hc_attn_base.shape, tensor_specs.hc_attn_base.dtype],
        attn_norm_weight: pl.Tensor[tensor_specs.attn_norm_weight.shape, tensor_specs.attn_norm_weight.dtype],
        wq_a: pl.Tensor[tensor_specs.wq_a.shape, tensor_specs.wq_a.dtype],
        wq_a_scale: pl.Tensor[tensor_specs.wq_a_scale.shape, tensor_specs.wq_a_scale.dtype],
        q_norm_weight: pl.Tensor[tensor_specs.q_norm_weight.shape, tensor_specs.q_norm_weight.dtype],
        wq_b: pl.Tensor[tensor_specs.wq_b.shape, tensor_specs.wq_b.dtype],
        wq_b_scale: pl.Tensor[tensor_specs.wq_b_scale.shape, tensor_specs.wq_b_scale.dtype],
        wkv: pl.Tensor[tensor_specs.wkv.shape, tensor_specs.wkv.dtype],
        wkv_scale: pl.Tensor[tensor_specs.wkv_scale.shape, tensor_specs.wkv_scale.dtype],
        kv_norm_weight: pl.Tensor[tensor_specs.kv_norm_weight.shape, tensor_specs.kv_norm_weight.dtype],
        attn_sink: pl.Tensor[tensor_specs.attn_sink.shape, tensor_specs.attn_sink.dtype],
        wo_a: pl.Tensor[tensor_specs.wo_a.shape, tensor_specs.wo_a.dtype],
        wo_b: pl.Tensor[tensor_specs.wo_b.shape, tensor_specs.wo_b.dtype],
        wo_b_scale: pl.Tensor[tensor_specs.wo_b_scale.shape, tensor_specs.wo_b_scale.dtype],
        rope_cos: pl.Tensor[tensor_specs.rope_cos.shape, tensor_specs.rope_cos.dtype],
        rope_sin: pl.Tensor[tensor_specs.rope_sin.shape, tensor_specs.rope_sin.dtype],
        window_slots: pl.Tensor[tensor_specs.window_slots.shape, tensor_specs.window_slots.dtype],
        window_indices: pl.Tensor[tensor_specs.window_indices.shape, tensor_specs.window_indices.dtype],
        window_cache: pl.InOut[pl.Tensor[tensor_specs.window_cache.shape, tensor_specs.window_cache.dtype]],
        window_cache_scale: pl.InOut[
            pl.Tensor[tensor_specs.window_cache_scale.shape, tensor_specs.window_cache_scale.dtype]
        ],
        attention_output: pl.InOut[
            pl.Tensor[tensor_specs.attention_output.shape, tensor_specs.attention_output.dtype]
        ],
        attention_hidden: pl.Out[
            pl.Tensor[tensor_specs.attention_hidden.shape, tensor_specs.attention_hidden.dtype]
        ],
        attention_pre_mix: pl.Out[
            pl.Tensor[tensor_specs.attention_pre_mix.shape, tensor_specs.attention_pre_mix.dtype]
        ],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        data_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for step in pl.range(epochs):
            for rank in pl.range(world_size):
                data = pld.window(data_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
                arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
                # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
                wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
                wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = wq_b_scale[rank]
                wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
                wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
                decode_swa_rank(
                    x_hc[rank],
                    incoming_pre_mix[rank],
                    hc_attn_fn[rank],
                    hc_attn_scale[rank],
                    hc_attn_base[rank],
                    attn_norm_weight[rank],
                    wq_a[rank],
                    wq_a_scale_r,
                    q_norm_weight[rank],
                    wq_b[rank],
                    wq_b_scale_r,
                    wkv[rank],
                    wkv_scale_r,
                    kv_norm_weight[rank],
                    attn_sink[rank],
                    wo_a[rank],
                    wo_b[rank],
                    wo_b_scale_r,
                    rope_cos[rank],
                    rope_sin[rank],
                    window_slots[rank],
                    window_indices[rank],
                    window_cache[rank],
                    window_cache_scale[rank],
                    attention_output[rank],
                    attention_hidden[rank],
                    attention_pre_mix[rank],
                    data,
                    arrived,
                    rank,
                    num_tokens,
                    attention_epoch + step,
                    device=rank,
                )

    return attention_group


def make_program(world_size, epochs, specs, sharded=False):
    """Build the Attention host; ``sharded`` selects the sequence-parallel wiring."""
    if sharded:
        return _make_program_sharded(world_size, epochs, specs)
    return _make_program(world_size, epochs, specs)


def _make_program_sharded(world_size, epochs, specs):
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be in [1, 1000]")
    tensor_specs = _decode_utils.check_program_specs(world_size, specs, SHARDED_ATTENTION_SPEC_NAMES)

    @pl.jit.host
    def attention_group(
        x_hc: pl.Tensor[tensor_specs.x_hc.shape, tensor_specs.x_hc.dtype],
        incoming_pre_mix: pl.Tensor[tensor_specs.incoming_pre_mix.shape, tensor_specs.incoming_pre_mix.dtype],
        hc_attn_fn: pl.Tensor[tensor_specs.hc_attn_fn.shape, tensor_specs.hc_attn_fn.dtype],
        hc_attn_scale: pl.Tensor[tensor_specs.hc_attn_scale.shape, tensor_specs.hc_attn_scale.dtype],
        hc_attn_base: pl.Tensor[tensor_specs.hc_attn_base.shape, tensor_specs.hc_attn_base.dtype],
        attn_norm_weight: pl.Tensor[tensor_specs.attn_norm_weight.shape, tensor_specs.attn_norm_weight.dtype],
        wq_a: pl.Tensor[tensor_specs.wq_a.shape, tensor_specs.wq_a.dtype],
        wq_a_scale: pl.Tensor[tensor_specs.wq_a_scale.shape, tensor_specs.wq_a_scale.dtype],
        q_norm_weight: pl.Tensor[tensor_specs.q_norm_weight.shape, tensor_specs.q_norm_weight.dtype],
        wq_b: pl.Tensor[tensor_specs.wq_b.shape, tensor_specs.wq_b.dtype],
        wq_b_scale: pl.Tensor[tensor_specs.wq_b_scale.shape, tensor_specs.wq_b_scale.dtype],
        wkv: pl.Tensor[tensor_specs.wkv.shape, tensor_specs.wkv.dtype],
        wkv_scale: pl.Tensor[tensor_specs.wkv_scale.shape, tensor_specs.wkv_scale.dtype],
        kv_norm_weight: pl.Tensor[tensor_specs.kv_norm_weight.shape, tensor_specs.kv_norm_weight.dtype],
        attn_sink: pl.Tensor[tensor_specs.attn_sink.shape, tensor_specs.attn_sink.dtype],
        wo_a: pl.Tensor[tensor_specs.wo_a.shape, tensor_specs.wo_a.dtype],
        wo_b: pl.Tensor[tensor_specs.wo_b.shape, tensor_specs.wo_b.dtype],
        wo_b_scale: pl.Tensor[tensor_specs.wo_b_scale.shape, tensor_specs.wo_b_scale.dtype],
        rope_cos: pl.Tensor[tensor_specs.rope_cos.shape, tensor_specs.rope_cos.dtype],
        rope_sin: pl.Tensor[tensor_specs.rope_sin.shape, tensor_specs.rope_sin.dtype],
        window_slots: pl.Tensor[tensor_specs.window_slots.shape, tensor_specs.window_slots.dtype],
        window_indices: pl.Tensor[tensor_specs.window_indices.shape, tensor_specs.window_indices.dtype],
        window_cache: pl.InOut[pl.Tensor[tensor_specs.window_cache.shape, tensor_specs.window_cache.dtype]],
        window_cache_scale: pl.InOut[
            pl.Tensor[tensor_specs.window_cache_scale.shape, tensor_specs.window_cache_scale.dtype]
        ],
        attention_output: pl.InOut[
            pl.Tensor[tensor_specs.attention_output.shape, tensor_specs.attention_output.dtype]
        ],
        attention_hidden: pl.Out[
            pl.Tensor[tensor_specs.attention_hidden.shape, tensor_specs.attention_hidden.dtype]
        ],
        attention_pre_mix: pl.Out[
            pl.Tensor[tensor_specs.attention_pre_mix.shape, tensor_specs.attention_pre_mix.dtype]
        ],
        gathered: pl.Out[pl.Tensor[tensor_specs.gathered.shape, tensor_specs.gathered.dtype]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        data_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        input_transport = pld.alloc_window_buffer([DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
        input_signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for step in pl.range(epochs):
            for rank in pl.range(world_size):
                data = pld.window(data_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
                arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
                input_window = pld.window(input_transport, [DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
                input_arrived = pld.window(input_signals, [TP_SIZE, 1], dtype=pl.INT32)
                # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
                wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
                wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = wq_b_scale[rank]
                wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
                wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
                decode_swa_rank_sharded(
                    x_hc[rank],
                    incoming_pre_mix[rank],
                    hc_attn_fn[rank],
                    hc_attn_scale[rank],
                    hc_attn_base[rank],
                    attn_norm_weight[rank],
                    wq_a[rank],
                    wq_a_scale_r,
                    q_norm_weight[rank],
                    wq_b[rank],
                    wq_b_scale_r,
                    wkv[rank],
                    wkv_scale_r,
                    kv_norm_weight[rank],
                    attn_sink[rank],
                    wo_a[rank],
                    wo_b[rank],
                    wo_b_scale_r,
                    rope_cos[rank],
                    rope_sin[rank],
                    window_slots[rank],
                    window_indices[rank],
                    window_cache[rank],
                    window_cache_scale[rank],
                    attention_output[rank],
                    attention_hidden[rank],
                    attention_pre_mix[rank],
                    gathered[rank],
                    input_window,
                    input_arrived,
                    data,
                    arrived,
                    rank,
                    num_tokens,
                    attention_epoch + step,
                    device=rank,
                )

    return attention_group


def build_specs(args, initial_state=None):
    del initial_state
    return _decode_utils.assemble_specs(args, build_swa_leaf_specs(args, "decode"), ATTENTION_SPEC_NAMES)


def build_specs_sharded(args):
    """Specs with per-rank token slabs plus the Attention-input gather scratch."""
    value_build_swa_specs = build_swa_leaf_specs(args, 'decode')
    return _decode_utils.assemble_specs(args, value_build_swa_specs, SHARDED_ATTENTION_SPEC_NAMES, sharded=True)


def make_golden(epochs):
    def golden_half(tensors):
        for _ in range(epochs):
            normalized, post, residual = _decode_hc_pre.golden_attention_pre(tensors)
            leaf = dict(tensors, x=normalized, output=tensors["attention_output"])
            golden_swa_leaf(leaf)
            _decode_hc_post.golden_attention_post(tensors, post, residual)

    return golden_half


def make_golden_sharded(epochs):
    def golden_half(tensors):
        for _ in range(epochs):
            gathered, post, residual = _decode_hc_pre.golden_attention_pre_sharded(tensors)
            leaf = dict(tensors, x=gathered, output=tensors["attention_output"])
            golden_swa_leaf(leaf, sharded=True)
            _decode_hc_post.golden_attention_post(tensors, post, residual)

    return golden_half


def comparisons(args, devices):
    leaf = ratio_allclose(atol=1e-2, rtol=1.0 / 128)
    compare = _decode_utils.make_boundary_comparisons(leaf)
    compare.update(
        {
            "attention_output": leaf,
            # Composed rows can flip one FP8 code at a quantization boundary: allow one
            # subnormal step (2**-9) plus one relative step (2**-3) on the payload, and less
            # than 1% of mapped rows to miss even that.
            **swa.window_cache_compare(
                len(devices), args.tokens, payload_atol=2 ** -9, payload_rtol=2 ** -3,
                payload_error_ratio=0.01,
            ),
        }
    )
    return compare


def comparisons_sharded(args, devices):
    leaf = ratio_allclose(atol=1e-2, rtol=1.0 / 128)
    compare = _decode_utils.make_boundary_comparisons(leaf)
    compare.update(
        {
            "attention_output": leaf,
            # One BF16 step is the noise floor of the rows rounded once by the local collapse.
            "gathered": ratio_allclose(atol=2 ** -6, rtol=2 ** -6),
            # Composed rows can flip one FP8 code at a quantization boundary: allow one
            # subnormal step (2**-9) plus one relative step (2**-3) on the payload, and less
            # than 1% of mapped rows to miss even that.
            **swa.window_cache_compare(
                len(devices), args.tokens, payload_atol=2 ** -9, payload_rtol=2 ** -3,
                payload_error_ratio=0.01,
            ),
        }
    )
    return compare


if __name__ == "__main__":
    import argparse
    from pypto.ir import DistributedConfig
    from golden import run
    from config import MAX_BATCH_PER_DP

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 SWA decode Attention composition")
    parser.add_argument("--stage", choices=("attention", "block"), default="attention")
    parser.add_argument("--cpu-golden", action="store_true")
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--tp", type=int, choices=(1, 4), default=TP_SIZE)
    parser.add_argument("--layer-id", type=int, default=REPRESENTATIVE_LAYER_ID)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=('mixed', 'shuffle', 'prefix'), default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--wiring", choices=("both", "replicated", "sharded"), default="both",
                        help="which wiring to run")
    parser.add_argument("--start-pos", type=str, default=None,
                        help="absolute decode start positions; a comma-separated list sets batch to its length")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()
    if args.start_pos is not None:
        try:
            positions = [int(value.strip()) for value in args.start_pos.split(",")]
        except ValueError:
            parser.error("--start-pos must contain comma-separated integers")
        if not 1 <= len(positions) <= min(MAX_BATCH_PER_DP, DECODE_MAX_TOKENS):
            parser.error("--start-pos batch exceeds the decode request capacity")
        if any(position < 0 or position >= M.max_position_embeddings for position in positions):
            parser.error(f"--start-pos values must be in [0, {M.max_position_embeddings})")
        if args.case not in ("random", "mixed"):
            parser.error("--start-pos requires the default random/mixed case")
        args.start_pos = positions
        args.tokens = args.requests = len(positions)
        if args.active_tokens is not None and args.active_tokens != args.tokens:
            parser.error("--start-pos requires every request to be active")
        args.active_tokens = args.tokens
    if args.cpu_golden:
        parser.error("Block CPU goldens are provided by decode_layer.py")
    if args.stage == "block":
        parser.error("Block composition is provided by decode_layer.py")
    if not 1 <= args.tokens <= DECODE_MAX_TOKENS or not 1 <= args.epochs <= 1000:
        parser.error("tokens or epochs out of range")
    args.active_tokens = args.tokens if args.active_tokens is None else args.active_tokens
    if not 1 <= args.active_tokens <= args.tokens or not 1 <= args.requests <= args.active_tokens:
        parser.error("require 1 <= requests <= active tokens <= tokens")
    if args.requests > MAX_BATCH_PER_DP:
        parser.error(f"requests must not exceed {MAX_BATCH_PER_DP}")
    if args.active_tokens != args.tokens:
        parser.error("this Attention mode requires active tokens to equal token capacity")
    args.dp, args.bench = 1, False
    if args.tp != TP_SIZE:
        parser.error("--tp must match the import-time tensor parallel configuration")
    devices = (
        list(range(args.tp))
        if args.device is None or args.compile_only
        else [int(device) for device in args.device.split(",")]
    )
    if len(devices) != args.tp or len(set(devices)) != args.tp or min(devices) < 0:
        parser.error("--device must name exactly TP distinct nonnegative device IDs")
    reason = skip_reason(args.layer_id)
    if reason:
        parser.error(reason)
    results = []
    wirings = ("replicated", "sharded") if args.wiring == "both" else (args.wiring,)
    for wiring in wirings:
        sharded = wiring == "sharded"
        if sharded:
            if args.tokens < 1 or args.tp < 1:
                parser.error("sequence-parallel requires positive tokens and tp")
            if args.active_tokens > args.tokens:
                parser.error("active tokens cannot exceed physical token capacity")
            args.local_tokens = (args.tokens + args.tp - 1) // args.tp
        specs = build_specs_sharded(args) if sharded else build_specs(args)
        compare = comparisons_sharded(args, devices) if sharded else comparisons(args, devices)
        program = make_program(args.tp, args.epochs, specs, sharded=sharded)
        golden_fn = make_golden_sharded(args.epochs) if sharded else make_golden(args.epochs)
        result = run(
            fn=program,
            specs=specs,
            golden_fn=golden_fn,
            compare_fn=compare,
            config={
                "platform": args.platform,
                "enable_chip_swimlane": args.enable_chip_swimlane,
                "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            },
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        kind_name = f"{KIND} sequence-parallel" if sharded else KIND
        print(f"[HALF] layer={args.layer_id} kind={kind_name} work_dir={result.work_dir}")
        results.append(result)
    for result in results:
        if not result.passed:
            raise SystemExit(result.error or 1)
