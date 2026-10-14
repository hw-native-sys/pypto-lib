# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C2A Reuse wired through mHC; the validation lives in ``prefill_c2a_full``."""


import pypto.language as pl
import pypto.language.distributed as pld


from config import (
    FLASH as M,
    CMP_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    T_DYN,
    TP_SIZE,
    COMPRESSED_CACHE_GROUP,
    PREFILL_MAX_TOKENS,
    WINDOW_CACHE_GROUP,
)

from config import SP_T_DYN
from prefill_cp_allgather import prefill_sp_input_allgather
from o_proj import prefill_sp_output_reduce_scatter
from hc_post import prefill_sp_post
from prefill_c2a_full import CMP_PACKED, CMP_SCALES, attention_hc_pre, run_prefill_c2a
from prefill_attn_c2a import (
    prefill_attn_c2a_reuse, prefill_attn_c2a_reuse_partial,
)
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
INDEX_TOPK = M.index_topk


@pl.jit.inline
def prefill_c2a_reuse_sp(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    x_hc_out: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Reuse attention sublayer between mHC pre and post; see ``prefill_c2a_full``."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attention_hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, next_pre_mix,
        post_mix, residual_mix, local_hidden,
    )
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, attn_input,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    group_tokens = pl.tensor.dim(attn_input, 0)
    partial = pl.create_tensor([group_tokens, D], dtype=pl.FP32)
    if num_tokens > 0:
        prefill_attn_c2a_reuse_partial(
            attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            compressed_indices, output_arrived, partial,
            num_tokens, attention_epoch,
        )
    prefill_sp_output_reduce_scatter(
        partial, output_window, output_arrived, attn_output,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    prefill_sp_post(
        attn_output, x_hc, post_mix, residual_mix, x_hc_out, next_pre_mix, tp_rank, num_tokens,
    )
    return x_hc_out


@pl.jit.inline
def prefill_c2a_reuse(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_output: pl.Tensor[[T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    x_hc_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Reuse attention sublayer between mHC pre and post; see ``prefill_c2a_full``."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    attention_hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, next_pre_mix,
        post_mix, residual_mix, attn_input,
    )
    prefill_attn_c2a_reuse(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
        attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        compressed_indices, output_window, output_arrived, attn_output, group_base, tp_rank,
        num_tokens, attention_epoch,
    )
    mhc_post(attn_output, x_hc, post_mix, residual_mix, x_hc_out)
    return x_hc_out


def make_hc_program(capacity, world_size, epochs):
    """Build the L3 group entry that runs one mHC-wrapped C2A Reuse sublayer on every rank."""

    @pl.jit
    def c2a_reuse_rank(
        x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
        input_window: pld.DistributedTensor[[capacity, D], pl.BF16],
        input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Tensor[[1], pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Bind the runtime shapes and run the sublayer once per epoch on one rank."""
        x_hc.bind_dynamic(0, SP_T_DYN)
        attn_input.bind_dynamic(0, T_DYN)
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
        active = pl.read(num_tokens, [0])
        for step in pl.range(epochs):
            prefill_c2a_reuse_sp(
                x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
                wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
                kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale,
                compressed_cache, compressed_cache_scale, compressed_indices, input_window, input_arrived, output_window,
                output_arrived, attn_input, attn_output, next_pre_mix, x_hc_out,
                rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active, attention_epoch + step,
            )
        return x_hc_out, next_pre_mix, attn_output, attn_input, window_cache, window_cache_scale

    @pl.jit.host
    def c2a_reuse_group(
        x_hc: pl.Tensor[[world_size, SP_T_DYN, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[world_size, SP_T_DYN, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[world_size, MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[world_size, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[world_size, MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[world_size, D], pl.BF16],
        wq_a: pl.Tensor[[world_size, D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[world_size, D // 32, Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[world_size, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[world_size, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[world_size, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[world_size, D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[world_size, D // 32, HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[world_size, HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[world_size, LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[world_size, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[world_size, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[world_size, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[world_size, T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[world_size, T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[world_size, T_DYN], pl.INT64],
        window_indices: pl.Tensor[[world_size, T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[world_size, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[world_size, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.Tensor[[world_size, CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[world_size, CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[world_size, T_DYN, INDEX_TOPK], pl.INT32],
        attn_input: pl.Out[pl.Tensor[[world_size, T_DYN, D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[world_size, SP_T_DYN, D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[world_size, SP_T_DYN, HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[world_size, SP_T_DYN, HC_MULT, D], pl.FP32]],
        num_tokens: pl.Tensor[[world_size, 1], pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Allocate the TP communication windows and launch one rank entry per device."""
        x_hc.bind_dynamic(1, SP_T_DYN)
        attn_input.bind_dynamic(1, T_DYN)
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
        input_buffer = pld.alloc_window_buffer([capacity, D], dtype=pl.BF16)
        input_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        data_buffer = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            input_data = pld.window(input_buffer, [capacity, D], dtype=pl.BF16)
            input_signal = pld.window(input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            data = pld.window(data_buffer, [capacity, D], dtype=pl.FP32)
            signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            # Each rank consumes packed MX_B_NN scale rows.
            wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
            wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
            wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
            wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
            c2a_reuse_rank(
                x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
                attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank],
                wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank],
                attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank],
                rope_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank],
                window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
                compressed_indices[rank], attn_input[rank], attn_output[rank], next_pre_mix[rank],
                x_hc_out[rank], input_data, input_signal, data, signal, rank, num_tokens[rank], attention_epoch, device=rank,
            )

    return c2a_reuse_group


def validate(argv=None):
    """Validate packed prefill C2A Reuse wired through mHC on A5."""
    return run_prefill_c2a(make_hc_program, "reuse", argv=argv)


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
