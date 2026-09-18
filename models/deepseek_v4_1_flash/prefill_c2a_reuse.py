# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C2A Reuse wired through mHC; the validation lives in ``prefill_c2a_full``."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import CMP_BLOCKS_DYN, D, HC_MULT, ORI_BLOCKS_DYN, T_DYN, TP_SIZE
from models.deepseek_v4_1_flash.decode_c2a_full import CMP_PACKED, CMP_SCALES
from models.deepseek_v4_1_flash.hc_post import mhc_post
from models.deepseek_v4_1_flash.prefill_c2a_full import attention_hc_pre, run_prefill_c2a
from models.deepseek_v4_1_flash.prefill_attn_c2a_reuse import prefill_attn_c2a_reuse


@pl.jit.inline
def prefill_c2a_reuse(
    x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    compressed_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    attn_input: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    attn_output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    next_pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    x_hc_out: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
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
        x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
        pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
        attn_input: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
        output_window: pld.DistributedTensor[[capacity, C.D], pl.FP32],
        output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Bind the runtime shapes and run the sublayer once per epoch on one rank."""
        x_hc.bind_dynamic(0, T_DYN)
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
        for step in pl.range(epochs):
            prefill_c2a_reuse(
                x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
                wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
                kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale,
                compressed_cache, compressed_cache_scale, compressed_indices, output_window,
                output_arrived, attn_input, attn_output, next_pre_mix, x_hc_out,
                rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, num_tokens, attention_epoch + step,
            )
        return x_hc_out, next_pre_mix, attn_output, attn_input, window_cache, window_cache_scale

    @pl.jit.host
    def c2a_reuse_group(
        x_hc: pl.Tensor[[world_size, C.T_DYN, C.HC_MULT, C.D], pl.FP32],
        pre_mix: pl.Tensor[[world_size, C.T_DYN, C.HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[world_size, C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[world_size, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[world_size, C.MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[world_size, C.D], pl.BF16],
        wq_a: pl.Tensor[[world_size, C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[world_size, C.D // 32, C.Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[world_size, C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[world_size, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[world_size, C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[world_size, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[world_size, C.D // 32, C.HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[world_size, C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[world_size, C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[world_size, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[world_size, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[world_size, C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[world_size, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[world_size, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[world_size, C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[world_size, C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[world_size, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[world_size, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.Tensor[[world_size, C.CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[world_size, C.CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[world_size, C.T_DYN, C.INDEX_TOPK], pl.INT32],
        attn_input: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Allocate the TP communication windows and launch one rank entry per device."""
        x_hc.bind_dynamic(1, T_DYN)
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
        data_buffer = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            data = pld.window(data_buffer, [capacity, D], dtype=pl.FP32)
            signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            c2a_reuse_rank(
                x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
                attn_norm_weight[rank], wq_a[rank], wq_a_scale[rank], q_norm_weight[rank], wq_b[rank],
                wq_b_scale[rank], wkv[rank], wkv_scale[rank], kv_norm_weight[rank],
                attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank], rope_cos[rank],
                rope_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank],
                window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
                compressed_indices[rank], attn_input[rank], attn_output[rank], next_pre_mix[rank],
                x_hc_out[rank], data, signal, rank, num_tokens, attention_epoch, device=rank,
            )

    return c2a_reuse_group


def main():
    """Validate packed prefill C2A Reuse wired through mHC on A5."""
    run_prefill_c2a(make_hc_program, "reuse")


__all__ = ["prefill_c2a_reuse"]


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    main()
