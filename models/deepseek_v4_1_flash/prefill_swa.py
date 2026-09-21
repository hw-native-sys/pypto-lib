# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-sharded packed-prefill SWA with local mHC and head-TP attention."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import math

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash.attention_common import golden_swa_attention
from models.deepseek_v4_1_flash.config import (
    D,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MIX_HC,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCKS_DYN,
    PREFILL_MAX_TOKENS,
    Q_LORA,
    ROPE_DIM,
    T_DYN,
    TP_SIZE,
    WINDOW_CACHE_GROUP,
)
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post, mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre
from models.deepseek_v4_1_flash.decode_common import attention_pre
from models.deepseek_v4_1_flash.golden import rms_norm
from models.deepseek_v4_1_flash.prefill_attn_swa import prefill_attn_swa_partial
from models.deepseek_v4_1_flash.attention_sp import (
    SP_T_DYN,
    prefill_sp_input_allgather,
    prefill_sp_output_reduce_scatter,
)


# Prefill SWA plus four-stream mHC overflows the default 256 MiB ring heap.
# One GiB per ring matches the V4 Pro prefill attention bring-up budget.
PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4


@pl.jit.inline(auto_scope=False)
def prefill_swa(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
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
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    output: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Gather local mHC inputs, reduce-scatter SWA partials, and update local streams."""
    tokens = pl.tensor.dim(x_hc, 0)
    collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post_mix, residual_mix = attention_pre(
        x_hc, incoming_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        collapsed, local_hidden, next_pre_mix, tokens,
    )
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, hidden,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    group_tokens = pl.tensor.dim(hidden, 0)
    partial = pl.create_tensor([group_tokens, D], dtype=pl.FP32)
    if num_tokens > 0:
        prefill_attn_swa_partial(
            hidden, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale,
            rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
            output_arrived, partial, num_tokens, attention_epoch,
        )
    prefill_sp_output_reduce_scatter(
        partial, output_window, output_arrived, attn_out,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    output_flat = pl.reshape(output, [tokens, HC_DIM])
    with pl.spmd(tokens, name_hint="swa_sp_zero_padding"):
        row = pl.tile.get_block_idx()
        if tp_rank * tokens + row >= num_tokens:
            zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
            for col in pl.range(0, HC_DIM, 512):
                zero = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                output_flat = pl.store(zero, [row, col], output_flat)
    return output, next_pre_mix


def golden_prefill_swa(
    x_hc: torch.Tensor,
    incoming_pre_mix: torch.Tensor,
    hc_attn_fn: torch.Tensor,
    hc_attn_scale: torch.Tensor,
    hc_attn_base: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official staggered mHC boundary, input RMSNorm, SWA, and residual expansion."""
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    hidden = rms_norm(golden_mhc_pre(x_hc, incoming_pre_mix), attn_norm_weight)
    result = golden_swa_attention(
        hidden, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
    )
    output = golden_mhc_post(result.output.to(torch.bfloat16), x_hc, post_mix, residual_mix)
    return output, result.window_cache, result.window_cache_scale, pre_mix


HC_INPUT_NAMES = (
    "x_hc", "incoming_pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
    "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
)


# Layer-0 mHC fixture statistics from DeepSeek-V4.1-Flash at
# dba1be0a40aa45a94ad051997016db3960a90277, model-00003-of-00048.safetensors.
HC_FIXTURE_FN_STD = 0.0225357748568058
HC_FIXTURE_SCALE = (0.06304993480443954, 0.011977050453424454, 0.06153993308544159)
HC_FIXTURE_BASE = (
    2.845095157623291, -3.343229293823242, -3.233515501022339, -2.217123031616211,
    -4.17180871963501, -3.9524407386779785, -19.57306671142578, -1.4172816276550293,
    0.20820771157741547, -25.451234817504883, -5.102978229522705, 0.09652144461870193,
    0.5044570565223694, -0.4407903552055359, -25.251018524169922, -21.587751388549805,
    -25.49090003967285, 0.1881018877029419, -0.11225703358650208, -22.04298973083496,
    -19.504518508911133, -21.67133140563965, -22.463973999023438, 13.438657760620117,
)


def make_hc_inputs(base: dict, seed: int, fixture: str = "checkpoint") -> dict:
    """Replace the collapsed hidden with four HC streams and mix weights."""
    gen = torch.Generator().manual_seed(seed + 2000)
    tokens = base["x"].shape[0]
    values = dict(base)
    values.pop("x")
    if fixture == "checkpoint":
        values["x_hc"] = torch.rand(tokens, HC_MULT, D, generator=gen) * 2 - 1
        values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM, generator=gen) * HC_FIXTURE_FN_STD
        values["hc_attn_scale"] = torch.tensor(HC_FIXTURE_SCALE)
        values["hc_attn_base"] = torch.tensor(HC_FIXTURE_BASE)
        values["incoming_pre_mix"] = torch.zeros(tokens, HC_MULT)
        values["incoming_pre_mix"][:, 0] = 1
    elif fixture == "random":
        values["x_hc"] = torch.randn(tokens, HC_MULT, D, generator=gen)
        values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM)
        values["hc_attn_scale"] = torch.randn(3, generator=gen)
        values["hc_attn_base"] = torch.randn(MIX_HC, generator=gen)
        values["incoming_pre_mix"] = torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen))
    else:
        raise ValueError(f"unsupported fixture: {fixture}")
    values["attn_norm_weight"] = torch.ones(D, dtype=torch.bfloat16)
    if bool((base["x"] == 0).all()):
        values["x_hc"].zero_()
    return values


def make_hc_program(capacity, world_size, epochs):
    """Wrap the HC-orchestrated prefill SWA operator for stacked TP/DP ranks."""
    @pl.jit
    def swa_rank(
        x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
        incoming_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
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
        window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]],
        output: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
        hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        attn_out: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
        input_window: pld.DistributedTensor[[capacity, D], pl.BF16],
        input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        num_tokens: pl.Tensor[[1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Run one rank of HC-orchestrated prefill SWA for ``epochs`` dispatches."""
        x_hc.bind_dynamic(0, SP_T_DYN)
        hidden.bind_dynamic(0, T_DYN)
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        active_tokens = pl.read(num_tokens, [0])
        for step in pl.range(epochs):
            prefill_swa(
                x_hc, incoming_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
                wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale,
                rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
                input_window, input_arrived, output_window, output_arrived, hidden, attn_out, output, next_pre_mix,
                rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active_tokens, attention_epoch + step,
            )
        return output, next_pre_mix, hidden, attn_out, window_cache, window_cache_scale

    @pl.jit.host
    def swa_group(
        x_hc: pl.Tensor[[world_size, SP_T_DYN, HC_MULT, D], pl.FP32],
        incoming_pre_mix: pl.Tensor[[world_size, SP_T_DYN, HC_MULT], pl.FP32],
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
        window_cache_scale: pl.InOut[pl.Tensor[[world_size, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]],
        output: pl.Out[pl.Tensor[[world_size, SP_T_DYN, HC_MULT, D], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[world_size, SP_T_DYN, HC_MULT], pl.FP32]],
        hidden: pl.Out[pl.Tensor[[world_size, T_DYN, D], pl.BF16]],
        attn_out: pl.Out[pl.Tensor[[world_size, SP_T_DYN, D], pl.BF16]],
        num_tokens: pl.Tensor[[world_size, 1], pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Dispatch stacked ranks onto TP windows for the HC prefill program."""
        x_hc.bind_dynamic(1, SP_T_DYN)
        hidden.bind_dynamic(1, T_DYN)
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        input_buf = pld.alloc_window_buffer([capacity, D], dtype=pl.BF16)
        input_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        data_buf = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            input_data = pld.window(input_buf, [capacity, D], dtype=pl.BF16)
            input_signal = pld.window(input_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
            data = pld.window(data_buf, [capacity, D], dtype=pl.FP32)
            signal = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
            # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
            wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
            wq_b_scale_r: pl.Tensor[
                [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
            ] = wq_b_scale[rank]
            wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
            wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
            swa_rank(
                x_hc[rank], incoming_pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank],
                wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
                wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank],
                wo_a[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank],
                window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank],
                output[rank], next_pre_mix[rank], hidden[rank], attn_out[rank], input_data, input_signal, data, signal,
                num_tokens[rank], rank, attention_epoch,
                device=rank,
            )

    return swa_group


def build_hc_specs(args):
    """Stacked TP/DP specs with HC streams in and HC streams out."""
    from golden import ScalarSpec, TensorSpec
    from models.deepseek_v4_1_flash.decode_attn_swa import make_inputs, make_packed_inputs

    world_size = TP_SIZE * args.dp
    packed = True
    if packed:
        lengths = [args.tokens // args.requests + (r < args.tokens % args.requests) for r in range(args.requests)]
        if args.requests > 1 and lengths[-1] > 16:
            lengths[0] += 7
            lengths[-1] -= 7
        prefixes = ([127, 128, 511, 1048000] * 8)[:args.requests] if args.case == "prefix" else [0] * args.requests
        pages = 1 + sum((p + n + 128) // 128 - max(0, p - 127) // 128 + 1 for p, n in zip(prefixes, lengths))
    else:
        pages = args.tokens + 1
    local_tokens = (args.tokens + TP_SIZE - 1) // TP_SIZE
    shapes = (
        [local_tokens, HC_MULT, D], [local_tokens, HC_MULT], [MIX_HC, HC_DIM], [3], [MIX_HC], [D],
        [D, Q_LORA], [D // 32, Q_LORA], [Q_LORA],
        [Q_LORA, LOCAL_H * HEAD_DIM], [Q_LORA // 32, LOCAL_H * HEAD_DIM],
        [D, HEAD_DIM], [D // 32, HEAD_DIM], [HEAD_DIM], [LOCAL_H],
        [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [LOCAL_O_WIDTH, D], [LOCAL_O_WIDTH // 32, D],
        [args.tokens, ROPE_DIM // 2], [args.tokens, ROPE_DIM // 2], [args.tokens], [args.tokens, 128],
        [pages, 128, 1, HEAD_DIM], [pages, 128, 1, HEAD_DIM // 32],
    )
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    dtypes = (torch.float32, torch.float32, torch.float32, torch.float32, torch.float32, bf,
              fp, mx, bf, fp, mx, fp, mx, bf, torch.float32, bf, fp, mx,
              torch.float32, torch.float32, torch.int64, torch.int32, fp, mx)
    values = {}

    def initialize(name):
        """Materialize one stacked HC input, sharing DP-group replicas."""
        if not values:
            ranks = []
            for rank in range(world_size):
                seed = args.seed + rank
                if packed:
                    value = make_packed_inputs(args.tokens, args.requests, seed, args.case)
                else:
                    value = make_inputs(args.tokens, seed)
                    if args.case == "masked":
                        value["window_slots"].fill_(-1)
                        value["window_indices"].fill_(-1)
                    elif args.case == "zero":
                        value["x"].zero_()
                        value["window_cache"].view(torch.uint8).zero_()
                    elif args.case == "sink":
                        value["attn_sink"].fill_(1000)
                ranks.append(make_hc_inputs(value, seed, args.fixture))
            replicated = (
                "x_hc", "incoming_pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
                "wq_a", "wq_a_scale", "q_norm_weight", "wkv", "wkv_scale",
                "kv_norm_weight", "rope_cos", "rope_sin", "window_slots", "window_indices",
                "window_cache", "window_cache_scale",
            )
            for rank, value in enumerate(ranks):
                for key in replicated:
                    value[key] = ranks[rank // TP_SIZE * TP_SIZE][key]
            counts = getattr(args, "dp_tokens", None) or [args.tokens] * args.dp
            for rank, value in enumerate(ranks):
                active = counts[rank // TP_SIZE]
                start = rank % TP_SIZE * local_tokens
                shard = torch.full((local_tokens, HC_MULT, D), 17.0)
                valid = max(0, min(local_tokens, active - start))
                shard[:valid].copy_(value["x_hc"][start:start + valid])
                value["x_hc"] = shard
                pre_shard = torch.zeros(local_tokens, HC_MULT)
                pre_shard[:valid].copy_(value["incoming_pre_mix"][start:start + valid])
                value["incoming_pre_mix"] = pre_shard
                value["window_slots"] = value["window_slots"].clone()
                value["window_slots"][active:] = -1
                value["window_indices"] = value["window_indices"].clone()
                value["window_indices"][active:] = -1
            for key in HC_INPUT_NAMES:
                dtype = ranks[0][key].dtype
                shards = [r[key].view(torch.uint8) if dtype in (fp, mx) else r[key] for r in ranks]
                values[key] = torch.stack(shards).view(dtype)
        return values[name]

    counts = getattr(args, "dp_tokens", None) or [args.tokens] * args.dp
    rank_counts = torch.tensor(counts, dtype=torch.int32).repeat_interleave(TP_SIZE).reshape(world_size, 1)
    specs = [TensorSpec(name, [world_size, *shape], dtype, init_value=lambda n=name: initialize(n), resident="stacked")
             for name, shape, dtype in zip(HC_INPUT_NAMES, shapes, dtypes)]
    specs += [TensorSpec("output", [world_size, local_tokens, HC_MULT, D], torch.float32, resident="stacked"),
              TensorSpec("next_pre_mix", [world_size, local_tokens, HC_MULT], torch.float32, resident="stacked"),
              TensorSpec("hidden", [world_size, args.tokens, D], bf, resident="stacked"),
              TensorSpec("attn_out", [world_size, local_tokens, D], bf, resident="stacked"),
              TensorSpec("num_tokens", [world_size, 1], torch.int32, init_value=lambda: rank_counts, resident="stacked"),
              ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True,
                         benchmark_step=args.epochs if args.bench else None)]
    return specs


def reference_attention(tensors, hidden, base):
    """One TP group's SWA on ``hidden``: the FP32 TP sum in BF16 and each rank's (cache, scale)."""
    from models.deepseek_v4_1_flash.decode_attn_swa import official_reference

    partials, caches = [], []
    active = int(tensors["num_tokens"][base, 0]) if "num_tokens" in tensors else hidden.shape[0]
    for rank in range(base, base + TP_SIZE):
        inputs = {name: tensors[name][rank] for name in HC_INPUT_NAMES if name not in (
            "x_hc", "incoming_pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
        )}
        inputs["x"] = hidden[:active]
        for name in ("rope_cos", "rope_sin", "window_slots", "window_indices"):
            inputs[name] = inputs[name][:active]
        partial = torch.zeros_like(hidden, dtype=torch.float32)
        cache, scale = inputs["window_cache"].clone(), inputs["window_cache_scale"].clone()
        if active:
            active_partial, cache, scale = official_reference(inputs)
            partial[:active].copy_(active_partial)
        partials.append(partial)
        caches.append((cache, scale))
    return sum(partials).bfloat16(), caches


def token_shards(value, local_tokens):
    """Pad a global token tensor and split it in TP rank order."""
    padded = value.new_zeros(TP_SIZE * local_tokens, *value.shape[1:])
    padded[:value.shape[0]].copy_(value)
    return padded.reshape(TP_SIZE, local_tokens, *value.shape[1:])


def golden_prefill_swa_case(tensors):
    """Reconstruct full residuals, run replicated attention, and split the reference."""
    world_size, local_tokens = tensors["x_hc"].shape[:2]
    tokens = tensors["hidden"].shape[1]
    for base in range(0, world_size, TP_SIZE):
        group = slice(base, base + TP_SIZE)
        active = int(tensors["num_tokens"][base, 0]) if "num_tokens" in tensors else tokens
        x_hc = tensors["x_hc"][group].flatten(0, 1)[:tokens].clone()
        x_hc[active:] = 0
        pre_mix, post_mix, residual_mix = golden_mhc_mixes(
            x_hc, tensors["hc_attn_fn"][base],
            tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base],
        )
        incoming = tensors["incoming_pre_mix"][group].flatten(0, 1)[:tokens]
        hidden = rms_norm(golden_mhc_pre(x_hc, incoming), tensors["attn_norm_weight"][base])
        pre_mix[active:] = 0
        tensors["next_pre_mix"][group].copy_(token_shards(pre_mix, local_tokens))
        reduced, caches = reference_attention(tensors, hidden, base)
        for rank, (cache, scale) in enumerate(caches, base):
            tensors["window_cache"][rank].copy_(cache)
            tensors["window_cache_scale"][rank].copy_(scale)
        output = golden_mhc_post(reduced, x_hc, post_mix, residual_mix)
        output[active:] = 0
        tensors["hidden"][group].copy_(hidden.unsqueeze(0).expand(TP_SIZE, -1, -1))
        tensors["attn_out"][group].copy_(token_shards(reduced, local_tokens))
        tensors["output"][group].copy_(token_shards(output, local_tokens))


# The device collapses the HC streams in another FP32 order than torch and flips isolated BF16
# ULPs of ``hidden``; the MXFP8 attention can turn one such ULP into a percent-level move of a
# token row. So every stage is held to its own budget on its own device input: ``hidden``
# against the golden, the attention re-run on the device's ``hidden`` (teacher forcing), and
# hc_post replayed on the device's ``attn_out``. A token row moving by one BF16 ULP everywhere
# is a real error that ratio_allclose's outlier allowance would still admit, hence ROW_BUDGET.
ROW_BUDGET = 2.0**-8
OUTPUT_ATOL = 1e-2
OUTPUT_RTOL = 1e-2
OUTPUT_MAX_ERROR_RATIO = 0.01


def report(name, actual, expected):
    """Print one stage's global and worst token-row rel L2; returns both."""
    diff = actual.double() - expected.double()
    reference = expected.double()
    rel_l2 = (diff.norm() / reference.norm().clamp_min(1e-12)).item()
    rows = diff.flatten(1).norm(dim=-1) / reference.flatten(1).norm(dim=-1).clamp_min(1e-12)
    print(f"[PRECISION] {name} rel_l2={rel_l2:.8g} max_row_rel_l2={rows.max().item():.8g}")
    return rel_l2, rows.max().item()


def make_staged_compare(
    *, output_atol=OUTPUT_ATOL, output_rtol=OUTPUT_RTOL, output_max_error_ratio=OUTPUT_MAX_ERROR_RATIO,
):
    """Stage-wise comparators: mHC pre vs golden, SWA teacher-forced, hc_post replayed."""
    from golden import ratio_allclose
    from models.deepseek_v4_1_flash.decode_attn_swa import compare_distributed_cache, compare_output, compare_scales

    bf16_close = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    forced = {}

    def teacher_forced(inputs, actual_outputs, expected_outputs):
        """The SWA reference on the device's own ``hidden``, computed once per validation."""
        if not forced:
            # The golden caches already hold this call's rows, which the reference rewrites from
            # the device's hidden before reading them; every other row is the initial cache.
            tensors = {**inputs, "window_cache": expected_outputs["window_cache"],
                       "window_cache_scale": expected_outputs["window_cache_scale"]}
            hidden = actual_outputs["hidden"]
            for name in ("attn_out", "window_cache", "window_cache_scale"):
                forced[name] = torch.empty_like(actual_outputs[name])
            for base in range(0, hidden.shape[0], TP_SIZE):
                reduced, caches = reference_attention(tensors, hidden[base], base)
                forced["attn_out"][base:base + TP_SIZE].copy_(token_shards(reduced, inputs["x_hc"].shape[1]))
                for rank, (cache, scale) in enumerate(caches, base):
                    forced["window_cache"][rank].copy_(cache)
                    forced["window_cache_scale"][rank].copy_(scale)
        return forced

    def staged(name, check):
        """Compare one attention stage against its teacher-forced reference."""
        def compare(actual, expected, *, inputs, actual_outputs, expected_outputs, **kwargs):
            """Supply device-input reference outputs to the stage comparator."""
            reference = teacher_forced(inputs, actual_outputs, expected_outputs)
            return check(actual, reference[name], inputs=inputs, actual_outputs=actual_outputs,
                         expected_outputs=reference, **kwargs)
        return compare

    def replicated(actual, base):
        """Check identical gathered inputs within one TP group."""
        return all(torch.equal(actual[base], actual[rank]) for rank in range(base + 1, base + TP_SIZE))

    def compare_hidden(actual, expected, **kwargs):
        """Check mHC-pre precision and gathered-input equality across TP ranks."""
        passed = True
        for base in range(0, actual.shape[0], TP_SIZE):
            _, worst_row = report("hidden", actual[base], expected[base])
            passed &= bf16_close(actual[base], expected[base], **kwargs)[0] and worst_row <= ROW_BUDGET
            passed &= replicated(actual, base)
        return passed, f"hc_pre + input norm budget, every token row <= {ROW_BUDGET:.3g} rel L2, TP replicas identical"

    def compare_attn_out(actual, expected, **kwargs):
        """Report end-to-end errors and check attention on its device input."""
        for rank in range(actual.shape[0]):
            report("attn_out(end-to-end, not gated)", actual[rank], expected[rank])
        def check_shards(actual, reference, **unused):
            """Apply the attention precision budget to every token shard."""
            passed = all(compare_output(actual[rank], reference[rank])[0] for rank in range(actual.shape[0]))
            return passed, "Every token shard must pass attention precision"
        return staged("attn_out", check_shards)(actual, expected, **kwargs)

    def compare_hc_output(actual, expected, *, inputs, actual_outputs, **kwargs):
        """Check local mHC-post replay, inactive rows, and end-to-end accuracy."""
        passed = True
        for base in range(actual.shape[0]):
            x_hc = inputs["x_hc"][base]
            _, post_mix, residual_mix = golden_mhc_mixes(
                x_hc, inputs["hc_attn_fn"][base], inputs["hc_attn_scale"][base], inputs["hc_attn_base"][base],
            )
            replay = golden_mhc_post(actual_outputs["attn_out"][base], x_hc, post_mix, residual_mix)
            local_active = x_hc.shape[0]
            if "num_tokens" in inputs:
                active = int(inputs["num_tokens"][base, 0])
                local_active = max(0, min(x_hc.shape[0], active - base % TP_SIZE * x_hc.shape[0]))
                replay[local_active:] = 0
            _, worst_row = report("output(hc_post replay)", actual[base], replay)
            close, _ = bf16_close(actual[base], replay, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
            passed &= close and worst_row <= ROW_BUDGET
            report("output(end-to-end, diagnostic)", actual[base], expected[base])
            budget = ratio_allclose(
                atol=output_atol, rtol=output_rtol, max_error_ratio=output_max_error_ratio,
                valid_rows=local_active, zero_tail=True,
            )
            within_budget, detail = budget(
                actual[base], expected[base], inputs=inputs, actual_outputs=actual_outputs, **kwargs,
            )
            actual_valid = actual[base, :local_active].float()
            expected_valid = expected[base, :local_active].float()
            tolerance = output_atol + output_rtol * expected_valid.abs()
            bad = int(((actual_valid - expected_valid).abs() > tolerance).sum())
            count = actual_valid.numel()
            print(f"[PRECISION] output rank={base} outliers={bad}/{count} "
                  f"({bad / max(1, count):.6%}) allowed={output_max_error_ratio:.6%} "
                  f"atol={output_atol:g} rtol={output_rtol:g}")
            if not within_budget:
                print(detail)
            passed &= within_budget
        return passed, "hc_post replay budget and per-shard end-to-end tolerance/outlier budget; zero padding"

    def compare_next_pre_mix(actual, expected, *, inputs, **kwargs):
        """Check outgoing staggered coefficients and exact inactive rows."""
        passed = True
        for rank in range(actual.shape[0]):
            rows = actual.shape[1]
            active = int(inputs["num_tokens"][rank, 0])
            valid = max(0, min(rows, active - rank % TP_SIZE * rows))
            compare = ratio_allclose(atol=2.5e-5, rtol=5e-3, valid_rows=valid, zero_tail=True)
            valid_result, detail = compare(actual[rank], expected[rank], inputs=inputs, **kwargs)
            if not valid_result:
                print(detail)
            passed &= valid_result
        return passed, "next pre-mix coefficients within budget; zero inactive rows"

    return {
        "next_pre_mix": compare_next_pre_mix,
        "hidden": compare_hidden,
        "attn_out": compare_attn_out,
        "window_cache": staged("window_cache", compare_distributed_cache),
        "window_cache_scale": staged("window_cache_scale", compare_scales),
        "output": compare_hc_output,
    }


def run_prefill_swa(argv=None):
    """Run A5 validation for packed prefill SWA wired through mHC."""
    import argparse
    import os

    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill SWA + mHC: A5 precision and timing")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--ep", type=int, default=8, choices=[2, 4, 8])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", "--batch", type=int, default=128)
    parser.add_argument("--dp-tokens", help="comma-separated active counts per DP group, each in [0, tokens]")
    parser.add_argument("--requests", type=int, help="packed requests; default min(tokens, 4)")
    parser.add_argument("--case", default="mixed", choices=["mixed", "prefix", "shuffle", "masked", "zero", "sink"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fixture", choices=["checkpoint", "random"], default="checkpoint",
                        help="checkpoint mHC scale/base with synthetic weights, or random stress parameters")
    parser.add_argument("--epochs", type=int, default=1, help="operator calls per dispatch; timing includes all epochs")
    parser.add_argument("--output-atol", type=float, default=OUTPUT_ATOL)
    parser.add_argument("--output-rtol", type=float, default=OUTPUT_RTOL)
    parser.add_argument("--output-max-error-ratio", type=float, default=OUTPUT_MAX_ERROR_RATIO,
                        help="allowed fraction of active output elements outside tolerance, in [0, 1]")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.output_atol) or args.output_atol < 0:
        parser.error("output-atol must be finite and nonnegative")
    if not math.isfinite(args.output_rtol) or args.output_rtol < 0:
        parser.error("output-rtol must be finite and nonnegative")
    if not math.isfinite(args.output_max_error_ratio) or not 0 <= args.output_max_error_ratio <= 1:
        parser.error("output-max-error-ratio must be a finite fraction in [0, 1]")
    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    args.requests = args.requests if args.requests is not None else min(args.tokens, 4)
    if args.epochs < 1:
        parser.error("epochs must be at least 1")
    try:
        devices = list(range(TP_SIZE * args.dp)) if args.device is None else [int(d) for d in args.device.split(",")]
    except ValueError:
        parser.error("device IDs must be comma-separated integers")
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("device IDs must be distinct nonnegative integers, with count TP * DP")
    if not 1 <= args.tokens <= PREFILL_MAX_TOKENS or not 1 <= args.requests <= min(32, args.tokens):
        parser.error(f"tokens must be in [1, {PREFILL_MAX_TOKENS}]; requests in [1, min(32, tokens)]")
    if args.dp_tokens is not None:
        try:
            args.dp_tokens = tuple(int(n) for n in args.dp_tokens.split(","))
        except ValueError:
            parser.error("dp-tokens must contain comma-separated integer counts")
        if len(args.dp_tokens) != args.dp or any(n < 0 or n > args.tokens for n in args.dp_tokens):
            parser.error("dp-tokens must contain one count per DP group, each in [0, tokens]")
    torch.set_num_threads(8)
    print(f"[SWA+HC] tokens={args.tokens} requests={args.requests} TP={TP_SIZE} DP={args.dp} "
          f"case={args.case} fixture={args.fixture} seed={args.seed} devices={devices}")
    result = run(
        fn=make_hc_program(PREFILL_MAX_TOKENS, len(devices), args.epochs),
        specs=build_hc_specs(args),
        golden_fn=golden_prefill_swa_case,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        config=dict(
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            ring_heap=PREFILL_ATTN_RING_HEAP,
        ),
        compare_fn=make_staged_compare(
            output_atol=args.output_atol, output_rtol=args.output_rtol,
            output_max_error_ratio=args.output_max_error_ratio,
        ),
    )
    print(f"[SWA+HC] work_dir={result.work_dir}")
    if args.compile_only and result.passed:
        print("[SWA+HC] Compilation passed; device accuracy was NOT validated.")
    return result


__all__ = [
    "golden_prefill_swa",
    "prefill_swa",
    "run_prefill_swa",
]


def validate(argv=None):
    """Validate packed prefill SWA wired through mHC on A5."""
    return run_prefill_swa(argv=argv)


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,dp", [(1, 1), (2, 2), (4, 1)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
