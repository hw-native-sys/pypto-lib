# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-sharded packed-prefill SWA with local mHC and head-TP attention."""


import math

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from sparse_attention import golden_swa_attention
from config import MAX_BATCH_PER_DP, PREFILL_MAX_TOKENS, WINDOW_CACHE_GROUP, FLASH as M, ORI_BLOCKS_DYN, T_DYN, TP_SIZE

from hc_mixes import golden_mhc_mixes
from hc_post import golden_mhc_post
from hc_pre import golden_mhc_pre
from hc_pre import attention_pre
from rmsnorm import golden_rms_norm
from prefill_attn_swa import prefill_attn_swa_partial
from config import (
    SP_T_DYN,
)
from prefill_cp_allgather import prefill_sp_input_allgather
from o_proj import golden_grouped_output, prefill_sp_output_reduce_scatter
from hc_post import prefill_sp_post
from qkv_proj_rope import golden_rope_tail
from quant import golden_mxfp8_linear, golden_mxfp8_quantize, pack_mx_b_scale, quantize_mxfp8_cache
from utils import window_metadata


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4

D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc


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
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
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
    prefill_sp_post(
        attn_out, x_hc, post_mix, residual_mix, output, next_pre_mix, tp_rank, num_tokens,
    )
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
    normalized_hidden = golden_rms_norm(golden_mhc_pre(x_hc, incoming_pre_mix), attn_norm_weight)
    result = golden_swa_attention(
        normalized_hidden, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
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
        window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
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
        window_cache_scale: pl.InOut[pl.Tensor[[world_size, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
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


def golden_reference(tensors):
    """CPU transcription of group-32 FP8 GEMM and block-64 online attention.

    DeepSeek-V4.1-Flash inference/kernel.py and model.py, revision
    dba1be0a40aa45a94ad051997016db3960a90277; not execution of the CUDA kernels.
    """
    t = tensors
    head_dim = t["window_cache"].shape[-1]
    local_heads = t["attn_sink"].numel()
    groups = t["wo_a"].shape[0]
    group_in = t["wo_a"].shape[-1]
    qr = golden_rms_norm(golden_mxfp8_linear(t["x"], t["wq_a"], t["wq_a_scale"]), t["q_norm_weight"])
    q = golden_mxfp8_linear(qr, t["wq_b"], t["wq_b_scale"]).unflatten(-1, (local_heads, head_dim))
    q = golden_rope_tail(q, t["rope_cos"], t["rope_sin"])
    kv = golden_rms_norm(golden_mxfp8_linear(t["x"], t["wkv"], t["wkv_scale"]), t["kv_norm_weight"])
    kv = golden_rope_tail(kv, t["rope_cos"], t["rope_sin"])
    payload, scales = quantize_mxfp8_cache(kv)
    cache = t["window_cache"].clone()
    cache_scale = t["window_cache_scale"].clone()
    slots = t["window_slots"].long()
    valid_slots = slots >= 0
    cache.view(torch.uint8).reshape(-1, head_dim)[slots[valid_slots]] = payload.view(torch.uint8)[valid_slots]
    cache_scale.reshape(-1, head_dim // 32)[slots[valid_slots]] = scales[valid_slots]
    values = cache.float() * cache_scale.float().repeat_interleave(32, -1)
    idx = t["window_indices"].long()
    selected = values.reshape(-1, head_dim)[idx.clamp_min(0)].to(torch.bfloat16)
    selected = selected.masked_fill((idx < 0)[..., None], 0)
    maximum = torch.full(q.shape[:2], -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q, dtype=torch.float32)
    for start in range(0, idx.shape[-1], 64):
        keys = selected[:, start:start + 64].float()
        logits = torch.einsum("thd,tkd->thk", q.float(), keys) * head_dim ** -0.5
        valid = idx[:, start:start + 64] >= 0
        logits = logits.masked_fill(~valid[:, None], -torch.inf)
        new_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = (maximum - new_maximum).exp()
        probabilities = (logits - new_maximum[..., None]).exp()
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None] + torch.einsum(
            "thk,tkd->thd", probabilities.to(torch.bfloat16).float(), keys)
        maximum = new_maximum
    final_max = torch.maximum(maximum, t["attn_sink"][None])
    correction = (maximum - final_max).exp()
    denominator = denominator * correction + (t["attn_sink"][None] - final_max).exp()
    attended = (numerator * (correction / denominator)[..., None]).to(torch.bfloat16)
    attended = golden_rope_tail(attended, t["rope_cos"], t["rope_sin"], inverse=True)
    grouped = attended.reshape(-1, groups, group_in)
    latent = golden_grouped_output(grouped, t["wo_a"])
    output = golden_mxfp8_linear(latent.flatten(1), t["wo_b"], t["wo_b_scale"], fp32=True)
    return output, cache, cache_scale

INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
)

def make_inputs(batch=32, seed=17, position=127):
    """Distinct request pages, ragged visibility, nontrivial scales and rotary phases."""
    gen = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=gen)
    def weight(k, n):
        payload, scales = golden_mxfp8_quantize(rand(n, k) / math.sqrt(k))
        return payload.T.contiguous(), pack_mx_b_scale(scales.T.contiguous()).view(torch.float8_e8m0fnu)
    qa, qas = weight(D, Q_LORA)
    qb, qbs = weight(Q_LORA, LOCAL_H * HEAD_DIM)
    kv, kvs = weight(D, HEAD_DIM)
    ob, obs = weight(LOCAL_O_WIDTH, D)
    positions = torch.full((batch,), position, dtype=torch.int64)
    if batch > 1:
        positions[:min(batch, 6)] = torch.tensor([0, 1, 63, 127, 128, 1048575])[:min(batch, 6)]
    freq = 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    angles = positions.float()[:, None] * freq
    physical = torch.randperm(batch + 1, generator=gen)[:batch]
    slots = physical.long() * 128 + positions.remainder(128)
    indices = torch.full((batch, 128), -1, dtype=torch.int32)
    for row, pos in enumerate(positions.tolist()):
        length = min(pos + 1, 128)
        logical = torch.arange(pos - length + 1, pos + 1)
        indices[row, :length] = (physical[row] * 128 + logical.remainder(128)).to(torch.int32)
    cache, cache_scale = quantize_mxfp8_cache(rand(batch + 1, 128, 1, HEAD_DIM).to(torch.bfloat16))
    values = (rand(batch, D).to(torch.bfloat16), qa, qas,
              (rand(Q_LORA) * 0.1 + 1).to(torch.bfloat16), qb, qbs, kv, kvs,
              (rand(HEAD_DIM) * 0.1 + 1).to(torch.bfloat16), rand(LOCAL_H) * 2,
              (rand(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16),
              ob, obs, angles.cos(), angles.sin(), slots, indices, cache,
              cache_scale)
    return dict(zip(INPUT_NAMES, values))

def make_packed_inputs(tokens=257, requests=4, seed=17, case="mixed", start_pos=None):
    if not 1 <= requests <= min(32, tokens):
        raise ValueError("requests must be in [1, min(32, tokens)]")
    values = make_inputs(1, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    lengths = torch.full((requests,), tokens // requests, dtype=torch.int64)
    lengths[:tokens % requests] += 1
    if requests > 1 and lengths[-1] > 16:
        lengths[0] += 7
        lengths[-1] -= 7
    prefixes = torch.zeros(requests, dtype=torch.int64)
    if case == "prefix":
        prefixes = torch.tensor([127, 128, 511, 1048000] * 8)[:requests]
    if start_pos is not None:
        prefixes = torch.tensor(start_pos, dtype=torch.int64)
    ids = torch.repeat_interleave(torch.arange(requests), lengths)
    positions = torch.cat([torch.arange(int(p), int(p + n)) for p, n in zip(prefixes, lengths)])
    logical_capacity = int((prefixes + lengths + 129).max()) // 128 + 1
    table = torch.full((requests, logical_capacity), -1, dtype=torch.int32)
    mappings = []
    for request, (prefix, length) in enumerate(zip(prefixes.tolist(), lengths.tolist())):
        start = max(0, prefix - 127) // 128
        end = (prefix + length + 128) // 128
        mappings.extend((request, block) for block in range(start, end + 1))
    pages = torch.randperm(len(mappings) + 1, generator=gen)[:len(mappings)]
    for (request, block), page in zip(mappings, pages):
        table[request, block] = page
    slots, indices, lens = window_metadata(positions, ids, table)
    assert slots.unique().numel() == tokens
    # Verify physical metadata independently, without relying on the helper's mask.
    for row, (request, position) in enumerate(zip(ids.tolist(), positions.tolist())):
        history = list(range(max(0, position - 127), position + 1))
        expected = [int(table[request, p // 128]) * 128 + p % 128 for p in history]
        assert indices[row, :len(expected)].tolist() == expected
        assert (indices[row, len(expected):] == -1).all()
    angles = positions.float()[:, None] * (10000.0 ** (-torch.arange(0, 64, 2).float() / 64))
    cache, scales = quantize_mxfp8_cache(torch.randn(len(mappings) + 1, 128, 1, 512, generator=gen).bfloat16())
    values.update(x=torch.randn(tokens, 5120, generator=gen).bfloat16(),
                  rope_cos=angles.cos(), rope_sin=angles.sin(),
                  window_slots=slots, window_indices=indices,
                  window_cache=cache, window_cache_scale=scales)
    if case == "shuffle":
        order = torch.randperm(tokens, generator=gen)
        for name in ("x", "rope_cos", "rope_sin", "window_slots", "window_indices"):
            values[name] = values[name][order].contiguous()
    if case == "masked":
        values["window_slots"].fill_(-1)
        values["window_indices"].fill_(-1)
    if case == "zero":
        values["x"].zero_()
        values["window_cache"].view(torch.uint8).zero_()
    if case == "sink":
        values["attn_sink"].fill_(1000)
    print(f"[FIXTURE] tokens={tokens} requests={requests} lengths={lengths.tolist()} prefixes={prefixes.tolist()}")
    return values

def window_cache_compare(world_size, tokens):
    """Compare the mapped window rows and require every other row to stay unchanged."""
    from golden import mapped_pool_ratio_allclose

    options = dict(
        mapping_shape=(world_size, tokens),
        block_size=128,
        leading_rank_axis=True,
        atol=0.0,
        rtol=0.0,
        max_error_ratio=0.0,
    )
    return {
        "window_cache": mapped_pool_ratio_allclose("window_slots", pool_name="window cache", **options),
        "window_cache_scale": mapped_pool_ratio_allclose(
            "window_slots", pool_name="window cache scale", **options
        ),
    }


def build_hc_specs(args):
    """Stacked TP/DP specs with HC streams in and HC streams out."""
    from golden import ScalarSpec, TensorSpec

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
              torch.float32, torch.float32, torch.int64, torch.int32, fp, bf)
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
            active_partial, cache, scale = golden_reference(inputs)
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
        hidden = golden_rms_norm(golden_mhc_pre(x_hc, incoming), tensors["attn_norm_weight"][base])
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


def run_prefill_swa(argv=None):
    """Run A5 validation for packed prefill SWA wired through mHC."""
    import argparse
    import os

    from golden import ratio_allclose, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill SWA + mHC: A5 precision and timing")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
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
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args(argv)
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
    if not 1 <= args.tokens <= PREFILL_MAX_TOKENS or not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"tokens must be in [1, {PREFILL_MAX_TOKENS}]; requests in [1, min({MAX_BATCH_PER_DP}, tokens)]")
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
        compare_fn={
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            # Zero, masked and sink outputs are exactly zero, so any residue is a failure.
            "attn_out": (
                ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
                if args.case in ("masked", "zero", "sink") else ratio_allclose(atol=1e-2, rtol=1.0 / 128)
            ),
            "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            **window_cache_compare(len(devices), args.tokens),
        },
    )
    print(f"[SWA+HC] work_dir={result.work_dir}")
    if args.compile_only and result.passed:
        print("[SWA+HC] Compilation passed; device accuracy was NOT validated.")
    return result


def validate(argv=None):
    """Validate packed prefill SWA wired through mHC on A5."""
    return run_prefill_swa(argv=argv)


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
