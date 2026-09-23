# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode SWA leaf attention for encoder layers 0 and 1."""

import argparse
import math
import os
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
import torch

from golden import ScalarSpec, TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.metadata import window_metadata
from models.deepseek_v4_1_flash.o_proj import o_proj
from models.deepseek_v4_1_flash.qkv_proj_rope import qkv_proj_rope
from models.deepseek_v4_1_flash.quantization import decode_e8m0, pack_mx_b_scale, unpack_mx_b_scale
from models.deepseek_v4_1_flash.attention_tp import (
    OUTPUT_T_DYN,
    decode_tp_output_all_reduce,
    decode_tp_output_reduce_scatter,
)
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult, golden_swa_attention
from models.deepseek_v4_1_flash.config import (
    D,
    DECODE_MAX_TOKENS,
    HEAD_DIM,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCKS_DYN,
    Q_LORA,
    ROPE_DIM,
    T_DYN,
    TP_SIZE,
    WINDOW_CACHE_GROUP,
)


SOFTMAX_SCALE = HEAD_DIM ** -0.5

# Tiling.
M_TILE = 16


@pl.jit.inline
def publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(num_tokens, name_hint="swa_cache_publish", deps=[cache_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(kv, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
            value = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM // 16, 32])
            amax = pl.maximum(pl.row_max(pl.abs(value)), 1e-4)
            raw = pl.mul(amax, 1.0 / 448.0)
            bits = pl.reinterpret_view(raw, pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            payload = pl.cast(pl.row_expand_div(value, scale), pl.FP8E4M3FN, mode="rint")
            flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
            signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
            codes = pl.cast(signed_exponent, pl.INT8)
            encoded = pl.reinterpret_view(pl.reinterpret_view(codes, pl.UINT8), pl.FP8E8M0)
            scale_flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(encoded, [1, HEAD_DIM // 16]),
                                                         1, HEAD_DIM // 32)
    return cache, scales


@pl.jit.inline
def gather_window(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    selected: pl.Tensor[[T_DYN, 128, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    tokens = pl.tensor.dim(selected, 0)
    selected_rows = tokens * 128
    gathered = pl.reshape(selected, [selected_rows, HEAD_DIM])
    with pl.spmd(num_tokens * 8, name_hint="swa_cache_gather") as gather_tid:
        block = pl.tile.get_block_idx()
        t = block // 8
        for i in pl.range(block % 8 * 16, block % 8 * 16 + 16):
            row_i32 = pl.read(indices, [t, i])
            dst = t * 128 + i
            if row_i32 >= 0:
                row = pl.cast(row_i32, pl.INDEX)
                value = pl.reshape(pl.cast(flat[row:row + 1, :], pl.FP32), [HEAD_DIM // 32, 32])
                scale_row = pl.slice(scale_flat, [1, 32], [row, 0], valid_shape=[1, HEAD_DIM // 32])
                raw_codes = pl.reinterpret_view(scale_row, pl.UINT8)
                signed_codes = pl.cast(pl.reinterpret_view(raw_codes, pl.INT8), pl.INT32)
                codes = pl.ands(signed_codes, 255)
                scale_values = pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32)
                scale = pl.reshape(scale_values[:, :HEAD_DIM // 32], [HEAD_DIM // 32, 1])
                decoded = pl.cast(pl.row_expand_mul(value, scale), pl.BF16, mode="rint")
                gathered[dst:dst + 1, :] = pl.reshape(decoded, [1, HEAD_DIM])
            else:
                gathered[dst:dst + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    return gather_tid


@pl.jit.inline
def attend_window(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, 128, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(query, 0)
    head_rows = tokens * LOCAL_H
    cache_rows = tokens * 128
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    for block in pl.spmd(num_tokens * (LOCAL_H // M_TILE), name_hint="swa_online_attention"):
        t = block // (LOCAL_H // M_TILE)
        h = block % (LOCAL_H // M_TILE) * M_TILE
        q0 = t * LOCAL_H + h
        maximum = pl.full([1, M_TILE], dtype=pl.FP32, value=-1e30)
        denominator = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
        numerator = pl.full([M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for part in pl.range(2):
            k0 = t * 128 + part * 64
            q = qflat[q0:q0 + M_TILE, :]
            kv = kflat[k0:k0 + 64, :]
            scores = pl.matmul(q, kv, b_trans=True)
            scores = pl.mul(scores, SOFTMAX_SCALE)
            idx = pl.cast(indices[t:t + 1, part * 64:part * 64 + 64], pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            scores = pl.col_expand_add(scores, bias)
            next_max = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, M_TILE]))
            correction = pl.exp(pl.sub(maximum, next_max))
            probabilities = pl.col_expand_mul(
                pl.exp(pl.row_expand_sub(scores, pl.reshape(next_max, [M_TILE, 1]))), valid)
            denominator = pl.add(pl.mul(denominator, correction),
                                 pl.reshape(pl.row_sum(probabilities), [1, M_TILE]))
            weights = pl.cast(probabilities, pl.BF16, mode="rint")
            weighted = pl.matmul(weights, kv)
            numerator = pl.add(pl.row_expand_mul(numerator, pl.reshape(correction, [M_TILE, 1])), weighted)
            maximum = next_max
        sinks = pl.reshape(sink[h:h + M_TILE], [1, M_TILE])
        final_max = pl.maximum(maximum, sinks)
        correction = pl.exp(pl.sub(maximum, final_max))
        denominator = pl.add(pl.mul(denominator, correction), pl.exp(pl.sub(sinks, final_max)))
        result = pl.row_expand_mul(numerator, pl.reshape(pl.div(correction, denominator), [M_TILE, 1]))
        oflat[q0:q0 + M_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
    return output


if TP_SIZE not in (1, 2, 4):
    raise ValueError("Decode SWA currently supports TP1, TP2, and TP4; TP8 requires head-tile padding")



def golden_decode_attn_swa(
    x: torch.Tensor,
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
) -> AttentionGoldenResult:
    return golden_swa_attention(
        x,
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
    )


@pl.jit.inline(auto_scope=False)
def decode_swa_partial(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
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
    output: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Write the FP32 local output; return the cache-read completion task for caller WAR ordering."""
    tokens = pl.tensor.dim(x, 0)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qkv_proj_rope(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, qr, q, kv, num_tokens,
    )
    publish_window(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
    selected = pl.create_tensor([tokens, 128, HEAD_DIM], dtype=pl.BF16)
    cache_consumed = gather_window(window_cache, window_cache_scale, window_indices, selected, num_tokens)
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attend_window(q, selected, window_indices, attn_sink, attended, num_tokens)
    o_proj(attended, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, output, num_tokens)
    return cache_consumed


def make_decode_attn_swa(output_reduce=None):
    """Build the decode_attn_swa leaf around one output collective."""
    reducer = output_reduce or decode_tp_output_all_reduce

    @pl.jit.inline(auto_scope=False)
    def decode_attn_swa(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
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
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Write BF16 TP output using zero-initialized windows and consecutive 1-based epochs."""
        # A later epoch must not overwrite cache or transport storage still being read.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_previous_epoch", allow_early_resolve=False) as cache_ready:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                                cmp=pld.WaitCmp.Ge)
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        decode_swa_partial(x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                           wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale,
                           rope_cos, rope_sin, window_slots, window_indices, window_cache,
                           window_cache_scale, partial, num_tokens, cache_ready)
        reducer(partial, output_window, output_arrived, output,
                                    group_base, tp_rank, num_tokens, attention_epoch)
        return output

    return decode_attn_swa


decode_attn_swa = make_decode_attn_swa()
decode_attn_swa_sharded = make_decode_attn_swa(decode_tp_output_reduce_scatter)


__all__ = ["decode_attn_swa", "decode_attn_swa_sharded", "golden_decode_attn_swa"]



def official_quantize(value):
    """Independent translation of official kernel.py act_quant_kernel, group size 32."""
    groups = value.float().unflatten(-1, (-1, 32))
    maximum = groups.abs().amax(-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(maximum * (1.0 / 448.0)))
    scale = torch.pow(2.0, exponent)
    payload = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2)
    return payload, (exponent + 127).to(torch.uint8)


def official_linear(x, weight, packed_scale, fp32=False):
    """Match official FP8 GEMM's group-32 product, A scale, B scale, FP32 sum."""
    payload, codes = official_quantize(x)
    activation, weights = payload.float(), weight.float()
    scale_a = decode_e8m0(codes)
    scale_b = decode_e8m0(unpack_mx_b_scale(packed_scale))
    result = torch.zeros(x.shape[0], weight.shape[1], dtype=torch.float32)
    for group in range(weight.shape[0] // 32):
        start = group * 32
        product = activation[:, start:start + 32] @ weights[start:start + 32]
        scaled = product * scale_a[:, group:group + 1]
        scaled = scaled * scale_b[group:group + 1]
        result = result + scaled
    return result if fp32 else result.bfloat16()


def official_rope(x, cos, sin, inverse=False):
    rope_dim = cos.shape[-1] * 2
    value = x.float().clone()
    c = cos.reshape(x.shape[0], *([1] * (x.ndim - 2)), -1)
    s = sin.reshape_as(c) * (-1 if inverse else 1)
    tail = value[..., -rope_dim:].unflatten(-1, (rope_dim // 2, 2))
    real, imag = tail[..., 0].clone(), tail[..., 1].clone()
    tail[..., 0] = real * c - imag * s
    tail[..., 1] = real * s + imag * c
    return value.to(torch.bfloat16)


def official_reference(tensors):
    """CPU transcription of group-32 FP8 GEMM and block-64 online attention.

    DeepSeek-V4.1-Flash inference/kernel.py and model.py, revision
    dba1be0a40aa45a94ad051997016db3960a90277; not execution of the CUDA kernels.
    """
    t = tensors
    head_dim = t["window_cache"].shape[-1]
    local_heads = t["attn_sink"].numel()
    groups = t["wo_a"].shape[0]
    group_in = t["wo_a"].shape[-1]
    def norm(x, weight):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-20)
                * weight.float()).to(torch.bfloat16)

    qr = norm(official_linear(t["x"], t["wq_a"], t["wq_a_scale"]), t["q_norm_weight"])
    q = official_linear(qr, t["wq_b"], t["wq_b_scale"]).unflatten(-1, (local_heads, head_dim))
    q = official_rope(q, t["rope_cos"], t["rope_sin"])
    kv = norm(official_linear(t["x"], t["wkv"], t["wkv_scale"]), t["kv_norm_weight"])
    kv = official_rope(kv, t["rope_cos"], t["rope_sin"])
    payload, scales = official_quantize(kv)
    cache = t["window_cache"].clone()
    cache_scale = t["window_cache_scale"].clone()
    slots = t["window_slots"].long()
    valid_slots = slots >= 0
    cache.view(torch.uint8).reshape(-1, head_dim)[slots[valid_slots]] = payload.view(torch.uint8)[valid_slots]
    cache_scale.view(torch.uint8).reshape(-1, head_dim // 32)[slots[valid_slots]] = scales[valid_slots]
    values = cache.float() * decode_e8m0(cache_scale).repeat_interleave(32, -1)
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
    attended = official_rope(attended, t["rope_cos"], t["rope_sin"], inverse=True)
    grouped = attended.reshape(-1, groups, group_in)
    latent = torch.einsum("tgd,grd->tgr", grouped.float(), t["wo_a"].float()).to(torch.bfloat16)
    output = official_linear(latent.flatten(1), t["wo_b"], t["wo_b_scale"], fp32=True)
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
        payload, scales = official_quantize(rand(n, k) / math.sqrt(k))
        return payload.T.contiguous(), pack_mx_b_scale(scales.T.contiguous()).view(torch.float8_e8m0fnu)
    qa, qas = weight(C.D, C.Q_LORA)
    qb, qbs = weight(C.Q_LORA, C.LOCAL_H * C.HEAD_DIM)
    kv, kvs = weight(C.D, C.HEAD_DIM)
    ob, obs = weight(C.LOCAL_O_WIDTH, C.D)
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
    cache, cache_scale = official_quantize(rand(batch + 1, 128, 1, C.HEAD_DIM).to(torch.bfloat16))
    values = (rand(batch, C.D).to(torch.bfloat16), qa, qas,
              (rand(C.Q_LORA) * 0.1 + 1).to(torch.bfloat16), qb, qbs, kv, kvs,
              (rand(C.HEAD_DIM) * 0.1 + 1).to(torch.bfloat16), rand(C.LOCAL_H) * 2,
              (rand(C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN) / math.sqrt(C.O_GROUP_IN)).to(torch.bfloat16),
              ob, obs, angles.cos(), angles.sin(), slots, indices, cache,
              cache_scale.view(torch.float8_e8m0fnu))
    return dict(zip(INPUT_NAMES, values))



def make_packed_inputs(tokens=257, requests=4, seed=17, case="mixed"):
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
    cache, scales = official_quantize(torch.randn(len(mappings) + 1, 128, 1, 512, generator=gen).bfloat16())
    values.update(x=torch.randn(tokens, 5120, generator=gen).bfloat16(),
                  rope_cos=angles.cos(), rope_sin=angles.sin(),
                  window_slots=slots, window_indices=indices,
                  window_cache=cache, window_cache_scale=scales.view(torch.float8_e8m0fnu))
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



def compare_output(actual, expected, **kwargs):
    """FP8 budget: global L2 <= 1%, every row <= 2%, cosine >= 0.9999; exact zero rows."""
    a, e = actual.double(), expected.double()
    error = (a - e).norm() / e.norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0).clamp(-1, 1)
    row_error = (a - e).norm(dim=-1) / e.norm(dim=-1).clamp_min(1e-12)
    zero_rows = e.norm(dim=-1) == 0
    zero_exact = bool((a.norm(dim=-1)[zero_rows] == 0).all())
    if bool((e == 0).all()):
        cosine = torch.tensor(1.0 if zero_exact else 0.0)
    max_abs = (a - e).abs().max()
    print(f"[PRECISION] output rel_l2={error.item():.8g} cosine={cosine.item():.8g} "
          f"max_abs={max_abs.item():.8g} max_row_rel_l2={row_error.max().item():.8g} zero_rows_exact={zero_exact}")
    passed = torch.isfinite(a).all() and error <= 0.01 and row_error.max() <= 0.02
    passed = passed and cosine >= 0.9999 and zero_exact
    return bool(passed), f"rel_l2={error.item():.8g}"


def compare_scales(actual, expected, **kwargs):
    equal = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    return equal, "E8M0 scale bytes differ"


def compare_cache(actual, expected, *, actual_outputs, expected_outputs, inputs, **kwargs):
    slots = inputs["window_slots"].long()
    rows = slots[slots >= 0]
    a = actual.view(torch.uint8).reshape(-1, C.HEAD_DIM)
    e = expected.view(torch.uint8).reshape_as(a)
    untouched = torch.ones(a.shape[0], dtype=torch.bool)
    untouched[rows] = False
    if not torch.equal(a[untouched], e[untouched]):
        return False, "Unmapped cache payload bytes were modified"
    af = actual.float().reshape(-1, C.HEAD_DIM)[rows]
    ef = expected.float().reshape(-1, C.HEAD_DIM)[rows]
    actual_scale = decode_e8m0(actual_outputs["window_cache_scale"]).reshape(-1, C.HEAD_DIM // 32)
    expected_scale = decode_e8m0(expected_outputs["window_cache_scale"]).reshape(-1, C.HEAD_DIM // 32)
    af *= actual_scale[rows].repeat_interleave(32, -1)
    ef *= expected_scale[rows].repeat_interleave(32, -1)
    rel = (af - ef).norm() / ef.norm().clamp_min(1e-12)
    mismatch = (a[rows] != e[rows]).float().mean().item() if rows.numel() else 0.0
    print(f"[PRECISION] cache rel_l2={rel.item():.8g} byte_mismatch_fraction={mismatch:.8g} untouched_exact=True")
    return bool(torch.isfinite(af).all() and rel <= 0.01), f"cache rel_l2={rel.item():.8g}"


def make_program(operator, capacity, world_size, epochs):
    """Wrap the production inline operator; epochs advance across persistent dispatches."""
    @pl.jit
    def swa_rank(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
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
        output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        x.bind_dynamic(0, T_DYN)
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        for step in pl.range(epochs):
            operator(
                x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale,
                rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
                output_window, output_arrived, output,
                rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, num_tokens, attention_epoch + step,
            )
        return output, window_cache, window_cache_scale

    @pl.jit.host
    def swa_group(
        x: pl.Tensor[[world_size, T_DYN, D], pl.BF16],
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
        output: pl.Out[pl.Tensor[[world_size, T_DYN, D], pl.BF16]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        x.bind_dynamic(1, T_DYN)
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        data_buf = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
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
                x[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
                wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank],
                wo_a[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank],
                window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank],
                output[rank], data, signal, rank, num_tokens, attention_epoch, device=rank,
            )

    return swa_group


def build_specs(args, mode):
    """Build shape-only specs; replay and compile-only never generate random weights."""
    world_size = TP_SIZE * args.dp
    packed = (mode == "prefill" or args.tokens > C.MAX_BATCH_PER_DP
              or args.requests != args.tokens or args.case in ("prefix", "shuffle"))
    if packed:
        lengths = [args.tokens // args.requests + (r < args.tokens % args.requests) for r in range(args.requests)]
        if args.requests > 1 and lengths[-1] > 16:
            lengths[0] += 7
            lengths[-1] -= 7
        prefixes = ([127, 128, 511, 1048000] * 8)[:args.requests] if args.case == "prefix" else [0] * args.requests
        pages = 1 + sum((p + n + 128) // 128 - max(0, p - 127) // 128 + 1 for p, n in zip(prefixes, lengths))
    else:
        pages = args.tokens + 1
    shapes = (
        [args.tokens, D], [D, Q_LORA], [D // 32, Q_LORA], [Q_LORA],
        [Q_LORA, LOCAL_H * HEAD_DIM], [Q_LORA // 32, LOCAL_H * HEAD_DIM],
        [D, HEAD_DIM], [D // 32, HEAD_DIM], [HEAD_DIM], [LOCAL_H],
        [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [LOCAL_O_WIDTH, D], [LOCAL_O_WIDTH // 32, D],
        [args.tokens, ROPE_DIM // 2], [args.tokens, ROPE_DIM // 2], [args.tokens], [args.tokens, 128],
        [pages, 128, 1, HEAD_DIM], [pages, 128, 1, HEAD_DIM // 32],
    )
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    dtypes = (bf, fp, mx, bf, fp, mx, fp, mx, bf, torch.float32, bf, fp, mx,
              torch.float32, torch.float32, torch.int64, torch.int32, fp, mx)
    values = {}

    def initialize(name):
        if not values:
            ranks = []
            for rank in range(world_size):
                seed = args.seed + rank
                if packed:
                    value = make_packed_inputs(args.tokens, args.requests, seed, args.case)
                else:
                    value = make_inputs(args.tokens, seed)
                if not packed:
                    if args.case == "masked":
                        value["window_slots"].fill_(-1)
                        value["window_indices"].fill_(-1)
                    elif args.case == "zero":
                        value["x"].zero_()
                        value["window_cache"].view(torch.uint8).zero_()
                    elif args.case == "sink":
                        value["attn_sink"].fill_(1000)
                ranks.append(value)
            replicated = ("x", "wq_a", "wq_a_scale", "q_norm_weight", "wkv", "wkv_scale",
                          "kv_norm_weight", "rope_cos", "rope_sin", "window_slots", "window_indices",
                          "window_cache", "window_cache_scale")
            for rank, value in enumerate(ranks):
                for key in replicated:
                    value[key] = ranks[rank // TP_SIZE * TP_SIZE][key]
            for key in INPUT_NAMES:
                dtype = ranks[0][key].dtype
                shards = [r[key].view(torch.uint8) if dtype in (fp, mx) else r[key] for r in ranks]
                values[key] = torch.stack(shards).view(dtype)
        return values[name]

    specs = [TensorSpec(name, [world_size, *shape], dtype, init_value=lambda n=name: initialize(n), resident="stacked")
             for name, shape, dtype in zip(INPUT_NAMES, shapes, dtypes)]
    specs += [TensorSpec("output", [world_size, args.tokens, D], bf, resident="stacked"),
              ScalarSpec("num_tokens", torch.int32, args.tokens),
              ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True,
                         benchmark_step=args.epochs if args.bench else None)]
    return specs


def golden_swa(tensors, sharded=False):
    """Reference each TP shard independently, then perform the FP32 TP reduction.

    With ``sharded`` the summed rows are handed out as the contiguous token slabs
    the kernel ReduceScatter produces, one per rank.
    """
    world_size = tensors["x"].shape[0]
    active = int(tensors["num_tokens"])
    for base in range(0, world_size, TP_SIZE):
        partials = []
        for rank in range(base, base + TP_SIZE):
            inputs = {name: tensors[name][rank] for name in INPUT_NAMES}
            partial, cache, scale = official_reference(inputs)
            partials.append(partial)
            tensors["window_cache"][rank].copy_(cache)
            tensors["window_cache_scale"][rank].copy_(scale)
        reduced = sum(partials).bfloat16()
        if sharded:
            # Ownership follows the fixed physical slab, so the active count only
            # masks its suffix and T < TP keeps the AllGather row mapping.
            width = tensors["output"].shape[1]
            for offset in range(TP_SIZE):
                first = min(offset * width, active)
                count = max(0, min(width, active - first))
                tensors["output"][base + offset].zero_()
                if count:
                    tensors["output"][base + offset][:count].copy_(reduced[first : first + count])
        else:
            tensors["output"][base:base + TP_SIZE].copy_(reduced.unsqueeze(0).expand(TP_SIZE, -1, -1))


def compare_reduced(actual, expected, **kwargs):
    passed = True
    for base in range(0, actual.shape[0], TP_SIZE):
        valid, _ = compare_output(actual[base], expected[base])
        passed &= valid
        passed &= all(torch.equal(actual[base], actual[rank]) for rank in range(base + 1, base + TP_SIZE))
    return passed, "Each DP group must pass precision and every TP replica must be byte-identical"


def compare_distributed_cache(actual, expected, *, actual_outputs, expected_outputs, inputs, **kwargs):
    passed = True
    for rank in range(actual.shape[0]):
        valid, detail = compare_cache(actual[rank], expected[rank],
            actual_outputs={"window_cache_scale": actual_outputs["window_cache_scale"][rank]},
            expected_outputs={"window_cache_scale": expected_outputs["window_cache_scale"][rank]},
            inputs={"window_slots": inputs["window_slots"][rank]})
        passed &= valid
        if not valid:
            print(f"[PRECISION] rank={rank}: {detail}")
    return passed, "Every rank must preserve unmapped cache bytes and pass updated-cache precision"


def run_swa(operator, mode, argv=None):
    """Run A5 validation for a production SWA operator."""
    capacity = C.DECODE_MAX_TOKENS if mode == "decode" else C.PREFILL_MAX_TOKENS
    parser = argparse.ArgumentParser(description=f"DeepSeek V4.1 {mode} SWA: A5 precision and timing")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--ep", type=int, default=C.EP_SIZE, choices=[2, 4, 8])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", "--batch", type=int, default=32 if mode == "decode" else 257)
    parser.add_argument("--requests", type=int, help="packed requests; default min(tokens, 32 decode / 4 prefill)")
    parser.add_argument("--case", default="mixed", choices=["mixed", "prefix", "shuffle", "masked", "zero", "sink"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1, help="operator calls per dispatch; timing includes all epochs")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true", help="save validated inputs and golden outputs for replay")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args(argv)
    if args.tp != TP_SIZE:
        parser.error(f"--tp {args.tp} does not match import-time TP_SIZE {TP_SIZE}")
    from pypto.ir import DistributedConfig

    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    args.requests = args.requests if args.requests is not None else min(args.tokens, 32 if mode == "decode" else 4)
    try:
        devices = list(range(TP_SIZE * args.dp)) if args.device is None else [int(d) for d in args.device.split(",")]
    except ValueError:
        parser.error("device IDs must be comma-separated integers")
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("device IDs must be distinct nonnegative integers, with count TP * DP")
    if not 1 <= args.tokens <= capacity or not 1 <= args.requests <= min(32, args.tokens):
        parser.error(f"tokens must be in [1, {capacity}]; requests in [1, min(32, tokens)]")
    if not 1 <= args.epochs <= 1000:
        parser.error("epochs must be in [1, 1000]")
    if args.bench and args.enable_chip_swimlane:
        parser.error("benchmark epoch stepping and multi-pass chip swimlane must be run separately")
    torch.set_num_threads(8)
    print(f"[SWA] mode={mode} tokens={args.tokens} requests={args.requests} TP={TP_SIZE} DP={args.dp} "
          f"case={args.case} seed={args.seed} epochs/dispatch={args.epochs} devices={devices}")
    if args.bench:
        print("[SWA] Resident device timing excludes compilation, input generation and CPU golden; "
              "each dispatch advances the communication epoch. Timing includes all epochs/dispatch.")
    result = run(fn=make_program(operator, capacity, len(devices), args.epochs), specs=build_specs(args, mode),
        golden_fn=golden_swa, compile_only=args.compile_only, save_data=args.save_data, golden_data=args.golden_data,
        config=dict(platform=args.platform, distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
                    enable_chip_swimlane=args.enable_chip_swimlane, enable_dep_gen=args.enable_dep_gen),
        compare_fn={"output": compare_reduced, "window_cache": compare_distributed_cache,
                    "window_cache_scale": compare_scales})
    print(f"[SWA] work_dir={result.work_dir}")
    if args.compile_only and result.passed:
        print("[SWA] Compilation passed; device accuracy was NOT validated.")
    elif args.save_data and result.passed:
        print(f"[SWA] Validated snapshot: {result.work_dir}/data")
    return result


def validate(argv=None):
    """Validate the Decode SWA production operator on A5."""
    return run_swa(decode_attn_swa, "decode", argv=argv)


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

    @pytest.mark.parametrize("tp,dp", [(2, 2)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error


if __name__ == _SCRIPT_ENTRY_POINT:
    main()
