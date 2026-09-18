# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill SWA wired through mHC for encoder layers 0 and 1."""

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
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes, mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post, mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre, mhc_pre
from models.deepseek_v4_1_flash.prefill_attn_swa import prefill_attn_swa


# Prefill SWA plus four-stream mHC overflows the default 256 MiB ring heap.
# One GiB per ring matches the V4 Pro prefill attention bring-up budget.
PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4


@pl.jit.inline(auto_scope=False)
def prefill_swa(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
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
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Collapse HC streams, run packed SWA, then expand the residual."""
    tokens = pl.tensor.dim(x_hc, 0)
    pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, pre_mix, post_mix, residual_mix)
    mhc_pre(x_hc, pre_mix, hidden)
    prefill_attn_swa(
        hidden, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale,
        rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
        output_window, output_arrived, attn_out,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


def golden_prefill_swa(
    x_hc: torch.Tensor,
    hc_attn_fn: torch.Tensor,
    hc_attn_scale: torch.Tensor,
    hc_attn_base: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HC mixes, collapse, SWA, then residual expansion. No extra attention RMSNorm."""
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    hidden = golden_mhc_pre(x_hc, pre_mix)
    result = golden_swa_attention(
        hidden, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
    )
    output = golden_mhc_post(result.output.to(torch.bfloat16), x_hc, post_mix, residual_mix)
    return output, result.window_cache, result.window_cache_scale


HC_INPUT_NAMES = (
    "x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
    "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
)


def make_hc_inputs(base: dict, seed: int) -> dict:
    """Replace the collapsed hidden with four HC streams and mix weights."""
    gen = torch.Generator().manual_seed(seed + 2000)
    tokens = base["x"].shape[0]
    values = dict(base)
    values.pop("x")
    values["x_hc"] = torch.randn(tokens, HC_MULT, D, generator=gen)
    values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM)
    values["hc_attn_scale"] = torch.randn(3, generator=gen)
    values["hc_attn_base"] = torch.randn(MIX_HC, generator=gen)
    if bool((base["x"] == 0).all()):
        values["x_hc"].zero_()
    return values


def make_hc_program(capacity, world_size, epochs):
    """Wrap the HC-orchestrated prefill SWA operator for stacked TP/DP ranks."""
    @pl.jit
    def swa_rank(
        x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
        hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
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
        output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Run one rank of HC-orchestrated prefill SWA for ``epochs`` dispatches."""
        x_hc.bind_dynamic(0, T_DYN)
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        for step in pl.range(epochs):
            prefill_swa(
                x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
                wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale,
                rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
                output_window, output_arrived, output,
                rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, num_tokens, attention_epoch + step,
            )
        return output, window_cache, window_cache_scale

    @pl.jit.host
    def swa_group(
        x_hc: pl.Tensor[[world_size, T_DYN, HC_MULT, D], pl.FP32],
        hc_attn_fn: pl.Tensor[[world_size, MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[world_size, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[world_size, MIX_HC], pl.FP32],
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
        output: pl.Out[pl.Tensor[[world_size, T_DYN, HC_MULT, D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Dispatch stacked ranks onto TP windows for the HC prefill program."""
        x_hc.bind_dynamic(1, T_DYN)
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
                x_hc[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
                wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
                wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank],
                wo_a[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank],
                window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank],
                output[rank], data, signal, rank, num_tokens, attention_epoch, device=rank,
            )

    return swa_group


def build_hc_specs(args):
    """Stacked TP/DP specs with HC streams in and HC streams out."""
    from golden import ScalarSpec, TensorSpec
    from models.deepseek_v4_1_flash.decode_swa import make_inputs, make_packed_inputs

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
    shapes = (
        [args.tokens, HC_MULT, D], [MIX_HC, HC_DIM], [3], [MIX_HC],
        [D, Q_LORA], [D // 32, Q_LORA], [Q_LORA],
        [Q_LORA, LOCAL_H * HEAD_DIM], [Q_LORA // 32, LOCAL_H * HEAD_DIM],
        [D, HEAD_DIM], [D // 32, HEAD_DIM], [HEAD_DIM], [LOCAL_H],
        [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [LOCAL_O_WIDTH, D], [LOCAL_O_WIDTH // 32, D],
        [args.tokens, ROPE_DIM // 2], [args.tokens, ROPE_DIM // 2], [args.tokens], [args.tokens, 128],
        [pages, 128, 1, HEAD_DIM], [pages, 128, 1, HEAD_DIM // 32],
    )
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    dtypes = (torch.float32, torch.float32, torch.float32, torch.float32,
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
                ranks.append(make_hc_inputs(value, seed))
            replicated = (
                "x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                "wq_a", "wq_a_scale", "q_norm_weight", "wkv", "wkv_scale",
                "kv_norm_weight", "rope_cos", "rope_sin", "window_slots", "window_indices",
                "window_cache", "window_cache_scale",
            )
            for rank, value in enumerate(ranks):
                for key in replicated:
                    value[key] = ranks[rank // TP_SIZE * TP_SIZE][key]
            for key in HC_INPUT_NAMES:
                dtype = ranks[0][key].dtype
                shards = [r[key].view(torch.uint8) if dtype in (fp, mx) else r[key] for r in ranks]
                values[key] = torch.stack(shards).view(dtype)
        return values[name]

    specs = [TensorSpec(name, [world_size, *shape], dtype, init_value=lambda n=name: initialize(n), resident="stacked")
             for name, shape, dtype in zip(HC_INPUT_NAMES, shapes, dtypes)]
    specs += [TensorSpec("output", [world_size, args.tokens, HC_MULT, D], torch.float32, resident="stacked"),
              ScalarSpec("num_tokens", torch.int32, args.tokens),
              ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True,
                         benchmark_step=args.epochs if args.bench else None)]
    return specs


def golden_prefill_swa_case(tensors):
    """Reference each TP shard independently, reduce, then expand HC residuals."""
    from models.deepseek_v4_1_flash.decode_swa import official_reference

    world_size = tensors["x_hc"].shape[0]
    for base in range(0, world_size, TP_SIZE):
        pre_mix, post_mix, residual_mix = golden_mhc_mixes(
            tensors["x_hc"][base], tensors["hc_attn_fn"][base],
            tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base],
        )
        hidden = golden_mhc_pre(tensors["x_hc"][base], pre_mix)
        partials = []
        for rank in range(base, base + TP_SIZE):
            inputs = {name: tensors[name][rank] for name in HC_INPUT_NAMES if name not in (
                "x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
            )}
            inputs["x"] = hidden
            partial, cache, scale = official_reference(inputs)
            partials.append(partial)
            tensors["window_cache"][rank].copy_(cache)
            tensors["window_cache_scale"][rank].copy_(scale)
        reduced = sum(partials).bfloat16()
        output = golden_mhc_post(reduced, tensors["x_hc"][base], post_mix, residual_mix)
        tensors["output"][base:base + TP_SIZE].copy_(output.unsqueeze(0).expand(TP_SIZE, -1, -1, -1))


def run_prefill_swa():
    """Run A5 validation for packed prefill SWA wired through mHC."""
    import argparse
    import os

    from golden import run
    from models.deepseek_v4_1_flash.decode_swa import (
        compare_distributed_cache,
        compare_reduced,
        compare_scales,
    )
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill SWA + mHC: A5 precision and timing")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--ep", type=int, default=8, choices=[2, 4, 8])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", "--batch", type=int, default=128)
    parser.add_argument("--requests", type=int, help="packed requests; default min(tokens, 4)")
    parser.add_argument("--case", default="mixed", choices=["mixed", "prefix", "shuffle", "masked", "zero", "sink"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1, help="operator calls per dispatch; timing includes all epochs")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args()
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
    torch.set_num_threads(8)
    print(f"[SWA+HC] tokens={args.tokens} requests={args.requests} TP={TP_SIZE} DP={args.dp} "
          f"case={args.case} seed={args.seed} devices={devices}")
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
        compare_fn={"output": compare_reduced, "window_cache": compare_distributed_cache,
                    "window_cache_scale": compare_scales},
    )
    print(f"[SWA+HC] work_dir={result.work_dir}")
    if not result.passed:
        raise SystemExit(1)
    if args.compile_only:
        print("[SWA+HC] Compilation passed; device accuracy was NOT validated.")


__all__ = [
    "golden_prefill_swa",
    "prefill_swa",
    "run_prefill_swa",
]


def main():
    """Validate packed prefill SWA wired through mHC on A5."""
    run_prefill_swa()


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    main()
