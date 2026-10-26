# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SWA decode attention with mHC."""


import pypto.language as pl
from config import FLASH as M, T_DYN, LOCAL_T_DYN, ORI_BLOCKS_DYN, WINDOW_CACHE_GROUP
from sparse_attention import publish_window_row, swa_attention_partial
from qkv_proj_rope import qkv_proj_rope
from o_proj import decode_o_proj
from hc import hc_pre, zero_bf16_padding, hc_post
from rmsnorm import rms_norm_tiled


# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
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

# tiling
Q_ROPE_WORKER_TILE = 64


@pl.jit.inline(auto_scope=False)
def decode_swa(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
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
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one SWA attention sublayer and emit the staggered mix for the next sublayer."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_attn = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(
        x_hc, incoming_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        x_attn, attention_pre_mix, post_mix, residual_mix,
    )
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    for norm_block in pl.spmd((tokens + 7) // 8, name_hint="norm_in"):
        rms_norm_tiled(x_attn, attn_norm_weight, x_norm, norm_block, tokens, output_fp32=False)

    zero_bf16_padding(x_norm, pl.min(num_tokens, pl.tensor.dim(x_norm, 0)))
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qkv_proj_rope(
        x_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin,
        qr, q, kv_projection, kv, num_tokens, max_workers=Q_ROPE_WORKER_TILE,
        fuse_qnorm=True, q_b_n_tile=256, q_b_max_workers=2147483647,
    )
    publish_window_row(kv, window_slots, window_cache, window_cache_scale, num_tokens)
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    swa_attention_partial(q, window_cache, window_cache_scale, window_indices, attn_sink, attended, num_tokens)
    decode_o_proj(
        attended, wo_a, wo_a_scale, wo_b,
        wo_b_scale, rope_cos, rope_sin,
        attention_output, num_tokens,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden, deps=hc_post_deps)
    return attention_hidden, attention_pre_mix


@pl.jit
def l2_decode_swa(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    attention_output: pl.InOut[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Dispatch one production SWA composition to a device-selected JIT entry."""
    return decode_swa(
        x_hc, incoming_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale,
        attention_output, attention_hidden, attention_pre_mix,
        num_tokens,
    )


@pl.jit.host
def l3_decode_swa(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    attention_output: pl.InOut[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    # MX_B_NN weight-scale views.
    wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale
    wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale
    wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale
    wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale
    l2_decode_swa(
        x_hc, incoming_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale_r, q_norm_weight, wq_b, wq_b_scale_r,
        wkv, wkv_scale_r, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale_r,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale,
        attention_output, attention_hidden, attention_pre_mix,
        num_tokens, device=0,
    )


def build_swa_specs(args):
    """Build SWA attention operands, mHC boundaries, and runtime specifications."""
    from golden import ScalarSpec, TensorSpec
    import torch
    from functools import partial
    from config import MAX_BATCH_PER_DP
    from utils import packed_query_lengths
    from sparse_attention import SWA_INPUT_NAMES, make_swa_inputs, make_packed_swa_inputs

    if getattr(args, "dp", 1) != 1:
        raise ValueError("decode SWA specs describe one device")
    packed = getattr(args, "start_pos", None) is not None or args.tokens > MAX_BATCH_PER_DP
    packed = packed or args.requests != args.tokens or args.case in ("prefix", "shuffle")
    if packed:
        query_lens = getattr(args, "query_lens", None)
        lengths = packed_query_lengths(args.tokens, args.requests, query_lens)
        if query_lens is None and args.requests > 1 and lengths[-1] > 16:
            lengths[0] += 7
            lengths[-1] -= 7
        prefixes = ([127, 128, 511, 1048000] * 8)[:args.requests] if args.case == "prefix" else [0] * args.requests
        if getattr(args, "start_pos", None) is not None:
            prefixes = args.start_pos
        pages = 1 + sum((p + n + 128) // 128 - max(0, p - 127) // 128 + 1 for p, n in zip(prefixes, lengths))
    else:
        pages = args.tokens + 1
    leaf_shapes = (
        [args.tokens, D], [D, Q_LORA], [D // 32, Q_LORA], [Q_LORA],
        [Q_LORA, LOCAL_H * HEAD_DIM], [Q_LORA // 32, LOCAL_H * HEAD_DIM],
        [D, HEAD_DIM], [D // 32, HEAD_DIM], [HEAD_DIM], [LOCAL_H],
        [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA],
        [LOCAL_O_WIDTH, D], [LOCAL_O_WIDTH // 32, D],
        [args.tokens, ROPE_DIM // 2], [args.tokens, ROPE_DIM // 2], [args.tokens], [args.tokens, 128],
        [pages, 128, 1, HEAD_DIM], [pages, 128, 1, HEAD_DIM // 32],
    )
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    dtypes = (bf, fp, mx, bf, fp, mx, fp, mx, bf, torch.float32, fp, mx, fp, mx,
              torch.float32, torch.float32, torch.int64, torch.int32, fp, bf)
    values = None
    if not getattr(args, "compile_only", False) and not getattr(args, "golden_data", None):
        if packed:
            values = make_packed_swa_inputs(
                args.tokens, args.requests, args.seed, args.case,
                start_pos=getattr(args, "start_pos", None), query_lens=getattr(args, "query_lens", None),
            )
        else:
            values = make_swa_inputs(args.tokens, args.seed)
            if args.case == "masked":
                values["window_slots"].fill_(-1)
                values["window_indices"].fill_(-1)
            elif args.case == "zero":
                values["x"].zero_()
                values["window_cache"].view(torch.uint8).zero_()
            elif args.case == "sink":
                values["attn_sink"].fill_(1000)
    specs = []
    for name, shape, dtype in zip(SWA_INPUT_NAMES, leaf_shapes, dtypes):
        if name == "x":
            continue
        initialize = None if values is None else partial(torch.clone, values[name])
        specs.append(TensorSpec(name, shape, dtype, init_value=initialize))
    local = getattr(args, "local_tokens", args.tokens)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    shapes = {
        "x_hc": [local, HC_MULT, D],
        "incoming_pre_mix": [local, HC_MULT],
        "hc_attn_fn": [MIX_HC, HC_DIM],
        "hc_attn_scale": [3],
        "hc_attn_base": [MIX_HC],
        "attn_norm_weight": [D],
    }
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator)
        if name == "hc_attn_fn":
            value /= HC_DIM**0.5
        elif name == "attn_norm_weight":
            value = torch.ones(shape, dtype=torch.bfloat16)
        elif name == "incoming_pre_mix":
            value = torch.softmax(value, dim=-1)
        if name in ("x_hc", "incoming_pre_mix"):
            stacked = torch.randn(shape, generator=generator)
            if name == "incoming_pre_mix":
                stacked = torch.softmax(stacked, dim=-1)
        else:
            stacked = value
        specs.append(TensorSpec(name, shape, value.dtype, init_value=stacked))
    for name, shape, dtype in (
        ("attention_output", [local, D], torch.bfloat16),
        ("attention_hidden", [local, HC_MULT, D], torch.float32),
        ("attention_pre_mix", [local, HC_MULT], torch.float32),
    ):
        sentinel = 13.0 if name == "attention_output" else 0.0
        specs.append(TensorSpec(name, shape, dtype, init_value=sentinel))
    specs.append(TensorSpec("gathered", [args.tokens, D], torch.bfloat16, init_value=13.0))
    specs.append(ScalarSpec("num_tokens", torch.int32, args.active_tokens))
    specs.append(ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True, benchmark_step=None))
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in l3_decode_swa.param_names]


def golden_swa(tensors, *, reference=None):
    """Compose TP1 mHC, normalization, QKV, attention and output references."""
    import torch
    from sparse_attention import SWA_INPUT_NAMES, swa_official_reference
    from hc import golden_hc_collapse, golden_mhc_mixes, golden_hc_post
    from rmsnorm import golden_rms_norm

    active = int(tensors["num_tokens"])
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"],
        tensors["hc_attn_fn"], tensors["hc_attn_scale"], tensors["hc_attn_base"],
    )
    tensors["attention_pre_mix"].copy_(pre_mix)
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"]).to(torch.bfloat16)
    normalized = golden_rms_norm(collapsed[:active], tensors["attn_norm_weight"])
    if reference is None:
        inputs = {name: normalized if name == "x" else tensors[name] for name in SWA_INPUT_NAMES}
        cache_rows = tensors["window_cache"].shape[0] * 128
        shared_cache = active >= 32 and cache_rows <= active * 128
        partial, cache, scale = swa_official_reference(inputs, shared_cache=shared_cache)
        tensors["attention_output"][:active].copy_(partial.bfloat16())
        tensors["window_cache"].view(torch.uint8).copy_(cache.contiguous().view(torch.uint8))
        tensors["window_cache_scale"].copy_(scale)
    else:
        import inspect

        parameters = inspect.signature(reference).parameters
        inputs = {name: normalized if name == "x" else tensors[name] for name in parameters}
        computed = reference(**inputs)
        tensors["attention_output"][:active].copy_(computed["output"])
        for name in ("window_cache", "window_cache_scale"):
            tensors[name].view(torch.uint8).copy_(computed[name].contiguous().view(torch.uint8))
    tensors["attention_hidden"].zero_()
    tensors["attention_output"][active:].zero_()
    if active:
        hidden = golden_hc_post(
            tensors["attention_output"][:active], tensors["x_hc"][:active],
            post_mix[:active], residual_mix[:active],
        ).to(torch.float32)
        tensors["attention_hidden"][:active].copy_(hidden)
    tensors["attention_input"] = collapsed


if __name__ == "__main__":
    from golden import mapped_pool_ratio_allclose, ratio_allclose

    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse

    from pypto.ir import DistributedConfig
    from golden import run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 TP1 SWA decode composition")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one starting position per request; omitted: default validation workload"
    parser.add_argument("--start-pos", "--startpos", dest="startpos", type=parse_decode_startpos, help=startpos_help)
    query_help = "one query length per start-pos; omitted: config.DECODE_ROWS_PER_REQUEST each"
    parser.add_argument("--query-lens", type=parse_decode_query_lens, help=query_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=("mixed", "shuffle", "prefix"), default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=72, requests=12, startpos=[8192] * 12)
    args = parser.parse_args()
    configure_decode_workload(args, parser)
    if args.tokens != 72:
        parser.error("this decode entry requires 72 query tokens")
    args.start_pos = args.startpos
    args.dp = 1
    args.active_tokens = args.tokens
    args.bench = False
    devices = [args.device]
    result = run(
        fn=l3_decode_swa,
        specs=build_swa_specs(args),
        golden_fn=golden_swa,
        compare_fn={
            "attention_hidden": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
            "attention_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache", mapping_shape=(args.tokens,),
                block_size=128, leading_rank_axis=False,
                atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.01,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache scale", mapping_shape=(args.tokens,),
                block_size=128, leading_rank_axis=False,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
        },
        config={
            "backend_type": args.backend_type, "execution_mode": args.execution_mode,
            "enable_chip_swimlane": args.enable_chip_swimlane,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
