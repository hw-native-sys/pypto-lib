# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""SWA decode attention with HC."""


from quant import packed_cache_compare

import pypto.language as pl

from quant import pack_quantized_cache, unpack_quantized_cache

from config import FLASH as M, T_DYN, ORI_BLOCKS_DYN, WINDOW_CACHE_BYTES, DECODE_BATCH
from sparse_attention import gather_sparse_attention
from o_proj import o_proj_hc_post
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj, window_publish
from hc import hc_head_norm, hc_project, hc_sinkhorn, zero_bf16_padding


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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    attention_hidden: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    attention_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one SWA attention sublayer and emit the staggered mix for the next sublayer."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    x_ready = hc_head_norm(x_hc, incoming_pre_mix, attn_norm_weight, x_norm, worker_limit=64)

    zero_bf16_padding(x_norm, pl.min(num_tokens, pl.tensor.dim(x_norm, 0)))
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    proj_payload, proj_scales, proj_payload_padded_rows = quantize_projection_input(x_norm, num_tokens, row_tile=80)
    q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q,
        num_tokens, qa_worker_limit=32, qb_worker_limit=32,
        padded_rows=proj_payload_padded_rows,
        qa_row_tile=80, qb_row_tile=80,
    )
    kv_projection = kv_proj(
        proj_payload, proj_scales,
        wkv, wkv_scale,
        kv_projection, num_tokens, worker_limit=32, row_tile=80,
    )
    cache_ready = window_publish(
        kv_projection, kv_norm_weight, rope_cos, rope_sin,
        window_slots, window_cache,
        num_tokens, x_ready,
    )
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    gather_sparse_attention(
        q, window_indices, window_cache, window_indices, window_cache,
        attn_sink, attended, num_tokens, cache_ready, compressed_keys=0,
    )
    inv_rms, partials = hc_project(x_hc, hc_attn_fn, rms_worker_limit=64)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix)
    o_proj_hc_post(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        x_hc, post_mix, residual_mix, attention_hidden, num_tokens,
        oa_row_tile=80, ob_row_tile=128,
    )
    return attention_hidden, attention_pre_mix


@pl.jit
def l2_decode_swa(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    attention_hidden: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
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
        window_cache,
        attention_hidden, attention_pre_mix,
        num_tokens,
    )


@pl.jit.host
def l3_decode_swa(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
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
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    attention_hidden: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, T_DYN)
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
        window_cache,
        attention_hidden, attention_pre_mix,
        num_tokens, device=0,
    )


def build_swa_specs(args):
    """Build SWA attention operands, HC boundaries, and runtime specifications."""
    from golden import ScalarSpec, TensorSpec
    import torch
    from functools import partial
    from utils import packed_query_lengths, window_metadata
    from quant import quantize_mxfp8_cache
    import math
    from quant import pack_fp8_nz, pack_mx_b_scale
    from quant import quantize_mxfp8_upward_reference, quantize_grouped_mxfp8_weight

    if args.requests < 1 or args.tokens < 1:
        raise ValueError("requests and tokens must be positive")
    if getattr(args, "dp", 1) != 1:
        raise ValueError("decode SWA specs describe one device")
    packed = getattr(args, "start_pos", None) is not None or args.tokens > DECODE_BATCH
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
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    leaf_specs = (
        ("wq_a", [D, Q_LORA], fp),
        ("wq_a_scale", [D // 32, Q_LORA], mx),
        ("q_norm_weight", [Q_LORA], bf),
        ("wq_b", [Q_LORA, LOCAL_H * HEAD_DIM], fp),
        ("wq_b_scale", [Q_LORA // 32, LOCAL_H * HEAD_DIM], mx),
        ("wkv", [D, HEAD_DIM], fp),
        ("wkv_scale", [D // 32, HEAD_DIM], mx),
        ("kv_norm_weight", [HEAD_DIM], bf),
        ("attn_sink", [LOCAL_H], torch.float32),
        ("wo_a", [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], fp),
        ("wo_a_scale", [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], mx),
        ("wo_b", [LOCAL_O_WIDTH, D], fp),
        ("wo_b_scale", [LOCAL_O_WIDTH // 32, D], mx),
        ("rope_cos", [args.tokens, ROPE_DIM // 2], torch.float32),
        ("rope_sin", [args.tokens, ROPE_DIM // 2], torch.float32),
        ("window_slots", [args.tokens], torch.int64),
        ("window_indices", [args.tokens, 128], torch.int32),
        ("window_cache", [pages, 128, 1, WINDOW_CACHE_BYTES], torch.uint8),
    )
    values = None
    if not getattr(args, "compile_only", False) and not getattr(args, "golden_data", None):
        gen = torch.Generator().manual_seed(args.seed)

        def rand(*shape):
            return torch.randn(*shape, generator=gen)

        def weight(k, n):
            payload, scales = quantize_mxfp8_upward_reference(rand(n, k) / math.sqrt(k))
            return payload.T.contiguous(), pack_mx_b_scale(scales.T.contiguous()).view(torch.float8_e8m0fnu)

        qa, qas = weight(D, Q_LORA)
        qb, qbs = weight(Q_LORA, LOCAL_H * HEAD_DIM)
        kv, kvs = weight(D, HEAD_DIM)
        ob, obs = weight(LOCAL_O_WIDTH, D)
        values = dict(
            wq_a=pack_fp8_nz(qa), wq_a_scale=qas,
            q_norm_weight=(rand(Q_LORA) * 0.1 + 1).to(torch.bfloat16),
            wq_b=pack_fp8_nz(qb), wq_b_scale=qbs,
            wkv=pack_fp8_nz(kv), wkv_scale=kvs,
            kv_norm_weight=(rand(HEAD_DIM) * 0.1 + 1).to(torch.bfloat16), attn_sink=rand(LOCAL_H) * 2,
            wo_b=pack_fp8_nz(ob), wo_b_scale=obs,
        )
        wo_a = (rand(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16)
        values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(wo_a)
        cache_gen = torch.Generator().manual_seed(args.seed + 1000)
        if packed:
            input_lengths = torch.tensor(lengths, dtype=torch.int64)
            prefix_tensor = torch.tensor(prefixes, dtype=torch.int64)
            ids = torch.repeat_interleave(torch.arange(args.requests), input_lengths)
            positions = torch.cat([torch.arange(p, p + n) for p, n in zip(prefixes, lengths)])
            logical_capacity = int((prefix_tensor + input_lengths + 129).max()) // 128 + 1
            table = torch.full((args.requests, logical_capacity), -1, dtype=torch.int32)
            mappings = []
            for request, (prefix, length) in enumerate(zip(prefixes, lengths)):
                start = max(0, prefix - 127) // 128
                end = (prefix + length + 128) // 128
                mappings.extend((request, block) for block in range(start, end + 1))
            physical_pages = torch.randperm(pages, generator=cache_gen)[:len(mappings)]
            for (request, block), page in zip(mappings, physical_pages):
                table[request, block] = page
            slots, indices, _ = window_metadata(positions, ids, table)
            if slots.unique().numel() != args.tokens:
                raise ValueError("SWA fixture slots must be unique")
            for row, (request, position) in enumerate(zip(ids.tolist(), positions.tolist())):
                history = range(max(0, position - 127), position + 1)
                expected = [int(table[request, p // 128]) * 128 + p % 128 for p in history]
                if indices[row, :len(expected)].tolist() != expected or not (indices[row, len(expected):] == -1).all():
                    raise ValueError("SWA fixture indices do not match the paged window")
        else:
            positions = torch.full((args.tokens,), 127, dtype=torch.int64)
            if args.tokens > 1:
                positions[:min(args.tokens, 6)] = torch.tensor([0, 1, 63, 127, 128, 1048575])[:min(args.tokens, 6)]
            physical_pages = torch.randperm(pages, generator=cache_gen)[:args.tokens]
            slots = physical_pages.long() * 128 + positions.remainder(128)
            indices = torch.full((args.tokens, 128), -1, dtype=torch.int32)
            for row, position in enumerate(positions.tolist()):
                length = min(position + 1, 128)
                logical = torch.arange(position - length + 1, position + 1)
                indices[row, :length] = (physical_pages[row] * 128 + logical.remainder(128)).to(torch.int32)
        frequency = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
        angles = positions.float()[:, None] * frequency
        cache_values = torch.randn(pages, 128, 1, HEAD_DIM, generator=cache_gen).bfloat16()
        cache, scales = quantize_mxfp8_cache(cache_values)
        values.update(
            rope_cos=angles.cos(), rope_sin=angles.sin(), window_slots=slots, window_indices=indices,
            window_cache=cache, window_cache_scale=scales,
        )
        if args.case == "shuffle":
            order = torch.randperm(args.tokens, generator=cache_gen)
            for name in ("rope_cos", "rope_sin", "window_slots", "window_indices"):
                values[name] = values[name][order].contiguous()
        if args.case == "masked":
            values["window_slots"].fill_(-1)
            values["window_indices"].fill_(-1)
        elif args.case == "zero":
            values["window_cache"].view(torch.uint8).zero_()
        elif args.case == "sink":
            values["attn_sink"].fill_(1000)
    capacity = getattr(args, "capacity", args.tokens)
    if capacity < args.tokens:
        raise ValueError("capacity must cover the scheduled tokens")
    row_metadata = {"rope_cos", "rope_sin", "window_slots", "window_indices"}
    specs = []
    for name, shape, dtype in leaf_specs:
        if name == "window_cache":
            cache = None if values is None else pack_quantized_cache(values[name], values["window_cache_scale"])
            initialize = None if cache is None else partial(torch.clone, cache)
        else:
            initialize = None if values is None else partial(torch.clone, values[name])
        if name in row_metadata and capacity != args.tokens:
            shape = [capacity, *shape[1:]]
            if values is not None:
                fill = -1 if name in ("window_slots", "window_indices") else 0
                padded = torch.full(shape, fill, dtype=dtype)
                padded[:args.tokens] = values[name]
                initialize = partial(torch.clone, padded)
        specs.append(TensorSpec(name, shape, dtype, init_value=initialize))
    generator = torch.Generator().manual_seed(args.seed + 1000)
    shapes = {
        "x_hc": [capacity, HC_MULT, D],
        "incoming_pre_mix": [capacity, HC_MULT],
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
        ("attention_hidden", [capacity, HC_MULT, D], torch.float32),
        ("attention_pre_mix", [capacity, HC_MULT], torch.float32),
    ):
        specs.append(TensorSpec(name, shape, dtype, init_value=0.0))
    specs.append(TensorSpec("gathered", [args.tokens, D], torch.bfloat16, init_value=13.0))
    specs.append(ScalarSpec("num_tokens", torch.int32, args.active_tokens))
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in l3_decode_swa.param_names]


def golden_swa(tensors):
    """Compose HC, normalization, QKV, attention and output references."""
    import torch
    from sparse_attention import golden_gather_sparse_attention
    from qkv_proj_rope import golden_qkv_proj_rope
    from o_proj import golden_o_proj
    from quant import quantize_mxfp8_cache
    from quant import _publish_quantized_cache
    from hc import golden_hc_collapse, golden_hc_norm, golden_hc_mixes, golden_hc_post

    fields = dict(tensors)
    fields["window_cache"], fields["window_cache_scale"] = unpack_quantized_cache(tensors["window_cache"], False, head_dim=tensors["kv_norm_weight"].numel())
    active = int(tensors["num_tokens"])
    # Projection exists only in the CPU reference after the fused device path.
    tensors["attention_output"] = torch.zeros_like(tensors["x_hc"][:, 0], dtype=torch.bfloat16)
    pre_mix, post_mix, residual_mix = golden_hc_mixes(
        tensors["x_hc"],
        tensors["hc_attn_fn"], tensors["hc_attn_scale"], tensors["hc_attn_base"],
    )
    tensors["attention_pre_mix"].copy_(pre_mix)
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"]).to(torch.bfloat16)
    normalized = golden_hc_norm(collapsed[:active], tensors["attn_norm_weight"])
    if active:
        inputs = dict(fields, x=normalized)
        for name in ("rope_cos", "rope_sin", "window_slots", "window_indices"):
            inputs[name] = inputs[name][:active]
        query, window_kv, _ = golden_qkv_proj_rope(
            inputs["x"], inputs["wq_a"], inputs["wq_a_scale"], inputs["q_norm_weight"],
            inputs["wq_b"], inputs["wq_b_scale"], inputs["wkv"], inputs["wkv_scale"], inputs["kv_norm_weight"],
            inputs["rope_cos"], inputs["rope_sin"], sequential_query=True,
        )
        cache, scale = _publish_quantized_cache(
            inputs["window_cache"], inputs["window_cache_scale"], window_kv, inputs["window_slots"], quantize_mxfp8_cache,
        )
        packed_cache = pack_quantized_cache(cache, scale)
        indices = inputs["window_indices"]
        attended = golden_gather_sparse_attention(
            query, indices, packed_cache, indices, packed_cache, inputs["attn_sink"], compressed_keys=0,
        )
        projected = golden_o_proj(
            attended, inputs["rope_cos"], inputs["rope_sin"], inputs["wo_a"], inputs["wo_b"], inputs["wo_b_scale"],
            wo_a_scale=inputs["wo_a_scale"], sequential_projection=True,
        )
        tensors["attention_output"][:active].copy_(projected)
        tensors["window_cache"].copy_(packed_cache)
    tensors["attention_hidden"].zero_()
    tensors["attention_output"][active:].zero_()
    if active:
        hidden = golden_hc_post(
            tensors["attention_output"][:active], tensors["x_hc"][:active],
            post_mix[:active], residual_mix[:active],
        )
        tensors["attention_hidden"][:active].copy_(hidden)
    tensors["attention_input"] = collapsed


if __name__ == "__main__":
    from golden import mapped_pool_ratio_allclose, ratio_allclose

    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse

    from pypto.ir import DistributedConfig
    from golden import run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 SWA decode composition")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one starting position per request; omitted: default validation workload"
    parser.add_argument("--start-pos", "--startpos", dest="startpos", type=parse_decode_startpos, help=startpos_help)
    query_help = "one query length per start-pos; omitted: config.DECODE_SEQ each"
    parser.add_argument("--query-lens", type=parse_decode_query_lens, help=query_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=("mixed", "shuffle", "prefix"), default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
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
            "window_cache": packed_cache_compare(
                mapped_pool_ratio_allclose(
                    "window_slots", pool_name="window cache", mapping_shape=(args.tokens,),
                    block_size=128, leading_rank_axis=False,
                    atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.01,
                ),
                mapped_pool_ratio_allclose(
                    "window_slots", pool_name="window cache scale", mapping_shape=(args.tokens,),
                    block_size=128, leading_rank_axis=False,
                    atol=0.0, rtol=0.0, max_error_ratio=0.0,
                ),
                fp4=False,
            ),
        },
        config={
            "backend_type": args.backend_type, "execution_mode": args.execution_mode,
            "enable_chip_swimlane": args.enable_chip_swimlane,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
