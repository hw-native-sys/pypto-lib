# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DSpark context KV publication and sliding-window attention with HC pre/post processing."""

from pathlib import Path

import pypto.language as pl
import torch
from pypto.runtime import pto_isa_include_dir

from quant import packed_cache_compare, pack_quantized_cache, unpack_quantized_cache

from config import (
    DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH, FLASH, MOE_TOKENS, ORI_BLOCKS_DYN,
    WINDOW_CACHE_BYTES,
)
from hc import hc_head_norm, hc_project, hc_sinkhorn, zero_bf16_padding
from o_proj import o_proj_hc_post
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj, window_publish
from qkv_proj_rope import golden_rope_interleave
from quant import quantize_mxfp8_cache
from rmsnorm import golden_rms_norm


# Dynamic shape variables.
CONTEXT_T_DYN = pl.dynamic("DSPARK_CONTEXT_T_DYN")
CONTEXT_PAYLOAD_T_DYN = pl.dynamic("DSPARK_CONTEXT_PAYLOAD_T_DYN")
CONTEXT_SCALE_T_DYN = pl.dynamic("DSPARK_CONTEXT_SCALE_T_DYN")

# model config
HC_MULT = FLASH.hc_mult
HC_DIM = FLASH.hc_dim
MIX_HC = FLASH.mix_hc
D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
LOCAL_H = FLASH.num_attention_heads
LOCAL_O_GROUPS = FLASH.o_groups
LOCAL_O_WIDTH = LOCAL_O_GROUPS * FLASH.o_lora_rank
O_GROUP_IN = FLASH.num_attention_heads * HEAD_DIM // FLASH.o_groups
O_LORA = FLASH.o_lora_rank
Q_LORA = FLASH.q_lora_rank
ROPE_DIM = FLASH.qk_rope_head_dim

SOFTMAX_SCALE = HEAD_DIM**-0.5

# tiling
QUERY_HEAD_TILE = max(32, LOCAL_H)


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "dspark_attention_core.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "dspark_attention_core.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _attend_dspark_native(
    query: pl.Tensor,
    cache: pl.Tensor,
    indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Gather packed KV into L1 and attend K64 tiles with BF16 probabilities."""
    ...


@pl.jit.inline(auto_scope=False)
def attend_dspark_window(
    query: pl.Tensor[[MOE_TOKENS, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[MOE_TOKENS, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    workers = pl.max(1, pl.min(num_tokens, 32))
    if LOCAL_H < 32:
        padded_query = pl.create_tensor([MOE_TOKENS, QUERY_HEAD_TILE * HEAD_DIM], dtype=pl.BF16)
        for row in pl.spmd(num_tokens, name_hint="dspark_pad_heads"):
            padded = pl.full([1, QUERY_HEAD_TILE * HEAD_DIM], dtype=pl.BF16, value=0.0)
            active_query = query[row : row + 1, :]
            padded[:, :LOCAL_H * HEAD_DIM] = active_query
            padded_query[row : row + 1, :] = padded
        with pl.spmd(workers, name_hint="dspark_online_attention"):
            output = _attend_dspark_native(padded_query, cache, indices, sink, output, num_tokens, LOCAL_H)
    else:
        with pl.spmd(workers, name_hint="dspark_online_attention"):
            output = _attend_dspark_native(query, cache, indices, sink, output, num_tokens, LOCAL_H)
    return output


@pl.jit.inline(auto_scope=False)
def dspark_context_kv(
    kv_input_payload: pl.Tensor,
    kv_input_scales: pl.Tensor,
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    context_slots: pl.Tensor,
    window_cache: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project shared quantized context inputs through one layer's own weights."""
    tokens = pl.tensor.dim(context_slots, 0)
    projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    window_input_ready = pl.system.task_dummy(deps=[])
    projected = kv_proj(
        kv_input_payload, kv_input_scales,
        wkv, wkv_scale,
        projected, num_tokens, worker_limit=32, row_tile=80,
    )
    window_publish(
        projected, kv_norm_weight, rope_cos, rope_sin,
        context_slots, window_cache,
        num_tokens, window_input_ready,
    )
    return window_cache


@pl.jit
def dspark_context_kv_test(
    kv_input_payload: pl.Tensor[[CONTEXT_PAYLOAD_T_DYN, D], pl.FP8E4M3FN],
    kv_input_scales: pl.Tensor[[1, CONTEXT_SCALE_T_DYN], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[CONTEXT_T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CONTEXT_T_DYN, ROPE_DIM // 2], pl.FP32],
    context_slots: pl.Tensor[[CONTEXT_T_DYN], pl.INT64],
    window_cache: pl.InOut[pl.Tensor[[3, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Publish through a middle-layer view while retaining both guard layers."""
    kv_input_payload.bind_dynamic(0, CONTEXT_PAYLOAD_T_DYN)
    kv_input_scales.bind_dynamic(1, CONTEXT_SCALE_T_DYN)
    rope_cos.bind_dynamic(0, CONTEXT_T_DYN)
    rope_sin.bind_dynamic(0, CONTEXT_T_DYN)
    context_slots.bind_dynamic(0, CONTEXT_T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    dspark_context_kv(
        kv_input_payload, kv_input_scales, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, context_slots, window_cache[1], num_tokens,
    )
    return window_cache


@pl.jit.inline(auto_scope=False)
def dspark_attn(
    x_hc: pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32],
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[MOE_TOKENS], pl.INT64],
    window_indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    attention_pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
    attention_hidden: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run HC pre-mix, query attention, output projection, and HC post."""
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    normalized_attention = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    hc_head_norm(x_hc, incoming_pre_mix, attn_norm_weight, normalized_attention, worker_limit=64)
    inv_rms, partials = hc_project(x_hc, hc_attn_fn, rms_worker_limit=64)

    post_mix = pl.create_tensor([MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([MOE_TOKENS, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix)
    zero_bf16_padding(normalized_attention, num_tokens)
    proj_payload, proj_scales, proj_rows = quantize_projection_input(normalized_attention, num_tokens, row_tile=80)
    qr = pl.create_tensor([MOE_TOKENS, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([MOE_TOKENS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    q_proj_result = q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q,
        num_tokens, qa_worker_limit=32, qb_worker_limit=32,
        padded_rows=proj_rows,
        qa_row_tile=80, qb_row_tile=80,
    )
    kv_projection = pl.create_tensor([MOE_TOKENS, HEAD_DIM], dtype=pl.BF16)
    window_input_ready = pl.system.task_dummy(deps=[])
    kv_projection = kv_proj(
        proj_payload, proj_scales,
        wkv, wkv_scale,
        kv_projection, num_tokens, worker_limit=32, row_tile=80,
    )
    window_publish(
        kv_projection, kv_norm_weight, rope_cos, rope_sin,
        window_slots, window_cache,
        num_tokens, window_input_ready,
    )
    attended = pl.create_tensor([MOE_TOKENS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attended = attend_dspark_window(q, window_cache, window_indices, attn_sink, attended, num_tokens)

    o_proj_hc_post(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        x_hc, post_mix, residual_mix, attention_hidden, num_tokens,
        oa_row_tile=80, ob_row_tile=128,
    )

    return attention_hidden, attention_pre_mix, window_cache


@pl.jit(auto_scope=False)
def l2_dspark_attn(
    x_hc: pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32],
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[MOE_TOKENS], pl.INT64],
    window_indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    attention_pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
    attention_hidden: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Dispatch the production DSpark attention composition."""
    return dspark_attn(
        x_hc, incoming_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin, window_slots, window_indices, window_cache,
        attention_pre_mix, attention_hidden,
        num_tokens,
    )


def golden_context_kv_cache(
    payload: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    cache_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish one draft layer's context KV into its own MXFP8 SWA cache."""
    from quant import decode_e8m0, unpack_fp8_nz, unpack_mx_b_scale

    if weight.dtype == torch.float8_e4m3fn:
        weight = unpack_fp8_nz(weight)
    if slots.shape != payload.shape[:1]:
        raise ValueError("context slots must have one entry per token")
    if cache.shape[:-1] != cache_scale.shape[:-1] or cache.shape[-1] != cache_scale.shape[-1] * 32:
        raise ValueError("context cache payload and scale shapes disagree")
    active = slots >= 0
    rows = slots[active].long()
    cache_rows = cache.numel() // cache.shape[-1]
    if rows.numel() and (int(rows.max()) >= cache_rows or rows.unique().numel() != rows.numel()):
        raise ValueError("context slots must be in range and unique")
    updated_cache = cache.clone()
    updated_scale = cache_scale.clone()
    if rows.numel():
        activation = payload.float().unflatten(-1, (-1, 32))
        weight_groups = weight.float().unflatten(0, (-1, 32))
        partials = torch.einsum("...gk,gkn->...gn", activation, weight_groups)
        scales_a = decode_e8m0(activation_scale)
        scales_b = decode_e8m0(unpack_mx_b_scale(weight_scale))
        partials = partials * scales_a.unsqueeze(-1) * scales_b
        projected = partials.sum(dim=-2).to(torch.bfloat16)
        projected = golden_rms_norm(projected, norm_weight)
        rope_dim = cos.shape[-1] * 2
        rotated = golden_rope_interleave(projected[..., -rope_dim:], cos, sin)
        kv = torch.cat((projected[..., :-rope_dim], rotated), dim=-1)
        payload, scale = quantize_mxfp8_cache(kv[active])
        updated_cache.reshape(-1, cache.shape[-1])[rows] = payload
        updated_scale.reshape(-1, cache_scale.shape[-1])[rows] = scale
    return updated_cache, updated_scale


def golden_dspark_attn(tensors):
    """Compose QKV, shared-history attention, output projection and HC references."""
    import torch
    from hc import golden_hc_head_norm, golden_hc_mixes, golden_hc_post
    from o_proj import golden_o_proj
    from qkv_proj_rope import golden_qkv_proj_rope
    from quant import quantize_mxfp8_cache
    from sparse_attention import golden_gather_sparse_attention
    from quant import _publish_quantized_cache

    active = int(tensors["num_tokens"])
    pre, post, residual = golden_hc_mixes(
        tensors["x_hc"], tensors["hc_attn_fn"], tensors["hc_attn_scale"], tensors["hc_attn_base"],
    )
    tensors["attention_pre_mix"].copy_(pre)
    normalized = golden_hc_head_norm(tensors["x_hc"], tensors["incoming_pre_mix"], tensors["attn_norm_weight"])
    payload, factors = unpack_quantized_cache(tensors["window_cache"], False)
    inputs = dict(tensors, x=normalized, window_cache=payload, window_cache_scale=factors)
    for name in ("x", "rope_cos", "rope_sin", "window_slots", "window_indices"):
        inputs[name] = inputs[name][:active]
    projection = torch.zeros((MOE_TOKENS, D), dtype=torch.bfloat16)
    packed_cache = tensors["window_cache"]
    if active:
        query, window_kv, _ = golden_qkv_proj_rope(
            inputs["x"], inputs["wq_a"], inputs["wq_a_scale"], inputs["q_norm_weight"],
            inputs["wq_b"], inputs["wq_b_scale"], inputs["wkv"], inputs["wkv_scale"], inputs["kv_norm_weight"],
            inputs["rope_cos"], inputs["rope_sin"], sequential_query=True,
        )
        cache, scale = _publish_quantized_cache(
            inputs["window_cache"], inputs["window_cache_scale"],
            window_kv, inputs["window_slots"], quantize_mxfp8_cache,
        )
        packed_cache = pack_quantized_cache(cache, scale)
        indices = inputs["window_indices"]
        attended = golden_gather_sparse_attention(
            query, indices, packed_cache, indices, packed_cache, inputs["attn_sink"],
            compressed_keys=0, key_tile=64,
        )
        projected = golden_o_proj(
            attended, inputs["rope_cos"], inputs["rope_sin"], inputs["wo_a"], inputs["wo_b"], inputs["wo_b_scale"],
            wo_a_scale=inputs["wo_a_scale"], sequential_projection=True,
        )
        projection[:active].copy_(projected)
    tensors["attention_hidden"].copy_(golden_hc_post(projection, tensors["x_hc"], post, residual))
    tensors["window_cache"].copy_(packed_cache)


def build_attention_specs(seed: int = 43):
    """Build five shared-history queries, HC inputs and runtime specifications."""
    import torch
    from golden import ScalarSpec, TensorSpec
    import inspect
    from hc import make_hc_inputs
    import math
    from quant import pack_fp8_nz, pack_mx_b_scale
    from quant import quantize_mxfp8_upward_reference, quantize_grouped_mxfp8_weight
    from quant import quantize_mxfp8_cache

    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen)

    def weight(k, n):
        payload, scales = quantize_mxfp8_upward_reference(rand(n, k) / math.sqrt(k))
        packed_scales = pack_mx_b_scale(scales.T.contiguous()).view(torch.float8_e8m0fnu)
        return payload.T.contiguous(), packed_scales

    qa, qas = weight(D, Q_LORA)
    qb, qbs = weight(Q_LORA, LOCAL_H * HEAD_DIM)
    kv, kvs = weight(D, HEAD_DIM)
    ob, obs = weight(LOCAL_O_WIDTH, D)
    qa = pack_fp8_nz(qa)
    qb = pack_fp8_nz(qb)
    kv = pack_fp8_nz(kv)
    ob = pack_fp8_nz(ob)
    cache, cache_scale = quantize_mxfp8_cache(rand(DSPARK_QUERY_WIDTH + 1, 128, 1, HEAD_DIM).to(torch.bfloat16))
    hc_inputs = make_hc_inputs(MOE_TOKENS, seed, "random")
    hc_inputs["x_hc"][DSPARK_QUERY_WIDTH:] = 0
    values = dict(
        x_hc=hc_inputs["x_hc"], incoming_pre_mix=hc_inputs["pre_mix"],
        hc_attn_fn=hc_inputs["hc_attn_fn"], hc_attn_scale=hc_inputs["hc_attn_scale"],
        hc_attn_base=hc_inputs["hc_attn_base"], attn_norm_weight=hc_inputs["attn_norm_weight"],
        wq_a=qa, wq_a_scale=qas, q_norm_weight=(rand(Q_LORA) * 0.1 + 1).to(torch.bfloat16),
        wq_b=qb, wq_b_scale=qbs,
        wkv=kv, wkv_scale=kvs, kv_norm_weight=(rand(HEAD_DIM) * 0.1 + 1).to(torch.bfloat16),
        attn_sink=rand(LOCAL_H) * 2,
    )
    wo_a = (rand(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16)
    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(wo_a)
    values.update(wo_b=ob, wo_b_scale=obs)
    positions = torch.arange(2, 2 + DSPARK_QUERY_WIDTH)
    frequencies = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
    angles = positions.float()[:, None] * frequencies
    values["rope_cos"] = angles.cos()
    values["rope_sin"] = angles.sin()
    values["window_slots"] = 128 + positions
    indices = torch.full((DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH), -1, dtype=torch.int32)
    # Contiguous panels in both lanes, followed by a partial panel.
    visible_keys = FLASH.sliding_window + DSPARK_QUERY_WIDTH
    indices[0, :visible_keys] = torch.arange(128, 128 + visible_keys, dtype=torch.int32)
    # Irregular physical slots across decode, gather, K64 and lane boundaries.
    boundary_columns = torch.tensor([0, 15, 16, 31, 32, 63, 64, 95, 96, 127, 128, 191])
    boundary_slots = torch.tensor([127, 128, 255, 256, 383, 384, 511, 512, 639, 640, 767, 0], dtype=torch.int32)
    indices[1, boundary_columns] = boundary_slots
    indices[2, boundary_columns] = boundary_slots.flip(0)
    indices[2, 95] = indices[2, 96]
    # Only lane one contributes; the last query remains fully masked.
    indices[3, 96:128] = torch.arange(512, 544, dtype=torch.int32)
    indices[3, 191] = 767
    values["window_indices"] = indices
    for name in ("rope_cos", "rope_sin", "window_slots", "window_indices"):
        current = values[name]
        fill = -1 if name in ("window_slots", "window_indices") else 0
        padded = torch.full((MOE_TOKENS, *current.shape[1:]), fill, dtype=current.dtype)
        padded[:DSPARK_QUERY_WIDTH] = current
        values[name] = padded
    values["window_cache"] = pack_quantized_cache(cache, cache_scale)
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
    specs.append(TensorSpec("attention_pre_mix", [MOE_TOKENS, HC_MULT], torch.float32))
    specs.append(TensorSpec("attention_hidden", [MOE_TOKENS, HC_MULT, D], torch.float32))
    specs.append(ScalarSpec("num_tokens", torch.int32, DSPARK_QUERY_WIDTH, compile_runtime=True))
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in inspect.signature(l2_dspark_attn._func).parameters]


def _compare_context_cache(
    _actual,
    _expected,
    *,
    actual_outputs,
    expected_outputs,
    inputs,
    **_kwargs,
):
    from quant import dequantize_mxfp8_cache

    slots = inputs["context_slots"][:int(inputs["num_tokens"])]
    layer_rows = actual_outputs["window_cache"].shape[1] * 128
    active = slots[slots >= 0].long().unique() + layer_rows
    actual_payload, actual_scale = unpack_quantized_cache(actual_outputs["window_cache"], False)
    expected_payload, expected_scale = unpack_quantized_cache(expected_outputs["window_cache"], False)
    actual = dequantize_mxfp8_cache(actual_payload, actual_scale).reshape(-1, HEAD_DIM)
    expected = dequantize_mxfp8_cache(expected_payload, expected_scale).reshape(-1, HEAD_DIM)
    error_norm = (actual[active].float() - expected[active].float()).norm()
    expected_norm = expected[active].float().norm().clamp_min(1e-12)
    relative_l2 = error_norm / expected_norm
    inactive = torch.ones(actual.shape[0], dtype=torch.bool)
    inactive[active] = False
    untouched = all(
        torch.equal(
            actual_field.contiguous().view(torch.uint8).reshape(actual.shape[0], -1)[inactive],
            expected_field.contiguous().view(torch.uint8).reshape(actual.shape[0], -1)[inactive],
        )
        for actual_field, expected_field in ((actual_payload, expected_payload), (actual_scale, expected_scale))
    )
    passed = bool(torch.isfinite(relative_l2) and relative_l2 <= 0.05 and untouched)
    detail = f"active context KV relative L2={relative_l2.item():.6g}; inactive rows unchanged={untouched}"
    return passed, detail


def run_context_kv_case(*, device_id: int = 0, compile_only: bool = False, execution_mode: bool = False):
    """Validate context KV publication and untouched guard layers."""
    from golden import ScalarSpec, TensorSpec, run
    from quant import pack_fp8_nz, pack_mx_b_scale, pack_mx_a_scale, unpack_mx_a_scale
    from quant import _quantize_mxfp8_activation, encode_e8m0

    torch.manual_seed(42)
    tokens = 8
    main_x = torch.randn(tokens, D, dtype=torch.bfloat16) * 0.1
    padded_rows = (tokens + 79) // 80 * 80
    activation, activation_scale = _quantize_mxfp8_activation(main_x, scale_alg=0)
    input_payload = torch.zeros(padded_rows, D, dtype=torch.float8_e4m3fn)
    input_payload[:tokens] = activation.flatten(-2)
    scale_codes = torch.zeros(padded_rows, D // 32, dtype=torch.uint8)
    scale_codes[:tokens] = encode_e8m0(activation_scale)
    input_scales = pack_mx_a_scale(scale_codes).reshape(1, -1).view(torch.float8_e8m0fnu)
    weight = (torch.randn(D, HEAD_DIM) * 0.01).to(torch.float8_e4m3fn)
    weight = pack_fp8_nz(weight)
    weight_scale = pack_mx_b_scale(torch.full((D // 32, HEAD_DIM), 127, dtype=torch.uint8))
    weight_scale = weight_scale.view(torch.float8_e8m0fnu)
    cos = torch.ones(tokens, ROPE_DIM // 2)
    sin = torch.zeros_like(cos)
    sin[0].fill_(1.0)
    cos[0].zero_()
    cache = torch.empty(3, 2, 128, 1, HEAD_DIM)
    cache_scale = torch.empty((3, 2, 128, 1, HEAD_DIM // 32), dtype=torch.bfloat16)
    for layer in range(3):
        cache[layer].fill_((layer + 1) * 0.25)
        cache_scale[layer].fill_((layer + 1) * 4.0)
    cache = cache.to(torch.float8_e4m3fn)
    slots = torch.tensor([0, 1, 127, 128, 255, -1, -1, -1], dtype=torch.int64)

    def golden(values):
        cache_payload, cache_factors = unpack_quantized_cache(values["window_cache"][1], False)
        active = int(values["num_tokens"])
        codes = values["kv_input_scales"].view(torch.uint8).reshape(-1, D // 32)
        activation_codes = unpack_mx_a_scale(codes)[:active]
        payload, scale = golden_context_kv_cache(
            values["kv_input_payload"][:active], activation_codes,
            values["wkv"], values["wkv_scale"], values["kv_norm_weight"],
            values["rope_cos"][:active], values["rope_sin"][:active],
            values["context_slots"][:active],
            cache_payload, cache_factors,
        )
        values["window_cache"][1].copy_(pack_quantized_cache(payload, scale))

    packed_cache = pack_quantized_cache(cache, cache_scale)
    specs = [
        TensorSpec("kv_input_payload", list(input_payload.shape), torch.float8_e4m3fn, init_value=input_payload),
        TensorSpec("kv_input_scales", list(input_scales.shape), torch.float8_e8m0fnu, init_value=input_scales),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=weight),
        TensorSpec("wkv_scale", [D // 32, HEAD_DIM], torch.float8_e8m0fnu, init_value=weight_scale),
        TensorSpec("kv_norm_weight", [HEAD_DIM], torch.bfloat16, init_value=1.0),
        TensorSpec("rope_cos", [tokens, ROPE_DIM // 2], torch.float32, init_value=cos),
        TensorSpec("rope_sin", [tokens, ROPE_DIM // 2], torch.float32, init_value=sin),
        TensorSpec("context_slots", [tokens], torch.int64, init_value=slots),
        TensorSpec("window_cache", [3, 2, 128, 1, WINDOW_CACHE_BYTES], torch.uint8, init_value=packed_cache),
        ScalarSpec("num_tokens", torch.int32, tokens, compile_runtime=True),
    ]
    compare = {"window_cache": _compare_context_cache}
    result = run(
        fn=dspark_context_kv_test,
        specs=specs,
        golden_fn=golden,
        compare_fn=compare,
        config={"backend_type": "a5", "device_id": device_id,
                "execution_mode": execution_mode},
        compile_only=compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark non-causal SWA validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--context-kv", action="store_true", help="validate context KV publication only")
    args = parser.parse_args()
    if args.context_kv:
        run_context_kv_case(
            device_id=args.device, compile_only=args.compile_only, execution_mode=args.execution_mode,
        )
        raise SystemExit(0)
    if args.execution_mode:
        parser.error("DSpark native attention requires A5 device execution; simulator support is unavailable")
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run

    result = run(
        fn=l2_dspark_attn,
        specs=build_attention_specs(),
        golden_fn=golden_dspark_attn,
        compare_fn={
            "attention_hidden": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
            "window_cache": packed_cache_compare(
                mapped_pool_ratio_allclose(
                    "window_slots", pool_name="window cache", mapping_shape=(MOE_TOKENS,),
                    block_size=128, leading_rank_axis=False,
                    atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.01,
                ),
                mapped_pool_ratio_allclose(
                    "window_slots", pool_name="window cache scale", mapping_shape=(MOE_TOKENS,),
                    block_size=128, leading_rank_axis=False,
                    atol=0.0, rtol=0.0, max_error_ratio=0.0,
                ),
                fp4=False,
            ),
        },
        config={"backend_type": args.backend_type, "device_id": args.device,
                "execution_mode": args.execution_mode},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
