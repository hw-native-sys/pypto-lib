# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch reference operations for the DeepSeek-V4.1-Flash text backbone."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.quantization import mxfp8_linear


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = FLASH.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)


def rope_interleave(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply adjacent-pair rotary embedding to the final dimension."""
    value = x.float().unflatten(-1, (-1, 2))
    real, imag = value.unbind(dim=-1)
    if inverse:
        sin = -sin
    while cos.ndim < real.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated = torch.stack((real * cos - imag * sin, imag * cos + real * sin), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def qkv_proj_rope(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized query latent and shared KV, then rotate their RoPE tails."""
    qr = rms_norm(mxfp8_linear(x, wq_a, wq_a_scale), q_norm_weight)
    head_dim = wkv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim
    q = mxfp8_linear(qr, wq_b, wq_b_scale).unflatten(-1, (num_heads, head_dim))
    kv = rms_norm(mxfp8_linear(x, wkv, wkv_scale), kv_norm_weight)
    rd = cos.shape[-1] * 2
    q = torch.cat((q[..., :-rd], rope_interleave(q[..., -rd:], cos, sin)), dim=-1)
    kv = torch.cat((kv[..., :-rd], rope_interleave(kv[..., -rd:], cos, sin)), dim=-1)
    return q, kv, qr


def publish_cache(cache: torch.Tensor, values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Write valid flattened physical rows into a paged cache in place."""
    flat = cache.flatten(0, 1)
    valid = slots >= 0
    flat[slots[valid].to(torch.long)] = values[valid].reshape_as(flat[slots[valid].to(torch.long)])
    return cache


def compressor_ratio1(
    x: torch.Tensor,
    wkv: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Reference the ratio-1 compressor: projection followed by RMSNorm."""
    projected = torch.matmul(x.float(), wkv.float()).to(x.dtype)
    return rms_norm(projected, norm_weight)


def compressor_ratio2(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    norm_weight: torch.Tensor,
    prior_kv: torch.Tensor | None = None,
    prior_score: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool pairs and return the incomplete FP32 KV/score tail as recurrent state."""
    kv = torch.matmul(x.float(), wkv.float())
    score = torch.matmul(x.float(), wgate.float())
    if prior_kv is not None:
        kv = torch.cat((prior_kv.float(), kv), dim=-2)
        score = torch.cat((prior_score.float(), score), dim=-2)
    complete = kv.shape[-2] // 2 * 2
    pooled_kv = kv[..., :complete, :].unflatten(-2, (-1, 2))
    pooled_score = score[..., :complete, :].unflatten(-2, (-1, 2))
    pooled = (pooled_kv * pooled_score.softmax(dim=-2)).sum(dim=-2)
    pooled = rms_norm(pooled.to(x.dtype), norm_weight)
    return pooled, kv[..., complete:, :], score[..., complete:, :]


def compressor_ratio2_paged(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    position_ids: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    state_block_table: torch.Tensor,
    state_cache: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ratio-2 compression with request-owned FP32 ring state.

    Query boundaries count valid packed tokens; any tensor tail is padding.
    Negative/out-of-range request or block indices suppress state access and
    publication. Valid chunks contain consecutive absolute positions.
    """
    kv = torch.matmul(x.float(), wkv.float())
    score = torch.matmul(x.float(), wgate.float())
    latent = torch.zeros_like(kv, dtype=x.dtype)
    publish = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    blocks, capacity, width = state_cache.shape
    if capacity < 1 or width != 2 * kv.shape[-1] or state_cache.dtype != torch.float32:
        raise ValueError("state_cache must be FP32 [blocks, positive capacity, 2 * head_dim]")
    head_dim = kv.shape[-1]
    for request in range(state_block_table.shape[0]):
        start = int(query_start_loc[request])
        end = min(int(query_start_loc[request + 1]), x.shape[0])
        block = int(state_block_table[request, 0])
        if start >= end or not 0 <= block < blocks:
            continue
        for token in range(start, end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) != request or position < 0:
                continue
            if position % 2:
                if token > start:
                    previous_kv, previous_score = kv[token - 1], score[token - 1]
                else:
                    previous = state_cache[block, (position - 1) % capacity]
                    previous_kv, previous_score = previous[:head_dim], previous[head_dim:]
                pair_kv = torch.stack((previous_kv, kv[token]))
                pair_score = torch.stack((previous_score, score[token]))
                pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
                latent[token] = rms_norm(pooled.to(x.dtype), norm_weight)
                publish[token] = True
        for token in range(max(start, end - capacity), end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) == request and position >= 0:
                state_cache[block, position % capacity, :head_dim] = kv[token]
                state_cache[block, position % capacity, head_dim:] = score[token]
    return latent, publish


def publish_index_key(
    latent: torch.Tensor,
    publish_mask: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index_slots: torch.Tensor,
    index_cache: torch.Tensor,
) -> torch.Tensor:
    """Project completed compressor rows and publish their paged index keys."""
    keys = index_key(latent, wk, norm_weight, cos, sin)
    slots = index_slots.masked_fill(~publish_mask.to(torch.bool), -1)
    return publish_cache(index_cache, keys, slots)


def select_candidate_blocks(
    logits: torch.Tensor,
    compressed_lens: torch.Tensor | int,
    topk_blocks: int = FLASH.candidate_topk_blocks,
    block_size: int = FLASH.candidate_block_size,
) -> torch.Tensor:
    """Return the first-level candidate mask used by late index-source layers."""
    width = logits.shape[-1]
    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last = (torch.as_tensor(compressed_lens, device=logits.device) - 1) // block_size
    while last.ndim < scores.ndim:
        last = last.unsqueeze(-1)
    block_ids = torch.arange(scores.shape[-1], device=logits.device)
    scores = scores.masked_fill(block_ids == last, torch.inf)
    top = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    index_k: torch.Tensor,
    wq_b: torch.Tensor,
    weights_proj: torch.Tensor,
    compressed_lens: torch.Tensor,
    candidates: torch.Tensor | None = None,
    offset: int = 0,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score compressed positions and return causal top-k indices in position order."""
    index_dim = index_k.shape[-1]
    index_heads = weights_proj.shape[-1]
    q = torch.matmul(qr, wq_b).unflatten(-1, (index_heads, index_dim))
    if cos is not None and sin is not None:
        rd = cos.shape[-1] * 2
        q = torch.cat((q[..., :-rd], rope_interleave(q[..., -rd:], cos, sin)), dim=-1)
    weights = torch.matmul(x, weights_proj)
    weights = weights * index_dim**-0.5 * index_heads**-0.5
    score = torch.einsum("...qhd,...kd->...qhk", q.float(), index_k.float()).relu()
    score = (score * weights.float().unsqueeze(-1)).sum(dim=-2)
    positions = torch.arange(index_k.shape[-2], device=x.device)
    lens = compressed_lens
    while lens.ndim < score.ndim:
        lens = lens.unsqueeze(-1)
    score = score.masked_fill(positions >= lens, -torch.inf)
    if candidates is not None:
        score = score.masked_fill(~candidates, -torch.inf)
    count = min(FLASH.index_topk, score.shape[-1])
    indices = score.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected_scores = score.gather(-1, indices)
    valid = torch.isfinite(selected_scores) & (indices < lens)
    return torch.where(valid, indices + offset, -1).to(torch.int32)


def paged_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    request_ids: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    compressed_lens: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    weights_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    candidates: torch.Tensor | None = None,
    topk: int = FLASH.index_topk,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score request-local paged index keys and return scores plus physical rows."""
    if compressed_lens.ndim != 1 or compressed_lens.shape[0] != x.shape[0]:
        raise ValueError("compressed_lens must contain one causal length per query token")
    max_len = int(compressed_lens.max().item()) if compressed_lens.numel() else 0
    scores = x.new_full((x.shape[0], max_len), -torch.inf, dtype=torch.float32)
    physical = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    if max_len == 0:
        return scores, physical

    positions = torch.arange(max_len, device=x.device)
    request_rows = request_ids.to(torch.long).unsqueeze(-1)
    blocks = torch.div(positions, 128, rounding_mode="floor")
    offsets = positions.remainder(128)
    physical_rows = block_table[request_rows, blocks] * 128 + offsets
    flat_keys = index_cache.flatten(0, 1).squeeze(-2)
    keys = flat_keys[physical_rows.to(torch.long)]

    index_dim = index_cache.shape[-1]
    index_heads = weights_proj.shape[-1]
    q = mxfp8_linear(qr, wq_b, wq_b_scale).unflatten(-1, (index_heads, index_dim))
    rd = cos.shape[-1] * 2
    q = torch.cat((q[..., :-rd], rope_interleave(q[..., -rd:], cos, sin)), dim=-1)
    weights = torch.matmul(x.float(), weights_proj.float()).to(x.dtype)
    weights = (weights.float() * (index_dim**-0.5 * index_heads**-0.5)).to(x.dtype)
    dots = torch.einsum("thd,tkd->thk", q.float(), keys.float()).to(q.dtype)
    weighted = (dots.float().relu() * weights.float().unsqueeze(-1)).to(q.dtype)
    scores = weighted.float().sum(dim=-2).to(q.dtype).float()
    valid_positions = positions.unsqueeze(0) < compressed_lens.to(torch.long).unsqueeze(-1)
    scores = scores.masked_fill(~valid_positions, -torch.inf)
    if candidates is not None:
        scores = scores.masked_fill(~candidates[..., :max_len].to(torch.bool), -torch.inf)

    count = min(topk, max_len)
    logical = scores.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected_scores = scores.gather(-1, logical)
    selected_rows = physical_rows.gather(-1, logical)
    for token in range(x.shape[0]):
        valid_rows = selected_rows[token, torch.isfinite(selected_scores[token])].to(torch.int32)
        physical[token, :valid_rows.numel()] = valid_rows
    return scores, physical


def index_key(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Project an unrotated compressor latent into a rotated index key."""
    projected = torch.matmul(latent.float(), wk.float()).to(latent.dtype)
    key = rms_norm(projected, norm_weight)
    rd = cos.shape[-1] * 2
    return torch.cat((key[..., :-rd], rope_interleave(key[..., -rd:], cos, sin)), dim=-1)


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Gather one shared latent KV stream and evaluate sink-augmented sparse attention."""
    safe = indices.clamp_min(0).to(torch.long)
    batch = torch.arange(q.shape[0], device=q.device)
    while batch.ndim < safe.ndim:
        batch = batch.unsqueeze(-1)
    selected = kv[batch, safe]
    logits = torch.einsum("bqhd,bqkd->bqhk", q.float(), selected.float()) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(indices.unsqueeze(-2) < 0, -torch.inf)
    sink_logits = sink.float().view(1, 1, -1, 1)
    denominator = torch.logsumexp(torch.cat((logits, sink_logits.expand_as(logits[..., :1])), dim=-1), dim=-1)
    weights = torch.exp(logits - denominator.unsqueeze(-1))
    return torch.einsum("bqhk,bqkd->bqhd", weights, selected.float()).to(q.dtype)


def sparse_attention_stats(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return online-softmax max, exponential sum, and weighted-value numerator."""
    safe = indices.clamp_min(0).to(torch.long)
    batch = torch.arange(q.shape[0], device=q.device)
    while batch.ndim < safe.ndim:
        batch = batch.unsqueeze(-1)
    selected = kv[batch, safe]
    logits = torch.einsum("bqhd,bqkd->bqhk", q.float(), selected.float()) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(indices.unsqueeze(-2) < 0, -torch.inf)
    maximum = logits.amax(dim=-1)
    finite_maximum = maximum.masked_fill(~torch.isfinite(maximum), 0.0)
    exponentials = torch.exp(logits - finite_maximum.unsqueeze(-1))
    exponentials = exponentials.masked_fill(~torch.isfinite(logits), 0.0)
    denominator = exponentials.sum(dim=-1)
    numerator = torch.einsum("bqhk,bqkd->bqhd", exponentials, selected.float())
    return maximum, denominator, numerator


def paged_sparse_attention_stats(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token-major online-softmax statistics from physical paged rows."""
    flat_cache = cache.flatten(0, 1).squeeze(-2)
    safe = indices.clamp_min(0).to(torch.long)
    selected = flat_cache[safe]
    logits = torch.einsum("thd,tkd->thk", q.float(), selected.float()) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(indices.unsqueeze(-2) < 0, -torch.inf)
    maximum = logits.amax(dim=-1)
    finite_maximum = maximum.masked_fill(~torch.isfinite(maximum), 0.0)
    exponentials = torch.exp(logits - finite_maximum.unsqueeze(-1))
    exponentials = exponentials.masked_fill(~torch.isfinite(logits), 0.0)
    denominator = exponentials.sum(dim=-1)
    numerator = torch.einsum("thk,tkd->thd", exponentials, selected.float())
    return maximum, denominator, numerator


def paged_sparse_attention(
    q: torch.Tensor,
    window_cache: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_indices: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Merge token-major paged window and compressed attention with one sink."""
    parts = (
        paged_sparse_attention_stats(q, window_cache, window_indices),
        paged_sparse_attention_stats(q, compressed_cache, compressed_indices),
    )
    return merge_attention_stats(parts, sink).to(q.dtype)


def merge_attention_stats(
    parts: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    sink: torch.Tensor,
) -> torch.Tensor:
    """Merge sparse sources with one attention sink in a common softmax denominator."""
    maxima = [part[0] for part in parts]
    sink_value = sink.float().view(*([1] * (maxima[0].ndim - 1)), -1)
    global_maximum = torch.maximum(torch.stack(maxima).amax(dim=0), sink_value)
    denominator = torch.exp(sink_value - global_maximum)
    numerator = torch.zeros_like(parts[0][2])
    for maximum, local_sum, local_numerator in parts:
        scale = torch.exp(maximum - global_maximum)
        scale = scale.masked_fill(~torch.isfinite(maximum), 0.0)
        denominator = denominator + local_sum * scale
        numerator = numerator + local_numerator * scale.unsqueeze(-1)
    return numerator / denominator.unsqueeze(-1)


def attention(
    q: torch.Tensor,
    window_kv: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_kv: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    sink: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
) -> torch.Tensor:
    """Evaluate both KV sources, inverse RoPE, and grouped output projection."""
    parts = [sparse_attention_stats(q, window_kv, window_indices)]
    if compressed_kv is not None and compressed_indices is not None:
        parts.append(sparse_attention_stats(q, compressed_kv, compressed_indices))
    output = merge_attention_stats(tuple(parts), sink).to(q.dtype)
    rd = cos.shape[-1] * 2
    output = torch.cat((output[..., :-rd], rope_interleave(output[..., -rd:], cos, sin, True)), dim=-1)
    grouped = output.flatten(-2).unflatten(-1, (FLASH.o_groups, -1))
    latent = torch.einsum("...gd,grd->...gr", grouped, wo_a)
    return torch.matmul(latent.flatten(-2), wo_b)


def gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sqrt-softplus routing, with bias affecting selection only."""
    scores = F.softplus(F.linear(x.float(), weight.float()) / FLASH.gate_temperature).sqrt()
    indices = (scores + correction_bias.float()).topk(FLASH.num_experts_per_tok, dim=-1).indices
    weights = scores.gather(-1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * FLASH.routed_scaling_factor, indices.to(torch.int32)


def expert(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    route_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate one clamp-stabilized SwiGLU expert."""
    gate_value = F.linear(x, w1).float().clamp(max=FLASH.swiglu_limit)
    up = F.linear(x, w3).float().clamp(-FLASH.swiglu_limit, FLASH.swiglu_limit)
    hidden = F.silu(gate_value) * up
    if route_weight is not None:
        hidden = hidden * route_weight
    return F.linear(hidden.to(x.dtype), w2)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse four Hyper-Connection streams into one sublayer input."""
    return (x.float() * pre_mix.float().unsqueeze(-1)).sum(dim=-2).to(x.dtype)


def identity_pre_mix(x_hc: torch.Tensor) -> torch.Tensor:
    """Create the initial one-hot HC mix that selects residual lane zero."""
    pre_mix = torch.zeros(*x_hc.shape[:-1], dtype=torch.float32, device=x_hc.device)
    pre_mix[..., 0] = 1.0
    return pre_mix


def hc_mixes(
    x: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = FLASH.hc_sinkhorn_iters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one HC stream into pre, post, and doubly-stochastic residual mixes."""
    flat = x.flatten(-2).float()
    inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + FLASH.rms_norm_eps)
    mixes = F.linear(flat, function.float()) * inverse_rms
    hc = FLASH.hc_mult
    pre_raw, post_raw, residual_raw = mixes.split((hc, hc, hc * hc), dim=-1)
    pre = torch.sigmoid(pre_raw * scale[0] + base[:hc]) + FLASH.hc_eps
    post = 2 * torch.sigmoid(post_raw * scale[1] + base[hc : 2 * hc])
    residual_base = base[2 * hc :].unflatten(-1, (hc, hc))
    residual = (residual_raw.unflatten(-1, (hc, hc)) * scale[2] + residual_base).softmax(dim=-1)
    residual = residual + FLASH.hc_eps
    residual = residual / (residual.sum(dim=-2, keepdim=True) + FLASH.hc_eps)
    for _ in range(iterations - 1):
        residual = residual / (residual.sum(dim=-1, keepdim=True) + FLASH.hc_eps)
        residual = residual / (residual.sum(dim=-2, keepdim=True) + FLASH.hc_eps)
    return pre, post, residual


def hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams."""
    update = post_mix.unsqueeze(-1) * sublayer.unsqueeze(-2)
    skip = (residual_mix.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    return (update.float() + skip.float()).to(sublayer.dtype)


def hc_head(
    x: torch.Tensor,
    pre_mix: torch.Tensor,
) -> torch.Tensor:
    """Collapse the final HC streams with the last layer's delayed pre-mix."""
    return hc_pre(x, pre_mix).to(torch.bfloat16)


# Independent CED references retain checkpoint values and select activation precision.

PREFILL_HC = FLASH.hc_mult


PREFILL_EPS = FLASH.hc_eps


PREFILL_ITERATIONS = FLASH.hc_sinkhorn_iters


PREFILL_TOPK = FLASH.num_experts_per_tok


PREFILL_ROUTE_SCALE = FLASH.routed_scaling_factor


PREFILL_LIMIT = FLASH.swiglu_limit


def _prefill_operand(value):
    """Round an operand to model-state precision before widening accumulation."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError("FP32 contraction operands must be real floating-point tensors")
    return value.to(torch.float32).to(torch.float64)


def prefill_round_activation(x, precision="fp32"):
    """Store BF16 values in FP32 transport buffers only in official mode."""
    if precision not in ("fp32", "official"):
        raise ValueError("precision must be 'fp32' or 'official'")
    return x.bfloat16().float() if precision == "official" else x


def prefill_quantize_dequantize(x, kind):
    """Independent released FP8/FP4 roundtrip, including BF16 boundaries.

    FP4 ties select the even code by distance, independently of the device's
    midpoint comparisons. Power-of-two scales use frexp, not device bit logic.
    """
    if kind not in ("mxfp8", "index_fp4", "kv_fp4"):
        raise ValueError("unknown activation/cache quantization kind")
    group_size = 16 if kind == "kv_fp4" else 32
    if x.shape[-1] % group_size:
        raise ValueError("quantized width must be a multiple of its group size")
    value = x.bfloat16().float().unflatten(-1, (-1, group_size))
    maximum = value.abs().amax(-1, keepdim=True)
    if kind == "kv_fp4":
        scale = (maximum.clamp_min(6 * 2.0**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        floor, inverse = (1e-4, 1 / 448) if kind == "mxfp8" else (6 * 2.0**-126, 1 / 6)
        fraction, exponent = torch.frexp(maximum.clamp_min(floor) * inverse)
        scale = torch.ldexp(torch.ones_like(fraction), exponent - (fraction == 0.5).int())
    normalized = value / scale
    if kind == "mxfp8":
        quantized = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    else:
        magnitude = normalized.abs().clamp(max=6)
        levels = magnitude.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        best = torch.zeros_like(magnitude, dtype=torch.long)
        distance = magnitude
        # Even codes precede odd codes, so equal distances implement RN-even.
        for code in (2, 4, 6, 1, 3, 5, 7):
            candidate = (magnitude - levels[code]).abs()
            closer = candidate < distance
            best = torch.where(closer, code, best)
            distance = torch.where(closer, candidate, distance)
        quantized = torch.copysign(levels[best], normalized)
    return (quantized * scale).flatten(-2).bfloat16().float()


def prefill_linear(x, weight, bias=None, *, precision="fp32", format="fp32"):
    """Contract output-major weights with explicit released matrix roles.

    Quantized roles use FP8 activations and sequential FP32 group-32 partials.
    Decoded weights retain their exact power-of-two scales. Dense FP32 roles
    (HC, router, ratio-two compressor and head) never narrow their outputs.
    """
    if format not in ("fp32", "bf16", "mxfp8", "mxfp4"):
        raise ValueError("unknown checkpoint linear format")
    if precision not in ("fp32", "official"):
        raise ValueError("precision must be 'fp32' or 'official'")
    if precision == "official" and format in ("mxfp8", "mxfp4"):
        if bias is not None or x.shape[-1] % 32:
            raise ValueError("official quantized linears require no bias and a group-32 width")
        quantized = prefill_quantize_dequantize(x, "mxfp8")
        output = x.new_zeros((*x.shape[:-1], weight.shape[0]), dtype=torch.float32)
        for start in range(0, x.shape[-1], 32):
            partial = F.linear(
                _prefill_operand(quantized[..., start:start + 32]), _prefill_operand(weight[..., start:start + 32])
            ).float()
            output = output + partial
        return output.bfloat16().float()
    if precision == "official" and format == "bf16":
        x, weight = x.bfloat16().float(), weight.bfloat16().float()
    bias = None if bias is None else _prefill_operand(bias)
    output = F.linear(_prefill_operand(x), _prefill_operand(weight), bias).float()
    return output.bfloat16().float() if precision == "official" and format == "bf16" else output


def prefill_rms_norm(x, weight, *, precision="fp32"):
    """Preserve the released input/output activation precision around FP32 norm."""
    return prefill_round_activation(rms_norm(prefill_round_activation(x, precision), weight), precision)


def prefill_matmul(left, right):
    """Matrix/vector/batched contraction with FP32 operands and result."""
    return torch.matmul(_prefill_operand(left), _prefill_operand(right)).float()


def prefill_einsum(equation, *operands):
    """Evaluate an explicit Einstein contraction with one final FP32 rounding."""
    return torch.einsum(equation, *(_prefill_operand(value) for value in operands)).float()


def prefill_hc_mixes(x, function, scale, base, *, norm_eps=None):
    """Project HC logits in FP64, retaining FP32 RMS and coefficient math."""
    flat = x.flatten(-2).float()
    epsilon = FLASH.rms_norm_eps if norm_eps is None else norm_eps
    inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + epsilon)
    logits = prefill_linear(flat, function) * inverse_rms
    return prefill_hc_coefficients_reference(logits, scale.float(), base.float())


def prefill_hc_coefficients_reference(logits, scale, base):
    """Independent Torch sigmoid and Sinkhorn coefficient transform."""
    pre = torch.sigmoid(logits[..., :PREFILL_HC] * scale[0] + base[:PREFILL_HC]) + PREFILL_EPS
    post = 2 * torch.sigmoid(logits[..., PREFILL_HC:2 * PREFILL_HC] * scale[1] + base[PREFILL_HC:2 * PREFILL_HC])
    residual = (logits[..., 2 * PREFILL_HC:] * scale[2] + base[2 * PREFILL_HC:]).unflatten(-1, (PREFILL_HC, PREFILL_HC))
    residual = residual.softmax(-1) + PREFILL_EPS
    residual = residual / (residual.sum(-2, keepdim=True) + PREFILL_EPS)
    for _ in range(PREFILL_ITERATIONS - 1):
        residual = residual / (residual.sum(-1, keepdim=True) + PREFILL_EPS)
        residual = residual / (residual.sum(-2, keepdim=True) + PREFILL_EPS)
    return pre, post, residual


def prefill_gate_reference(logits, bias):
    """Independent Torch router math; correction bias affects selection only."""
    scores = F.softplus(logits.float()).sqrt()
    indices = torch.argsort(-(scores + bias.float()), dim=-1, stable=True)[..., :PREFILL_TOPK]
    selected = scores.gather(-1, indices)
    return indices, selected / (selected.sum(-1, keepdim=True) + 1e-20) * PREFILL_ROUTE_SCALE


def prefill_expert_reference(x, matrices, route_weights=None, *, precision="fp32", formats=None):
    """Apply routing weights before the released BF16 hidden boundary and W2."""
    w1, w3, w2 = matrices
    formats = formats or ("fp32",) * 3
    gate = prefill_linear(x, w1, precision=precision, format=formats[0]).clamp(max=PREFILL_LIMIT)
    up = prefill_linear(x, w3, precision=precision, format=formats[1]).clamp(-PREFILL_LIMIT, PREFILL_LIMIT)
    hidden = F.silu(gate) * up
    if route_weights is not None:
        hidden = hidden * route_weights[:, None]
    hidden = prefill_round_activation(hidden, precision)
    return prefill_linear(hidden, w2, precision=precision, format=formats[2])


def prefill_moe_reference(x, checkpoint, layer_id, *, precision="fp32"):
    """Independent unsharded MoE with FP32 expert accumulation in both modes."""
    stem = f"layers.{layer_id}.ffn."
    logits = prefill_linear(x, checkpoint.tensor(stem + "gate.weight"))
    indices, weights = prefill_gate_reference(logits, checkpoint.tensor(stem + "gate.bias"))
    output = torch.zeros_like(x)
    for expert in sorted(indices.unique().tolist()):
        rows, slots = torch.where(indices == expert)
        formats = None if precision == "fp32" else tuple(
            checkpoint.linear_format(stem + f"experts.{expert}." + name) for name in ("w1", "w3", "w2")
        )
        value = prefill_expert_reference(
            x[rows], checkpoint.expert(layer_id, expert), weights[rows, slots],
            precision=precision, formats=formats,
        )
        output.index_add_(0, rows, value)
    formats = None if precision == "fp32" else tuple(
        checkpoint.linear_format(stem + "shared_experts." + name) for name in ("w1", "w3", "w2")
    )
    output = output + prefill_expert_reference(
        x, checkpoint.shared_expert(layer_id), precision=precision, formats=formats
    )
    return {"output": prefill_round_activation(output, precision), "indices": indices, "weights": weights}


@dataclass(frozen=True)
class PrefillAttentionMetadata:
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    request_ids: torch.Tensor | None = None
    compressed_lens: torch.Tensor | None = None
    index_block_table: torch.Tensor | None = None
    compressed_slots: torch.Tensor | None = None
    compressed_rope_cos: torch.Tensor | None = None
    compressed_rope_sin: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    state_block_table: torch.Tensor | None = None
    attention_extents: torch.Tensor | None = None


@dataclass
class PrefillAttentionCaches:
    """Window/global/index cache rows use [blocks,128,dim], with no scales."""

    window: torch.Tensor
    compressed: torch.Tensor | None = None
    index: torch.Tensor | None = None
    state: torch.Tensor | None = None

    def clone(self):
        return PrefillAttentionCaches(
            *(
                value.clone() if value is not None else None
                for value in (self.window, self.compressed, self.index, self.state)
            )
        )


@dataclass
class PrefillAttentionResult:
    output: torch.Tensor
    caches: PrefillAttentionCaches
    topk_indices: torch.Tensor | None = None
    candidate_mask: torch.Tensor | None = None


def prefill_rotate(value, cosine, sine, *, inverse=False, precision="fp32"):
    """Rotate adjacent pairs in FP32, then restore the selected activation precision."""
    width = cosine.shape[-1] * 2
    return prefill_round_activation(torch.cat(
        (
            value[..., :-width],
            rope_interleave(value[..., -width:], cosine, sine, inverse=inverse),
        ),
        dim=-1,
    ), precision)


def prefill_publish(cache, values, slots):
    """Publish only caller-owned physical rows, preserving all other bytes."""
    if cache.dtype != torch.float32 or values.dtype != torch.float32:
        raise ValueError("FP32 publication cannot narrow values or caches")
    flat = cache.flatten(0, 1)
    valid = slots >= 0
    flat[slots[valid].long()] = values[valid].reshape_as(flat[slots[valid].long()])


def _prefill_online_attention(query, window, window_indices, compressed, compressed_indices, sink, extents):
    """Released online64 order, including holes and unrounded FP32 denominator."""
    query = query.bfloat16().float()
    output = torch.empty_like(query)
    window = window.flatten(0, 1).bfloat16().float()
    compressed = compressed.flatten(0, 1).bfloat16().float() if compressed is not None else None
    for token in range(query.shape[0]):
        window_width = int(extents[token, 0]) if extents is not None else window_indices.shape[1]
        global_width = int(extents[token, 1]) if extents is not None else (
            compressed_indices.shape[1] if compressed_indices is not None else 0
        )
        indices = [window_indices[token, :window_width]]
        caches = [window]
        if global_width:
            indices.append(compressed_indices[token, :global_width])
            caches.append(compressed)
        masks = [rows >= 0 for rows in indices]
        keys = torch.cat([
            torch.where(mask[:, None], cache[rows.clamp_min(0).long()], 0)
            for rows, mask, cache in zip(indices, masks, caches, strict=True)
        ])
        valid = torch.cat(masks)
        padding = -keys.shape[0] % 64
        keys = F.pad(keys, (0, 0, 0, padding))
        valid = F.pad(valid, (0, padding), value=False)
        maximum = query.new_full((query.shape[1],), -1e30)
        denominator = torch.zeros_like(maximum)
        accumulator = torch.zeros_like(query[token])
        for start in range(0, keys.shape[0], 64):
            block = keys[start:start + 64]
            mask = valid[start:start + 64]
            scores = prefill_matmul(query[token], block.T) * query.shape[-1] ** -0.5
            scores = scores.masked_fill(~mask[None, :], -torch.inf)
            updated_maximum = torch.maximum(maximum, scores.amax(-1))
            alpha = torch.exp(maximum - updated_maximum)
            probability = torch.exp(scores - updated_maximum[:, None])
            denominator = denominator * alpha + probability.sum(-1)
            accumulator = accumulator * alpha[:, None] + prefill_matmul(probability.bfloat16().float(), block)
            maximum = updated_maximum
        denominator = denominator + torch.exp(sink - maximum)
        output[token] = (accumulator / denominator[:, None]).bfloat16().float()
    return output


def prefill_sparse_attention_reference(
    query, window, window_indices, compressed, compressed_indices, sink, *, precision="fp32", extents=None
):
    """Independent global FP32 or released online64 mixed-precision attention."""
    if precision == "official":
        return _prefill_online_attention(query, window, window_indices, compressed, compressed_indices, sink, extents)
    if query.dtype != torch.float32:
        raise ValueError("FP32 attention requires FP32 queries")
    output = torch.empty_like(query)
    window_rows = window.flatten(0, 1)
    global_rows = compressed.flatten(0, 1) if compressed is not None else None
    for token in range(query.shape[0]):
        rows = window_indices[token]
        selected = [window_rows[rows[rows >= 0].long()]]
        if global_rows is not None and compressed_indices is not None:
            rows = compressed_indices[token]
            selected.append(global_rows[rows[rows >= 0].long()])
        keys = torch.cat(selected)
        scores = prefill_matmul(query[token], keys.T) * query.shape[-1] ** -0.5
        probability = torch.cat((scores, sink[:, None]), dim=-1).softmax(dim=-1)[:, :-1]
        output[token] = prefill_matmul(probability, keys)
    return output


def _prefill_compressor_ratio2_paged(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    position_ids: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    state_block_table: torch.Tensor,
    state_cache: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    precision="fp32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ratio-2 compression with request-owned FP32 ring state.

    Query boundaries count valid packed tokens; any tensor tail is padding.
    Negative/out-of-range request or block indices suppress state access and
    publication. Valid chunks contain consecutive absolute positions.
    """
    kv = prefill_matmul(x.float(), wkv.float())
    score = prefill_matmul(x.float(), wgate.float())
    latent = torch.zeros_like(kv, dtype=x.dtype)
    prefill_publish = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    blocks, capacity, width = state_cache.shape
    if capacity < 1 or width != 2 * kv.shape[-1] or state_cache.dtype != torch.float32:
        raise ValueError("state_cache must be FP32 [blocks, positive capacity, 2 * head_dim]")
    head_dim = kv.shape[-1]
    for request in range(state_block_table.shape[0]):
        start = int(query_start_loc[request])
        end = min(int(query_start_loc[request + 1]), x.shape[0])
        block = int(state_block_table[request, 0])
        if start >= end or not 0 <= block < blocks:
            continue
        for token in range(start, end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) != request or position < 0:
                continue
            if position % 2:
                if token > start:
                    previous_kv, previous_score = kv[token - 1], score[token - 1]
                else:
                    previous = state_cache[block, (position - 1) % capacity]
                    previous_kv, previous_score = previous[:head_dim], previous[head_dim:]
                pair_kv = torch.stack((previous_kv, kv[token]))
                pair_score = torch.stack((previous_score, score[token]))
                pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
                latent[token] = prefill_rms_norm(prefill_round_activation(pooled.to(x.dtype), precision), norm_weight, precision=precision)
                prefill_publish[token] = True
        for token in range(max(start, end - capacity), end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) == request and position >= 0:
                state_cache[block, position % capacity, :head_dim] = kv[token]
                state_cache[block, position % capacity, head_dim:] = score[token]
    return latent, prefill_publish


def _prefill_paged_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    request_ids: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    compressed_lens: torch.Tensor,
    wq_b: torch.Tensor,
    weights_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    candidates: torch.Tensor | None = None,
    topk: int = FLASH.index_topk,
    *,
    precision="fp32",
    formats=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score request-local paged index keys and return scores plus physical rows."""
    if compressed_lens.ndim != 1 or compressed_lens.shape[0] != x.shape[0]:
        raise ValueError("compressed_lens must contain one causal length per query token")
    max_len = int(compressed_lens.max().item()) if compressed_lens.numel() else 0
    scores = x.new_full((x.shape[0], max_len), -torch.inf, dtype=torch.float32)
    physical = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    if max_len == 0:
        return scores, physical

    positions = torch.arange(max_len, device=x.device)
    request_rows = request_ids.to(torch.long).unsqueeze(-1)
    blocks = torch.div(positions, 128, rounding_mode="floor")
    offsets = positions.remainder(128)
    physical_rows = block_table[request_rows, blocks] * 128 + offsets
    flat_keys = index_cache.flatten(0, 1).squeeze(-2)
    keys = flat_keys[physical_rows.to(torch.long)]

    index_dim = index_cache.shape[-1]
    index_heads = weights_proj.shape[-1]
    if precision == "official":
        formats = formats or {}
        q = prefill_linear(qr, wq_b.T, precision=precision, format=formats["index_wq_b"])
        q = q.unflatten(-1, (index_heads, index_dim))
        q = prefill_quantize_dequantize(prefill_rotate(q, cos, sin, precision=precision), "index_fp4")
        weights = prefill_linear(x, weights_proj.T, precision=precision, format=formats["index_weights_proj"])
        weights = prefill_round_activation(weights * (index_dim**-0.5 * index_heads**-0.5), precision)
        dots = prefill_round_activation(prefill_einsum("thd,tkd->thk", q, keys), precision)
        weighted = prefill_round_activation(dots.relu() * weights.unsqueeze(-1), precision)
        scores = prefill_round_activation(weighted.sum(dim=-2), precision)
    else:
        q = prefill_matmul(qr, wq_b).unflatten(-1, (index_heads, index_dim))
        rd = cos.shape[-1] * 2
        q = torch.cat((q[..., :-rd], rope_interleave(q[..., -rd:], cos, sin)), dim=-1)
        weights = prefill_matmul(x.float(), weights_proj.float()).to(x.dtype)
        weights = (weights.float() * (index_dim**-0.5 * index_heads**-0.5)).to(x.dtype)
        dots = prefill_einsum("thd,tkd->thk", q.float(), keys.float()).to(q.dtype)
        weighted = (dots.float().relu() * weights.float().unsqueeze(-1)).to(q.dtype)
        scores = weighted.float().sum(dim=-2).to(q.dtype).float()
    valid_positions = positions.unsqueeze(0) < compressed_lens.to(torch.long).unsqueeze(-1)
    scores = scores.masked_fill(~valid_positions, -torch.inf)
    if candidates is not None:
        scores = scores.masked_fill(~candidates[..., :max_len].to(torch.bool), -torch.inf)

    count = min(topk, max_len)
    logical = scores.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected_scores = scores.gather(-1, logical)
    selected_rows = physical_rows.gather(-1, logical)
    for token in range(x.shape[0]):
        valid_rows = selected_rows[token, torch.isfinite(selected_scores[token])].to(torch.int32)
        physical[token, : valid_rows.numel()] = valid_rows
    return scores, physical


def _prefill_project(x, weights, name, precision):
    format = weights["formats"][name] if precision == "official" else "fp32"
    return prefill_linear(x, weights[name], precision=precision, format=format)


def _prefill_publish_latent(latent, weights, metadata, caches, *, precision="fp32"):
    if metadata.compressed_rope_cos is None or metadata.compressed_rope_sin is None:
        raise ValueError("global publication needs source-position compressed RoPE")
    cosine, sine = metadata.compressed_rope_cos, metadata.compressed_rope_sin
    # Index keys consume the unrotated latent, before compressed-cache quantization.
    keys = prefill_rms_norm(
        _prefill_project(latent, weights, "index_wk", precision), weights["index_norm_weight"], precision=precision
    )
    global_values = prefill_rotate(latent, cosine, sine, precision=precision)
    index_values = prefill_rotate(keys, cosine, sine, precision=precision)
    if precision == "official":
        global_values = prefill_quantize_dequantize(global_values, "kv_fp4")
        index_values = prefill_quantize_dequantize(index_values, "index_fp4")
    prefill_publish(caches.compressed, global_values, metadata.compressed_slots)
    prefill_publish(caches.index, index_values, metadata.compressed_slots)


def prefill_publish_decoder_reference(x, weights, metadata, caches, *, precision="fp32"):
    """Publish full encoder rows after layer20's HC collapse and attention norm."""
    if x.dtype != torch.float32:
        raise ValueError("decoder publication requires FP32 transport buffers")
    result = caches.clone()
    latent = prefill_rms_norm(
        _prefill_project(x, weights, "compressor_wkv", precision), weights["compressor_norm_weight"], precision=precision
    )
    _prefill_publish_latent(latent, weights, metadata, result, precision=precision)
    return result


def prefill_attention_reference(
    x,
    weights,
    metadata,
    caches,
    *,
    mode,
    ratio=0,
    compressed_indices=None,
    candidate_mask=None,
    precision="fp32",
):
    """Evaluate one unsharded attention stage on normalized FP32 token rows.

    All linear matrices are output-major [N,K]. Full decoder mode may consume
    already-published global caches by passing negative compressed_slots; the
    learned index query and candidate selection still run on decoder rows.
    """
    if x.dtype != torch.float32:
        raise ValueError("FP32 attention requires normalized FP32 inputs")
    if any(
        v is not None and v.dtype != torch.float32
        for v in (caches.window, caches.compressed, caches.index, caches.state)
    ):
        raise ValueError("FP32 attention cache state must remain FP32")
    mode = getattr(mode, "value", mode).rsplit("_", 1)[-1].lower()
    result = caches.clone()
    qr = prefill_rms_norm(_prefill_project(x, weights, "wq_a", precision), weights["q_norm_weight"], precision=precision)
    width = weights["wkv"].shape[0]
    query = _prefill_project(qr, weights, "wq_b", precision).unflatten(-1, (-1, width))
    query = prefill_rotate(query, metadata.rope_cos, metadata.rope_sin, precision=precision)
    kv = prefill_rms_norm(_prefill_project(x, weights, "wkv", precision), weights["kv_norm_weight"], precision=precision)
    kv = prefill_rotate(kv, metadata.rope_cos, metadata.rope_sin, precision=precision)
    if precision == "official":
        kv = prefill_quantize_dequantize(kv, "mxfp8")
    prefill_publish(result.window, kv, metadata.window_slots)

    if mode == "full" and (
        ratio == 2 or (metadata.compressed_slots is not None and bool((metadata.compressed_slots >= 0).any()))
    ):
        if ratio == 1:
            latent = prefill_rms_norm(
                _prefill_project(x, weights, "compressor_wkv", precision), weights["compressor_norm_weight"], precision=precision
            )
        elif ratio == 2:
            latent, publication_mask = _prefill_compressor_ratio2_paged(
                x,
                metadata.query_start_loc,
                metadata.position_ids,
                metadata.request_ids,
                metadata.state_block_table,
                result.state,
                weights["compressor_wkv"].T,
                weights["compressor_wgate"].T,
                weights["compressor_norm_weight"],
                precision=precision,
            )
            if bool(((metadata.compressed_slots >= 0) & ~publication_mask).any()):
                raise ValueError("ratio-two publication slots include incomplete pairs")
        else:
            raise ValueError("compressed full mode requires ratio one or two")
        _prefill_publish_latent(latent, weights, metadata, result, precision=precision)

    topk, candidates = compressed_indices, candidate_mask
    if mode in ("full", "reindex"):
        scores, topk = _prefill_paged_indexer(
            x,
            qr,
            metadata.request_ids,
            result.index,
            metadata.index_block_table,
            metadata.compressed_lens,
            weights["index_wq_b"].T,
            weights["index_weights_proj"].T,
            metadata.rope_cos,
            metadata.rope_sin,
            candidates=candidate_mask,
            precision=precision,
            formats=weights.get("formats"),
        )
        if ratio == 1 and mode == "full":
            candidates = select_candidate_blocks(scores, metadata.compressed_lens)
    elif mode not in ("swa", "reuse"):
        raise ValueError(f"unknown FP32 attention mode {mode}")
    if mode != "swa" and topk is None:
        raise ValueError("compressed attention requires source Top-K indices")

    attended = prefill_sparse_attention_reference(
        query,
        result.window,
        metadata.window_indices,
        result.compressed,
        topk,
        weights["attn_sink"],
        precision=precision,
        extents=metadata.attention_extents,
    )
    rotated = prefill_rotate(attended, metadata.rope_cos, metadata.rope_sin, inverse=True, precision=precision)
    groups = weights["wo_a"].shape[0]
    grouped = rotated.reshape(x.shape[0], groups, -1)
    wo_a = weights["wo_a"].bfloat16().float() if precision == "official" else weights["wo_a"]
    projected = prefill_round_activation(prefill_einsum("tgk,gnk->tgn", grouped, wo_a), precision)
    output = _prefill_project(projected.flatten(1), weights, "wo_b", precision)
    return PrefillAttentionResult(output, result, topk, candidates)


def prefill_engram_gate_reference(x, kv, weight, *, eps=FLASH.rms_norm_eps, token_mask=None, precision="fp32"):
    """Released normalized signed-square-root Engram gate, entirely FP32."""
    if x.dtype != torch.float32 or kv.dtype != torch.float32:
        raise ValueError("FP32 Engram requires FP32 stream and projected values")
    copies, width = x.shape[-2:]
    key, value = kv.split((copies * width, width), dim=-1)
    key = key.unflatten(-1, (copies, width))
    rstd = torch.rsqrt(x.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (x * weight * key).sum(-1) * rstd * width**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return prefill_round_activation(x + gate.unsqueeze(-1) * value.unsqueeze(-2), precision)


def prefill_engram_reference(x, weights, *, eps=FLASH.rms_norm_eps, token_mask=None, precision="fp32"):
    """Project independently decoded original lookup rows and update the HC stream."""
    lookup = prefill_round_activation(weights["lookup"], precision)
    kv = _prefill_project(lookup, weights, "wkv", precision)
    return prefill_engram_gate_reference(x, kv, weights["weight"], eps=eps, token_mask=token_mask, precision=precision)
