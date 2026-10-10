# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch reference operations for the DeepSeek-V4.1-Flash text backbone."""

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
