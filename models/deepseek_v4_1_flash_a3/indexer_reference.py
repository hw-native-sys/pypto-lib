# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged fixtures and Torch scores for the A3 indexers."""

import torch

from .config import FLASH as M
from .gemm import golden_dynamic_int8
from .qkv_proj_rope import golden_rope_tail
from .quant import FP4_VALUES, dequantize_mxfp4_cache, quantize_mxfp4_cache


def make_index_case(checkpoint, layer, history, *, ties=False, num_tokens=12):
    """Create page permutations, a zero row, causal boundaries and initialized inactive rows."""
    pages = (history + 127) // 128
    generator = torch.Generator().manual_seed(41051)
    lengths = torch.tensor([0, 1, 127, 128, 129, 511, 512, 513, 8191, 8192, 8193, 16513, 0, 0, 0, 0])
    lengths = lengths.clamp_max(history).int()
    if num_tokens == 1:
        lengths[0] = history
    x = torch.randn(16, M.hidden_size, generator=generator).bfloat16()
    latent = torch.randn(16, M.q_lora_rank, generator=generator).bfloat16()
    latent[1] = 0
    if ties:
        latent.zero_()
    if checkpoint is None:
        columns = torch.arange(M.index_n_heads * M.index_head_dim)
        weight = torch.zeros(M.q_lora_rank, columns.numel(), dtype=torch.int8)
        weight[columns % M.q_lora_rank, columns] = (columns % 11 - 5).to(torch.int8)
        scale = torch.full((1, columns.numel()), 1.0 / 32)
        heads = torch.arange(M.index_n_heads)
        head_weight = torch.zeros(M.hidden_size, M.index_n_heads, dtype=torch.bfloat16)
        head_weight[heads, heads] = 1.0
    else:
        prefix = f"layers.{layer}.attn.indexer."
        weight = checkpoint.tensor(prefix + "wq_b.weight").t().contiguous()
        scale = checkpoint.tensor(prefix + "wq_b.weight_scale").reshape(1, -1)
        offset = checkpoint.tensor(prefix + "wq_b.weight_offset")
        if weight.dtype != torch.int8 or bool(offset.count_nonzero()):
            raise ValueError("index query projection requires symmetric INT8 checkpoint weights")
        head_weight = checkpoint.tensor(prefix + "weights_proj.weight").t().contiguous()
    values = torch.randn(pages + 1, 128, 1, M.index_head_dim, generator=generator).bfloat16()
    cache, scales = quantize_mxfp4_cache(values, 32, "e8m0", round_to_even=True)
    table = torch.randperm(pages, generator=generator).reshape(1, -1).int()
    positions = torch.arange(history - 16, history).float().unsqueeze(-1)
    frequency = 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    angles = positions * frequency
    return dict(x=x, latent=latent, query_weight=weight, query_weight_scale=scale,
                weight_projection=head_weight, cos=angles.cos(), sin=angles.sin(),
                cache=cache, cache_scale=scales, table=table,
                request_ids=torch.zeros(16, dtype=torch.int32), lengths=lengths, num_tokens=num_tokens)


def indexer_reference(inputs):
    """Evaluate exact W8A8 projection, FP4 rounding and the BF16 score boundaries."""
    t = inputs
    tokens, heads, dim = t["x"].shape[0], M.index_n_heads, M.index_head_dim
    width = t["table"].shape[-1] * 128
    projected = golden_dynamic_int8(t["latent"], t["query_weight"], t["query_weight_scale"])
    query = golden_rope_tail(projected.reshape(tokens, heads, dim), t["cos"], t["sin"])
    codes, scales = quantize_mxfp4_cache(query, 32, "e8m0", round_to_even=True)
    rounded = dequantize_mxfp4_cache(codes, scales, 32, "e8m0").bfloat16()
    raw_weights = (t["x"].float() @ t["weight_projection"].float()).bfloat16()
    weights = (raw_weights.float() * (heads * dim) ** -0.5).bfloat16()
    keys = dequantize_mxfp4_cache(t["cache"], t["cache_scale"], 32, "e8m0").bfloat16()
    keys = keys.reshape(-1, dim)
    scores = torch.full((tokens, width), -torch.inf)
    c1a = torch.full((tokens, M.index_topk), -1, dtype=torch.int32)
    c2a = c1a.clone()
    physical = torch.full((tokens, width), -1, dtype=torch.int32)
    for row in range(int(t["num_tokens"])):
        count = int(t["lengths"][row])
        logical = torch.arange(count)
        pages = t["table"][t["request_ids"][row].long(), logical // 128]
        selected = pages.long() * 128 + logical.remainder(128)
        physical[row, :count] = selected.int()
        if not count:
            continue
        dot = (rounded[row].float() @ keys[selected].float().t()).bfloat16()
        product = (dot.float().relu() * weights[row].float().unsqueeze(-1)).bfloat16()
        values = product.float().sum(0).bfloat16().float()
        scores[row, :count] = values
        chosen = torch.argsort(values, descending=True, stable=True)[:M.index_topk]
        c1a[row, :chosen.numel()] = selected[chosen.sort().values].int()
        if count <= M.index_topk:
            c2a[row, :count] = selected.int()
        else:
            c2a[row, :chosen.numel()] = selected[chosen].sort().values.int()
    carrier = torch.stack((FP4_VALUES[(codes & 15).long()], FP4_VALUES[(codes >> 4).long()]), dim=-1)
    carrier = carrier.flatten(-2).to(torch.float8_e4m3fn).view(torch.uint8).flatten(1)
    return dict(codes=codes.flatten(1), fp8_codes=carrier, scales=scales.flatten(1), query=rounded.flatten(1),
                raw_weights=raw_weights, weights=weights, scores=scores, c1a=c1a, c2a=c2a, physical=physical)


def compare_index_scores(actual, expected, *, inputs, **kwargs):
    """Compare valid scores with the standard gate and retain exact masked/tail values."""
    from golden import ratio_allclose

    active = int(inputs["num_tokens"])
    positions = torch.arange(actual.shape[-1]).unsqueeze(0)
    valid = positions < inputs["lengths"].unsqueeze(-1)
    valid[active:] = False
    gate = ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.0)
    passed, detail = gate(actual[valid], expected[valid], inputs=inputs, **kwargs)
    preserved = torch.equal(actual[~valid], expected[~valid])
    return passed and preserved, f"{detail}; exact_mask_and_tail={preserved}"


def compare_pool_scores(actual, expected, *, inputs, **kwargs):
    """Compare compact candidate scores and retain exact invalid and inactive lanes."""
    from golden import ratio_allclose

    active = int(inputs["num_tokens"])
    valid = torch.isfinite(expected)
    valid[active:] = False
    gate = ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.0)
    passed, detail = gate(actual[valid], expected[valid], inputs=inputs, **kwargs)
    preserved = torch.equal(actual[~valid], expected[~valid])
    return passed and preserved, f"{detail}; exact_pool_mask_and_tail={preserved}"
