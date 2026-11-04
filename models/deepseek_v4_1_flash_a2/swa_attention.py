# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 BF16 sink-aware sliding-window attention kernel and reference."""

from __future__ import annotations

import pypto.language as pl

from .swa_cache import (
    gather_swa_cache_reference,
    make_swa_cache_gather,
    make_swa_cache_publish,
    write_swa_cache_reference,
)


def swa_attention_reference(
    query: torch.Tensor,
    selected: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    head_dim: int,
    *,
    valid_tokens: int | None = None,
) -> torch.Tensor:
    """Compute sink-aware attention over gathered BF16 SWA rows."""

    import torch

    if query.ndim != 2 or query.shape[-1] % head_dim:
        raise ValueError("query must be [T, H*D] with a divisible head dimension")
    tokens = query.shape[0]
    heads = query.shape[-1] // head_dim
    if selected.ndim == 2:
        if selected.shape[0] != tokens * indices.shape[1]:
            raise ValueError("flat selected rows do not match indices")
        selected = selected.reshape(tokens, indices.shape[1], head_dim)
    if selected.shape[:2] != indices.shape or selected.shape[-1] != head_dim:
        raise ValueError("selected and indices shape mismatch")
    if sink.shape != (heads,):
        raise ValueError(f"sink must have shape {(heads,)}, got {tuple(sink.shape)}")
    active = tokens if valid_tokens is None else int(valid_tokens)
    output = torch.zeros_like(query)
    if active == 0:
        return output
    q = query[:active].float().reshape(active, heads, head_dim)
    k = selected[:active].float()
    score = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)
    valid = indices[:active].ge(0).unsqueeze(1)
    score = score.masked_fill(~valid, float("-inf"))
    sink_score = sink.float().reshape(1, heads, 1).expand(active, -1, -1)
    all_score = torch.cat((score, sink_score), dim=-1)
    probability = torch.softmax(all_score, dim=-1)
    result = torch.matmul(probability[..., :-1], k)
    output[:active] = result.reshape(active, heads * head_dim).to(torch.bfloat16)
    return output


def make_swa_attention(
    capacity: int,
    heads: int,
    head_dim: int,
    window: int,
    *,
    decode_state: bool = False,
):
    """Build the sink-aware A2/A3 attention reduction kernel.

    The default keeps the prefill/layer ABI. Decode requests opt into the
    request-state controls explicitly so existing callers retain their
    public signature.
    """

    if capacity <= 0 or heads <= 0 or head_dim <= 0 or window <= 0:
        raise ValueError("attention dimensions must be positive")
    if heads % 16:
        raise ValueError("A2/A3 attention uses a 16-head vector tile")
    publish = make_swa_cache_publish(capacity, 2, head_dim, decode_state=True)
    gather = make_swa_cache_gather(capacity, window, 2, head_dim, decode_state=True)
    head_tile = 16
    scale = head_dim ** -0.5

    @pl.jit.inline
    def decode_attend(
        query: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        cache_source_ids: pl.Tensor[[capacity], pl.INT32],
        finished: pl.Tensor[[capacity], pl.INT8],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        publish(window_kv, slots, positions, request_ids, cache_source_ids, finished, cache, active)
        selected_flat = pl.create_tensor([capacity * window, head_dim], dtype=pl.BF16)
        gather(cache, indices, finished, selected_flat, active)
        qflat = pl.reshape(query, [capacity * heads, head_dim])
        oflat = pl.reshape(output, [capacity * heads, head_dim])
        kflat = selected_flat
        for block in pl.spmd((active + 0) * (heads // head_tile), name_hint="dsv41_a2a3_swa_attention"):
            token = block // (heads // head_tile)
            head = (block % (heads // head_tile)) * head_tile
            q = qflat[token * heads + head:token * heads + head + head_tile, :]
            k = kflat[token * window:token * window + window, :]
            scores = pl.mul(pl.matmul(q, k, b_trans=True, out_dtype=pl.FP32), scale)
            idx = pl.cast(indices[token:token + 1, :], pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
            scores = pl.col_expand_add(scores, pl.mul(pl.sub(valid, 1.0), 1.0e30))
            maximum = pl.reshape(pl.row_max(scores), [1, head_tile])
            probabilities = pl.col_expand_mul(
                pl.exp(pl.row_expand_sub(scores, pl.reshape(maximum, [head_tile, 1]))), valid
            )
            denominator = pl.row_sum(probabilities)
            sink_score = pl.reshape(sink[head:head + head_tile], [1, head_tile])
            final_max = pl.maximum(maximum, sink_score)
            correction = pl.exp(pl.sub(maximum, final_max))
            denominator = pl.add(
                pl.mul(pl.reshape(denominator, [1, head_tile]), correction),
                pl.exp(pl.sub(sink_score, final_max)),
            )
            weighted = pl.matmul(pl.cast(probabilities, pl.BF16, mode="rint"), k, out_dtype=pl.FP32)
            result = pl.row_expand_mul(weighted, pl.reshape(pl.div(correction, denominator), [head_tile, 1]))
            oflat[token * heads + head:token * heads + head + head_tile, :] = pl.cast(
                result, pl.BF16, mode="rint"
            )
        return output, cache, selected_flat

    if decode_state:
        return decode_attend

    @pl.jit.inline
    def attend(
        query: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        output: pl.Tensor[[capacity, heads * head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        positions = pl.full([capacity], dtype=pl.INT64, value=0)
        request_ids = pl.full([capacity], dtype=pl.INT32, value=0)
        cache_source_ids = pl.full([capacity], dtype=pl.INT32, value=0)
        finished = pl.full([capacity], dtype=pl.INT8, value=0)
        return decode_attend(
            query,
            window_kv,
            slots,
            indices,
            sink,
            positions,
            request_ids,
            cache_source_ids,
            finished,
            cache,
            output,
            active,
        )

    return attend


__all__ = [
    "gather_swa_cache_reference",
    "make_swa_attention",
    "swa_attention_reference",
    "write_swa_cache_reference",
]
