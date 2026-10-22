# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Q/KV projection, normalization, and RoPE stages."""


from gemm import gemm_bf16_via_mxfp8_to_bf16_staged

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN, TP_SIZE
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum
from rmsnorm import golden_rms_norm, rms_norm


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank

# tiling
N_TILE = 128
K_TILE = 256
Q_A_N_TILE = 128
Q_A_K_TILE = 256


@pl.jit.inline
def rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.constexpr,
    name_hint: pl.constexpr,
    max_workers: pl.constexpr,
):
    blocks = num_tokens * heads
    workers = pl.min(blocks, max_workers)
    for worker in pl.spmd(workers, name_hint=name_hint):
        for block in pl.range(worker, blocks, workers):
            t = block // heads
            h = block % heads
            base = h * HEAD_DIM
            output[t : t + 1, base : base + (HEAD_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (HEAD_DIM - ROPE_DIM)]
            tail = pl.cast(x[t : t + 1, base + (HEAD_DIM - ROPE_DIM) : base + HEAD_DIM], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[t : t + 1, :]
            s = pl.mul(sin[t : t + 1, :], 1.0)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            im = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[t : t + 1, base + (HEAD_DIM - ROPE_DIM) : base + HEAD_DIM] = pl.cast(rotated, pl.BF16, mode="rint")
    return output


@pl.jit.inline(auto_scope=False)
def q_proj_qr(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    gemm_bf16_via_mxfp8_to_bf16_staged(
        x, wq_a, wq_a_scale, projected, num_tokens,
        D, Q_LORA, "attention_q_a", "attention_q_a_quant", Q_A_N_TILE, Q_A_K_TILE,  1,
    )
    norm_blocks = (num_tokens + 7) // 8
    for norm_worker in pl.spmd(norm_blocks, name_hint="attention_q_norm"):
        rms_norm(projected, q_norm_weight, query_latent, norm_worker, num_tokens, width=Q_LORA)
    return query_latent


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    max_workers: pl.constexpr = 2147483647,
):
    tokens = pl.tensor.dim(query_latent, 0)
    expanded = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    gemm_bf16_via_mxfp8_to_bf16_staged(
        query_latent, wq_b, wq_b_scale, expanded, num_tokens,
        Q_LORA, LOCAL_H * HEAD_DIM, "attention_q_b", "attention_q_b_quant", N_TILE, K_TILE,  1,
    )
    rope(expanded, rope_cos, rope_sin, query, num_tokens, LOCAL_H, "attention_q_rope", max_workers)
    return query


@pl.jit.inline(auto_scope=False)
def kv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    projected: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x, 0)
    gemm_bf16_via_mxfp8_to_bf16_staged(
        x, wkv, wkv_scale, projected,
        num_tokens,
        D, HEAD_DIM, "attention_kv", "attention_kv_quant", N_TILE, K_TILE,  1,
    )
    normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    for norm_worker in pl.spmd(norm_blocks, name_hint="attention_kv_norm"):
        rms_norm(projected, kv_norm_weight, normalized, norm_worker, num_tokens, width=HEAD_DIM)
    rope(normalized, rope_cos, rope_sin, window_kv, num_tokens, 1, "attention_kv_rope", 2147483647)
    return window_kv


def golden_rope_interleave(
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


def golden_rope_tail(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing 2*cos.shape[-1] lanes as adjacent (real, imag) pairs."""
    rope_dim = cos.shape[-1] * 2
    rotated = golden_rope_interleave(x[..., -rope_dim:], cos, sin, inverse)
    return torch.cat((x[..., :-rope_dim], rotated), dim=-1)


def golden_qkv_proj_rope(
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
    *,
    sequential_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized query latent and shared KV, then rotate their RoPE tails."""
    def project(value, weight, scale):
        if sequential_query and scale is not None:
            return golden_gemm_bf16_via_mxfp8_to_bf16_sequential(value, weight, scale)
        return golden_mxfp8_linear_grouped_sum(value, weight, scale)

    if wq_a_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(x, wq_a, wq_a_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(x, wq_a, wq_a_scale)
    qr = golden_rms_norm(projected, q_norm_weight)
    head_dim = wkv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim
    q = project(qr, wq_b, wq_b_scale).unflatten(-1, (num_heads, head_dim))
    kv = golden_rms_norm(project(x, wkv, wkv_scale), kv_norm_weight)
    q = golden_rope_tail(q, cos, sin)
    kv = golden_rope_tail(kv, cos, sin)
    return q, kv, qr
