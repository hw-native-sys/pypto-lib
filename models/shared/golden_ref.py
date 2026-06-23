# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared PyTorch golden reference helpers for Qwen3 test harnesses.

Usage (in a model file's golden function):

    from models.shared.golden import (
        golden_rmsnorm,
        golden_rope_rotate_half,
        golden_online_softmax_step,
        golden_swiglu,
    )

Extract duplicated arithmetic without pulling in the entire pipeline
orchestration.  Each helper is a plain Python/PyTorch function — no
``@pl.inline``, just numeric reference logic.
"""

from __future__ import annotations

import torch


# ── RMSNorm ──


def golden_rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference RMSNorm: ``(x / sqrt(mean(x^2) + eps)) * gamma``.

    Input ``x`` shape ``[*, hidden]``, ``gamma`` shape ``[1, hidden]``.
    Returns ``bfloat16`` tensor matching the kernel's output dtype.
    """
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True) + eps
    inv_rms = torch.rsqrt(variance)
    return (x_f32 * inv_rms * gamma.float()).bfloat16()


# ── RoPE rotate-half ──


def golden_rope_rotate_half(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reference RoPE rotate-half over the last dim.

    Splits ``x``, ``cos``, ``sin`` into lo/hi halves along the last axis:
        lo' = x_lo * cos_lo - x_hi * sin_lo
        hi' = x_hi * cos_hi + x_lo * sin_hi

    ``cos`` and ``sin`` may be full-length (same as ``x.shape[-1]``) or
    already half-length.  Returns concatenated result (same shape as ``x``).
    """
    half = x.shape[-1] // 2
    x_lo, x_hi = x[..., :half], x[..., half:]
    if cos.shape[-1] == x.shape[-1]:
        cos_lo, cos_hi = cos[..., :half], cos[..., half:]
        sin_lo, sin_hi = sin[..., :half], sin[..., half:]
    else:
        cos_lo = cos_hi = cos
        sin_lo = sin_hi = sin
    rot_lo = x_lo * cos_lo - x_hi * sin_lo
    rot_hi = x_hi * cos_hi + x_lo * sin_hi
    return torch.cat([rot_lo, rot_hi], dim=-1)


# ── Online softmax (single step) ──


def golden_online_softmax_step(
    oi: torch.Tensor,
    mi: torch.Tensor,
    li: torch.Tensor,
    oi_tmp: torch.Tensor,
    cur_mi: torch.Tensor,
    cur_li: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step of online-softmax accumulation.

    Returns updated ``(oi, mi, li)`` — caller must rebind each.
    """
    mi_new = torch.maximum(mi, cur_mi)
    alpha = torch.exp(mi - mi_new)
    beta = torch.exp(cur_mi - mi_new)
    li = alpha * li + beta * cur_li
    oi = oi * alpha + oi_tmp * beta
    mi = mi_new
    return oi, mi, li


# ── SwiGLU ──


def golden_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference SiLU(gate) * up = (gate * sigmoid(gate)) * up."""
    return gate * torch.sigmoid(gate) * up


# ── Decode attention core (scope 1 + 2) ──


def golden_decode_scope1(
    hidden_states: torch.Tensor,
    input_rms_weight: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    *,
    hidden: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm + Q/K/V projection (shared scope 1).

    Returns ``(normed_bf16, q_proj, k_proj, v_proj)``.
    """
    normed_bf16 = golden_rmsnorm(hidden_states, input_rms_weight, eps=eps)
    normed_f32 = normed_bf16.float()
    wq_f = wq.float()
    wk_f = wk.float()
    wv_f = wv.float()
    q_proj = (normed_f32 @ wq_f).float()
    k_proj = (normed_f32 @ wk_f).float()
    v_proj = (normed_f32 @ wv_f).float()
    return normed_bf16, q_proj, k_proj, v_proj


def golden_decode_scope2(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq: int,
    seq_tile: int,
    q_per_kv: int,
    q_head_batch: int,
    attn_scale: float,
) -> torch.Tensor:
    """RoPE + flat-cache write + GQA attention (shared scope 2 for 2D layout).

    Returns ``attn_out`` of shape ``[batch, hidden]`` with per-head attention
    results accumulated in FP32.

    Assumes flat KV cache indexed as ``[b * num_kv_heads * max_seq + ki * max_seq + pos]``.
    For 4D block-major cache, use ``golden_decode_scope2_4d`` instead.
    """
    half = head_dim // 2
    hidden = num_heads * head_dim
    attn_out = torch.zeros(batch, hidden, dtype=torch.float32)

    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + seq_tile - 1) // seq_tile

        cos_row = rope_cos[pos: pos + 1, :]
        sin_row = rope_sin[pos: pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        # K RoPE + cache write
        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_rot = golden_rope_rotate_half(k_heads, cos_row, sin_row)
        for ki in range(num_kv_heads):
            cr = b * num_kv_heads * max_seq + ki * max_seq + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * head_dim: (ki + 1) * head_dim].to(torch.bfloat16)

        # Q RoPE
        q_heads = q_proj[b].view(num_heads, head_dim)
        q_rot = golden_rope_rotate_half(q_heads, cos_row, sin_row)
        q_rot_bf16 = q_rot.to(torch.bfloat16)

        # GQA attention
        for kvh in range(num_kv_heads):
            for qg in range(q_per_kv // q_head_batch):
                q_base = kvh * q_per_kv + qg * q_head_batch
                q_grp_bf16 = q_rot_bf16[q_base: q_base + q_head_batch, :]

                oi = torch.zeros(q_head_batch, head_dim, dtype=torch.float32)
                li = torch.zeros(q_head_batch, 1, dtype=torch.float32)
                mi = torch.zeros(q_head_batch, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * seq_tile
                    valid_len = min(seq_tile, ctx_len - s0)
                    cb = b * num_kv_heads * max_seq + kvh * max_seq + s0

                    k_tile = k_cache[cb: cb + seq_tile, :]
                    v_tile = v_cache[cb: cb + seq_tile, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < seq_tile:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * attn_scale

                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        oi, mi, li = golden_online_softmax_step(oi, mi, li, oi_tmp, cur_mi, cur_li)

                ctx = oi / li
                for qi in range(q_head_batch):
                    qh = q_base + qi
                    attn_out[b, qh * head_dim: (qh + 1) * head_dim] = ctx[qi]

    return attn_out


def golden_decode_scope2_4d(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq: int,
    seq_tile: int,
    q_per_kv: int,
    q_head_batch: int,
    attn_scale: float,
    out_proj_k_blocks: int,
    out_proj_k_chunk: int,
) -> torch.Tensor:
    """RoPE + 4D block-major cache write + GQA attention.

    Returns ``attn_proj_tile`` of shape ``[out_proj_k_blocks, batch, out_proj_k_chunk]``
    — the per-Q-head attention output in block-major layout, ready for 4D output
    projection.

    Assumes 4D KV cache indexed as ``[cache_idx, pos_block, pos_offset, :]``.
    """
    half = head_dim // 2
    hidden = num_heads * head_dim
    attn_proj_tile = torch.zeros(out_proj_k_blocks, batch, out_proj_k_chunk, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = int(seq_lens[b, 0, 0, 0].item())
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + seq_tile - 1) // seq_tile

        cos_row = rope_cos[pos, 0, :, :]
        sin_row = rope_sin[pos, 0, :, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        # K RoPE + 4D cache write
        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_rot = golden_rope_rotate_half(k_heads, cos_row, sin_row)
        for ki in range(num_kv_heads):
            cache_idx = b * num_kv_heads + ki
            pos_block = pos // seq_tile
            pos_offset = pos - pos_block * seq_tile
            k_cache[cache_idx, pos_block, pos_offset, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_idx, pos_block, pos_offset, :] = (
                v_proj[b, ki * head_dim: (ki + 1) * head_dim].to(torch.bfloat16)
            )

        # Q RoPE
        q_heads = q_proj[b].view(num_heads, head_dim)
        q_rot = golden_rope_rotate_half(q_heads, cos_row, sin_row)
        q_rot_bf16 = q_rot.to(torch.bfloat16)

        # GQA attention
        for kvh in range(num_kv_heads):
            for qg in range(q_per_kv // q_head_batch):
                q_base = kvh * q_per_kv + qg * q_head_batch
                q_grp_bf16 = q_rot_bf16[q_base: q_base + q_head_batch, :]

                oi = torch.zeros(q_head_batch, head_dim, dtype=torch.float32)
                li = torch.zeros(q_head_batch, 1, dtype=torch.float32)
                mi = torch.zeros(q_head_batch, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * seq_tile
                    valid_len = min(seq_tile, ctx_len - s0)
                    cache_idx = b * num_kv_heads + kvh
                    k_tile = k_cache[cache_idx, sb, :, :]
                    v_tile = v_cache[cache_idx, sb, :, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < seq_tile:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * attn_scale

                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        oi, mi, li = golden_online_softmax_step(oi, mi, li, oi_tmp, cur_mi, cur_li)

                ctx = oi / li
                for qi in range(q_head_batch):
                    qh = q_base + qi
                    attn_proj_tile[qh, b, :] = ctx[qi].to(torch.bfloat16)

    return attn_proj_tile


# ── Scope 3: output projection + residual + post-RMSNorm + SwiGLU MLP ──


def golden_decode_scope3(
    attn_out: torch.Tensor,
    hidden_states: torch.Tensor,
    wo: torch.Tensor,
    post_rms_weight: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    hidden: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Output projection → residual → post-RMSNorm → SwiGLU MLP → residual (2D layout).

    Returns ``[batch, hidden]`` BF16 output.
    """
    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    normed_bf16 = golden_rmsnorm(resid1, post_rms_weight, eps=eps)

    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp = golden_swiglu(gate, up).bfloat16()
    down = torch.matmul(mlp.float(), w_down.float())

    return (down + resid1).bfloat16()
