# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared Torch reference operations for the GLM-5.3-Flash text backbone.

Only the transforms that more than one work item depends on live here. Every
operator file owns the golden for its own kernel, so that one engineer can land a
golden and a kernel body together without touching a shared file.

References are checked against ``transformers`` ``models/glm5_next`` and the
released ``zai-org/GLM-5.3-Flash`` checkpoint.
"""

import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import FLASH, KV_LORA, QK_DIM, V_DIM


def _sum_fp32_reference(
    value: torch.Tensor, dim: int = -1, keepdim: bool = True, tile_size: int | None = None,
) -> torch.Tensor:
    """Round an independent wide reduction to FP32, preserving requested tile boundaries."""
    if tile_size is None:
        return value.double().sum(dim=dim, keepdim=keepdim).float()
    total = None
    for tile in value.split(tile_size, dim=dim):
        partial = tile.float().sum(dim=dim, keepdim=True)
        total = partial if total is None else total + partial
    return total if keepdim else total.squeeze(dim)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = FLASH.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    square_mean = value.double().square().mean(dim=-1, keepdim=True).float()
    value = value * torch.rsqrt((square_mean + eps).double()).float()
    return (weight.to(dtype).float() * value.to(dtype).float()).to(dtype)


def rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float = FLASH.rms_norm_eps,
) -> torch.Tensor:
    """KDA output norm: strict FP32 RMSNorm over ``linear_head_dim``, then a sigmoid gate.

    Mirrors ``Glm5NextTextRMSNormGated``: the weights are *not* downcast, and the
    gate is applied after the norm, in FP32.
    """
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt((_sum_fp32_reference(value.square()) / value.shape[-1] + eps).double()).float()
    value = weight.float() * value
    sigmoid = torch.reciprocal(1.0 + torch.exp(-gate.double()).float())
    return (value * sigmoid).to(dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalisation used on the KDA query and key.

    Note the ``+ eps`` inside the square root rather than ``max(norm, eps)``; the
    reference is explicit that this differs from ``F.normalize``.
    """
    value = x.float()
    norm = torch.sqrt((_sum_fp32_reference(value.square()) + eps).double()).float()
    return (value / norm).to(x.dtype)


def swiglu(gate_value: torch.Tensor, up_value: torch.Tensor) -> torch.Tensor:
    """Clamp-stabilised SwiGLU shared by the dense MLP, shared expert and routed experts.

    ``gate`` is clamped from above only; ``up`` is clamped on both sides.
    """
    limit = FLASH.swiglu_limit
    gate_value = gate_value.float().clamp(max=limit)
    up_value = up_value.float().clamp(-limit, limit)
    return F.silu(gate_value) * up_value


def expert(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    route_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate one clamp-stabilised SwiGLU expert (routed, shared, or dense)."""
    hidden = swiglu(F.linear(x, w_gate), F.linear(x, w_up))
    if route_weight is not None:
        hidden = hidden * route_weight
    return F.linear(hidden.to(x.dtype), w_down)


def gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid ``noaux_tc`` routing; the correction bias affects selection only.

    ``n_group`` and ``topk_group`` are both 1 for GLM-5.3-Flash, so the grouped
    masking of ``DeepseekV3TopkRouter`` degenerates to a plain top-k.
    """
    logits = (x.double() @ weight.double().t()).float()
    scores = (1.0 / (1.0 + torch.exp(-logits.double()).float()).double()).float()
    biased = scores + correction_bias.float()
    # Exact corrected-score ties keep ascending expert ids.
    indices = torch.argsort(-biased, dim=-1, stable=True)[..., :FLASH.num_experts_per_tok]
    weights = scores.gather(-1, indices)
    if FLASH.norm_topk_prob:
        denominator = weights.double().sum(dim=-1, keepdim=True).float() + 1e-20
        weights = (weights.double() / denominator.double()).float()
    return weights * FLASH.routed_scaling_factor, indices.to(torch.int32)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the four hyper-connection streams into one sublayer input."""
    return (x.float() * pre_mix.float().unsqueeze(-1)).double().sum(dim=-2).float().to(x.dtype)


def hc_seed(inputs_embeds: torch.Tensor, hc_mult: int = FLASH.hc_mult) -> torch.Tensor:
    """Seed the hyper-connection stream by replicating the embedding.

    ``Glm5NextTextModel.forward`` does
    ``inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()``, so the
    stream inherits the embedding's dtype: **BF16, not FP32**. DeepSeek-V4 carries
    its HC stream in FP32, so the ABIs are not interchangeable.
    """
    return inputs_embeds.unsqueeze(-2).expand(*inputs_embeds.shape[:-1], hc_mult, -1).contiguous()


def hc_mixes(
    x: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = FLASH.hc_sinkhorn_iters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC mixes with FP32 projections, unaries and reductions."""
    flat = x.flatten(-2).float()
    square_sum = flat.double().square().sum(dim=-1, keepdim=True).float()
    inv = torch.rsqrt((square_sum / flat.shape[-1] + FLASH.rms_norm_eps).double()).float()
    mixes = (flat.double() @ function.double().t()).float() * inv
    hc = FLASH.hc_mult
    (p, q, r) = mixes.split((hc, hc, hc * hc), -1)
    pre_logits = p * scale[0] + base[:hc]
    post_logits = q * scale[1] + base[hc:2 * hc]
    pre = (1.0 / (1.0 + torch.exp(-pre_logits.double()).float()).double()).float() + FLASH.hc_eps
    post = 2 * (1.0 / (1.0 + torch.exp(-post_logits.double()).float()).double()).float()
    logits = r.unflatten(-1, (hc, hc)) * scale[2] + base[2 * hc:].unflatten(-1, (hc, hc))
    exps = torch.exp((logits - logits.amax(-1, keepdim=True)).double()).float()
    residual = (exps.double() / _sum_fp32_reference(exps).double()).float() + FLASH.hc_eps
    residual = (residual.double() / (_sum_fp32_reference(residual, -2) + FLASH.hc_eps).double()).float()
    for _ in range(iterations - 1):
        residual = (residual.double() / (_sum_fp32_reference(residual) + FLASH.hc_eps).double()).float()
        residual = (residual.double() / (_sum_fp32_reference(residual, -2) + FLASH.hc_eps).double()).float()
    return (pre, post, residual)


def hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams.

    Preserve the official decoder's stream-dtype coefficient, multiplication,
    matmul-output and addition boundaries.
    """
    dtype = residual.dtype
    post = post_mix.to(dtype).float().unsqueeze(-1)
    update = (post * sublayer.to(dtype).float().unsqueeze(-2)).to(dtype)
    coefficients = residual_mix.to(dtype).double().transpose(-1, -2)
    mixed_residual = (coefficients @ residual.double()).float().to(dtype)
    return (update.float() + mixed_residual.float()).to(dtype)


def hc_head(x: torch.Tensor) -> torch.Tensor:
    """Collapse the final HC streams.

    Unlike DeepSeek-V4, which applies the last layer's delayed pre-mix, GLM-5.3-Flash
    uses an unweighted mean (``Glm5NextTextHyperHead.forward`` is
    ``hidden_streams.mean(dim=2)``).
    """
    return x.double().mean(dim=-2).float().to(torch.bfloat16)


def golden_split_kv_b(w_kv_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into its key and value halves.

    Args:
        w_kv_b: ``[H * (QK_DIM + V_DIM), KV_LORA]`` as stored in the checkpoint.

    Returns:
        ``w_k`` of ``[H, QK_DIM, KV_LORA]`` and ``w_v`` of ``[H, V_DIM, KV_LORA]``.
    """
    heads = w_kv_b.shape[0] // (QK_DIM + V_DIM)
    reshaped = w_kv_b.reshape(heads, QK_DIM + V_DIM, KV_LORA)
    return reshaped[:, :QK_DIM], reshaped[:, QK_DIM:]


def golden_absorb_output(w_v: torch.Tensor, w_o: torch.Tensor) -> torch.Tensor:
    """Fold the value half into ``o_proj``; returns ``[D, H * KV_LORA]``.

    Decode attends in latent space, so its context rows are ``[H, KV_LORA]`` and the
    value expansion never has to happen per token: ``o_proj @ (ctx @ w_v^T)`` equals
    ``(o_proj folded with w_v) @ ctx``. This is a weight-time transform, and both
    operands have been prepared as BF16 by the loader, so nothing is requantized.

    Args:
        w_v: ``[H, V_DIM, KV_LORA]`` value half from :func:`golden_split_kv_b`.
        w_o: ``[D, H * V_DIM]`` head-space output projection.

    There is no kernel for this: it runs once per layer when the weights are staged,
    so it belongs to the weight loader rather than to any per-token scope. Only the
    query half (:func:`absorb_query`) is per-token.

    Returns:
        ``[D, H * KV_LORA]`` absorbed output projection, in ``w_o``'s dtype.
    """
    heads, v_dim, kv_lora = w_v.shape
    per_head = w_o.float().unflatten(-1, (heads, v_dim))
    absorbed = torch.einsum("dhv,hvk->dhk", per_head, w_v.float())
    return absorbed.reshape(w_o.shape[0], heads * kv_lora).to(w_o.dtype)


__all__ = [
    "expert",
    "gate",
    "golden_absorb_output",
    "golden_split_kv_b",
    "hc_head",
    "hc_mixes",
    "hc_post",
    "hc_pre",
    "hc_seed",
    "l2norm",
    "rms_norm",
    "rms_norm_gated",
    "swiglu",
]


if __name__ == "__main__":
    from models.glm5_3_flash._golden_smoke import run_expert_golden
    from models.glm5_3_flash._golden_smoke import run_moe_gate_golden
    from models.glm5_3_flash._golden_smoke import run_norm_goldens
    from models.glm5_3_flash._golden_smoke import run_swiglu_golden

    run_norm_goldens(rms_norm, rms_norm_gated, l2norm)
    run_swiglu_golden(swiglu)
    run_moe_gate_golden(gate)
    run_expert_golden(expert)
