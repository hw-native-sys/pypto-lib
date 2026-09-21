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

from models.glm5_3_flash.config import FLASH


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = FLASH.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)


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
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    value = weight.float() * value
    return (value * torch.sigmoid(gate.float())).to(dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalisation used on the KDA query and key.

    Note the ``+ eps`` inside the square root rather than ``max(norm, eps)``; the
    reference is explicit that this differs from ``F.normalize``.
    """
    return x / torch.sqrt(x.square().sum(dim=-1, keepdim=True) + eps)


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
    scores = torch.sigmoid(F.linear(x.float(), weight.float()))
    indices = (scores + correction_bias.float()).topk(FLASH.num_experts_per_tok, dim=-1).indices
    weights = scores.gather(-1, indices)
    if FLASH.norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * FLASH.routed_scaling_factor, indices.to(torch.int32)


def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the four hyper-connection streams into one sublayer input."""
    return (x.float() * pre_mix.float().unsqueeze(-1)).sum(dim=-2).to(x.dtype)


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
    """Project one HC stream into pre, post, and doubly-stochastic residual mixes.

    ``Glm5NextTextHyperConnection`` subclasses ``DeepseekV4HyperConnection`` with no
    changes, so this is the DeepSeek-V4 reference verbatim. The checkpoint supplies
    ``function`` as ``hc_{attn,ffn}_fn``, ``scale`` as ``hc_{attn,ffn}_scale`` and
    ``base`` as ``hc_{attn,ffn}_base``, once per site and twice per layer.
    """
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
    """Expand a sublayer result and mix the four residual streams.

    The official implementation performs this add in the stream dtype
    (``post.to(dtype) * sublayer + matmul(comb.to(dtype).T, residual)`` with
    ``dtype`` captured from the incoming stream, i.e. BF16). This reference
    accumulates in FP32 and rounds once, which is what a kernel with an FP32
    accumulator produces; a bit-exact comparison against HuggingFace needs the
    all-BF16 variant instead.
    """
    update = post_mix.unsqueeze(-1) * sublayer.unsqueeze(-2)
    skip = (residual_mix.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    return (update.float() + skip.float()).to(sublayer.dtype)


def hc_head(x: torch.Tensor) -> torch.Tensor:
    """Collapse the final HC streams.

    Unlike DeepSeek-V4, which applies the last layer's delayed pre-mix, GLM-5.3-Flash
    uses an unweighted mean (``Glm5NextTextHyperHead.forward`` is
    ``hidden_streams.mean(dim=2)``).
    """
    return x.float().mean(dim=-2).to(torch.bfloat16)


__all__ = [
    "expert",
    "gate",
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
    from models.glm5_3_flash._golden_smoke import run_moe_gate_golden
    from models.glm5_3_flash._golden_smoke import run_norm_goldens
    from models.glm5_3_flash._golden_smoke import run_swiglu_golden

    run_norm_goldens(rms_norm, rms_norm_gated, l2norm)
    run_swiglu_golden(swiglu)
    run_moe_gate_golden(gate)
