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


def _rsqrt_fp32_reference(value: torch.Tensor) -> torch.Tensor:
    """Round an accurate reciprocal square root to FP32 for host references.

    CPU torch.rsqrt(FP32) can differ by one ULP from the device high-precision
    path. Keep the argument and result FP32; evaluate the unary operation in
    FP64 so a host approximation does not flip a later BF16 rounding boundary.
    This is an accuracy reference, not a bit-exact device approximation.
    """
    return torch.rsqrt(value.double()).float()



def _sum_fp32_reference(value: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    """Evaluate an FP32 reduction accurately and round its result to FP32."""
    return value.double().sum(dim=dim, keepdim=keepdim).float()


def _exp_fp32_reference(value: torch.Tensor) -> torch.Tensor:
    """Evaluate an exponential accurately at the original FP32 boundary."""
    return torch.exp(value.double()).float()


def _sigmoid_fp32_reference(value: torch.Tensor) -> torch.Tensor:
    """Evaluate a sigmoid accurately at the original FP32 boundary."""
    return torch.sigmoid(value.double()).float()


def _silu_fp32_reference(value: torch.Tensor) -> torch.Tensor:
    """Evaluate SiLU accurately at the original FP32 boundary."""
    wide = value.double()
    return (wide * torch.sigmoid(wide)).float()


def _linear_fp32_reference(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Evaluate BF16 dot products accurately and round their outputs to FP32."""
    return (value.double() @ weight.double().t()).float()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = FLASH.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    value = value * _rsqrt_fp32_reference(_sum_fp32_reference(value.square()) / value.shape[-1] + eps)
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
    value = value * _rsqrt_fp32_reference(_sum_fp32_reference(value.square()) / value.shape[-1] + eps)
    value = weight.float() * value
    return (value * _sigmoid_fp32_reference(gate.float())).to(dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalisation used on the KDA query and key.

    Note the ``+ eps`` inside the square root rather than ``max(norm, eps)``; the
    reference is explicit that this differs from ``F.normalize``.
    """
    inverse = _rsqrt_fp32_reference(_sum_fp32_reference(x.square()) + eps)
    return (x * inverse).to(x.dtype)


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
    """Compute FP32 mHC mixes with accurately evaluated dots, unaries and reductions."""
    flat = x.flatten(-2).float()
    inv = torch.rsqrt((_sum_fp32_reference(flat.square()) / flat.shape[-1] + FLASH.rms_norm_eps).double()).float()
    mixes = (flat.double() @ function.double().t()).float() * inv
    hc = FLASH.hc_mult
    (p, q, r) = mixes.split((hc, hc, hc * hc), -1)
    pre = torch.sigmoid((p * scale[0] + base[:hc]).double()).float() + FLASH.hc_eps
    post = 2 * torch.sigmoid((q * scale[1] + base[hc:2 * hc]).double()).float()
    logits = r.unflatten(-1, (hc, hc)) * scale[2] + base[2 * hc:].unflatten(-1, (hc, hc))
    exps = torch.exp((logits - logits.amax(-1, keepdim=True)).double()).float()
    residual = exps / _sum_fp32_reference(exps) + FLASH.hc_eps
    residual = residual / (_sum_fp32_reference(residual, -2) + FLASH.hc_eps)
    for _ in range(iterations - 1):
        residual = residual / (_sum_fp32_reference(residual) + FLASH.hc_eps)
        residual = residual / (_sum_fp32_reference(residual, -2) + FLASH.hc_eps)
    return (pre, post, residual)


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
    Residual contributions follow the sublayer update in stream order.
    """
    update = post_mix.float().unsqueeze(-1) * sublayer.float().unsqueeze(-2)
    for stream in range(residual.shape[-2]):
        coefficient = residual_mix[..., stream, :].float().unsqueeze(-1)
        source = residual[..., stream, :].float().unsqueeze(-2)
        update = update + coefficient * source
    return update.to(sublayer.dtype)


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
    from models.glm5_3_flash._golden_smoke import (
        run_expert_golden,
        run_moe_gate_golden,
        run_norm_goldens,
        run_swiglu_golden,
    )

    run_norm_goldens(rms_norm, rms_norm_gated, l2norm)
    run_swiglu_golden(swiglu)
    run_moe_gate_golden(gate)
    run_expert_golden(expert)
