# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small deterministic CPU fixtures for directly executing operator goldens.

Every operator file's script-entry guard calls one runner from here, so that
``python models/glm5_3_flash/<file>.py`` proves the reference before any kernel body
exists. The a2a3 daily CI selects a file purely by grepping for that guard, so a
file gains one only once its golden is real — a file whose golden is still a stub
must stay without it, or the sweep runs an operator that cannot pass. This module
deliberately spells the guard nowhere, so it is not selected as a case itself.

The fixtures are deliberately tiny and shape-reduced. They check that a reference
runs, keeps its declared shapes and dtypes, and satisfies the invariants an
operator owner can rely on. They are not accuracy tests — those belong to the
golden harness in ``golden/`` once the kernel exists.
"""

import torch

from models.glm5_3_flash.config import FLASH


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _report(name: str) -> None:
    print(f"[GOLDEN] PASS {name}")


def run_mhc_goldens(golden_mixes, golden_pre, golden_post, golden_head) -> None:
    """Exercise the four mHC references on one reduced hyper-connection site."""
    torch.manual_seed(11)
    tokens, width = 6, FLASH.hc_mult
    hidden = 64
    hc_dim = width * hidden
    mix_hc = FLASH.mix_hc

    x_hc = torch.randn(tokens, width, hidden, dtype=torch.bfloat16)
    function = torch.randn(mix_hc, hc_dim) * 0.02
    scale = torch.randn(3)
    base = torch.randn(mix_hc) * 0.1

    pre_mix, post_mix, residual_mix = golden_mixes(x_hc, function, scale, base)
    _check(pre_mix.shape == (tokens, width), f"pre_mix shape {tuple(pre_mix.shape)}")
    _check(post_mix.shape == (tokens, width), f"post_mix shape {tuple(post_mix.shape)}")
    _check(
        residual_mix.shape == (tokens, width, width),
        f"residual_mix shape {tuple(residual_mix.shape)}",
    )
    # Sinkhorn must leave the residual mix doubly stochastic.
    for axis in (-1, -2):
        deviation = (residual_mix.sum(dim=axis) - 1.0).abs().max()
        _check(deviation < 1e-3, f"residual_mix is not doubly stochastic on axis {axis}: {deviation}")
    _check(bool((pre_mix > 0).all()), "pre_mix must be positive")

    sublayer_input = golden_pre(x_hc, pre_mix)
    _check(
        sublayer_input.shape == (tokens, hidden),
        f"mhc_pre shape {tuple(sublayer_input.shape)}",
    )
    _check(sublayer_input.dtype is torch.bfloat16, f"mhc_pre dtype {sublayer_input.dtype}")

    streams = golden_post(sublayer_input, x_hc, post_mix, residual_mix)
    _check(streams.shape == x_hc.shape, f"mhc_post shape {tuple(streams.shape)}")
    _check(streams.dtype is torch.bfloat16, f"mhc_post dtype {streams.dtype}")

    collapsed = golden_head(x_hc)
    _check(collapsed.shape == (tokens, hidden), f"mhc_head shape {tuple(collapsed.shape)}")
    _check(collapsed.dtype is torch.bfloat16, f"mhc_head dtype {collapsed.dtype}")
    # GLM-5.3-Flash collapses with an unweighted mean, not a learned head.
    expected = x_hc.float().mean(dim=-2).to(torch.bfloat16)
    _check(torch.equal(collapsed, expected), "mhc_head must be the unweighted stream mean")

    _report("mhc")


def run_moe_gate_golden(golden_gate) -> None:
    """Exercise the sigmoid ``noaux_tc`` router on a reduced expert count."""
    torch.manual_seed(13)
    tokens, hidden, experts = 8, 64, 32
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    weight = torch.randn(experts, hidden)
    correction_bias = torch.randn(experts)

    weights, indices = golden_gate(x, weight, correction_bias)
    topk = FLASH.num_experts_per_tok
    _check(weights.shape == (tokens, topk), f"router weight shape {tuple(weights.shape)}")
    _check(indices.shape == (tokens, topk), f"router index shape {tuple(indices.shape)}")
    _check(indices.dtype is torch.int32, f"router index dtype {indices.dtype}")
    _check(bool((indices >= 0).all() and (indices < experts).all()), "router index out of range")
    for row in indices.tolist():
        _check(len(set(row)) == topk, f"router selected a duplicate expert: {row}")
    if FLASH.norm_topk_prob:
        total = weights.sum(dim=-1) / FLASH.routed_scaling_factor
        _check((total - 1.0).abs().max() < 1e-4, f"router weights are not normalised: {total}")

    _report("moe_gate")


def run_swiglu_golden(golden_swiglu) -> None:
    """Check the clamped SwiGLU's saturation behaviour at the configured limit."""
    torch.manual_seed(17)
    limit = FLASH.swiglu_limit
    gate_value = torch.tensor([[-100.0, 0.0, 100.0]])
    up_value = torch.tensor([[-100.0, 1.0, 100.0]])

    hidden = golden_swiglu(gate_value, up_value)
    _check(hidden.shape == gate_value.shape, f"swiglu shape {tuple(hidden.shape)}")
    # `up` is clamped on both sides, `gate` only from above.
    expected_high = torch.nn.functional.silu(torch.tensor(limit)) * limit
    _check(
        torch.allclose(hidden[0, 2], expected_high, atol=1e-5),
        f"swiglu did not clamp the upper tail: {hidden[0, 2]} vs {expected_high}",
    )
    expected_low = torch.nn.functional.silu(torch.tensor(-100.0)) * -limit
    _check(
        torch.allclose(hidden[0, 0], expected_low, atol=1e-5),
        f"swiglu clamped the gate from below: {hidden[0, 0]} vs {expected_low}",
    )

    _report("swiglu")


def run_norm_goldens(golden_rms_norm, golden_rms_norm_gated, golden_l2norm) -> None:
    """Exercise the three normalisations the backbone shares."""
    torch.manual_seed(19)
    tokens, hidden = 5, 64
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    weight = torch.randn(hidden, dtype=torch.bfloat16)

    normalized = golden_rms_norm(x, weight)
    _check(normalized.shape == x.shape, f"rms_norm shape {tuple(normalized.shape)}")
    _check(normalized.dtype is x.dtype, f"rms_norm dtype {normalized.dtype}")

    heads, head_dim = 4, 16
    value = torch.randn(tokens, heads, head_dim)
    gate_value = torch.randn(tokens, heads, head_dim)
    gated = golden_rms_norm_gated(value, torch.ones(head_dim), gate_value)
    _check(gated.shape == value.shape, f"rms_norm_gated shape {tuple(gated.shape)}")
    ungated = golden_rms_norm_gated(value, torch.ones(head_dim), torch.full_like(gate_value, 1e9))
    _check(
        torch.allclose(gated, ungated * torch.sigmoid(gate_value), atol=1e-4),
        "rms_norm_gated must apply a sigmoid gate after the norm",
    )

    unit = golden_l2norm(torch.randn(tokens, head_dim))
    _check(
        (unit.square().sum(dim=-1) - 1.0).abs().max() < 1e-4,
        "l2norm must return unit rows",
    )

    _report("norms")


def run_quantization_goldens(quantize_per_channel, quantize_per_token, dynamic_linear) -> None:
    """Check the W8A8 references reconstruct a float matmul within int8 error."""
    torch.manual_seed(23)
    tokens, in_features, out_features = 6, 128, 64
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    weight_int8, weight_scale = quantize_per_channel(weight)
    _check(weight_int8.dtype is torch.int8, f"weight dtype {weight_int8.dtype}")
    _check(
        weight_scale.shape == (out_features, 1),
        f"per-channel scale shape {tuple(weight_scale.shape)}",
    )

    x_int8, token_scale = quantize_per_token(x)
    _check(x_int8.dtype is torch.int8, f"activation dtype {x_int8.dtype}")
    _check(token_scale.shape == (tokens, 1), f"per-token scale shape {tuple(token_scale.shape)}")

    reference = torch.nn.functional.linear(x.float(), weight.float())
    quantized = dynamic_linear(x, weight_int8, weight_scale, out_dtype=torch.float32)
    error = (quantized - reference).abs().max() / reference.abs().max()
    _check(float(error) < 0.05, f"W8A8 reconstruction error is too large: {float(error)}")

    _report("quantization")


__all__ = [
    "run_mhc_goldens",
    "run_moe_gate_golden",
    "run_norm_goldens",
    "run_quantization_goldens",
    "run_swiglu_golden",
]
