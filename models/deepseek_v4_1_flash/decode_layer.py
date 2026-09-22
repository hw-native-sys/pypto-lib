# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode Block golden composition and mode-selected program dispatch."""

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult
from models.deepseek_v4_1_flash.decode_layer_plan import (
    REPRESENTATIVE_LAYER_IDS,
    DecodeLayerKind,
    DecodeLayerPlan,
    load_decode_attention_module,
    resolve_decode_layer_plan,
)
from models.deepseek_v4_1_flash.golden import rms_norm
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre
from models.deepseek_v4_1_flash.moe import golden_moe


@dataclass(frozen=True)
class DecodeLayerGoldenResult:
    """Block outputs, intermediate boundaries, and attention state updates."""

    output: torch.Tensor
    next_pre_mix: torch.Tensor
    attention_input: torch.Tensor
    attention_output: torch.Tensor
    attention_hidden: torch.Tensor
    ffn_input: torch.Tensor
    ffn_output: torch.Tensor
    attention: AttentionGoldenResult


_MOE_KERNEL_READY = False


def _attention_golden(mode, kind: DecodeLayerKind):
    golden = getattr(mode, "ATTENTION_GOLDEN", None) or getattr(mode, "GOLDEN", None)
    if golden is not None:
        return golden
    names = {
        DecodeLayerKind.C1A_FULL: "golden_decode_attn_c1a_full",
        DecodeLayerKind.C1A_REINDEX: "golden_decode_attn_c1a_reindex",
        DecodeLayerKind.C1A_REUSE: "golden_decode_attn_c1a_reuse",
    }
    try:
        return getattr(mode, names[kind])
    except (AttributeError, KeyError) as error:
        raise AttributeError(f"{mode.__name__} does not expose the {kind.name} Attention golden") from error


def attention_half_skip_reason(layer_id: int) -> str | None:
    """Return the selected mode's Attention-only readiness reason."""
    plan = resolve_decode_layer_plan(layer_id)
    mode = load_decode_attention_module(plan.kind)
    skip_reason = getattr(mode, "skip_reason", None)
    if skip_reason is not None:
        return skip_reason(layer_id)
    return None if getattr(mode, "KERNEL_READY", True) else f"{plan.kind.name} attention kernel is pending"


def decode_layer_kernel_skip_reason(layer_id: int) -> str | None:
    """Return dependencies that prevent full Block device compilation."""
    missing = []
    attention_reason = attention_half_skip_reason(layer_id)
    if attention_reason:
        missing.append(attention_reason)
    if not _MOE_KERNEL_READY:
        missing.append("EP8 MoE kernel integration")
    return None if not missing else "decode_layer requires " + " and ".join(missing)


def decode_layer_attention_inputs(layer_id: int) -> tuple[str, ...]:
    """Return the exact golden inputs consumed by the resolved attention mode."""
    plan = resolve_decode_layer_plan(layer_id)
    mode = load_decode_attention_module(plan.kind)
    golden = _attention_golden(mode, plan.kind)
    return tuple(inspect.signature(golden).parameters)


def _select_inputs(function, values: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    selected = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in overrides:
            selected[name] = overrides[name]
        elif name in values:
            selected[name] = values[name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise KeyError(f"missing {function.__name__} input {name}")
    return selected


def golden_decode_layer(
    layer_id: int,
    x_hc: torch.Tensor,
    incoming_pre_mix: torch.Tensor,
    hc_attn_fn: torch.Tensor,
    hc_attn_scale: torch.Tensor,
    hc_attn_base: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    hc_ffn_fn: torch.Tensor,
    hc_ffn_scale: torch.Tensor,
    hc_ffn_base: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    attention_inputs: Mapping[str, Any],
    moe_inputs: Mapping[str, Any],
    num_tokens: int | None = None,
) -> DecodeLayerGoldenResult:
    """Evaluate the mHC-Attention-mHC-MoE-mHC Block order."""
    if num_tokens is not None and num_tokens != x_hc.shape[0]:
        raise ValueError("Block golden currently requires all capacity rows to be active")
    plan = resolve_decode_layer_plan(layer_id)
    mode = load_decode_attention_module(plan.kind)
    attention_golden = _attention_golden(mode, plan.kind)
    attn_pre, attn_post, attn_residual = golden_mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    attention_input = golden_mhc_pre(x_hc, incoming_pre_mix)
    normalized_attention = rms_norm(attention_input, attn_norm_weight)
    attention_kwargs = _select_inputs(attention_golden, attention_inputs, {"x": normalized_attention})
    attention = attention_golden(**attention_kwargs)
    attention_hidden = golden_mhc_post(attention.output, x_hc, attn_post, attn_residual)

    next_pre_mix, ffn_post, ffn_residual = golden_mhc_mixes(
        attention_hidden, hc_ffn_fn, hc_ffn_scale, hc_ffn_base
    )
    ffn_input = golden_mhc_pre(attention_hidden, attn_pre)
    moe_overrides = {"x": ffn_input, "norm_weight": ffn_norm_weight}
    if num_tokens is not None:
        moe_overrides["num_tokens"] = num_tokens
    moe_kwargs = _select_inputs(golden_moe, moe_inputs, moe_overrides)
    ffn_output = golden_moe(**moe_kwargs)
    output = golden_mhc_post(ffn_output, attention_hidden, ffn_post, ffn_residual)
    return DecodeLayerGoldenResult(
        output=output,
        next_pre_mix=next_pre_mix,
        attention_input=attention_input,
        attention_output=attention.output,
        attention_hidden=attention_hidden,
        ffn_input=ffn_input,
        ffn_output=ffn_output,
        attention=attention,
    )


def make_decode_layer_program(layer_id, world_size, epochs, *, stage="attention", specs=None):
    """Dispatch program construction to the selected mode before JIT discovery."""
    if stage == "attention":
        if specs is None:
            raise ValueError("Attention host entry requires tensor specs")
        plan = resolve_decode_layer_plan(layer_id)
        mode = load_decode_attention_module(plan.kind)
        if getattr(mode, "COMPOSITION_ABI", "spec-driven") == "native":
            raise NotImplementedError(
                f"{plan.kind.name} uses its mode-native make_program(tokens, pages, epochs) entry"
            )
        return mode.make_program(world_size, epochs, specs)
    if stage == "block":
        reason = decode_layer_kernel_skip_reason(layer_id)
        raise NotImplementedError(reason or "Block hardware fixture is pending integration")
    raise ValueError(f"unknown decode stage: {stage}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("attention", "block"), default="attention")
    parser.add_argument("--cpu-golden", action="store_true")
    parser.add_argument("--layer-id", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.cpu_golden:
        if args.stage != "block":
            parser.error("--cpu-golden requires --stage block")
        from models.deepseek_v4_1_flash._golden_smoke import (
            run_decode_layer_goldens,
            run_two_layer_decode_chain,
        )

        run_decode_layer_goldens(golden_decode_layer, REPRESENTATIVE_LAYER_IDS.values())
        run_two_layer_decode_chain(golden_decode_layer)
        return
    if args.stage == "block":
        parser.error(decode_layer_kernel_skip_reason(args.layer_id) or "Block hardware fixture is pending")
    plan = resolve_decode_layer_plan(args.layer_id)
    mode = load_decode_attention_module(plan.kind)
    mode.main()


__all__ = [
    "DecodeLayerGoldenResult",
    "DecodeLayerKind",
    "DecodeLayerPlan",
    "REPRESENTATIVE_LAYER_IDS",
    "attention_half_skip_reason",
    "decode_layer_attention_inputs",
    "decode_layer_kernel_skip_reason",
    "golden_decode_layer",
    "make_decode_layer_program",
    "resolve_decode_layer_plan",
]


if __name__ == "__main__":
    main()
