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


_MOE_KERNEL_READY = True

# The decode MoE uses the same packed expert ABI as the production EP driver.  Keep
# the names here in the order consumed by ``moe.moe`` so a block spec can be built
# without depending on the implementation signature hidden by ``@pl.jit``.
_MOE_INPUT_NAMES = (
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm_weight",
    "gate_weight", "correction_bias",
    "routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale",
    "routed_w3", "routed_w3_scale", "mxfp4_pair_lut",
    "shared_w1", "shared_w1_scale", "shared_w2", "shared_w2_scale",
    "shared_w3", "shared_w3_scale",
)
_MOE_OUTPUT_NAMES = ("next_pre_mix", "x_hc_out")
_BLOCK_SPEC_CACHE = {}


def _block_spec_names(kind):
    """Return the public input/output order for one complete decode Block."""
    mode = load_decode_attention_module(kind)
    return (*mode.ATTENTION_SPEC_NAMES, *_MOE_INPUT_NAMES, *_MOE_OUTPUT_NAMES)


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
    plan = resolve_decode_layer_plan(layer_id)
    if plan.kind not in (DecodeLayerKind.SWA, DecodeLayerKind.C2A_FULL, DecodeLayerKind.C2A_REUSE):
        missing.append(f"{plan.kind.name} Block composition is not part of the causal encoder")
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


def _golden_moe_fixture(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    values: Mapping[str, Any],
    num_tokens: int | None,
) -> torch.Tensor:
    """Evaluate the compact single-rank MoE fixture used by CPU Block goldens.

    The production ``moe.golden_moe`` owns the EP8 transport fixture and now
    takes one tensor dictionary.  Decode's small layer fixture intentionally
    keeps only local expert weights so all 20 layers remain cheap to execute;
    this helper applies the same RMSNorm, gate, shared SwiGLU, and routed
    SwiGLU equations without pretending to validate distributed transport.
    """
    import torch.nn.functional as F

    from models.deepseek_v4_1_flash.config import FLASH
    from models.deepseek_v4_1_flash.golden import gate
    from models.deepseek_v4_1_flash.quantization import (
        dequantize_mxfp4,
        dequantize_mxfp8,
        unpack_mx_b_scale,
    )

    shape = x.shape
    normalized = rms_norm(x.reshape(-1, shape[-1]), norm_weight)
    active = normalized.shape[0] if num_tokens is None else min(max(num_tokens, 0), normalized.shape[0])
    normalized_active = normalized[:active]
    route_weights, expert_indices = gate(
        normalized_active,
        values["gate_weight"],
        values["correction_bias"],
    )
    routed_w1 = dequantize_mxfp4(values["routed_w1"], values["routed_w1_scale"])
    routed_w2 = dequantize_mxfp4(values["routed_w2"], values["routed_w2_scale"])
    routed_w3 = dequantize_mxfp4(values["routed_w3"], values["routed_w3_scale"])
    shared_w1 = dequantize_mxfp8(
        values["shared_w1"], unpack_mx_b_scale(values["shared_w1_scale"])
    ).transpose(-2, -1)
    shared_w2 = dequantize_mxfp8(
        values["shared_w2"], unpack_mx_b_scale(values["shared_w2_scale"])
    ).transpose(-2, -1)
    shared_w3 = dequantize_mxfp8(
        values["shared_w3"], unpack_mx_b_scale(values["shared_w3_scale"])
    ).transpose(-2, -1)

    def evaluate_expert(
        expert_input: torch.Tensor,
        weight_one: torch.Tensor,
        weight_two: torch.Tensor,
        weight_three: torch.Tensor,
    ) -> torch.Tensor:
        gate_value = F.linear(expert_input.float(), weight_one.float()).clamp(max=FLASH.swiglu_limit)
        up_value = F.linear(expert_input.float(), weight_three.float()).clamp(
            -FLASH.swiglu_limit, FLASH.swiglu_limit
        )
        hidden = F.silu(gate_value) * up_value
        return F.linear(hidden, weight_two.float())

    output = evaluate_expert(normalized_active, shared_w1, shared_w2, shared_w3)
    for expert_id in range(routed_w1.shape[0]):
        token_rows, route_columns = torch.where(expert_indices == expert_id)
        if token_rows.numel() == 0:
            continue
        routed = evaluate_expert(
            normalized_active[token_rows],
            routed_w1[expert_id],
            routed_w2[expert_id],
            routed_w3[expert_id],
        )
        output[token_rows] += routed * route_weights[token_rows, route_columns].unsqueeze(-1)
    result = x.reshape(-1, shape[-1]).clone()
    result[:active] = output.to(x.dtype)
    return result.reshape(shape)


def _golden_moe_production(values: Mapping[str, Any], ffn_input: torch.Tensor) -> torch.Tensor:
    """Run the production tensor-dictionary MoE golden and return its output."""
    production_tensors = dict(values["tensors"])
    production_tensors.setdefault("x_hc", ffn_input.unsqueeze(0))
    golden_moe(production_tensors)
    return production_tensors["output"]


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
    if "tensors" in inspect.signature(golden_moe).parameters:
        if "tensors" in moe_inputs:
            ffn_output = _golden_moe_production(moe_inputs, ffn_input)
        else:
            ffn_output = _golden_moe_fixture(ffn_input, ffn_norm_weight, moe_inputs, num_tokens)
    else:
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


def _block_scale_alias(name):
    """Return the MX_B_NN shape needed when a rank slice enters a matmul."""
    shapes = {
        "wq_a_scale": "[D // MX_GROUP, Q_LORA]",
        "wq_b_scale": "[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM]",
        "wkv_scale": "[D // MX_GROUP, HEAD_DIM]",
        "wo_b_scale": "[LOCAL_O_WIDTH // MX_GROUP, D]",
        "index_wq_b_scale": "[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM]",
        "routed_w1_scale": "[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER]",
        "routed_w2_scale": "[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D]",
        "routed_w3_scale": "[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER]",
        "shared_w1_scale": "[D // MX_GROUP, MOE_INTER]",
        "shared_w2_scale": "[MOE_INTER // MX_GROUP, D]",
        "shared_w3_scale": "[D // MX_GROUP, MOE_INTER]",
    }
    return shapes.get(name)


def _make_decode_block_program(layer_id, world_size, epochs, specs):
    """Build one complete Attention-mHC-MoE-mHC EP/TP Block entry.

    The generated host keeps one rank-local Attention entry and one rank-local
    packed MoE entry in the same launch.  Attention collectives use the TP group
    containing the rank; the MoE transport uses the full EP world.  The function
    is generated from the checked spec order because PyPTO host entries require
    explicit tensor annotations for every parameter.
    """
    if specs is None:
        raise ValueError("Block host construction requires TensorSpec inputs")
    from types import SimpleNamespace

    import pypto.language as pl
    import pypto.language.distributed as pld

    from models.deepseek_v4_1_flash import config as C
    from models.deepseek_v4_1_flash import moe as moe_module

    plan = resolve_decode_layer_plan(layer_id)
    reason = decode_layer_kernel_skip_reason(layer_id)
    if reason:
        raise NotImplementedError(reason)
    mode = load_decode_attention_module(plan.kind)
    expected_names = _block_spec_names(plan.kind)
    actual_names = tuple(spec.name for spec in specs)
    if actual_names != expected_names:
        raise ValueError(
            f"Block specs must match the host parameter names and order: "
            f"expected {expected_names}, got {actual_names}"
        )
    if world_size != C.EP_SIZE:
        raise ValueError(f"Block world size must match EP_SIZE={C.EP_SIZE}")
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be in [1, 1000]")

    # ``@pl.jit`` wrappers intentionally hide their Python signatures.  The
    # mode and MoE entries are nevertheless stable positional ABIs, so emit the
    # host body with the exact spec-driven annotations used by existing hosts.
    dtype_map = {
        torch.bfloat16: pl.BF16,
        torch.float32: pl.FP32,
        torch.float8_e4m3fn: pl.FP8E4M3FN,
        torch.int32: pl.INT32,
        torch.int64: pl.INT64,
        torch.uint8: pl.UINT8,
        torch.int16: pl.INT16,
    }
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None:
        dtype_map[e8m0_dtype] = pl.FP8E8M0
    tensor_specs = SimpleNamespace(
        **{
            spec.name: SimpleNamespace(shape=spec.shape, dtype=dtype_map.get(spec.dtype, spec.dtype))
            for spec in specs
            if hasattr(spec, "shape")
        }
    )
    mode_rank_names = {
        DecodeLayerKind.SWA: "decode_swa_rank",
        DecodeLayerKind.C2A_FULL: "decode_c2a_full_rank",
        DecodeLayerKind.C2A_REUSE: "decode_c2a_reuse_rank",
    }
    mode_rank = getattr(mode, mode_rank_names[plan.kind])
    source_lines = [
        "@pl.jit.host",
        "def block_group(",
    ]
    for name in expected_names:
        spec = f"tensor_specs.{name}"
        if name == "attention_epoch":
            source_lines.append(f"    {name}: pl.Scalar[pl.INT32],")
        else:
            direction = "pl.InOut" if name in {
                "window_cache", "window_cache_scale", "compressed_cache", "compressed_cache_scale",
                "index_cache", "index_cache_scale", "state_cache", "attention_output",
                "topk_indices",
            } else "pl.Out" if name in {"attention_hidden", "attention_pre_mix", *_MOE_OUTPUT_NAMES} else "pl.Tensor"
            annotation = f"pl.Tensor[{spec}.shape, {spec}.dtype]"
            if direction != "pl.Tensor":
                annotation = f"{direction}[{annotation}]"
            source_lines.append(
                f"    {name}: {annotation},"
            )
    source_lines.extend([
        "):",
        "    attention_data_buf = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)",
        "    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)",
        "    recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)",
        "    recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)",
        "    recv_scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)",
        "    recv_weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)",
        "    recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)",
        "    arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)",
        "    data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)",
        "    routed_output_buf = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)",
        "    combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)",
        "    ffn_input_buf = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)",
        "    for step in pl.range(EPOCHS):",
        "        for rank in pl.range(WORLD_SIZE):",
        "            attention_data = pld.window(attention_data_buf, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)",
        "            attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)",
        "            recv_meta = pld.window(recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)",
        "            recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)",
        "            recv_scale = pld.window(recv_scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)",
        "            recv_weights = pld.window(recv_weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)",
        "            recv_routes = pld.window(recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)",
        "            arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
        "            data_arrived = pld.window(data_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
        "            routed_output = pld.window(routed_output_buf, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)",
        "            combine_arrived = pld.window(combine_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
        "            ffn_input = ffn_input_buf",
    ])
    for name in mode.LEAF_NAMES:
        shape = _block_scale_alias(name)
        if shape:
            source_lines.append(
                f"            {name}_r: pl.Tensor[{shape}, pl.FP8E8M0, pl.MX_B_NN] = {name}[rank]"
            )
    attention_args = [
        "x_hc[rank]", "incoming_pre_mix[rank]", "hc_attn_fn[rank]", "hc_attn_scale[rank]",
        "hc_attn_base[rank]", "attn_norm_weight[rank]",
    ]
    for name in mode.LEAF_NAMES:
        attention_args.append(f"{name}_r" if _block_scale_alias(name) else f"{name}[rank]")
    attention_args.extend([
        "attention_output[rank]", "attention_hidden[rank]", "attention_pre_mix[rank]",
        "attention_data", "attention_signal", "rank", "pl.read(num_tokens, [rank])",
        "attention_epoch + step",
        "device=rank",
    ])
    source_lines.append(f"            attention_rank({', '.join(attention_args)})")
    source_lines.extend(
        [
            "        # Launch the EP MoE only after every Attention rank has arrived.",
            "        for rank in pl.range(WORLD_SIZE):",
            "            recv_meta = pld.window(recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)",
            "            recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)",
            "            recv_scale = pld.window(recv_scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)",
            "            recv_weights = pld.window(recv_weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)",
            "            recv_routes = pld.window(recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)",
            "            arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
            "            data_arrived = pld.window(data_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
            "            routed_output = pld.window(routed_output_buf, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)",
            "            combine_arrived = pld.window(combine_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
            "            ffn_input = ffn_input_buf",
        ]
    )
    for name in _MOE_INPUT_NAMES[6:19]:
        shape = _block_scale_alias(name)
        if shape:
            source_lines.append(
                f"            {name}_r: pl.Tensor[{shape}, pl.FP8E8M0, pl.MX_B_NN] = {name}[rank]"
            )
    moe_args = [
        "attention_hidden[rank]", "attention_pre_mix[rank]",
        "hc_ffn_fn[rank]", "hc_ffn_scale[rank]", "hc_ffn_base[rank]", "ffn_norm_weight[rank]",
        "gate_weight[rank]", "correction_bias[rank]",
    ]
    for name in ("routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale", "mxfp4_pair_lut"):
        moe_args.append(f"{name}_r" if _block_scale_alias(name) else f"{name}[rank]")
    for name in ("shared_w1", "shared_w1_scale", "shared_w2", "shared_w2_scale", "shared_w3", "shared_w3_scale"):
        moe_args.append(f"{name}_r" if _block_scale_alias(name) else f"{name}[rank]")
    moe_args.extend([
        "next_pre_mix[rank]", "ffn_input", "x_hc_out[rank]",
        "recv_meta", "recv_x", "recv_scale", "recv_weights", "recv_routes",
        "arrived", "data_arrived", "routed_output", "combine_arrived",
        "num_tokens", "rank", "attention_epoch + step",
        "device=rank",
    ])
    source_lines.append(f"            decode_moe({', '.join(moe_args)})")
    source = "\n".join(source_lines)
    namespace = {
        "pl": pl,
        "pld": pld,
        "tensor_specs": tensor_specs,
        "attention_rank": mode_rank,
        "decode_moe": moe_module.moe_test,
        "DECODE_MAX_TOKENS": C.DECODE_MAX_TOKENS,
        "TP_SIZE": C.TP_SIZE,
        "EP_SIZE": C.EP_SIZE,
        "N_LOCAL_EXPERTS": C.N_LOCAL_EXPERTS,
        "RECV_MAX": C.RECV_MAX,
        "D": C.D,
        "MX_GROUP": C.MX_GROUP,
        "Q_LORA": C.Q_LORA,
        "HEAD_DIM": C.HEAD_DIM,
        "LOCAL_H": C.LOCAL_H,
        "LOCAL_O_WIDTH": C.LOCAL_O_WIDTH,
        "INDEX_H": C.INDEX_H,
        "INDEX_DIM": C.INDEX_DIM,
        "MOE_INTER": C.MOE_INTER,
        "AUX_WIDTH": C.AUX_WIDTH,
        "ROUTE_WIDTH": C.ROUTE_WIDTH,
        "MOE_TOKENS": moe_module.MOE_TOKENS,
        "TOPK": C.TOPK,
        "EPOCHS": epochs,
        "WORLD_SIZE": world_size,
    }
    filename = f"<deepseek_v41_decode_block_{layer_id}_{plan.kind.name.lower()}>"
    import linecache

    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    compiled = compile(source, filename, "exec")
    exec(compiled, namespace)
    return namespace["block_group"]


def build_decode_block_specs(layer_id: int, *, world_size: int | None = None, tokens: int = 16, seed: int = 17):
    """Build a runnable packed-EP Block fixture for the selected encoder layer."""
    from models.deepseek_v4_1_flash import config as C
    from models.deepseek_v4_1_flash import moe as moe_module
    from models.deepseek_v4_1_flash.expert_routed import (
        MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS,
    )
    from golden import ScalarSpec, TensorSpec

    world_size = C.EP_SIZE if world_size is None else world_size
    if world_size != C.EP_SIZE:
        raise ValueError(f"world_size must match EP_SIZE={C.EP_SIZE}")
    if tokens != moe_module.MOE_TOKENS:
        raise ValueError(f"decode MoE currently requires tokens={moe_module.MOE_TOKENS}")
    plan = resolve_decode_layer_plan(layer_id)
    reason = decode_layer_kernel_skip_reason(layer_id)
    if reason:
        raise NotImplementedError(reason)
    mode = load_decode_attention_module(plan.kind)
    cache_key = (layer_id, plan.kind, world_size, tokens, seed)
    if cache_key in _BLOCK_SPEC_CACHE:
        return _BLOCK_SPEC_CACHE[cache_key]
    requests = min(tokens, 4)
    pages = tokens + 1
    compressed_pages = requests * 4 + 1
    bf, fp8, e8m0 = torch.bfloat16, torch.float8_e4m3fn, getattr(torch, "float8_e8m0fnu", torch.uint8)
    selected = []

    def add(name, shape, dtype, *, resident="stacked"):
        selected.append(TensorSpec(name, [world_size, *shape], dtype, resident=resident))

    # Replicated mHC boundaries and visible Attention outputs.
    add("x_hc", [tokens, C.HC_MULT, C.D], torch.float32)
    add("incoming_pre_mix", [tokens, C.HC_MULT], torch.float32)
    add("hc_attn_fn", [C.MIX_HC, C.HC_DIM], torch.float32)
    add("hc_attn_scale", [3], torch.float32)
    add("hc_attn_base", [C.MIX_HC], torch.float32)
    add("attn_norm_weight", [C.D], bf)
    leaf_shapes = {
        "wq_a": ([C.D, C.Q_LORA], fp8), "wq_a_scale": ([C.D // C.MX_GROUP, C.Q_LORA], e8m0),
        "q_norm_weight": ([C.Q_LORA], bf), "wq_b": ([C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], fp8),
        "wq_b_scale": ([C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], e8m0),
        "wkv": ([C.D, C.HEAD_DIM], fp8), "wkv_scale": ([C.D // C.MX_GROUP, C.HEAD_DIM], e8m0),
        "kv_norm_weight": ([C.HEAD_DIM], bf), "attn_sink": ([C.LOCAL_H], torch.float32),
        "wo_a": ([C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], bf), "wo_b": ([C.LOCAL_O_WIDTH, C.D], fp8),
        "wo_b_scale": ([C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], e8m0),
        "rope_cos": ([tokens, C.ROPE_DIM // 2], torch.float32), "rope_sin": ([tokens, C.ROPE_DIM // 2], torch.float32),
        "window_slots": ([tokens], torch.int64), "window_indices": ([tokens, C.BLOCK_SIZE], torch.int32),
        "window_cache": ([pages, C.BLOCK_SIZE, 1, C.HEAD_DIM], fp8),
        "window_cache_scale": ([pages, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], e8m0),
    }
    if plan.kind is not DecodeLayerKind.SWA:
        leaf_shapes.update({
            "compressed_cache": ([compressed_pages, C.BLOCK_SIZE, 1, C.HEAD_DIM // 2], torch.uint8),
            "compressed_cache_scale": ([compressed_pages, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], fp8),
            "topk_indices": ([tokens, C.INDEX_TOPK], torch.int32),
        })
    if plan.kind is DecodeLayerKind.C2A_FULL:
        leaf_shapes.update({
            "token_to_req_indices": ([tokens], torch.int32), "compressed_lens": ([tokens], torch.int32),
            "index_cache": ([compressed_pages, C.BLOCK_SIZE, 1, C.INDEX_DIM // 2], torch.uint8),
            "index_cache_scale": ([compressed_pages, C.BLOCK_SIZE, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], e8m0),
            "index_block_table": ([requests, 8], torch.int32), "position_ids": ([tokens], torch.int32),
            "compressed_rope_cos": ([tokens, C.ROPE_DIM // 2], torch.float32),
            "compressed_rope_sin": ([tokens, C.ROPE_DIM // 2], torch.float32),
            "compressor_wkv": ([C.D, C.HEAD_DIM], torch.float32), "compressor_wgate": ([C.D, C.HEAD_DIM], torch.float32),
            "query_start_loc": ([requests + 1], torch.int32), "state_block_table": ([requests, 1], torch.int32),
            "state_cache": ([C.MAX_BATCH_PER_DP + 1, C.STATE_CAPACITY, C.STATE_WIDTH], torch.float32),
            "compressor_norm_weight": ([C.HEAD_DIM], bf), "compressed_slots": ([tokens], torch.int64),
            "index_wk": ([C.HEAD_DIM, C.INDEX_DIM], bf), "index_norm_weight": ([C.INDEX_DIM], bf),
            "index_wq_b": ([C.Q_LORA, C.INDEX_H * C.INDEX_DIM], fp8),
            "index_wq_b_scale": ([C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], e8m0),
            "index_weights_proj": ([C.D, C.INDEX_H], bf),
        })
    for name in mode.LEAF_NAMES:
        add(name, *leaf_shapes[name])
    add("attention_output", [tokens, C.D], bf)
    add("attention_hidden", [tokens, C.HC_MULT, C.D], torch.float32)
    add("attention_pre_mix", [tokens, C.HC_MULT], torch.float32)
    selected.extend(
        [
            TensorSpec("num_tokens", [world_size], torch.int32, init_value=tokens),
            ScalarSpec("attention_epoch", torch.int32, 1),
        ]
    )

    moe_shapes = {
        "hc_ffn_fn": ([C.MIX_HC, C.HC_DIM], torch.float32), "hc_ffn_scale": ([3], torch.float32),
        "hc_ffn_base": ([C.MIX_HC], torch.float32), "ffn_norm_weight": ([C.D], bf),
        "gate_weight": ([C.N_EXPERTS, C.D], torch.float32), "correction_bias": ([C.N_EXPERTS], torch.float32),
        "routed_w1": ([C.N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8),
        "routed_w1_scale": ([C.N_LOCAL_EXPERTS * (C.D // C.MX_GROUP), C.MOE_INTER], e8m0),
        "routed_w2": ([C.N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8),
        "routed_w2_scale": ([C.N_LOCAL_EXPERTS * (C.MOE_INTER // C.MX_GROUP), C.D], e8m0),
        "routed_w3": ([C.N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8),
        "routed_w3_scale": ([C.N_LOCAL_EXPERTS * (C.D // C.MX_GROUP), C.MOE_INTER], e8m0),
        "mxfp4_pair_lut": ([2, 256], torch.int16), "shared_w1": ([C.D, C.MOE_INTER], fp8),
        "shared_w1_scale": ([C.D // C.MX_GROUP, C.MOE_INTER], e8m0), "shared_w2": ([C.MOE_INTER, C.D], fp8),
        "shared_w2_scale": ([C.MOE_INTER // C.MX_GROUP, C.D], e8m0), "shared_w3": ([C.D, C.MOE_INTER], fp8),
        "shared_w3_scale": ([C.D // C.MX_GROUP, C.MOE_INTER], e8m0),
    }
    for name in _MOE_INPUT_NAMES:
        add(name, *moe_shapes[name])
    selected.extend([
        TensorSpec("next_pre_mix", [world_size, tokens, C.HC_MULT], torch.float32, resident="stacked"),
        TensorSpec("x_hc_out", [world_size, tokens, C.HC_MULT, C.D], torch.float32, resident="stacked"),
    ])
    _BLOCK_SPEC_CACHE[cache_key] = selected
    return selected


def make_decode_layer_program(layer_id, world_size, epochs, *, stage="attention", specs=None):
    """Build the selected Attention half or the complete decode Block."""
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
        return _make_decode_block_program(layer_id, world_size, epochs, specs)
    raise ValueError(f"unknown decode stage: {stage}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("attention", "block"), default="attention")
    parser.add_argument("--cpu-golden", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
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
        reason = decode_layer_kernel_skip_reason(args.layer_id)
        if reason:
            parser.error(reason)
        specs = build_decode_block_specs(args.layer_id)
        from models.deepseek_v4_1_flash import config as C

        program = make_decode_layer_program(args.layer_id, C.EP_SIZE, 1, stage="block", specs=specs)
        if args.compile_only:
            from golden import run

            result = run(
                fn=program,
                specs=specs,
                config={"platform": "a5sim", "device_id": 0},
                compile_only=True,
            )
            if not result.passed:
                raise SystemExit(result.error or 1)
        else:
            print(f"[BLOCK] READY layer={args.layer_id} world_size={C.EP_SIZE} tokens=16")
        return
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
    "build_decode_block_specs",
    "golden_decode_layer",
    "make_decode_layer_program",
    "resolve_decode_layer_plan",
]


if __name__ == "__main__":
    main()
