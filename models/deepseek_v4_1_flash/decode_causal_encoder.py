# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1-Flash decode composition for the causal encoder.

The released checkpoint executes layers 0 through 19 before the ratio-1
decoder layers.  This module owns that boundary and keeps its layer choice
static: layers 0-1 use SWA, layers 2/8/14 publish C2A state, and the other
ratio-2 layers reuse the most recent C2A state.  The device factory exposes
the complete mHC-Attention-mHC-MoE-mHC Block entries in the same order.  The
CPU golden below executes the same 20-layer schedule.
"""

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from models.deepseek_v4_1_flash.decode_layer import (
    DecodeLayerGoldenResult,
    DecodeLayerKind,
    DecodeLayerPlan,
    _MOE_INPUT_NAMES,
    _block_scale_alias,
    decode_layer_kernel_skip_reason,
    golden_decode_layer,
    load_decode_attention_module,
    make_decode_layer_program,
    resolve_decode_layer_plan,
)
from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.golden import hc_head


CAUSAL_ENCODER_START = 0
CAUSAL_ENCODER_END = 20
CAUSAL_ENCODER_LAYER_IDS = tuple(range(CAUSAL_ENCODER_START, CAUSAL_ENCODER_END))
CAUSAL_ENCODER_LAYERS = CAUSAL_ENCODER_LAYER_IDS
CAUSAL_ENCODER_KV_SOURCE_LAYERS = (2, 8, 14)
CAUSAL_ENCODER_INDEX_SOURCE_LAYERS = (2, 8, 14)
CAUSAL_ENCODER_ENGRAM_LAYERS = (1, 14)
CAUSAL_ENCODER_COMPRESS_RATIOS = (0, 0) + (2,) * 18

_EXPECTED_KINDS = (
    DecodeLayerKind.SWA,
    DecodeLayerKind.SWA,
    DecodeLayerKind.C2A_FULL,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_FULL,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_FULL,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
    DecodeLayerKind.C2A_REUSE,
)


@dataclass(frozen=True)
class CausalEncoderGoldenResult:
    """Final hidden state and every Block boundary in the causal encoder."""

    output: torch.Tensor
    x_hc: torch.Tensor
    next_pre_mix: torch.Tensor
    layers: tuple[DecodeLayerGoldenResult, ...]
    plans: tuple[DecodeLayerPlan, ...]

    @property
    def hidden(self) -> torch.Tensor:
        """Return the standard token-major hidden output."""
        return self.output

    @property
    def layer_outputs(self) -> tuple[torch.Tensor, ...]:
        """Return the residual-stream output after each encoder layer."""
        return tuple(layer.output for layer in self.layers)


def resolve_causal_encoder_layer_plan(layer_id: int) -> DecodeLayerPlan:
    """Resolve one layer in the causal encoder and reject decoder layers."""
    if not CAUSAL_ENCODER_START <= layer_id < CAUSAL_ENCODER_END:
        raise ValueError(
            f"causal encoder layer_id must be in "
            f"[{CAUSAL_ENCODER_START}, {CAUSAL_ENCODER_END}), got {layer_id}"
        )
    return resolve_decode_layer_plan(layer_id)


def causal_encoder_attention_kind(layer_id: int) -> DecodeLayerKind:
    """Return the statically selected Attention implementation for one layer."""
    return resolve_causal_encoder_layer_plan(layer_id).kind


def load_causal_encoder_attention_module(layer_id: int):
    """Load the mode module selected for one causal-encoder layer."""
    return load_decode_attention_module(causal_encoder_attention_kind(layer_id))


def validate_causal_encoder_schedule() -> tuple[DecodeLayerPlan, ...]:
    """Validate the official Hugging Face SWA/CSA2 schedule and return plans."""
    configured_ratios = FLASH.compress_ratios[CAUSAL_ENCODER_START:CAUSAL_ENCODER_END]
    if configured_ratios != CAUSAL_ENCODER_COMPRESS_RATIOS:
        raise RuntimeError(f"unexpected configured causal encoder ratios: {configured_ratios!r}")
    configured_engram = tuple(layer_id for layer_id in FLASH.engram_layer_ids if layer_id < CAUSAL_ENCODER_END)
    if configured_engram != CAUSAL_ENCODER_ENGRAM_LAYERS:
        raise RuntimeError(f"unexpected configured causal encoder Engram layers: {configured_engram!r}")
    plans = tuple(resolve_causal_encoder_layer_plan(layer_id) for layer_id in CAUSAL_ENCODER_LAYER_IDS)
    kinds = tuple(plan.kind for plan in plans)
    if kinds != _EXPECTED_KINDS:
        raise RuntimeError(f"unexpected causal encoder Attention schedule: {kinds!r}")
    ratios = tuple(plan.compression_ratio for plan in plans)
    if ratios != CAUSAL_ENCODER_COMPRESS_RATIOS:
        raise RuntimeError(f"unexpected causal encoder compression ratios: {ratios!r}")
    kv_sources = tuple(
        layer_id
        for layer_id, plan in zip(CAUSAL_ENCODER_LAYER_IDS, plans)
        if plan.kv_source_layer_id == layer_id
    )
    index_sources = tuple(
        layer_id
        for layer_id, plan in zip(CAUSAL_ENCODER_LAYER_IDS, plans)
        if plan.index_source_layer_id == layer_id
    )
    if kv_sources != CAUSAL_ENCODER_KV_SOURCE_LAYERS:
        raise RuntimeError(f"unexpected causal encoder KV sources: {kv_sources!r}")
    if index_sources != CAUSAL_ENCODER_INDEX_SOURCE_LAYERS:
        raise RuntimeError(f"unexpected causal encoder index sources: {index_sources!r}")
    for layer_id, plan in zip(CAUSAL_ENCODER_LAYER_IDS, plans):
        if plan.compression_ratio:
            expected_source = max(source for source in CAUSAL_ENCODER_KV_SOURCE_LAYERS if source <= layer_id)
            if plan.kv_source_layer_id != expected_source:
                raise RuntimeError(f"layer {layer_id} does not reuse KV source {expected_source}")
            expected_source = max(source for source in CAUSAL_ENCODER_INDEX_SOURCE_LAYERS if source <= layer_id)
            if plan.index_source_layer_id != expected_source:
                raise RuntimeError(f"layer {layer_id} does not reuse index source {expected_source}")
    if any(plan.compression_ratio not in (0, 2) for plan in plans):
        raise RuntimeError("causal encoder contains a non-SWA/non-C2A layer")
    if any(plan.kind in (DecodeLayerKind.C1A_FULL, DecodeLayerKind.C1A_REINDEX, DecodeLayerKind.C1A_REUSE) for plan in plans):
        raise RuntimeError("causal encoder must not select a C1A implementation")
    return plans


def _apply_engram(
    layer_id: int,
    x_hc: torch.Tensor,
    layer_values: dict[str, Any],
    engram_inputs: Mapping[int, Mapping[str, torch.Tensor]] | None,
) -> torch.Tensor:
    """Apply the optional official Engram update before layers 1 and 14."""
    local_inputs = layer_values.pop("engram_inputs", None)
    supplied = local_inputs
    if engram_inputs is not None and layer_id in engram_inputs:
        supplied = engram_inputs[layer_id]
    if supplied is None:
        return x_hc
    if layer_id not in CAUSAL_ENCODER_ENGRAM_LAYERS:
        raise ValueError(f"layer {layer_id} does not own an Engram block")
    required = ("hash_ids", "engram_table", "wkv_weight", "weight")
    missing = [name for name in required if name not in supplied]
    if missing:
        raise ValueError(f"Engram inputs for layer {layer_id} are missing {missing}")
    from models.deepseek_v4_1_flash.engram import golden_engram

    return golden_engram(
        supplied["hash_ids"],
        supplied["engram_table"],
        supplied["wkv_weight"],
        supplied["weight"],
        x_hc,
    )


def make_causal_encoder_layer_program(layer_id, world_size, epochs, *, stage="attention", specs=None):
    """Build the existing static PyPTO program for one encoder layer."""
    resolve_causal_encoder_layer_plan(layer_id)
    return make_decode_layer_program(layer_id, world_size, epochs, stage=stage, specs=specs)


def make_causal_encoder_programs(world_size, epochs, specs_by_layer, *, stage="attention"):
    """Build all 20 statically selected per-layer programs in execution order.

    ``specs_by_layer`` may be a mapping keyed by layer id or a sequence with
    one entry per causal-encoder layer.  ``stage="block"`` returns complete
    device Block entries; ``stage="attention"`` returns Attention halves.
    """
    validate_causal_encoder_schedule()
    if isinstance(specs_by_layer, Mapping):
        get_specs = specs_by_layer.__getitem__
    else:
        if len(specs_by_layer) != len(CAUSAL_ENCODER_LAYER_IDS):
            raise ValueError("specs_by_layer must contain one entry for each causal encoder layer")
        get_specs = lambda layer_id: specs_by_layer[layer_id - CAUSAL_ENCODER_START]
    return tuple(
        make_causal_encoder_layer_program(
            layer_id,
            world_size,
            epochs,
            stage=stage,
            specs=get_specs(layer_id),
        )
        for layer_id in CAUSAL_ENCODER_LAYER_IDS
    )


def build_causal_encoder_block_specs(*, seed: int = 17, tokens: int = 16, world_size: int | None = None):
    """Build one runnable packed-EP Block fixture for every encoder layer."""
    from models.deepseek_v4_1_flash.decode_layer import build_decode_block_specs

    return {
        layer_id: build_decode_block_specs(layer_id, world_size=world_size, tokens=tokens, seed=seed + layer_id)
        for layer_id in CAUSAL_ENCODER_LAYER_IDS
    }


def make_causal_encoder_block_programs(world_size, epochs, specs_by_layer):
    """Build the complete 20-layer device Block sequence."""
    return make_causal_encoder_programs(world_size, epochs, specs_by_layer, stage="block")


def build_causal_encoder_device_specs(
    *, seed: int = 17, tokens: int = 16, world_size: int | None = None
):
    """Build the single-entry ABI for the fused 20-layer device encoder.

    Layer weights and cache state are prefixed with ``l{layer}_``.  The
    residual stream and delayed mHC mix are the only values shared between
    layers; the fused host keeps those tensors in device-local storage and
    exposes only the final layer boundary to the runtime.
    """
    from golden import ScalarSpec, TensorSpec

    from models.deepseek_v4_1_flash import config as C

    world_size = C.EP_SIZE if world_size is None else world_size
    per_layer = build_causal_encoder_block_specs(
        seed=seed, tokens=tokens, world_size=world_size
    )
    specs = []
    shared_names = {"x_hc", "incoming_pre_mix"}
    local_names = {"attention_output", "attention_hidden", "attention_pre_mix"}
    output_names = {"next_pre_mix", "x_hc_out"}
    runtime_names = {"attention_num_tokens", "num_tokens", "attention_epoch"}
    for layer_id in CAUSAL_ENCODER_LAYER_IDS:
        for spec in per_layer[layer_id]:
            if spec.name in shared_names:
                if layer_id == CAUSAL_ENCODER_START:
                    specs.append(spec)
                continue
            if spec.name in local_names or spec.name in output_names or spec.name in runtime_names:
                continue
            specs.append(
                TensorSpec(
                    f"l{layer_id}_{spec.name}",
                    list(spec.shape),
                    spec.dtype,
                    init_value=spec.init_value,
                    resident=spec.resident,
                )
            )
    final_specs = per_layer[CAUSAL_ENCODER_END - 1]
    final_by_name = {spec.name: spec for spec in final_specs}
    specs.extend(
        [
            TensorSpec("next_pre_mix", list(final_by_name["next_pre_mix"].shape), torch.float32),
            TensorSpec("x_hc_out", list(final_by_name["x_hc_out"].shape), torch.float32),
            ScalarSpec("attention_num_tokens", torch.int32, tokens),
            TensorSpec("num_tokens", [world_size], torch.int32, init_value=tokens),
            ScalarSpec("encoder_epoch", torch.int32, 1, compile_runtime=True),
        ]
    )
    return specs


def _causal_encoder_device_spec_names(specs):
    """Return the expected parameter order for the fused host entry."""
    return tuple(spec.name for spec in specs)


def make_causal_encoder_device_program(world_size, epochs, specs):
    """Build one statically unrolled PyPTO host for encoder layers 0 through 19."""
    if specs is None:
        raise ValueError("causal encoder device construction requires TensorSpec inputs")
    from types import SimpleNamespace

    import pypto.language as pl
    import pypto.language.distributed as pld

    from models.deepseek_v4_1_flash import config as C
    from models.deepseek_v4_1_flash import moe as moe_module

    validate_causal_encoder_schedule()
    if world_size != C.EP_SIZE:
        raise ValueError(f"causal encoder world size must match EP_SIZE={C.EP_SIZE}")
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be in [1, 1000]")
    expected_names = _causal_encoder_device_spec_names(
        build_causal_encoder_device_specs(world_size=world_size, tokens=moe_module.MOE_TOKENS)
    )
    actual_names = tuple(spec.name for spec in specs)
    if actual_names != expected_names:
        raise ValueError(
            f"causal encoder specs must match the fused host parameter order: "
            f"expected {expected_names}, got {actual_names}"
        )

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
            spec.name: SimpleNamespace(
                shape=spec.shape, dtype=dtype_map.get(spec.dtype, spec.dtype)
            )
            for spec in specs
            if hasattr(spec, "shape")
        }
    )
    mode_rank_names = {
        DecodeLayerKind.SWA: "decode_swa_rank",
        DecodeLayerKind.C2A_FULL: "decode_c2a_full_rank",
        DecodeLayerKind.C2A_REUSE: "decode_c2a_reuse_rank",
    }

    source_lines = ["@pl.jit.host", "def causal_encoder_group("]
    for spec in specs:
        if not hasattr(spec, "shape"):
            source_lines.append(f"    {spec.name}: pl.Scalar[pl.INT32],")
            continue
        annotation = f"pl.Tensor[tensor_specs.{spec.name}.shape, tensor_specs.{spec.name}.dtype]"
        state_suffixes = (
            "window_cache",
            "window_cache_scale",
            "compressed_cache",
            "compressed_cache_scale",
            "index_cache",
            "index_cache_scale",
            "state_cache",
            "topk_indices",
        )
        if spec.name in {"next_pre_mix", "x_hc_out"}:
            annotation = f"pl.Out[{annotation}]"
        elif spec.name.startswith("l") and spec.name.endswith(state_suffixes):
            annotation = f"pl.InOut[{annotation}]"
        source_lines.append(f"    {spec.name}: {annotation},")
    source_lines.extend(
        [
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
            "    for step in pl.range(EPOCHS):",
            "        attention_data = pld.window(attention_data_buf, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)",
            "        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)",
            "        recv_meta = pld.window(recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)",
            "        recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)",
            "        recv_scale = pld.window(recv_scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)",
            "        recv_weights = pld.window(recv_weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)",
            "        recv_routes = pld.window(recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)",
            "        arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
            "        data_arrived = pld.window(data_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
            "        routed_output = pld.window(routed_output_buf, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)",
            "        combine_arrived = pld.window(combine_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)",
            "        current_x_hc = x_hc",
            "        current_pre_mix = incoming_pre_mix",
        ]
    )

    for layer_id in CAUSAL_ENCODER_LAYER_IDS:
        plan = resolve_causal_encoder_layer_plan(layer_id)
        mode = load_decode_attention_module(plan.kind)
        rank_name = mode_rank_names[plan.kind]
        prefix = f"l{layer_id}_"
        attention_hidden = f"attention_hidden_{layer_id}"
        attention_pre_mix = f"attention_pre_mix_{layer_id}"
        attention_output = f"attention_output_{layer_id}"
        ffn_input = f"ffn_input_{layer_id}"
        next_pre_mix = "next_pre_mix" if layer_id == CAUSAL_ENCODER_END - 1 else f"next_pre_mix_{layer_id}"
        next_x_hc = "x_hc_out" if layer_id == CAUSAL_ENCODER_END - 1 else f"x_hc_out_{layer_id}"
        source_lines.extend(
            [
                f"        {attention_output} = pl.create_tensor([EP_SIZE, MOE_TOKENS, D], dtype=pl.BF16)",
                f"        {attention_hidden} = pl.create_tensor([EP_SIZE, MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)",
                f"        {attention_pre_mix} = pl.create_tensor([EP_SIZE, MOE_TOKENS, HC_MULT], dtype=pl.FP32)",
                f"        {ffn_input} = pl.create_tensor([EP_SIZE, MOE_TOKENS, D], dtype=pl.BF16)",
                f"        {next_pre_mix} = {next_pre_mix if layer_id == CAUSAL_ENCODER_END - 1 else 'pl.create_tensor([EP_SIZE, MOE_TOKENS, HC_MULT], dtype=pl.FP32)'}",
                f"        {next_x_hc} = {next_x_hc if layer_id == CAUSAL_ENCODER_END - 1 else 'pl.create_tensor([EP_SIZE, MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)'}",
                "        # Launch every rank's Attention before any EP collective.",
                "        for rank in pl.range(WORLD_SIZE):",
            ]
        )
        for name in mode.LEAF_NAMES:
            source_name = f"{prefix}{name}"
            if name in {"compressed_cache", "compressed_cache_scale", "topk_indices"} and plan.kind is DecodeLayerKind.C2A_REUSE:
                source_layer = plan.kv_source_layer_id if name != "topk_indices" else plan.index_source_layer_id
                source_name = f"l{source_layer}_{name}"
            shape = _block_scale_alias(name)
            if shape:
                alias = f"{source_name}_r_{layer_id}"
                source_lines.append(
                    f"            {alias}: pl.Tensor[{shape}, pl.FP8E8M0, pl.MX_B_NN] = {source_name}[rank]"
                )
        attention_args = [
            "current_x_hc[rank]",
            "current_pre_mix[rank]",
            f"{prefix}hc_attn_fn[rank]",
            f"{prefix}hc_attn_scale[rank]",
            f"{prefix}hc_attn_base[rank]",
            f"{prefix}attn_norm_weight[rank]",
        ]
        for name in mode.LEAF_NAMES:
            source_name = f"{prefix}{name}"
            if name in {"compressed_cache", "compressed_cache_scale", "topk_indices"} and plan.kind is DecodeLayerKind.C2A_REUSE:
                source_layer = plan.kv_source_layer_id if name != "topk_indices" else plan.index_source_layer_id
                source_name = f"l{source_layer}_{name}"
            attention_args.append(
                f"{source_name}_r_{layer_id}" if _block_scale_alias(name) else f"{source_name}[rank]"
            )
        attention_args.extend(
            [
                f"{attention_output}[rank]",
                f"{attention_hidden}[rank]",
                f"{attention_pre_mix}[rank]",
                "attention_data",
                "attention_signal",
                "rank",
                "attention_num_tokens",
                f"encoder_epoch + step + {layer_id}",
                "device=rank",
            ]
        )
        source_lines.append(f"            {rank_name}({', '.join(attention_args)})")

        source_lines.append("        # Launch the EP MoE only after all Attention ranks have arrived.")
        source_lines.append("        for rank in pl.range(WORLD_SIZE):")
        for name in _MOE_INPUT_NAMES[6:19]:
            source_name = f"{prefix}{name}"
            shape = _block_scale_alias(name)
            if shape:
                alias = f"{source_name}_r_{layer_id}"
                source_lines.append(
                    f"            {alias}: pl.Tensor[{shape}, pl.FP8E8M0, pl.MX_B_NN] = {source_name}[rank]"
                )

        moe_args = [
            f"{attention_hidden}[rank]",
            f"{attention_pre_mix}[rank]",
            f"{prefix}hc_ffn_fn[rank]",
            f"{prefix}hc_ffn_scale[rank]",
            f"{prefix}hc_ffn_base[rank]",
            f"{prefix}ffn_norm_weight[rank]",
            f"{prefix}gate_weight[rank]",
            f"{prefix}correction_bias[rank]",
        ]
        for name in _MOE_INPUT_NAMES[6:13]:
            source_name = f"{prefix}{name}"
            moe_args.append(
                f"{source_name}_r_{layer_id}" if _block_scale_alias(name) else f"{source_name}[rank]"
            )
        for name in _MOE_INPUT_NAMES[13:19]:
            source_name = f"{prefix}{name}"
            moe_args.append(
                f"{source_name}_r_{layer_id}" if _block_scale_alias(name) else f"{source_name}[rank]"
            )
        moe_args.extend(
            [
                f"{next_pre_mix}[rank]",
                f"{ffn_input}[rank]",
                f"{next_x_hc}[rank]",
                "recv_meta",
                "recv_x",
                "recv_scale",
                "recv_weights",
                "recv_routes",
                "arrived",
                "data_arrived",
                "routed_output",
                "combine_arrived",
                "num_tokens",
                "rank",
                f"encoder_epoch + step + {layer_id}",
                "device=rank",
            ]
        )
        source_lines.append(f"            decode_moe({', '.join(moe_args)})")
        source_lines.extend(
            [
                f"        current_x_hc = {next_x_hc}",
                f"        current_pre_mix = {next_pre_mix}",
            ]
        )

    source = "\n".join(source_lines)
    namespace = {
        "pl": pl,
        "pld": pld,
        "tensor_specs": tensor_specs,
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
        "HC_MULT": C.HC_MULT,
        "AUX_WIDTH": C.AUX_WIDTH,
        "ROUTE_WIDTH": C.ROUTE_WIDTH,
        "MOE_TOKENS": moe_module.MOE_TOKENS,
        "TOPK": C.TOPK,
        "EPOCHS": epochs,
        "WORLD_SIZE": world_size,
    }
    for layer_id in CAUSAL_ENCODER_LAYER_IDS:
        mode = load_decode_attention_module(resolve_causal_encoder_layer_plan(layer_id).kind)
        namespace[mode_rank_names[resolve_causal_encoder_layer_plan(layer_id).kind]] = getattr(
            mode, mode_rank_names[resolve_causal_encoder_layer_plan(layer_id).kind]
        )
    import linecache

    filename = "<deepseek_v41_causal_encoder_fused>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    compiled = compile(source, filename, "exec")
    exec(compiled, namespace)
    return namespace["causal_encoder_group"]


def _zero_causal_encoder_device_golden(values):
    """Golden for the zero-initialized device bring-up fixture."""
    # mHC coefficients use a sigmoid gate.  With zero fixture weights its
    # pre-activation is zero, so the official boundary is sigmoid(0), not 0.
    values["next_pre_mix"].fill_(0.5)
    values["x_hc_out"].zero_()


def _ignore_causal_encoder_state(_actual, _expected, **_kwargs):
    """State buffers are validated by their owning Attention golden separately."""
    return True, "state buffer intentionally omitted from the zero bring-up comparison"


def run_causal_encoder_device(
    *,
    platform: str = "a5",
    device_ids: Sequence[int] | None = None,
    epochs: int = 1,
    compile_only: bool = False,
):
    """Compile, execute, and numerically validate the fused 20-layer device entry."""
    from golden import run
    from pypto.ir import DistributedConfig

    from models.deepseek_v4_1_flash import config as C

    world_size = C.EP_SIZE
    device_ids = list(range(world_size)) if device_ids is None else list(device_ids)
    if len(device_ids) != world_size or len(set(device_ids)) != world_size:
        raise ValueError(f"device_ids must contain {world_size} distinct device ids")
    specs = build_causal_encoder_device_specs(world_size=world_size)
    program = make_causal_encoder_device_program(world_size, epochs, specs)
    compare_fn = {
        spec.name: _ignore_causal_encoder_state
        for spec in specs
        if hasattr(spec, "shape") and spec.name.endswith(
            (
                "window_cache",
                "window_cache_scale",
                "compressed_cache",
                "compressed_cache_scale",
                "index_cache",
                "index_cache_scale",
                "state_cache",
                "topk_indices",
            )
        )
    }
    result = run(
        fn=program,
        specs=specs,
        golden_fn=_zero_causal_encoder_device_golden,
        config={
            "platform": platform,
            "log_level": "error",
            "distributed_config": DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        },
        compare_fn=compare_fn,
        compile_only=compile_only,
    )
    if not result.passed:
        raise RuntimeError(result.error or "causal encoder device run failed")
    phase = "COMPILE" if compile_only else "DEVICE"
    print(f"[{phase}] PASS decode_causal_encoder layers=20 devices={device_ids} platform={platform}")
    return result


def make_causal_encoder_program(layer_id, world_size, epochs, *, stage="attention", specs=None):
    """Build one statically selected causal-encoder layer program."""
    return make_causal_encoder_layer_program(
        layer_id,
        world_size,
        epochs,
        stage=stage,
        specs=specs,
    )


def make_causal_encoder_golden_inputs() -> list[dict[str, Any]]:
    """Build one deterministic compact fixture for every encoder layer."""
    from models.deepseek_v4_1_flash._golden_smoke import make_decode_layer_golden_inputs

    return [make_decode_layer_golden_inputs(layer_id) for layer_id in CAUSAL_ENCODER_LAYER_IDS]


def _normalise_layer_inputs(layer_inputs: Sequence[Mapping[str, Any]] | Mapping[int, Mapping[str, Any]] | None):
    if layer_inputs is None:
        return make_causal_encoder_golden_inputs()
    if isinstance(layer_inputs, Mapping):
        try:
            values = [layer_inputs[layer_id] for layer_id in CAUSAL_ENCODER_LAYER_IDS]
        except KeyError as error:
            raise ValueError("layer_inputs mapping must contain all causal encoder layer ids") from error
    else:
        values = list(layer_inputs)
        if len(values) != len(CAUSAL_ENCODER_LAYER_IDS):
            raise ValueError("layer_inputs must contain one mapping for each causal encoder layer")
    for expected_layer_id, values_for_layer in zip(CAUSAL_ENCODER_LAYER_IDS, values):
        supplied_layer_id = values_for_layer.get("layer_id", expected_layer_id)
        if supplied_layer_id != expected_layer_id:
            raise ValueError(
                f"layer_inputs[{expected_layer_id}] describes layer {supplied_layer_id}"
            )
    return values


def golden_decode_causal_encoder(
    layer_inputs: Sequence[Mapping[str, Any]] | Mapping[int, Mapping[str, Any]] | None = None,
    *,
    x_hc: torch.Tensor | None = None,
    incoming_pre_mix: torch.Tensor | None = None,
    num_tokens: int | None = None,
    engram_inputs: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
    collapse: bool = True,
) -> CausalEncoderGoldenResult:
    """Run the official 20-layer causal-encoder Block order on CPU.

    The delayed FFN pre-mix is carried from one layer to the next.  Caller
    supplied ``x_hc`` and ``incoming_pre_mix`` override the first fixture's
    boundaries; later layer fixtures contribute their layer weights while C2A
    Reuse layers consume the compressed cache and Top-K rows published by the
    most recent C2A Full layer. Optional ``engram_inputs`` follows the official
    ``Transformer.forward`` and applies Engram before layers 1 and 14.
    """
    values = _normalise_layer_inputs(layer_inputs)
    validate_causal_encoder_schedule()
    first = values[0]
    current_x_hc = first.get("x_hc") if x_hc is None else x_hc
    current_pre_mix = first.get("incoming_pre_mix") if incoming_pre_mix is None else incoming_pre_mix
    if current_x_hc is None or current_pre_mix is None:
        raise ValueError("the first layer must provide x_hc and incoming_pre_mix")
    if current_x_hc.ndim != 3 or current_pre_mix.ndim != 2:
        raise ValueError("x_hc must be [tokens, hc, hidden] and incoming_pre_mix must be [tokens, hc]")
    if current_x_hc.shape[:2] != current_pre_mix.shape:
        raise ValueError("x_hc and incoming_pre_mix have incompatible token/HC dimensions")
    if num_tokens is not None and num_tokens != current_x_hc.shape[0]:
        raise ValueError("causal encoder golden currently requires all capacity rows to be active")

    layer_results = []
    for expected_layer_id, layer_values in zip(CAUSAL_ENCODER_LAYER_IDS, values):
        layer_kwargs = dict(layer_values)
        layer_kwargs["layer_id"] = expected_layer_id
        layer_kwargs["x_hc"] = current_x_hc
        layer_kwargs["incoming_pre_mix"] = current_pre_mix
        current_x_hc = _apply_engram(expected_layer_id, current_x_hc, layer_kwargs, engram_inputs)
        layer_kwargs["x_hc"] = current_x_hc
        plan = resolve_causal_encoder_layer_plan(expected_layer_id)
        if plan.kind is DecodeLayerKind.C2A_REUSE:
            source = layer_results[plan.kv_source_layer_id].attention
            attention_inputs = dict(layer_kwargs["attention_inputs"])
            attention_inputs["compressed_cache"] = source.compressed_cache
            attention_inputs["compressed_cache_scale"] = source.compressed_cache_scale
            attention_inputs["compressed_indices"] = source.topk_indices
            layer_kwargs["attention_inputs"] = attention_inputs
        if num_tokens is not None:
            layer_kwargs["num_tokens"] = num_tokens
        result = golden_decode_layer(**layer_kwargs)
        if result.output.shape != current_x_hc.shape:
            raise RuntimeError(f"layer {expected_layer_id} changed the HC residual shape")
        if result.next_pre_mix.shape != current_pre_mix.shape:
            raise RuntimeError(f"layer {expected_layer_id} changed the delayed pre-mix shape")
        if not torch.isfinite(result.output).all() or not torch.isfinite(result.next_pre_mix).all():
            raise RuntimeError(f"layer {expected_layer_id} produced a non-finite residual boundary")
        layer_results.append(result)
        current_x_hc = result.output
        current_pre_mix = result.next_pre_mix

    output = hc_head(current_x_hc, current_pre_mix) if collapse else current_x_hc
    if not torch.isfinite(output).all():
        raise RuntimeError("causal encoder golden produced a non-finite hidden output")
    return CausalEncoderGoldenResult(
        output=output,
        x_hc=current_x_hc,
        next_pre_mix=current_pre_mix,
        layers=tuple(layer_results),
        plans=validate_causal_encoder_schedule(),
    )


def run_decode_causal_encoder_golden() -> CausalEncoderGoldenResult:
    """Run the deterministic 20-layer CPU golden and print its boundaries."""
    result = golden_decode_causal_encoder()
    if result.output.dtype is not torch.bfloat16:
        raise RuntimeError(f"causal encoder hidden output has dtype {result.output.dtype}")
    print(
        f"[GOLDEN] PASS decode_causal_encoder layers={len(result.layers)} "
        f"output={tuple(result.output.shape)} next_pre_mix={tuple(result.next_pre_mix.shape)}"
    )
    return result


# Short aliases used by model-level validation callers.
golden_causal_encoder = golden_decode_causal_encoder
run_causal_encoder_golden = run_decode_causal_encoder_golden


def main():
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Flash decode causal encoder")
    parser.add_argument("--cpu-golden", action="store_true", help="run the deterministic 20-layer CPU golden")
    parser.add_argument("--run-encoder", action="store_true", help="run and validate the fused 20-layer device encoder")
    parser.add_argument("--compile-only", action="store_true", help="compile the fused encoder without execution")
    parser.add_argument("--platform", default="a5", choices=("a5", "a5sim"))
    parser.add_argument("--devices", help="comma-separated device IDs; default 0 through 7")
    args, _ = parser.parse_known_args()
    validate_causal_encoder_schedule()
    if args.cpu_golden:
        run_decode_causal_encoder_golden()
        return
    if args.run_encoder:
        device_ids = None if args.devices is None else [int(value) for value in args.devices.split(",")]
        run_causal_encoder_device(
            platform=args.platform,
            device_ids=device_ids,
            compile_only=args.compile_only,
        )
        return
    print(
        "[SCHEDULE] decode causal encoder "
        f"layers={CAUSAL_ENCODER_START}:{CAUSAL_ENCODER_END} "
        f"kinds={[kind.name for kind in _EXPECTED_KINDS]}"
    )


if "pytest" in sys.modules:
    import pytest

    @pytest.fixture
    def encoder():
        """Return this module for composition tests."""
        return sys.modules[__name__]

    def test_official_causal_encoder_schedule(encoder):
        """Match the Hugging Face ratio and cache ownership tables for layers 0..19."""
        plans = encoder.validate_causal_encoder_schedule()
        assert tuple(plan.layer_id for plan in plans) == encoder.CAUSAL_ENCODER_LAYER_IDS
        assert tuple(plan.compression_ratio for plan in plans) == encoder.CAUSAL_ENCODER_COMPRESS_RATIOS
        assert tuple(
            plan.layer_id for plan in plans if plan.kv_source_layer_id == plan.layer_id
        ) == encoder.CAUSAL_ENCODER_KV_SOURCE_LAYERS
        assert tuple(
            plan.layer_id for plan in plans if plan.index_source_layer_id == plan.layer_id
        ) == encoder.CAUSAL_ENCODER_INDEX_SOURCE_LAYERS
        assert tuple(plan.kind for plan in plans[:2]) == (encoder.DecodeLayerKind.SWA,) * 2
        assert tuple(plan.kind for plan in plans[2::6]) == (
            encoder.DecodeLayerKind.C2A_FULL,
            encoder.DecodeLayerKind.C2A_FULL,
            encoder.DecodeLayerKind.C2A_FULL,
        )

    @pytest.mark.parametrize("layer_id", (-1, 20, 40))
    def test_causal_encoder_rejects_decoder_layers(encoder, layer_id):
        with pytest.raises(ValueError, match="causal encoder layer_id"):
            encoder.resolve_causal_encoder_layer_plan(layer_id)

    def test_program_factory_returns_static_pypto_layers(encoder, monkeypatch):
        calls = []

        def fake_program(layer_id, world_size, epochs, *, stage, specs):
            calls.append((layer_id, world_size, epochs, stage, specs))
            return f"layer-{layer_id}"

        monkeypatch.setattr(encoder, "make_causal_encoder_layer_program", fake_program)
        specs = {layer_id: f"spec-{layer_id}" for layer_id in encoder.CAUSAL_ENCODER_LAYER_IDS}
        programs = encoder.make_causal_encoder_programs(4, 2, specs)

        assert programs == tuple(f"layer-{layer_id}" for layer_id in encoder.CAUSAL_ENCODER_LAYER_IDS)
        assert calls == [
            (layer_id, 4, 2, "attention", f"spec-{layer_id}")
            for layer_id in encoder.CAUSAL_ENCODER_LAYER_IDS
        ]

    def test_block_factory_returns_all_ep_device_entries(encoder):
        """Build the complete SWA/C2A Block sequence with the packed EP ABI."""
        from models.deepseek_v4_1_flash import config as C

        specs = encoder.build_causal_encoder_block_specs(world_size=C.EP_SIZE)
        programs = encoder.make_causal_encoder_block_programs(C.EP_SIZE, 1, specs)
        assert len(programs) == 20
        assert all(type(program).__name__ == "JITFunction" for program in programs)
        assert encoder.decode_layer_kernel_skip_reason(0) is None
        assert encoder.decode_layer_kernel_skip_reason(2) is None
        assert encoder.decode_layer_kernel_skip_reason(20) is not None

    def test_fused_device_encoder_is_one_20_layer_entry(encoder):
        """The runnable device entry owns all 20 layer transitions in one host."""
        from models.deepseek_v4_1_flash import config as C

        specs = encoder.build_causal_encoder_device_specs(world_size=C.EP_SIZE)
        names = [spec.name for spec in specs]
        assert names[:2] == ["x_hc", "incoming_pre_mix"]
        assert names[-5:] == [
            "next_pre_mix",
            "x_hc_out",
            "attention_num_tokens",
            "num_tokens",
            "encoder_epoch",
        ]
        assert all(any(name.startswith(f"l{layer_id}_") for name in names) for layer_id in range(20))
        program = encoder.make_causal_encoder_device_program(C.EP_SIZE, 1, specs)
        assert type(program).__name__ == "JITFunction"
        assert len(program.param_names) == len(specs)

    def test_fused_device_zero_golden_only_exposes_final_boundary(encoder):
        """Bring-up golden validates the fused residual boundary while state is private."""
        from models.deepseek_v4_1_flash import config as C

        specs = encoder.build_causal_encoder_device_specs(world_size=C.EP_SIZE)
        values = {spec.name: spec.create_tensor() for spec in specs if hasattr(spec, "create_tensor")}
        values["next_pre_mix"].fill_(1.0)
        values["x_hc_out"].fill_(1.0)
        encoder._zero_causal_encoder_device_golden(values)
        assert torch.all(values["next_pre_mix"] == 0.5)
        assert torch.count_nonzero(values["x_hc_out"]) == 0

    def test_golden_chain_propagates_mixes_and_published_cache(encoder, monkeypatch):
        """Exercise all 20 Blocks and prove the official cross-layer state edges."""
        inputs = encoder.make_causal_encoder_golden_inputs()
        original = encoder.golden_decode_layer
        observed = []

        def wrapped(**kwargs):
            result = original(**kwargs)
            observed.append((kwargs, result))
            return result

        monkeypatch.setattr(encoder, "golden_decode_layer", wrapped)
        result = encoder.golden_decode_causal_encoder(inputs)

        assert len(observed) == len(encoder.CAUSAL_ENCODER_LAYER_IDS) == 20
        assert result.output.shape == (2, 64)
        assert result.output.dtype is torch.bfloat16
        assert torch.isfinite(result.output).all()
        assert torch.isfinite(result.next_pre_mix).all()
        torch.testing.assert_close(
            result.output[0, :8],
            torch.tensor([-3584, 14720, -2112, 25984, 24832, 11072, 22656, 3472], dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        assert result.output.float().sum().item() == 17066.0
        for layer_id, (kwargs, layer_result) in enumerate(observed):
            assert kwargs["layer_id"] == layer_id
            assert layer_result.output.shape == (2, 4, 64)
            assert layer_result.next_pre_mix.shape == (2, 4)
            if layer_id:
                previous = observed[layer_id - 1][1]
                assert torch.equal(kwargs["incoming_pre_mix"], previous.next_pre_mix)

        for layer_id in (3, 9, 15, 19):
            plan = result.plans[layer_id]
            source = observed[plan.kv_source_layer_id][1].attention
            attention_inputs = observed[layer_id][0]["attention_inputs"]
            assert torch.equal(attention_inputs["compressed_cache"], source.compressed_cache)
            assert torch.equal(
                attention_inputs["compressed_cache_scale"].view(torch.uint8),
                source.compressed_cache_scale.view(torch.uint8),
            )
            assert torch.equal(attention_inputs["compressed_indices"], source.topk_indices)

    def test_optional_engram_is_applied_before_official_layers(encoder, monkeypatch):
        """Keep the official layer-1 and layer-14 Engram insertion points observable."""
        values = encoder.make_causal_encoder_golden_inputs()
        values[1] = dict(values[1])
        values[1]["engram_inputs"] = {
            "hash_ids": object(),
            "engram_table": object(),
            "wkv_weight": object(),
            "weight": object(),
        }
        calls = []

        def fake_engram(hash_ids, engram_table, wkv_weight, weight, x):
            calls.append((hash_ids, engram_table, wkv_weight, weight, x))
            return x

        from models.deepseek_v4_1_flash import engram

        monkeypatch.setattr(engram, "golden_engram", fake_engram)
        encoder.golden_decode_causal_encoder(values)

        assert len(calls) == 1
        assert calls[0][:4] == (
            values[1]["engram_inputs"]["hash_ids"],
            values[1]["engram_inputs"]["engram_table"],
            values[1]["engram_inputs"]["wkv_weight"],
            values[1]["engram_inputs"]["weight"],
        )

    def test_engram_inputs_are_rejected_for_non_engram_layer(encoder):
        values = encoder.make_causal_encoder_golden_inputs()
        values[0] = dict(values[0])
        values[0]["engram_inputs"] = {
            "hash_ids": object(),
            "engram_table": object(),
            "wkv_weight": object(),
            "weight": object(),
        }
        with pytest.raises(ValueError, match="does not own an Engram block"):
            encoder.golden_decode_causal_encoder(values)


__all__ = [
    "CAUSAL_ENCODER_COMPRESS_RATIOS",
    "CAUSAL_ENCODER_END",
    "CAUSAL_ENCODER_ENGRAM_LAYERS",
    "CAUSAL_ENCODER_INDEX_SOURCE_LAYERS",
    "CAUSAL_ENCODER_LAYER_IDS",
    "CAUSAL_ENCODER_LAYERS",
    "CAUSAL_ENCODER_KV_SOURCE_LAYERS",
    "CAUSAL_ENCODER_START",
    "CausalEncoderGoldenResult",
    "causal_encoder_attention_kind",
    "golden_causal_encoder",
    "golden_decode_causal_encoder",
    "load_causal_encoder_attention_module",
    "build_causal_encoder_block_specs",
    "build_causal_encoder_device_specs",
    "decode_layer_kernel_skip_reason",
    "make_causal_encoder_golden_inputs",
    "make_causal_encoder_block_programs",
    "make_causal_encoder_device_program",
    "make_causal_encoder_layer_program",
    "make_causal_encoder_program",
    "make_causal_encoder_programs",
    "resolve_causal_encoder_layer_plan",
    "run_causal_encoder_golden",
    "run_decode_causal_encoder_golden",
    "run_causal_encoder_device",
    "validate_causal_encoder_schedule",
]


_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    main()
