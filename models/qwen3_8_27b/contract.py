# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B text-stack dispatch, dimensions, and checkpoint contract.

Q5 has no full-forward kernel to register with the repository-wide serving
contract yet. This module pins the part that already exists independently of
those kernels: which mixer each decoder layer runs, every model dimension, and
the exact text tensors in the Eco-Tech W8A8 checkpoint. Q6 and Q7 can build
their kernel ABI from this table instead of rediscovering checkpoint names.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .config import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        QWEN3_8_27B,
        Qwen38Config,
        Qwen38Dimensions,
        dimensions,
    )
except ImportError:
    from config import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        QWEN3_8_27B,
        Qwen38Config,
        Qwen38Dimensions,
        dimensions,
    )


BF16 = "bf16"
FP32 = "fp32"
INT8 = "int8"

_DTYPE_BYTES = {BF16: 2, FP32: 4, INT8: 1}


@dataclass(frozen=True)
class WeightSpec:
    """One tensor in the pinned Eco-Tech checkpoint."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "value"

    @property
    def numel(self) -> int:
        value = 1
        for dim in self.shape:
            value *= dim
        return value

    @property
    def nbytes(self) -> int:
        return self.numel * _DTYPE_BYTES[self.dtype]

    @property
    def is_zero_offset(self) -> bool:
        return self.role == "zero_offset"


@dataclass(frozen=True)
class MemoryBudget:
    """Persistent single-sequence memory fixed before the full forward exists."""

    checkpoint_weight_bytes: int
    resident_weight_bytes: int
    kv_cache_bytes_per_token: int
    max_kv_cache_bytes: int
    gdn_state_bytes_per_sequence: int
    conv_state_bytes_per_sequence: int
    hbm_capacity_bytes: int

    @property
    def max_context_headroom_bytes(self) -> int:
        """HBM left for activations/runtime at the native maximum context."""
        return (
            self.hbm_capacity_bytes
            - self.resident_weight_bytes
            - self.max_kv_cache_bytes
            - self.gdn_state_bytes_per_sequence
            - self.conv_state_bytes_per_sequence
        )


@dataclass(frozen=True)
class Qwen38TextContract:
    """Host-visible contract shared by loading, references, and future kernels."""

    config: Qwen38Config
    dims: Qwen38Dimensions
    layer_types: tuple[str, ...]
    weight_specs: tuple[WeightSpec, ...]
    memory: MemoryBudget

    def layer_type(self, layer: int) -> str:
        if not 0 <= layer < len(self.layer_types):
            raise IndexError(f"layer index {layer} outside [0, {len(self.layer_types)})")
        return self.layer_types[layer]

    def layer_specs(self, layer: int) -> tuple[WeightSpec, ...]:
        self.layer_type(layer)
        prefix = f"model.language_model.layers.{layer}."
        return tuple(spec for spec in self.weight_specs if spec.name.startswith(prefix))

    @property
    def prepared_weight_specs(self) -> tuple[WeightSpec, ...]:
        """Tensors kept after symmetric zero offsets are validated and dropped."""
        return tuple(spec for spec in self.weight_specs if not spec.is_zero_offset)


def _plain(name: str, shape: tuple[int, ...], dtype: str = BF16) -> WeightSpec:
    return WeightSpec(name=name, shape=shape, dtype=dtype)


def _quantized_linear(prefix: str, out_features: int, in_features: int) -> tuple[WeightSpec, ...]:
    """ModelSlim W8A8_DYNAMIC: I8 weight plus column scale and zero offset."""
    return (
        WeightSpec(prefix + ".weight", (out_features, in_features), INT8, "quantized_weight"),
        WeightSpec(prefix + ".weight_scale", (out_features, 1), FP32, "weight_scale"),
        WeightSpec(prefix + ".weight_offset", (out_features, 1), FP32, "zero_offset"),
    )


def layer_weight_specs(layer: int, config: Qwen38Config = QWEN3_8_27B) -> tuple[WeightSpec, ...]:
    """Raw checkpoint tensors for one decoder layer in checkpoint-native layout."""
    if not 0 <= layer < config.num_hidden_layers:
        raise IndexError(f"layer index {layer} outside [0, {config.num_hidden_layers})")
    prefix = f"model.language_model.layers.{layer}"
    specs = [
        _plain(prefix + ".input_layernorm.weight", (config.hidden_size,)),
        _plain(prefix + ".post_attention_layernorm.weight", (config.hidden_size,)),
    ]

    mixer_type = config.layer_types[layer]
    if mixer_type == LINEAR_ATTENTION:
        mixer = prefix + ".linear_attn"
        specs.extend(
            (
                *_quantized_linear(mixer + ".in_proj_qkv", config.qkv_width, config.hidden_size),
                _plain(
                    mixer + ".conv1d.weight",
                    (config.qkv_width, 1, config.linear_conv_kernel_dim),
                ),
                _plain(mixer + ".in_proj_a.weight", (config.linear_num_value_heads, config.hidden_size)),
                _plain(mixer + ".in_proj_b.weight", (config.linear_num_value_heads, config.hidden_size)),
                _plain(mixer + ".A_log", (config.linear_num_value_heads,)),
                _plain(mixer + ".dt_bias", (config.linear_num_value_heads,)),
                *_quantized_linear(mixer + ".in_proj_z", config.value_width, config.hidden_size),
                _plain(mixer + ".norm.weight", (config.linear_value_head_dim,)),
                # Eco-Tech deliberately leaves this projection in BF16.
                _plain(mixer + ".out_proj.weight", (config.hidden_size, config.value_width)),
            )
        )
    elif mixer_type == FULL_ATTENTION:
        mixer = prefix + ".self_attn"
        specs.extend(
            (
                *_quantized_linear(
                    mixer + ".q_proj",
                    2 * config.attention_q_width,
                    config.hidden_size,
                ),
                *_quantized_linear(mixer + ".k_proj", config.attention_kv_width, config.hidden_size),
                *_quantized_linear(mixer + ".v_proj", config.attention_kv_width, config.hidden_size),
                *_quantized_linear(mixer + ".o_proj", config.hidden_size, config.attention_q_width),
                _plain(mixer + ".q_norm.weight", (config.head_dim,)),
                _plain(mixer + ".k_norm.weight", (config.head_dim,)),
            )
        )
    else:
        raise ValueError(f"unsupported layer type at {layer}: {mixer_type!r}")

    mlp = prefix + ".mlp"
    specs.extend(
        (
            *_quantized_linear(mlp + ".gate_proj", config.intermediate_size, config.hidden_size),
            *_quantized_linear(mlp + ".up_proj", config.intermediate_size, config.hidden_size),
            *_quantized_linear(mlp + ".down_proj", config.hidden_size, config.intermediate_size),
        )
    )
    return tuple(specs)


def text_weight_specs(config: Qwen38Config = QWEN3_8_27B) -> tuple[WeightSpec, ...]:
    """Every text-prefill tensor, excluding the vision tower and MTP layer."""
    specs = [_plain("model.language_model.embed_tokens.weight", (config.vocab_size, config.hidden_size))]
    for layer in range(config.num_hidden_layers):
        specs.extend(layer_weight_specs(layer, config))
    specs.extend(
        (
            _plain("model.language_model.norm.weight", (config.hidden_size,)),
            _plain("lm_head.weight", (config.vocab_size, config.hidden_size)),
        )
    )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise AssertionError("the Qwen3.8 text weight contract contains duplicate names")
    return tuple(specs)


def make_text_contract(config: Qwen38Config = QWEN3_8_27B) -> Qwen38TextContract:
    """Construct a contract for production or an architecture-faithful fixture."""
    specs = text_weight_specs(config)
    checkpoint_bytes = sum(spec.nbytes for spec in specs)
    resident_bytes = sum(spec.nbytes for spec in specs if not spec.is_zero_offset)
    # Only the 16 full-attention layers own KV. Cache values remain BF16.
    kv_per_token = (
        config.num_full_attention_layers
        * 2
        * config.num_key_value_heads
        * config.head_dim
        * _DTYPE_BYTES[BF16]
    )
    # Each GDN layer owns one FP32 [value_head, key_dim, value_dim] state.
    gdn_state = (
        config.num_linear_layers
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * _DTYPE_BYTES[FP32]
    )
    conv_state = (
        config.num_linear_layers
        * (config.linear_conv_kernel_dim - 1)
        * config.qkv_width
        * _DTYPE_BYTES[BF16]
    )
    return Qwen38TextContract(
        config=config,
        dims=dimensions(config),
        layer_types=config.layer_types,
        weight_specs=specs,
        memory=MemoryBudget(
            checkpoint_weight_bytes=checkpoint_bytes,
            resident_weight_bytes=resident_bytes,
            kv_cache_bytes_per_token=kv_per_token,
            max_kv_cache_bytes=kv_per_token * config.max_position_embeddings,
            gdn_state_bytes_per_sequence=gdn_state,
            conv_state_bytes_per_sequence=conv_state,
            hbm_capacity_bytes=64 * 2**30,
        ),
    )


QWEN3_8_27B_TEXT_CONTRACT = make_text_contract()


# Pin the published contract, rather than merely deriving internally consistent
# values. A changed checkpoint or accidental dtype edit must move these numbers.
assert len(QWEN3_8_27B_TEXT_CONTRACT.weight_specs) == 1555
assert QWEN3_8_27B_TEXT_CONTRACT.memory.checkpoint_weight_bytes == 31_004_212_224
assert QWEN3_8_27B_TEXT_CONTRACT.memory.resident_weight_bytes == 30_989_597_696
assert QWEN3_8_27B_TEXT_CONTRACT.memory.kv_cache_bytes_per_token == 65_536
assert QWEN3_8_27B_TEXT_CONTRACT.memory.gdn_state_bytes_per_sequence == 144 * 2**20
