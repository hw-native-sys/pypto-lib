# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B text-model dimensions, layer schedule and checkpoint quantization."""

from dataclasses import dataclass


LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

# The KV cache's page, in tokens. The serving layer's choice rather than a kernel tile,
# and a boundary width like the ones `Qwen38Dimensions` carries: the prefill write, the
# decode append and the decode attention all address the pool with it, and a
# disagreement between them writes rows the reader skips with nothing in either
# signature to notice.
KV_PAGE_SIZE = 128


@dataclass(frozen=True)
class Qwen38Config:
    """Follows HuggingFace config.json."""

    name: str

    # ---- shared ----
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float

    # ---- linear attention (Gated DeltaNet) ----
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    # ---- full attention, every `full_attention_interval`-th layer ----
    full_attention_interval: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: float
    attention_bias: bool

    @property
    def num_linear_layers(self) -> int:
        """Layers running Gated DeltaNet rather than full attention."""
        return self.num_hidden_layers - self.num_full_attention_layers

    @property
    def num_full_attention_layers(self) -> int:
        return self.num_hidden_layers // self.full_attention_interval

    @property
    def head_groups(self) -> int:
        """Value heads sharing one QK head. Value head h reads key head h // this."""
        return self.linear_num_value_heads // self.linear_num_key_heads

    @property
    def qkv_width(self) -> int:
        """Output width of `in_proj_qkv`: q and k at Hg heads, v at H."""
        return (2 * self.linear_num_key_heads * self.linear_key_head_dim
                + self.linear_num_value_heads * self.linear_value_head_dim)

    @property
    def value_width(self) -> int:
        """Output width of `in_proj_z`, and the input width of `out_proj`."""
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def attention_q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def attention_kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def layer_types(self) -> tuple[str, ...]:
        """The checkpoint's repeated `LLLA` token-mixer schedule."""
        return tuple(
            LINEAR_ATTENTION if (i + 1) % self.full_attention_interval else FULL_ATTENTION
            for i in range(self.num_hidden_layers)
        )

    def layer_index(self, layer: int) -> tuple[str, int]:
        """One layer's mixer type and its index **within that type**.

        Mixer weights stack over the type-local index and not the global one, because only
        48 of the 64 layers have an `in_proj_qkv` and only 16 have a `qkv_proj`; the MLP and
        the two layer norms stack over the global index, since every layer has them. The
        `LLLA` schedule is periodic, so both are closed form -- three integer operations --
        which is what lets a kernel compute a stacked weight's row offset from a
        `layer_idx` scalar instead of reading an index table out of GM on every layer.
        """
        if not 0 <= layer < self.num_hidden_layers:
            raise IndexError(f"layer {layer} outside [0, {self.num_hidden_layers})")
        interval = self.full_attention_interval
        if (layer + 1) % interval == 0:
            return FULL_ATTENTION, layer // interval
        return LINEAR_ATTENTION, layer - (layer + 1) // interval

    def layers_of_type(self, mixer: str) -> tuple[int, ...]:
        """The global layer indices of one mixer type, in order."""
        return tuple(i for i, t in enumerate(self.layer_types) if t == mixer)


@dataclass(frozen=True)
class Qwen38Dimensions:
    """The widths the whole-model kernels pass across their boundaries."""

    hidden: int
    intermediate: int
    vocab: int
    layers: int
    linear_layers: int
    attention_layers: int
    linear_qk_heads: int
    linear_value_heads: int
    linear_head_dim: int
    linear_qkv_width: int
    linear_value_width: int
    attention_q_heads: int
    attention_kv_heads: int
    attention_head_dim: int
    attention_q_width: int
    attention_q_gate_width: int
    attention_kv_width: int
    rotary_dim: int


@dataclass(frozen=True)
class GdnQuant:
    """The symmetric W8A8 arithmetic the GDN projection kernels use.

    The `Eco-Tech/Qwen3.8-27B-w8a8` checkpoint uses the same per-output-channel weight
    and per-token activation min/max recipe, and stores zero `weight_offset` tensors
    because the scheme is symmetric. Its GDN `out_proj` stays BF16 and so does ours,
    so the two names below are both this tree's kernel path and the checkpoint's
    quantized set for the block.
    """

    scale_max: float
    amax_eps: float
    # Which projections are int8. `out_proj` is not among them: the checkpoint publishes
    # it in BF16 (`contract.py`) and both mixers match it. `in_proj_a` and `in_proj_b` stay
    # BF16 for a different reason: at 0.5 MB and
    # 0.1% of the FLOPs int8 there moves the block output by 3e-5, so BF16 costs nothing
    # (`test_block_reference.py --quant`). The convolution's 80 KB of taps likewise.
    weights: tuple[str, ...]


QWEN3_8_27B = Qwen38Config(
    name="Qwen3.8-27B",
    hidden_size=5120,
    intermediate_size=17408,
    num_hidden_layers=64,
    vocab_size=248320,
    max_position_embeddings=262144,
    rms_norm_eps=1e-06,
    linear_num_value_heads=48,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    partial_rotary_factor=0.25,
    rope_theta=10_000_000.0,
    attention_bias=False,
)


GDN_QUANT = GdnQuant(
    scale_max=127.0,
    amax_eps=1e-4,
    weights=("in_proj_qkv.weight", "in_proj_z.weight"),
)


def dimensions(config: Qwen38Config) -> Qwen38Dimensions:
    """The dimensions table of one config, production or a reduced test config."""
    return Qwen38Dimensions(
        hidden=config.hidden_size,
        intermediate=config.intermediate_size,
        vocab=config.vocab_size,
        layers=config.num_hidden_layers,
        linear_layers=config.num_linear_layers,
        attention_layers=config.num_full_attention_layers,
        linear_qk_heads=config.linear_num_key_heads,
        linear_value_heads=config.linear_num_value_heads,
        linear_head_dim=config.linear_value_head_dim,
        linear_qkv_width=config.qkv_width,
        linear_value_width=config.value_width,
        attention_q_heads=config.num_attention_heads,
        attention_kv_heads=config.num_key_value_heads,
        attention_head_dim=config.head_dim,
        attention_q_width=config.attention_q_width,
        attention_q_gate_width=2 * config.attention_q_width,
        attention_kv_width=config.attention_kv_width,
        rotary_dim=config.rotary_dim,
    )


QWEN3_8_27B_DIMS = dimensions(QWEN3_8_27B)
QWEN3_8_27B_LAYER_TYPES = QWEN3_8_27B.layer_types

# The closed-form `layer_index` computes a stacked weight's row offset from a layer
# scalar; a wrong schedule here is a silently wrong weight row, so it is checked against
# the enumeration at import.
assert QWEN3_8_27B_LAYER_TYPES == (
    LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION) * 16
assert QWEN3_8_27B_DIMS.rotary_dim == 64
