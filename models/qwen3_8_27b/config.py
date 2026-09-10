# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B configuration, and the tiling the GDN kernels choose.

Model values follow HuggingFace `config.json` and carry its field names, so a
reader can check them against the published config without a translation table.
They were read from `Qwen/Qwen3.8-27B` and from the safetensors headers of its
first shard; the layer they describe is `model.language_model.layers.N.linear_attn`.

Only the linear-attention block is implemented here. The gated-attention layers,
the MoE-free dense MLP, the vision tower and the MTP layer are model fields for
context, not things this tree builds.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen38Config:
    """Follows HuggingFace config.json."""

    name: str

    # ---- shared ----
    hidden_size: int
    num_hidden_layers: int
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


@dataclass(frozen=True)
class GdnTiling:
    """Tiling the GDN kernels choose. Not from the model config."""

    # The chunk length the delta rule is blocked over. megagdn-pto fixes it at 128
    # and so do we; the reference Triton kernels default to 64.
    chunk: int


QWEN3_8_27B = Qwen38Config(
    name="Qwen3.8-27B",
    hidden_size=5120,
    num_hidden_layers=64,
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
)

GDN_TILING = GdnTiling(chunk=128)
