# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select CED prefill attention kernels and global cache ownership by layer."""

from dataclasses import dataclass
from importlib import import_module
from typing import Literal

from models.deepseek_v4_1_flash.config import AttentionMode, DeepSeekV41LayerConfig, FLASH


@dataclass(frozen=True)
class PrefillLayerPlan:
    """One checkpoint layer's CED stage, attention kernel, and cache publisher."""

    layer: DeepSeekV41LayerConfig
    stage: Literal["encoder", "decoder"]
    attention_module: str
    attention_symbol: str
    attention_host_symbol: str
    global_publisher_module: str | None = None
    global_publisher_symbol: str | None = None

    @property
    def layer_id(self) -> int:
        return self.layer.layer_id


_ENCODER_ATTENTION = {
    AttentionMode.SWA: ("prefill_swa", "prefill_swa", "make_hc_program"),
    AttentionMode.FULL: ("prefill_c2a_full", "prefill_c2a_full", "make_hc_program"),
    AttentionMode.REUSE: ("prefill_c2a_reuse", "prefill_c2a_reuse", "make_hc_program"),
}
_DECODER_ATTENTION = {
    AttentionMode.FULL: ("prefill_c1a_full", "prefill_c1a_seeded", "l3_prefill_c1a_seeded"),
    AttentionMode.REINDEX: ("prefill_c1a_reindex", "prefill_c1a_reindex", "l3_prefill_c1a_reindex_test"),
    AttentionMode.REUSE: ("prefill_c1a_reuse", "prefill_c1a_reuse", "l3_prefill_c1a_reuse_test"),
}


def resolve_prefill_layer_plan(layer_id: int) -> PrefillLayerPlan:
    """Resolve an encoder or decoder layer without importing device modules."""
    layer = FLASH.layer_config(layer_id)
    encoder = layer_id < FLASH.num_hidden_layers // 2
    stage: Literal["encoder", "decoder"] = "encoder" if encoder else "decoder"
    table = _ENCODER_ATTENTION if encoder else _DECODER_ATTENTION
    if layer.mode not in table:
        raise ValueError(f"unsupported {stage} prefill mode at layer {layer_id}: {layer.mode.value}")
    if encoder and layer.compression_ratio not in (0, 2):
        raise ValueError(f"encoder layer {layer_id} has an unexpected compression ratio")
    if not encoder and layer.compression_ratio != 1:
        raise ValueError(f"decoder layer {layer_id} must have ratio-one compressed attention")
    module, symbol, host_symbol = table[layer.mode]
    publisher_module = "prefill_decoder_kv" if not encoder and layer.is_kv_source else None
    publisher_symbol = "publish_decoder_global_from_encoder_rank" if publisher_module else None
    return PrefillLayerPlan(layer, stage, module, symbol, host_symbol, publisher_module, publisher_symbol)


def load_prefill_attention(plan: PrefillLayerPlan):
    """Load the selected mHC-wrapped attention entry only when a backend needs it."""
    module = import_module(f"models.deepseek_v4_1_flash.{plan.attention_module}")
    return getattr(module, plan.attention_symbol)


def load_prefill_attention_host(plan: PrefillLayerPlan):
    """Load the mode's TP/DP host entry or host-entry factory."""
    module = import_module(f"models.deepseek_v4_1_flash.{plan.attention_module}")
    return getattr(module, plan.attention_host_symbol)


def load_prefill_ffn_host():
    """Load the common EP MoE plus TP-restore host-entry factory."""
    module = import_module("models.deepseek_v4_1_flash.prefill_layer")
    return module.make_ffn_program


def load_prefill_global_publisher(plan: PrefillLayerPlan):
    """Load the decoder Full Mode encoder-stream publisher, if one is required."""
    if plan.global_publisher_module is None:
        return None
    module = import_module(f"models.deepseek_v4_1_flash.{plan.global_publisher_module}")
    return getattr(module, plan.global_publisher_symbol)
