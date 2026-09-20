# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static decode layer planning and mode-module discovery."""

from dataclasses import dataclass
from enum import IntEnum
from importlib import import_module

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import AttentionMode


class DecodeLayerKind(IntEnum):
    """Static decode implementation selected for one backbone layer."""

    SWA = 0
    C2A_FULL = 1
    C2A_REUSE = 2
    C1A_FULL = 3
    C1A_REINDEX = 4
    C1A_REUSE = 5


@dataclass(frozen=True)
class DecodeLayerPlan:
    """Resolved attention implementation and source ownership for one layer."""

    layer_id: int
    kind: DecodeLayerKind
    compression_ratio: int
    kv_source_layer_id: int | None
    index_source_layer_id: int | None
    is_candidate_source: bool


REPRESENTATIVE_LAYER_IDS = {
    DecodeLayerKind.SWA: 0,
    DecodeLayerKind.C2A_FULL: 2,
    DecodeLayerKind.C2A_REUSE: 3,
    DecodeLayerKind.C1A_FULL: 20,
    DecodeLayerKind.C1A_REINDEX: 24,
    DecodeLayerKind.C1A_REUSE: 21,
}


MODE_MODULES = {
    DecodeLayerKind.SWA: "decode_swa",
    DecodeLayerKind.C2A_FULL: "decode_c2a_full",
    DecodeLayerKind.C2A_REUSE: "decode_c2a_reuse",
    DecodeLayerKind.C1A_FULL: "decode_attn_c1a_full",
    DecodeLayerKind.C1A_REINDEX: "decode_attn_c1a_reindex",
    DecodeLayerKind.C1A_REUSE: "decode_attn_c1a_reuse",
}


def resolve_decode_layer_plan(layer_id: int) -> DecodeLayerPlan:
    """Resolve one layer through the checkpoint-backed model configuration."""
    layer = C.FLASH.layer_config(layer_id)
    if layer.mode == AttentionMode.SWA:
        kind = DecodeLayerKind.SWA
    elif layer.compression_ratio == 2 and layer.mode == AttentionMode.FULL:
        kind = DecodeLayerKind.C2A_FULL
    elif layer.compression_ratio == 2 and layer.mode == AttentionMode.REUSE:
        kind = DecodeLayerKind.C2A_REUSE
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.FULL:
        kind = DecodeLayerKind.C1A_FULL
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.REINDEX:
        kind = DecodeLayerKind.C1A_REINDEX
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.REUSE:
        kind = DecodeLayerKind.C1A_REUSE
    else:
        raise ValueError(
            f"unsupported decode layer {layer_id}: ratio={layer.compression_ratio}, mode={layer.mode.value}"
        )
    return DecodeLayerPlan(
        layer_id=layer_id,
        kind=kind,
        compression_ratio=layer.compression_ratio,
        kv_source_layer_id=layer.kv_source_layer_id,
        index_source_layer_id=layer.index_source_layer_id,
        is_candidate_source=layer.is_candidate_source,
    )


def load_decode_attention_module(kind: DecodeLayerKind):
    """Import only the selected mode composition module."""
    module_name = MODE_MODULES[kind]
    return import_module(f"models.deepseek_v4_1_flash.{module_name}")


__all__ = [
    "DecodeLayerKind",
    "DecodeLayerPlan",
    "MODE_MODULES",
    "REPRESENTATIVE_LAYER_IDS",
    "load_decode_attention_module",
    "resolve_decode_layer_plan",
]
