# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 C2A reuse decoder path for layer 19 and layer 15."""

from __future__ import annotations

from typing import Any

from .decode_c2a_common import (
    LAYER15,
    Layer15A2A3Contract,
    layer15_reuse_reference,
    make_layer15_stages,
    validate_layer14_15_contract,
)
from .decode_layer_stages import LAYER19, c2a_reuse_reference, make_decoder_layer_stages
from .decode_layer_stages import DecoderLayerA2A3Contract


decode_c2a_reuse = c2a_reuse_reference
decode_c2a_reuse_reference = c2a_reuse_reference


def make_stages(capacity: int, layer_id: int = LAYER19.layer_id) -> dict[str, Any]:
    """Build the C2A reuse stages for L15 or L19 explicitly."""

    if int(layer_id) not in (LAYER15.layer_id, LAYER19.layer_id):
        raise ValueError("C2A reuse stages support only layers 15 and 19")
    if int(layer_id) == LAYER15.layer_id:
        return make_layer15_stages(capacity)
    return make_decoder_layer_stages(int(layer_id), capacity)


def validate_contract() -> tuple[Layer15A2A3Contract, DecoderLayerA2A3Contract]:
    """Validate the L15 and L19 C2A reuse contracts."""

    _, layer15 = validate_layer14_15_contract()
    return layer15, LAYER19


def make_layer19_stages(capacity: int) -> dict[str, Any]:
    """Build the L19 C2A reuse stages."""

    return make_decoder_layer_stages(LAYER19.layer_id, capacity)


__all__ = [
    "LAYER15",
    "LAYER19",
    "Layer15A2A3Contract",
    "decode_c2a_reuse",
    "decode_c2a_reuse_reference",
    "layer15_reuse_reference",
    "make_layer15_stages",
    "make_layer19_stages",
    "make_stages",
    "validate_contract",
]
