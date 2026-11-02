"""A2/A3 C2A reuse decoder path for layer 19 and layer 15."""

from __future__ import annotations

from typing import Any

from .layer14_15 import (
    LAYER15,
    Layer15A2A3Contract,
    layer15_reuse_reference,
    make_layer15_stages,
    validate_layer14_15_contract,
)
from .layers19_25 import LAYER19, c2a_reuse_reference, make_decoder_layer_stages
from .layers19_25 import DecoderLayerA2A3Contract


decode_c2a_reuse = c2a_reuse_reference
decode_c2a_reuse_reference = c2a_reuse_reference
make_stages = make_decoder_layer_stages


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
