"""A2/A3 C1A reuse decoder path for layers 21 and 25."""

from __future__ import annotations

from typing import Any

from .layers19_25 import (
    LAYER21,
    LAYER25,
    DecoderLayerA2A3Contract,
    c1a_reuse_reference,
    make_decoder_layer_stages,
    validate_layer19_20_21_24_25_contracts,
)


decode_c1a_reuse = c1a_reuse_reference
decode_c1a_reuse_reference = c1a_reuse_reference


def make_stages(capacity: int, layer_id: int = 21) -> dict[str, Any]:
    """Build the C1A reuse stages for L21 or L25."""

    if int(layer_id) not in (LAYER21.layer_id, LAYER25.layer_id):
        raise ValueError("C1A reuse stages support only layers 21 and 25")
    return make_decoder_layer_stages(int(layer_id), capacity)


def validate_contracts() -> tuple[DecoderLayerA2A3Contract, DecoderLayerA2A3Contract]:
    """Validate the L21 and L25 C1A reuse contracts."""

    validate_layer19_20_21_24_25_contracts()
    return LAYER21, LAYER25


__all__ = [
    "DecoderLayerA2A3Contract",
    "LAYER21",
    "LAYER25",
    "c1a_reuse_reference",
    "decode_c1a_reuse",
    "decode_c1a_reuse_reference",
    "make_stages",
    "validate_contracts",
]
