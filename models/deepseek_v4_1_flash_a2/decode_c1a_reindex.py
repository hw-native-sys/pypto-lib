"""A2/A3 C1A reindex decoder path for layer 24."""

from __future__ import annotations

from typing import Any

from .layers19_25 import (
    LAYER24,
    DecoderLayerA2A3Contract,
    c1a_reindex_reference,
    make_decoder_layer_stages,
    validate_layer19_20_21_24_25_contracts,
)


decode_c1a_reindex = c1a_reindex_reference
decode_c1a_reindex_reference = c1a_reindex_reference


def make_stages(capacity: int) -> dict[str, Any]:
    """Build the L24 C1A reindex stages."""

    return make_decoder_layer_stages(LAYER24.layer_id, capacity)


def validate_contract() -> DecoderLayerA2A3Contract:
    """Validate the L24 C1A reindex contract."""

    validate_layer19_20_21_24_25_contracts()
    return LAYER24


__all__ = [
    "DecoderLayerA2A3Contract",
    "LAYER24",
    "c1a_reindex_reference",
    "decode_c1a_reindex",
    "decode_c1a_reindex_reference",
    "make_stages",
    "validate_contract",
]
