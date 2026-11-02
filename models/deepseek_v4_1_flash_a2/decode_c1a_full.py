"""A2/A3 C1A full decoder path for layer 20."""

from __future__ import annotations

from typing import Any

from .layers19_25 import (
    LAYER20,
    DecoderLayerA2A3Contract,
    c1a_full_reference,
    make_decoder_layer_stages,
    select_candidate_block_pool_reference,
    validate_layer19_20_21_24_25_contracts,
)


decode_c1a_full = c1a_full_reference
decode_c1a_full_reference = c1a_full_reference


def make_stages(capacity: int) -> dict[str, Any]:
    """Build the L20 C1A source stages."""

    return make_decoder_layer_stages(LAYER20.layer_id, capacity)


def validate_contract() -> DecoderLayerA2A3Contract:
    """Validate the L20 C1A source contract."""

    validate_layer19_20_21_24_25_contracts()
    return LAYER20


__all__ = [
    "DecoderLayerA2A3Contract",
    "LAYER20",
    "c1a_full_reference",
    "decode_c1a_full",
    "decode_c1a_full_reference",
    "make_stages",
    "select_candidate_block_pool_reference",
    "validate_contract",
]
