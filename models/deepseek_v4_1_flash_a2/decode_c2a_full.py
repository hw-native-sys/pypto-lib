"""A2/A3 C2A full decoder path for layer 14."""

from __future__ import annotations

from .layer14_15 import (
    LAYER14,
    Layer14A2A3Contract,
    apply_layer14_engram_reference,
    dequantize_engram_rows_reference,
    engram_a2a3_reference,
    index_key_reference,
    layer14_source_reference,
    load_a2a3_layer14_15_bundle,
    make_index_key_publish,
    make_index_key_publish_rope,
    make_layer14_stages,
    make_ratio2_projection,
    quantize_engram_rows_reference,
    ratio2_pool_reference,
    update_ratio2_state_reference,
    validate_layer14_15_contract,
)


decode_c2a_full = layer14_source_reference
decode_c2a_full_reference = layer14_source_reference
make_stages = make_layer14_stages


def validate_contract() -> Layer14A2A3Contract:
    """Validate the L14 C2A source contract."""

    layer14, _ = validate_layer14_15_contract()
    return layer14


__all__ = [
    "LAYER14",
    "Layer14A2A3Contract",
    "apply_layer14_engram_reference",
    "decode_c2a_full",
    "decode_c2a_full_reference",
    "dequantize_engram_rows_reference",
    "engram_a2a3_reference",
    "index_key_reference",
    "layer14_source_reference",
    "load_a2a3_layer14_15_bundle",
    "make_index_key_publish",
    "make_index_key_publish_rope",
    "make_layer14_stages",
    "make_ratio2_projection",
    "make_stages",
    "quantize_engram_rows_reference",
    "ratio2_pool_reference",
    "update_ratio2_state_reference",
    "validate_contract",
    "validate_layer14_15_contract",
]
