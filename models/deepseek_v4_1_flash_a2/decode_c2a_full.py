# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 C2A full decoder path for layer 14."""

from __future__ import annotations

from .decode_c2a_common import (
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
