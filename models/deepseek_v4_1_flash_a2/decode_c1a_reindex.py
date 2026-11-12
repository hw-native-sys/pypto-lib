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
"""A2/A3 C1A reindex decoder path for layer 24."""

from __future__ import annotations

from typing import Any

from .decode_layer_stages import (
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
