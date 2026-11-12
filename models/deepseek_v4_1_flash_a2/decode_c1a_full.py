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
"""A2/A3 C1A full decoder path for layer 20."""

from __future__ import annotations

from typing import Any

from .decode_layer_stages import (
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
