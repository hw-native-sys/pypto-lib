# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Search K=2 candidates for the independent A2/A3 Decode product route."""

from __future__ import annotations

from dataclasses import dataclass


SEARCH_K = 2
BASELINE_ID = "addb655f7b4f4a4ce56af875724a04d79d3fe325"
INPUT_CONTRACT_ID = "decode-contract-v1"
TOPOLOGY_CONTRACT_ID = "dp8-tp2-ep16-p1-pcp1-dcp1"
TOLERANCE_CONTRACT_ID = "rtol-0.05-atol-0.05-discrete-exact"
LAYER_CONTRACT_ID = "physical-layer-v1"
QUANTIZATION_CONTRACT_ID = "w8a8-int8-engram-block128-v1"
STATE_CONTRACT_ID = "decode-token-cache-request-state-v1"


@dataclass(frozen=True)
class DecodeCandidate:
    """One executable PyPTO route evaluated under the shared Decode contract."""

    name: str
    entrypoint: str
    factory: str
    cache_layout: str
    page_size: int
    packing: str
    scale_dtype: str
    status: str
    baseline: str = BASELINE_ID
    input_contract: str = INPUT_CONTRACT_ID
    topology_contract: str = TOPOLOGY_CONTRACT_ID
    tolerance_contract: str = TOLERANCE_CONTRACT_ID
    layer_contract: str = LAYER_CONTRACT_ID
    quantization_contract: str = QUANTIZATION_CONTRACT_ID
    state_contract: str = STATE_CONTRACT_ID
    product_scope: str = "bounded_swa_only"
    formal_status: str = "not_run"

    def contract_identity(self) -> tuple[object, ...]:
        """Return the fields that must remain equal across candidates."""

        return (
            self.baseline,
            self.input_contract,
            self.topology_contract,
            self.tolerance_contract,
            self.layer_contract,
            self.quantization_contract,
            self.state_contract,
            self.page_size,
            self.scale_dtype,
        )


CANDIDATES = (
    DecodeCandidate(
        name="A",
        entrypoint="models.deepseek_v4_1_flash_a2.decode_fwd",
        factory="make_decode_step",
        cache_layout="[block, row, kv_head, head_dim]",
        page_size=128,
        packing="canonical_KN_int8",
        scale_dtype="fp32_channel",
        status="runnable_bounded_swa",
        product_scope="bounded_swa_only",
    ),
    DecodeCandidate(
        name="B",
        entrypoint="models.deepseek_v4_1_flash_a2.decode_replay",
        factory="make_decode_replay(steps=1)",
        cache_layout="[block, row, kv_head, head_dim] padded_state_envelope",
        page_size=128,
        packing="canonical_KN_int8_padded_state",
        scale_dtype="fp32_channel",
        status="runnable_bounded_swa",
        product_scope="bounded_swa_only",
    ),
)


def validate_candidates() -> tuple[DecodeCandidate, ...]:
    """Validate Search K=2 and the shared baseline contract."""

    if len(CANDIDATES) != SEARCH_K:
        raise ValueError(f"Search K={SEARCH_K} requires exactly {SEARCH_K} candidates")
    if len({candidate.name for candidate in CANDIDATES}) != SEARCH_K:
        raise ValueError("Decode candidate names must be unique")
    identities = {candidate.contract_identity() for candidate in CANDIDATES}
    if len(identities) != 1:
        raise ValueError("candidates must share baseline, layer, input, state, topology, quantization, page and scale contracts")
    for candidate in CANDIDATES:
        if not candidate.entrypoint.startswith("models.deepseek_v4_1_flash_a2.decode_"):
            raise ValueError(f"candidate {candidate.name} is outside the Decode package")
        if candidate.status != "runnable_bounded_swa":
            raise ValueError(f"candidate {candidate.name} is not an executable PyPTO route")
        if candidate.product_scope != "bounded_swa_only":
            raise ValueError(f"candidate {candidate.name} has an unsupported product scope")
        if candidate.formal_status != "not_run":
            raise ValueError(f"candidate {candidate.name} cannot be promoted before a formal NPU run")
    return CANDIDATES


__all__ = [
    "BASELINE_ID",
    "CANDIDATES",
    "DecodeCandidate",
    "INPUT_CONTRACT_ID",
    "LAYER_CONTRACT_ID",
    "QUANTIZATION_CONTRACT_ID",
    "SEARCH_K",
    "STATE_CONTRACT_ID",
    "TOLERANCE_CONTRACT_ID",
    "TOPOLOGY_CONTRACT_ID",
    "validate_candidates",
]
