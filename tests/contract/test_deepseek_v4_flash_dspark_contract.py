# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the DSpark block-table and compressor-state slot helpers."""

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_dspark"
sys.path.insert(0, str(MODEL_DIR))

from utils import block_table, state_slot_mapping  # noqa: E402


def test_block_table_can_model_hca_deployment_request_slots() -> None:
    physical_blocks = 1088
    blocks_per_request = 17
    table = block_table(
        batch=2,
        table_blocks=blocks_per_request + 1,
        physical_blocks=physical_blocks,
        request_slots=64,
    )

    assert table.shape == (2, blocks_per_request + 1)
    assert table[0, :blocks_per_request].tolist() == [64 * block for block in range(blocks_per_request)]
    assert table[1, :blocks_per_request].tolist() == [64 * block + 1 for block in range(blocks_per_request)]
    assert table[0, blocks_per_request].item() == table[0, 0].item()
    assert set(table[0].tolist()).isdisjoint(table[1].tolist())


def test_block_table_request_slots_default_is_backward_compatible() -> None:
    default = block_table(batch=2, table_blocks=7, physical_blocks=32)
    explicit = block_table(batch=2, table_blocks=7, physical_blocks=32, request_slots=2)
    assert default.equal(explicit)


def test_block_table_rejects_fewer_request_slots_than_rows() -> None:
    with pytest.raises(ValueError, match="request_slots must be >= batch"):
        block_table(batch=2, table_blocks=7, physical_blocks=32, request_slots=1)


def test_prefill_csa_state_handoff_matches_decode_ring_across_wraps() -> None:
    block_size = 2
    storage_len = 16
    ring_pages = storage_len // block_size
    prefill_table = block_table(
        batch=1,
        table_blocks=268,
        physical_blocks=512,
        request_slots=64,
    )
    prefill_tables = prefill_table.unsqueeze(0).expand(4, -1, -1).clone()
    decode_tables = torch.full((4, 64, ring_pages), -1, dtype=torch.int32)
    decode_tables[:, 0].copy_(prefill_tables[:, 0, :ring_pages])

    for decode_start in range(512, 528):
        live_positions = torch.arange(
            decode_start - 8,
            decode_start + 8,
            dtype=torch.int64,
        ).unsqueeze(0).expand(4, -1)
        prefill_rows = state_slot_mapping(
            live_positions,
            prefill_tables[:, 0],
            state_block_size=block_size,
        )
        decode_rows = state_slot_mapping(
            live_positions % storage_len,
            decode_tables[:, 0],
            state_block_size=block_size,
        )

        assert torch.equal(decode_rows, prefill_rows)
        assert all(torch.unique(rank_rows).numel() == storage_len for rank_rows in decode_rows)
