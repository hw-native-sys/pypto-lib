# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Checkpoint fixture allocation rejects incomplete and aliased publication."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
MODEL = Path(__file__).resolve().parents[2] / "models/deepseek_v4_flash_dspark"
sys.path.insert(0, str(MODEL))
from utils import prefill_state_fixture, validate_prefill_state_checkpoint


GEOMETRY = [(128, 8, 136), (8, 2, 8)]


def fixture(history, page, start, count, checkpoint, *, legacy_ring=None):
    ring = legacy_ring or (history + 8 + page - 1) // page
    table = torch.arange((1048576 + page - 1) // page, dtype=torch.int32) % ring
    positions = torch.arange(start, start + count, dtype=torch.int32)
    if legacy_ring:
        mapping, blocks = table[positions.long() // page].long() * page + positions % page, ring
    else:
        table, mapping, blocks = prefill_state_fixture(
            start, count, checkpoint, history=history, block_size=page, table=table, physical_blocks=ring,
        )
    return dict(position_ids=positions, state_slot_mapping=mapping, compress_state_block_table=table[None],
                state_checkpoint_ends=torch.tensor([checkpoint]), compress_state=torch.empty(blocks, page, 0))


@pytest.mark.parametrize("history,page,tail", GEOMETRY)
@pytest.mark.parametrize("start,count,checkpoint", [(0, 512, 384), (33, 1024, 640), (127, 513, 128), (511, 513, 640),
                                                   (0, 8192, 8064), (8205, 8192, 16256),
                                                   (1048063, 513, 1048448)])
def test_checkpoint_and_terminal_have_distinct_live_rows(history, page, tail, start, count, checkpoint):
    tensors = fixture(history, page, start, count, checkpoint)
    validate_prefill_state_checkpoint(tensors, history=history, block_size=page, tail=tail)
    table = tensors["compress_state_block_table"]
    positions = torch.arange(max(0, start - history), start)
    read_rows = table[0, positions // page].long() * page + positions % page
    assert not set(read_rows.tolist()) & set(tensors["state_slot_mapping"].tolist())
    checkpoint_pages = {}
    for position in range(max(0, checkpoint - history), checkpoint):
        physical = (int(table[0, position // page]) if position < start
                    else int(tensors["state_slot_mapping"][position - start]) // page)
        assert checkpoint_pages.setdefault(position // page, physical) == physical


@pytest.mark.parametrize("history,page,tail", GEOMETRY)
@pytest.mark.parametrize("count,ring_rows", [(512, None), (896, 512)])
def test_reference_rejects_publication_aliases(history, page, tail, count, ring_rows):
    # The first case aliases within one tile; the second overwrites a retained
    # checkpoint in a later tile, despite a valid first publication.
    ring = ring_rows // page if ring_rows else (history + 8 + page - 1) // page
    tensors = fixture(history, page, 0, count, 384, legacy_ring=ring)
    with pytest.raises(ValueError, match="alias"):
        validate_prefill_state_checkpoint(tensors, history=history, block_size=page, tail=tail)


@pytest.mark.parametrize("history,page,tail", GEOMETRY)
@pytest.mark.parametrize("corruption", ["missing_checkpoint", "missing_terminal", "out_of_range", "missing_history"])
def test_reference_rejects_missing_required_rows(history, page, tail, corruption):
    tensors = fixture(history, page, 33, 1024, 640)
    if corruption == "missing_history":
        tensors["compress_state_block_table"][0, 32 // page] = -1
    else:
        index = 640 - 1 - 33 if corruption == "missing_checkpoint" else 1023
        value = tensors["compress_state"].shape[0] * page if corruption == "out_of_range" else -1
        tensors["state_slot_mapping"][index] = value
    with pytest.raises(ValueError, match="missing"):
        validate_prefill_state_checkpoint(tensors, history=history, block_size=page, tail=tail)


@pytest.mark.parametrize("checkpoint", [-1, 1048577])
def test_fixture_rejects_checkpoint_outside_context(checkpoint):
    with pytest.raises(ValueError, match="context"):
        fixture(128, 8, 33, 512, checkpoint)


@pytest.mark.parametrize("history,page,tail", GEOMETRY)
@pytest.mark.parametrize("start,count,checkpoint", [(33, 567, 640), (700, 512, 640), (0, 512, 8192)])
def test_target_in_other_chunk_validates_only_available_state(history, page, tail, start, count, checkpoint):
    tensors = fixture(history, page, start, count, checkpoint)
    validate_prefill_state_checkpoint(tensors, history=history, block_size=page, tail=tail)


def test_partial_checkpoint_cannot_hide_a_missing_publication():
    tensors = fixture(128, 8, 33, 567, 640)
    tensors["state_slot_mapping"][550 - 33] = -1
    with pytest.raises(ValueError, match="missing"):
        validate_prefill_state_checkpoint(tensors, history=128, block_size=8, tail=136)


def test_zero_checkpoint_keeps_existing_ring_geometry():
    table = torch.arange(64, dtype=torch.int32) % 17
    actual, mapping, blocks = prefill_state_fixture(
        33, 127, 0, history=128, block_size=8, table=table, physical_blocks=17,
    )
    positions = torch.arange(33, 160)
    assert actual is table and blocks == 17
    assert torch.equal(mapping, table[positions // 8].long() * 8 + positions % 8)
