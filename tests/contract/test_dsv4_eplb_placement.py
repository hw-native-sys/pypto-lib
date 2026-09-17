# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify the matched EP8x32 DeepSeek EPLB placement control."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp"
sys.path.insert(0, str(_MODEL_DIR))

from eplb_placement import balanced_pack_no_redundancy  # noqa: E402
from stats_placement_fixture import eplb_replay_placement, load_stats_placement_manifest  # noqa: E402
from stats_route_fixture import apportion_route_counts  # noqa: E402


def test_no_redundancy_balanced_pack_uses_assignment_order_for_local_slots() -> None:
    placement = balanced_pack_no_redundancy(
        [10, 9, 8, 7],
        ranks=2,
        capacity_per_rank=2,
    )

    assert placement.rank_to_logical == ((0, 3), (1, 2))
    assert placement.physical_to_logical == (0, 3, 1, 2)
    assert placement.logical_to_physical == (0, 2, 3, 1)
    assert placement.estimated_rank_loads == (17, 17)


def test_no_redundancy_balanced_pack_matches_upstream_torch_sort_ties() -> None:
    first = balanced_pack_no_redundancy(
        [1, 1, 1, 1],
        ranks=2,
        capacity_per_rank=2,
    )
    second = balanced_pack_no_redundancy(
        [1, 1, 1, 1],
        ranks=2,
        capacity_per_rank=2,
    )
    upstream_order = torch.ones(4, dtype=torch.float32).sort(descending=True).indices.tolist()

    assert first == second
    assert first.rank_to_logical == (tuple(upstream_order[0::2]), tuple(upstream_order[1::2]))


@pytest.mark.parametrize(
    ("loads", "ranks", "capacity", "error", "match"),
    [
        ([1, 2, 3], 2, 2, ValueError, "must contain 4 entries"),
        ([1, -1], 1, 2, ValueError, "must be nonnegative"),
        ([1, 1], 0, 2, ValueError, "ranks must be a positive integer"),
        ([1, 1], 1, 0, ValueError, "capacity_per_rank must be a positive integer"),
    ],
)
def test_no_redundancy_balanced_pack_rejects_invalid_inputs(
    loads,
    ranks,
    capacity,
    error,
    match,
) -> None:
    with pytest.raises(error, match=match):
        balanced_pack_no_redundancy(
            loads,
            ranks=ranks,
            capacity_per_rank=capacity,
        )


def test_eplb_control_uses_exact_replay_counts_and_improves_the_stats_peak_proxy() -> None:
    manifest = load_stats_placement_manifest()
    stats_peaks = []
    eplb_peaks = []
    strict_improvements = 0

    for layer_id in range(44):
        route_counts = apportion_route_counts(
            manifest["expert_loads"][layer_id],
            total_routes=384,
        )
        placement = eplb_replay_placement(manifest, layer_id)
        stats_loads = tuple(
            sum(route_counts[expert_id] for expert_id in logical_experts)
            for logical_experts in manifest["layers"][layer_id]["rank_to_logical"]
        )

        assert all(len(rank) == 32 for rank in placement.rank_to_logical)
        assert sorted(placement.physical_to_logical) == list(range(256))
        assert placement.estimated_rank_loads == tuple(
            sum(route_counts[expert_id] for expert_id in rank)
            for rank in placement.rank_to_logical
        )
        assert placement.logical_to_physical != tuple(
            manifest["layers"][layer_id]["logical_to_physical"]
        )

        stats_peak = max(stats_loads)
        eplb_peak = max(placement.estimated_rank_loads)
        stats_peaks.append(stats_peak)
        eplb_peaks.append(eplb_peak)
        strict_improvements += eplb_peak < stats_peak
        assert eplb_peak <= stats_peak

    assert strict_improvements == 32
    assert sum(eplb_peaks) / len(eplb_peaks) == pytest.approx(48.97727272727273)
    assert sum(stats_peaks) / len(stats_peaks) == pytest.approx(49.79545454545455)
    assert max(eplb_peaks) == max(stats_peaks) == 61


@pytest.mark.parametrize("layer_id", [-1, 44, True])
def test_eplb_replay_rejects_invalid_layer_ids(layer_id) -> None:
    with pytest.raises(ValueError, match="layer_id must be"):
        eplb_replay_placement(load_stats_placement_manifest(), layer_id)
