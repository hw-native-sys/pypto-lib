# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools

import pytest
import torch

from models.deepseek_v4_flash_mtp.stats_route_fixture import (
    apportion_route_counts,
    make_route_table_from_counts,
    make_stats_shaped_route_table,
)


def _route_counts(table: torch.Tensor, *, num_experts: int) -> tuple[int, ...]:
    counts = torch.bincount(table.to(torch.int64).reshape(-1), minlength=num_experts)
    return tuple(counts.tolist())


def _assert_legal_table(table: torch.Tensor, *, num_tokens: int, topk: int) -> None:
    assert table.dtype == torch.int32
    assert tuple(table.shape) == (num_tokens, topk)
    for row in table.tolist():
        assert len(row) == len(set(row))


def test_apportion_route_counts_uses_exact_integer_largest_remainders() -> None:
    assert apportion_route_counts([5, 3, 2, 0], total_routes=6) == (3, 2, 1, 0)


def test_apportion_route_counts_breaks_equal_remainders_by_expert_id() -> None:
    assert apportion_route_counts([1, 1, 1, 0], total_routes=2) == (1, 1, 0, 0)


def test_apportion_route_counts_preserves_an_exact_histogram() -> None:
    counts = (3, 2, 2, 1, 0)

    assert apportion_route_counts(counts, total_routes=sum(counts)) == counts
    assert apportion_route_counts(counts, total_routes=0) == (0, 0, 0, 0, 0)


def test_make_route_table_from_counts_is_exact_distinct_and_deterministic() -> None:
    counts = (3, 2, 2, 1)
    expected = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 1],
            [2, 3],
        ],
        dtype=torch.int32,
    )

    first = make_route_table_from_counts(counts, num_tokens=4, topk=2)
    second = make_route_table_from_counts(counts, num_tokens=4, topk=2)

    _assert_legal_table(first, num_tokens=4, topk=2)
    assert _route_counts(first, num_experts=len(counts)) == counts
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_make_route_table_from_counts_handles_every_small_feasible_histogram() -> None:
    num_tokens = 4
    topk = 2
    num_experts = 4
    for counts in itertools.product(range(num_tokens + 1), repeat=num_experts):
        if sum(counts) != num_tokens * topk:
            continue
        table = make_route_table_from_counts(counts, num_tokens=num_tokens, topk=topk)
        _assert_legal_table(table, num_tokens=num_tokens, topk=topk)
        assert _route_counts(table, num_experts=num_experts) == counts


def test_make_stats_shaped_route_table_apportions_then_constructs() -> None:
    aggregate_counts = (10, 6, 4, 0)
    table = make_stats_shaped_route_table(aggregate_counts, num_tokens=3, topk=2)

    _assert_legal_table(table, num_tokens=3, topk=2)
    assert _route_counts(table, num_experts=len(aggregate_counts)) == (3, 2, 1, 0)


@pytest.mark.parametrize(
    ("counts", "total_routes", "error", "match"),
    [
        ([], 1, ValueError, "must not be empty"),
        ([0, 0], 1, ValueError, "all-zero histogram"),
        ([1, -1], 1, ValueError, "must be nonnegative"),
        ([1, 0.5], 1, TypeError, "must be an integer"),
        ([1, 1], -1, ValueError, "must be nonnegative"),
    ],
)
def test_apportion_route_counts_rejects_invalid_inputs(counts, total_routes, error, match) -> None:
    with pytest.raises(error, match=match):
        apportion_route_counts(counts, total_routes=total_routes)


@pytest.mark.parametrize(
    ("counts", "num_tokens", "topk", "match"),
    [
        ((2, 1, 1), 3, 2, "sum to 4"),
        ((4, 1, 1), 3, 2, "exceeding one route per token"),
        ((1,), 1, 2, "exceeds the 1 available experts"),
        ((0, 0), 0, 1, "num_tokens must be positive"),
        ((0, 0), 1, 0, "topk must be positive"),
    ],
)
def test_make_route_table_from_counts_rejects_impossible_inputs(
    counts,
    num_tokens,
    topk,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        make_route_table_from_counts(counts, num_tokens=num_tokens, topk=topk)
