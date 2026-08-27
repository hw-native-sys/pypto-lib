# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deterministic route fixtures derived from expert-usage histograms."""

from __future__ import annotations

import heapq
import operator
from collections.abc import Sequence


def _as_nonnegative_counts(expert_counts: Sequence[int]) -> tuple[int, ...]:
    if len(expert_counts) == 0:
        raise ValueError("expert_counts must not be empty")

    counts = []
    for expert_id, value in enumerate(expert_counts):
        if isinstance(value, bool):
            raise TypeError(f"expert count at index {expert_id} must be an integer, got bool")
        try:
            count = operator.index(value)
        except TypeError as error:
            raise TypeError(f"expert count at index {expert_id} must be an integer, got {value!r}") from error
        if count < 0:
            raise ValueError(f"expert count at index {expert_id} must be nonnegative, got {count}")
        counts.append(count)
    return tuple(counts)


def _as_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}")
    return result


def apportion_route_counts(
    aggregate_counts: Sequence[int],
    *,
    total_routes: int,
) -> tuple[int, ...]:
    """Scale counts by largest remainder, breaking remainder ties by expert ID."""
    counts = _as_nonnegative_counts(aggregate_counts)
    total_routes = _as_nonnegative_int(total_routes, name="total_routes")
    aggregate_total = sum(counts)
    if aggregate_total == 0:
        if total_routes == 0:
            return counts
        raise ValueError("cannot apportion a positive route budget from an all-zero histogram")

    apportioned = []
    remainders = []
    for expert_id, count in enumerate(counts):
        floor, remainder = divmod(count * total_routes, aggregate_total)
        apportioned.append(floor)
        if remainder:
            remainders.append((-remainder, expert_id))

    unassigned = total_routes - sum(apportioned)
    remainders.sort()
    if unassigned > len(remainders):
        raise RuntimeError("largest-remainder apportionment invariant failed")
    for _remainder, expert_id in remainders[:unassigned]:
        apportioned[expert_id] += 1
    return tuple(apportioned)


def make_route_table_from_counts(
    expert_counts: Sequence[int],
    *,
    num_tokens: int,
    topk: int,
):
    """Build exact distinct rows, preferring highest residual count then expert ID."""
    import torch

    counts = _as_nonnegative_counts(expert_counts)
    num_tokens = _as_nonnegative_int(num_tokens, name="num_tokens")
    topk = _as_nonnegative_int(topk, name="topk")
    if num_tokens == 0:
        raise ValueError("num_tokens must be positive")
    if topk == 0:
        raise ValueError("topk must be positive")
    if topk > len(counts):
        raise ValueError(f"topk={topk} exceeds the {len(counts)} available experts")

    expected_routes = num_tokens * topk
    actual_routes = sum(counts)
    if actual_routes != expected_routes:
        raise ValueError(
            f"expert counts sum to {actual_routes}, expected num_tokens * topk = {expected_routes}"
        )

    hottest_count = max(counts)
    if hottest_count > num_tokens:
        hottest_expert = counts.index(hottest_count)
        raise ValueError(
            f"expert {hottest_expert} has count {hottest_count}, exceeding one route per token "
            f"across {num_tokens} tokens"
        )

    remaining = [(-count, expert_id) for expert_id, count in enumerate(counts) if count]
    heapq.heapify(remaining)
    rows = []
    for _token_id in range(num_tokens):
        if len(remaining) < topk:
            raise RuntimeError("route construction invariant failed: fewer than topk experts remain")

        selected = []
        row = []
        for _slot in range(topk):
            negative_count, expert_id = heapq.heappop(remaining)
            next_count = -negative_count - 1
            row.append(expert_id)
            if next_count:
                selected.append((-next_count, expert_id))
        for entry in selected:
            heapq.heappush(remaining, entry)
        rows.append(row)

    if remaining:
        raise RuntimeError("route construction invariant failed: expert routes remain unassigned")
    return torch.tensor(rows, dtype=torch.int32)


def make_stats_shaped_route_table(
    aggregate_counts: Sequence[int],
    *,
    num_tokens: int,
    topk: int,
):
    """Apportion one aggregate histogram and build its exact route table."""
    num_tokens = _as_nonnegative_int(num_tokens, name="num_tokens")
    topk = _as_nonnegative_int(topk, name="topk")
    apportioned = apportion_route_counts(
        aggregate_counts,
        total_routes=num_tokens * topk,
    )
    return make_route_table_from_counts(
        apportioned,
        num_tokens=num_tokens,
        topk=topk,
    )
