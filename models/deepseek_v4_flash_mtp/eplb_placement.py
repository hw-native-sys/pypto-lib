# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deterministic no-redundancy DeepSeek EPLB packing for host-side fixtures."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass


EPLB_UPSTREAM_COMMIT = "d52c72d5b2f2fb4c41afbf8eb21366820239913d"
EPLB_ALGORITHM = "deepseek-eplb-balanced-packing-no-redundancy"


@dataclass(frozen=True)
class EplbPlacement:
    """One rank-major, single-copy expert placement."""

    logical_to_physical: tuple[int, ...]
    rank_to_logical: tuple[tuple[int, ...], ...]
    estimated_rank_loads: tuple[int, ...]

    @property
    def physical_to_logical(self) -> tuple[int, ...]:
        return tuple(expert_id for rank in self.rank_to_logical for expert_id in rank)


def _as_nonnegative_loads(expert_loads: Sequence[int]) -> tuple[int, ...]:
    loads = []
    for expert_id, value in enumerate(expert_loads):
        if isinstance(value, bool):
            raise TypeError(f"expert load at index {expert_id} must be an integer, got bool")
        try:
            load = operator.index(value)
        except TypeError as error:
            raise TypeError(
                f"expert load at index {expert_id} must be an integer, got {value!r}"
            ) from error
        if load < 0:
            raise ValueError(f"expert load at index {expert_id} must be nonnegative, got {load}")
        loads.append(load)
    return tuple(loads)


def balanced_pack_no_redundancy(
    expert_loads: Sequence[int],
    *,
    ranks: int,
    capacity_per_rank: int,
) -> EplbPlacement:
    """Pack one copy of every expert using DeepSeek EPLB balanced-packing semantics.

    This is the zero-redundancy specialization of DeepSeek EPLB's
    ``rebalance_experts`` at ``deepseek-ai/EPLB@d52c72d5``. Expert ordering uses
    the upstream FP32 descending torch sort, including its equal-load tie
    behavior. Destination ties select the lowest eligible rank, and the physical
    local slot follows assignment order, matching ``rank_in_pack``.
    """
    loads = _as_nonnegative_loads(expert_loads)
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
        raise ValueError(f"ranks must be a positive integer, got {ranks!r}")
    if (
        isinstance(capacity_per_rank, bool)
        or not isinstance(capacity_per_rank, int)
        or capacity_per_rank < 1
    ):
        raise ValueError(
            f"capacity_per_rank must be a positive integer, got {capacity_per_rank!r}"
        )
    expected_experts = ranks * capacity_per_rank
    if len(loads) != expected_experts:
        raise ValueError(f"expert_loads must contain {expected_experts} entries, got {len(loads)}")

    rank_loads = [0] * ranks
    rank_members: list[list[int]] = [[] for _ in range(ranks)]
    logical_to_physical = [-1] * expected_experts
    import torch

    expert_order = (
        torch.tensor(loads, dtype=torch.float32)
        .sort(descending=True)
        .indices.cpu()
        .tolist()
    )

    for expert_id in expert_order:
        rank = min(
            (candidate for candidate in range(ranks) if len(rank_members[candidate]) < capacity_per_rank),
            key=lambda candidate: (rank_loads[candidate], candidate),
        )
        local_slot = len(rank_members[rank])
        rank_members[rank].append(expert_id)
        rank_loads[rank] += loads[expert_id]
        logical_to_physical[expert_id] = rank * capacity_per_rank + local_slot

    return EplbPlacement(
        logical_to_physical=tuple(logical_to_physical),
        rank_to_logical=tuple(tuple(members) for members in rank_members),
        estimated_rank_loads=tuple(rank_loads),
    )
