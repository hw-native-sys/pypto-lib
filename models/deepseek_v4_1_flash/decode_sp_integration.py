# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-side sequence-parallel Decode boundary checks.

The device Attention entry points use the same row ownership rule as this
module.  Keeping the rule in a small torch-only helper makes the cross-layer
contract testable without requiring an EP8 device fixture: local mHC state is
split once, the gathered Attention rows are restored in token order, and the
next layer receives the owner-local rows again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecodeTokenShard:
    """One TP rank's fixed-capacity token slab and its request metadata."""

    hidden: torch.Tensor
    token_ids: torch.Tensor
    position_ids: torch.Tensor
    valid_mask: torch.Tensor


def combine_validation(results: Sequence[object]) -> object:
    """Return the failing result when any wiring missed, else the last one.

    The decode Attention entries validate the replicated wiring and the
    sequence-parallel wiring in one call.  Returning the replicated result as soon
    as it passes would hide a sequence-parallel mismatch, and a failing replicated
    run would hide behind a passing sequence-parallel one, so the aggregate has to
    fail whenever *either* wiring failed.  Returns the last (sequence-parallel)
    result only when every wiring passed.
    """
    ordered = list(results)
    if not ordered:
        raise ValueError("at least one validation result is required")
    for result in ordered:
        if not getattr(result, "passed", False):
            return result
    return ordered[-1]


def sequence_parallel_bounds(
    num_tokens: int, tp_size: int, tp_rank: int, capacity: int | None = None
) -> tuple[int, int, int]:
    """Return ``(first, count, width)`` for one contiguous TP slab.

    ``width`` is the *physical* slab every rank publishes, which the device takes
    from the tensor extent (``ceil(capacity / tp)``) rather than from the active row
    count.  ``capacity`` defaults to ``num_tokens`` for the common case where every
    capacity row is active; pass it explicitly whenever a padded batch or ``T < TP``
    makes the two differ, otherwise the rank mapping would disagree with the
    AllGather and ReduceScatter.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    physical = num_tokens if capacity is None else capacity
    if physical < num_tokens:
        raise ValueError(f"capacity {physical} cannot be smaller than {num_tokens} active tokens")
    width = (physical + tp_size - 1) // tp_size if physical else 0
    first = min(tp_rank * width, num_tokens)
    count = max(0, min(width, num_tokens - first))
    return first, count, width


def shard_decode_batch(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    position_ids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    tp_size: int,
    capacity: int | None = None,
) -> tuple[DecodeTokenShard, ...]:
    """Split a global Decode batch into fixed-width owner slabs.

    Padding rows are physically present so every rank can participate in the
    same collective.  They carry ``token_id=-1`` and ``valid_mask=False`` and
    are never included by :func:`gather_decode_batch`.
    """
    if hidden.ndim < 1:
        raise ValueError("hidden must have a token dimension")
    num_tokens = hidden.shape[0]
    if token_ids.shape != (num_tokens,) or position_ids.shape != (num_tokens,):
        raise ValueError("token_ids and position_ids must have one entry per token")
    if valid_mask is None:
        valid_mask = torch.ones(num_tokens, dtype=torch.bool, device=hidden.device)
    if valid_mask.shape != (num_tokens,) or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be a bool vector with one entry per token")
    if not bool(valid_mask.all()):
        raise ValueError("sequence-parallel ownership requires a contiguous valid token prefix")

    shards: list[DecodeTokenShard] = []
    for rank in range(tp_size):
        first, count, width = sequence_parallel_bounds(num_tokens, tp_size, rank, capacity)
        local_hidden = hidden.new_zeros((width, *hidden.shape[1:]))
        local_tokens = torch.full((width,), -1, dtype=token_ids.dtype, device=token_ids.device)
        local_positions = torch.full((width,), -1, dtype=position_ids.dtype, device=position_ids.device)
        local_valid = torch.zeros(width, dtype=torch.bool, device=hidden.device)
        if count:
            local_hidden[:count].copy_(hidden[first : first + count])
            local_tokens[:count].copy_(token_ids[first : first + count])
            local_positions[:count].copy_(position_ids[first : first + count])
            local_valid[:count] = True
        shards.append(DecodeTokenShard(local_hidden, local_tokens, local_positions, local_valid))
    return tuple(shards)


def gather_decode_batch(shards: tuple[DecodeTokenShard, ...] | list[DecodeTokenShard], num_tokens: int) -> DecodeTokenShard:
    """Restore global token order from owner-local slabs after Attention.

    The slab width carried by the shards is authoritative: it is the physical
    capacity the collectives published, so a caller does not have to restate it.
    """
    if not shards:
        raise ValueError("at least one TP shard is required")
    tp_size = len(shards)
    slab_width = shards[0].hidden.shape[0]
    pieces = []
    token_pieces = []
    position_pieces = []
    mask_pieces = []
    for rank, shard in enumerate(shards):
        if shard.hidden.shape[0] != slab_width:
            raise ValueError(
                f"rank {rank} has slab width {shard.hidden.shape[0]}, expected {slab_width}"
            )
        first, count, _ = sequence_parallel_bounds(
            num_tokens, tp_size, rank, capacity=slab_width * tp_size
        )
        if count:
            pieces.append(shard.hidden[:count])
            token_pieces.append(shard.token_ids[:count])
            position_pieces.append(shard.position_ids[:count])
            mask_pieces.append(shard.valid_mask[:count])
    if pieces:
        hidden = torch.cat(pieces, dim=0)
        token_ids = torch.cat(token_pieces, dim=0)
        position_ids = torch.cat(position_pieces, dim=0)
        valid_mask = torch.cat(mask_pieces, dim=0)
    else:
        template = shards[0].hidden
        hidden = template[:0]
        token_ids = shards[0].token_ids[:0]
        position_ids = shards[0].position_ids[:0]
        valid_mask = shards[0].valid_mask[:0]
    if hidden.shape[0] != num_tokens or not bool(valid_mask.all()):
        raise RuntimeError("sequence-parallel gather did not restore every active token")
    return DecodeTokenShard(hidden, token_ids, position_ids, valid_mask)


def validate_two_layer_metadata(*, num_tokens: int = 2, tp_size: int = 4) -> DecodeTokenShard:
    """Exercise token IDs, positions, masks, and final ordering for two layers."""
    hidden = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    token_ids = torch.arange(100, 100 + num_tokens, dtype=torch.int64)
    position_ids = torch.arange(17, 17 + num_tokens, dtype=torch.int32)
    first = gather_decode_batch(
        shard_decode_batch(hidden, token_ids, position_ids, tp_size=tp_size), num_tokens
    )
    second_input = tuple(
        DecodeTokenShard(shard.hidden + 1, shard.token_ids, shard.position_ids, shard.valid_mask)
        for shard in shard_decode_batch(hidden, token_ids, position_ids, tp_size=tp_size)
    )
    second = gather_decode_batch(second_input, num_tokens)
    if not torch.equal(first.token_ids, token_ids) or not torch.equal(second.token_ids, token_ids):
        raise RuntimeError("token IDs changed order across consecutive Decode layers")
    if not torch.equal(first.position_ids, position_ids) or not torch.equal(second.position_ids, position_ids):
        raise RuntimeError("position IDs changed order across consecutive Decode layers")
    if not bool(first.valid_mask.all() and second.valid_mask.all()):
        raise RuntimeError("valid mask lost an active token across Decode layers")
    return second


__all__ = [
    "DecodeTokenShard",
    "combine_validation",
    "gather_decode_batch",
    "sequence_parallel_bounds",
    "shard_decode_batch",
    "validate_two_layer_metadata",
]
