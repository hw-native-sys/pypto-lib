# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch metadata lowering for packed prefill and continuous-batch decode.

GLM-5.3-Flash has a **hybrid** cache, so one dispatch carries three different kinds
of per-token addressing:

* the 12 MLA layers write a 512-wide latent row into a paged pool of
  :data:`BLOCK_SIZE`-token blocks;
* the same 12 layers write a 256-wide indexer state row — ``[key(128),
  gate_scores(128)]`` — into a second paged pool, and separately read a **pooled**
  state addressed one row per ``index_kpool`` tokens. Those are two different
  mappings and must not be conflated: ``index_slots`` is per token at
  :data:`BLOCK_SIZE` pages, ``pool_slots`` is per pool of four;
* the 34 KDA layers hold no per-token cache at all, only a per-request conv window
  and recurrent state, addressed by a request row that never grows with context.

The same lowering serves prefill and decode; decode is just the case where every
request contributes ``1 + MTP_SPEC_TOKENS`` rows.
"""

from dataclasses import dataclass

import torch

from models.glm5_3_flash.config import BLOCK_SIZE
from models.glm5_3_flash.config import INDEX_KPOOL
from models.glm5_3_flash.config import TP_SIZE


@dataclass(frozen=True)
class ForwardMetadata:
    """Canonical token-major metadata consumed by every layer kernel.

    Attributes:
        query_start_loc: ``[requests + 1]`` cumulative packed query offsets.
        query_lens: ``[requests]`` tokens contributed by each request.
        request_ids: ``[tokens]`` request id per packed row.
        position_ids: ``[tokens]`` absolute position of each packed row.
        logit_row_indices: rows whose logits the sampler needs.
        moe_token_owners: ``[tokens]`` TP rank that owns each replicated row
            before EP dispatch, so an attention row replicated across TP is not
            dispatched ``TP_SIZE`` times.
        kv_seq_lens: ``[requests]`` cache length before this dispatch.
        new_kv_seq_lens: ``[requests]`` cache length after this dispatch.
        mla_slots: ``[tokens]`` flattened physical row in the 512-latent pool.
        index_slots: ``[tokens]`` flattened physical row in the per-token indexer
            state pool, paged at :data:`BLOCK_SIZE` like the latent cache.
        pool_slots: ``[tokens]`` flattened physical row of the **pooled** state this
            token belongs to, one row per ``index_kpool`` tokens. Tokens inside the
            same pool share a row, and the row is only complete once the pool closes.
        kda_state_rows: ``[requests]`` row in the KDA conv and recurrent state pools.
        pool_count: ``[requests]`` number of complete-or-partial indexer pools.
        tail_start: ``[requests]`` first raw index of the incomplete tail pool.
        tail_count: ``[requests]`` tokens in that tail (``0 .. index_kpool - 1``).
    """

    query_start_loc: torch.Tensor
    query_lens: torch.Tensor
    request_ids: torch.Tensor
    position_ids: torch.Tensor
    logit_row_indices: torch.Tensor
    moe_token_owners: torch.Tensor
    kv_seq_lens: torch.Tensor
    new_kv_seq_lens: torch.Tensor
    mla_slots: torch.Tensor
    index_slots: torch.Tensor
    pool_slots: torch.Tensor
    kda_state_rows: torch.Tensor
    pool_count: torch.Tensor
    tail_start: torch.Tensor
    tail_count: torch.Tensor


def request_ids_from_starts(query_start_loc: torch.Tensor) -> torch.Tensor:
    """Expand packed cumulative query lengths into one request id per token."""
    lengths = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=lengths.device), lengths
    ).to(torch.int32)


def tp_token_owners(
    num_tokens: int,
    tp_size: int = TP_SIZE,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Assign each replicated TP token row to exactly one rank before EP dispatch."""
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    return torch.arange(num_tokens, device=device, dtype=torch.int32).remainder(tp_size)


def paged_slots(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Map logical token positions to flattened physical cache rows."""
    logical_block = torch.div(positions, storage_block_size, rounding_mode="floor")
    offset = positions.remainder(storage_block_size)
    physical = block_table[request_ids.to(torch.long), logical_block.to(torch.long)]
    return physical.to(torch.int64) * storage_block_size + offset.to(torch.int64)


def kpool_metadata(
    kv_seq_lens: torch.Tensor,
    index_kpool: int = INDEX_KPOOL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive the indexer's pool count and incomplete-tail window per request.

    The reference pools from the first *valid* key rather than raw slot zero, which
    matters only for a left-padded dense batch. A paged cache has no left padding —
    request-local position zero is always a real token — so ``first_key`` is zero
    here and the pools are simply ``[4p, 4p+3]``. A kernel that ever sees a padded
    dense layout must add ``first_key`` back.

    Returns:
        ``pool_count`` (complete pools available to the last query of the request),
        ``tail_start`` (first raw index of the incomplete tail) and ``tail_count``
        (how many raw indices that tail holds, ``0 .. index_kpool - 1``).
    """
    visible = kv_seq_lens.to(torch.int64)
    tail_count = visible.remainder(index_kpool)
    tail_start = visible - tail_count
    pool_count = torch.div(visible, index_kpool, rounding_mode="floor")
    return pool_count.to(torch.int32), tail_start.to(torch.int32), tail_count.to(torch.int32)


def build_forward_metadata(
    query_start_loc: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    mla_block_table: torch.Tensor,
    index_block_table: torch.Tensor,
    pool_block_table: torch.Tensor,
    kda_state_rows: torch.Tensor,
    tp_size: int = TP_SIZE,
) -> ForwardMetadata:
    """Lower a scheduler batch into the canonical token-major metadata."""
    device = query_start_loc.device
    query_lens = (query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)).to(
        torch.int32
    )
    request_ids = request_ids_from_starts(query_start_loc)
    num_tokens = int(request_ids.numel())

    local_positions = torch.arange(num_tokens, device=device, dtype=torch.int64) - query_start_loc[
        request_ids.to(torch.long)
    ].to(torch.int64)
    position_ids = (local_positions + kv_seq_lens[request_ids.to(torch.long)].to(torch.int64)).to(
        torch.int32
    )
    new_kv_seq_lens = kv_seq_lens.to(torch.int32) + query_lens
    logit_row_indices = (query_start_loc[1:].to(torch.int64) - 1).to(torch.int32)

    pool_count, tail_start, tail_count = kpool_metadata(new_kv_seq_lens)
    return ForwardMetadata(
        query_start_loc=query_start_loc.to(torch.int32),
        query_lens=query_lens,
        request_ids=request_ids,
        position_ids=position_ids,
        logit_row_indices=logit_row_indices,
        moe_token_owners=tp_token_owners(num_tokens, tp_size, device),
        kv_seq_lens=kv_seq_lens.to(torch.int32),
        new_kv_seq_lens=new_kv_seq_lens,
        mla_slots=paged_slots(position_ids, request_ids, mla_block_table, BLOCK_SIZE).to(
            torch.int32
        ),
        index_slots=paged_slots(position_ids, request_ids, index_block_table, BLOCK_SIZE).to(
            torch.int32
        ),
        pool_slots=paged_slots(
            torch.div(position_ids, INDEX_KPOOL, rounding_mode="floor"),
            request_ids,
            pool_block_table,
            BLOCK_SIZE,
        ).to(torch.int32),
        kda_state_rows=kda_state_rows.to(torch.int32),
        pool_count=pool_count,
        tail_start=tail_start,
        tail_count=tail_count,
    )


__all__ = [
    "ForwardMetadata",
    "build_forward_metadata",
    "kpool_metadata",
    "paged_slots",
    "request_ids_from_starts",
    "tp_token_owners",
]
