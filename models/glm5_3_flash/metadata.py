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

* the 11 backbone MLA layers write a 512-wide latent row into a paged pool of
  :data:`BLOCK_SIZE`-token blocks;
* the same 11 layers write a 256-wide indexer state row — ``[key(128),
  gate_scores(128)]`` — into a second paged pool, and separately read a **pooled**
  state addressed one row per ``index_kpool`` tokens. Those are two different
  mappings and must not be conflated: ``index_slots`` is per token at
  :data:`BLOCK_SIZE` pages, ``pool_slots`` is per pool of four;
* the 34 KDA layers hold no per-token cache at all, only a per-request conv window
  and recurrent state, addressed by a request row that never grows with context.

The same lowering serves prefill and decode. Ordinary decode contributes one
new token per request; speculative callers may contribute additional draft rows.
"""

from dataclasses import dataclass

import torch

from models.glm5_3_flash.config import BLOCK_SIZE
from models.glm5_3_flash.config import INDEX_KPOOL
from models.glm5_3_flash.config import INDEX_STATE_BLOCK_SIZE
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
        pool_count: ``[requests]`` number of complete indexer pools.
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


class SharedCachePages:
    """Own a bounded pool of raw/latent pages shared by persistent request slots."""

    def __init__(self, requests: int, max_context: int, cache_tokens: int):
        if min(requests, max_context, cache_tokens) <= 0 or cache_tokens % BLOCK_SIZE:
            raise ValueError("cache geometry must be positive with a page-aligned token budget")
        self.max_context = max_context
        self.table = torch.full((requests, (max_context + BLOCK_SIZE - 1) // BLOCK_SIZE), -1, dtype=torch.int32)
        self.owners = torch.full((cache_tokens // BLOCK_SIZE,), -1, dtype=torch.int32)

    def reserve(self, lengths: torch.Tensor) -> list[tuple[int, int, int]]:
        """Reserve all extensions atomically; keep pages owned after a failed device dispatch."""
        lengths = torch.as_tensor(lengths, dtype=torch.int64)
        if lengths.shape != self.table.shape[:1] or ((lengths < 0) | (lengths > self.max_context)).any():
            raise ValueError("request lengths must fit the configured context limit")
        required = (lengths + BLOCK_SIZE - 1) // BLOCK_SIZE
        allocated = (self.table >= 0).sum(dim=1)
        missing = (required - allocated).clamp_min(0)
        free = torch.nonzero(self.owners < 0, as_tuple=False).flatten().tolist()
        if int(missing.sum()) > len(free):
            raise MemoryError("shared cache token budget exhausted; reset completed requests or increase --cache-tokens")
        reservations = []
        cursor = 0
        for request, count in enumerate(missing.tolist()):
            for logical in range(int(allocated[request]), int(allocated[request]) + count):
                physical = free[cursor]
                cursor += 1
                self.table[request, logical] = physical
                self.owners[physical] = request
                reservations.append((request, logical, physical))
        return reservations

    def cancel_reservation(self, reservations: list[tuple[int, int, int]]) -> None:
        """Cancel extensions only before any device write can have occurred."""
        for request, logical, physical in reversed(reservations):
            if int(self.table[request, logical]) != physical or int(self.owners[physical]) != request:
                raise ValueError("cache reservation no longer belongs to its request")
            self.table[request, logical] = -1
            self.owners[physical] = -1

    def release(self, requests: list[int]) -> None:
        """Release pages after the caller has cleared their device payloads and scales."""
        if len(set(requests)) != len(requests) or any(type(slot) is not int or slot not in range(self.table.shape[0]) for slot in requests):
            raise ValueError("cache release needs unique valid request slots")
        for request in requests:
            physical = self.table[request]
            physical = physical[physical >= 0].long()
            self.owners[physical] = -1
            self.table[request].fill_(-1)

    def pool_table(self) -> torch.Tensor:
        """Map each logical four-pool page onto its raw page's pooled-key storage."""
        pools_per_raw_page = BLOCK_SIZE // INDEX_KPOOL
        pages_per_raw_page = pools_per_raw_page // INDEX_STATE_BLOCK_SIZE
        logical = torch.arange((self.max_context + INDEX_KPOOL * INDEX_STATE_BLOCK_SIZE - 1)
                               // (INDEX_KPOOL * INDEX_STATE_BLOCK_SIZE))
        raw = self.table[:, logical // pages_per_raw_page]
        physical = raw * pages_per_raw_page + logical.remainder(pages_per_raw_page)
        return torch.where(raw >= 0, physical, -1).int()

    def pool_offsets(self) -> torch.Tensor:
        """Pack score segments in request order, retaining page-aligned private visibility."""
        spans = (self.table >= 0).sum(dim=1) * (BLOCK_SIZE // INDEX_KPOOL)
        return torch.cat((torch.zeros(1, dtype=torch.int64), spans.cumsum(0)[:-1])).int()


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
            INDEX_STATE_BLOCK_SIZE,
        ).to(torch.int32),
        kda_state_rows=kda_state_rows.to(torch.int32),
        pool_count=pool_count,
        tail_start=tail_start,
        tail_count=tail_count,
    )


def build_indexer_chunk_metadata(
    metadata: ForwardMetadata,
    index_block_table: torch.Tensor,
    pool_offsets: torch.Tensor,
    *,
    pools_cap: int,
    pool_rows: int,
    pool_block_table: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Lower one chunk against persistent raw and pooled indexer caches.

    ``pool_offsets`` gives each batch request's persistent pooled-row segment.
    Only pools completed by this chunk produce write events; their four raw
    members may include rows cached by an earlier chunk. Query positions and
    expanded attention indices remain absolute positions within each request.
    Empty requests produce no events, and padded events target the scratch row.
    """
    if pools_cap <= 0 or (pool_block_table is None and pool_rows < pools_cap * INDEX_STATE_BLOCK_SIZE):
        raise ValueError("indexer pool storage must cover the identity pool-block gather")
    device = metadata.query_start_loc.device
    old_lengths = metadata.kv_seq_lens.to(device=device, dtype=torch.int64)
    new_lengths = metadata.new_kv_seq_lens.to(device=device, dtype=torch.int64)
    offsets = pool_offsets.to(device=device, dtype=torch.int64)
    requests = old_lengths.numel()
    if offsets.ndim != 1 or offsets.numel() != requests or new_lengths.shape != old_lengths.shape:
        raise ValueError("indexer chunk metadata needs one length and pool offset per request")
    if index_block_table.ndim != 2 or index_block_table.shape[0] != requests:
        raise ValueError("indexer chunk metadata needs one raw block-table row per request")
    if bool(((old_lengths < 0) | (new_lengths < old_lengths)).any()):
        raise ValueError("indexer chunk lengths must extend a nonnegative cached prefix")

    prior_pools = torch.div(old_lengths, INDEX_KPOOL, rounding_mode="floor")
    new_pools = torch.div(new_lengths, INDEX_KPOOL, rounding_mode="floor")
    event_counts = new_pools - prior_pools
    if int(event_counts.sum().item()) > pools_cap:
        raise ValueError("indexer pool-close events exceed the compiled event capacity")
    active_requests = new_lengths > old_lengths
    invalid_windows = (offsets < 0) | (offsets + new_pools > pools_cap)
    if bool((active_requests & invalid_windows).any()):
        raise ValueError("indexer request pool segment exceeds the compiled score width")

    members = torch.full((pools_cap, INDEX_KPOOL), -1, dtype=torch.int32, device=device)
    destinations = torch.full((pools_cap,), pool_rows - 1, dtype=torch.int32, device=device)
    raw_table = index_block_table.to(device=device, dtype=torch.int64)
    lanes = torch.arange(INDEX_KPOOL, dtype=torch.int64, device=device)
    event = 0
    for request in range(requests):
        count = int(event_counts[request].item())
        if not count:
            continue
        pools = torch.arange(prior_pools[request], new_pools[request], device=device)
        positions = pools.unsqueeze(1) * INDEX_KPOOL + lanes.unsqueeze(0)
        logical_blocks = torch.div(positions, BLOCK_SIZE, rounding_mode="floor")
        if int(logical_blocks.max().item()) >= raw_table.shape[1]:
            raise ValueError("indexer pool members exceed the request's raw block table")
        pages = raw_table[request, logical_blocks]
        if bool((pages < 0).any()):
            raise ValueError("indexer pool-close event references an unallocated raw page")
        rows = pages * BLOCK_SIZE + positions.remainder(BLOCK_SIZE)
        members[event : event + count] = rows.to(torch.int32)
        if pool_block_table is None:
            destinations[event : event + count] = (offsets[request] + pools).to(torch.int32)
        else:
            page = pool_block_table[request, pools // INDEX_STATE_BLOCK_SIZE].long()
            if (page < 0).any():
                raise ValueError("pool-close event references an unallocated pooled page")
            destinations[event : event + count] = (page * INDEX_STATE_BLOCK_SIZE + pools.remainder(INDEX_STATE_BLOCK_SIZE)).int()
        event += count

    request_ids = metadata.request_ids.to(device=device, dtype=torch.int64)
    positions = metadata.position_ids.to(device=device, dtype=torch.int64)
    if request_ids.shape != positions.shape:
        raise ValueError("indexer chunk token positions must match token request ids")
    valid = request_ids >= 0
    if bool((request_ids[valid] >= requests).any()):
        raise ValueError("indexer chunk token references an unknown batch request")
    safe_requests = request_ids.clamp(min=0)
    length_through = torch.where(valid, positions + 1, torch.zeros_like(positions))
    tails = length_through.remainder(INDEX_KPOOL)
    visible = torch.div(length_through, INDEX_KPOOL, rounding_mode="floor")
    visible = torch.where(valid, torch.minimum(visible, new_pools[safe_requests]), torch.zeros_like(visible))
    segments = torch.where(valid, offsets[safe_requests], torch.zeros_like(positions))
    pool_blocks = torch.arange(pools_cap, dtype=torch.int32, device=device) * INDEX_STATE_BLOCK_SIZE
    if pool_block_table is not None:
        scratch_base = pool_rows - INDEX_STATE_BLOCK_SIZE
        pool_blocks.fill_(scratch_base)
        for request in range(requests):
            pages = (int(new_pools[request]) + INDEX_STATE_BLOCK_SIZE - 1) // INDEX_STATE_BLOCK_SIZE
            begin = int(offsets[request]) // INDEX_STATE_BLOCK_SIZE
            sources = pool_block_table[request, :pages].int() * INDEX_STATE_BLOCK_SIZE
            if (sources < 0).any() or (sources + INDEX_STATE_BLOCK_SIZE > scratch_base).any():
                raise ValueError("pooled page lies outside shared cache storage")
            pool_blocks[begin:begin + pages] = sources
    return {
        "seg_start": segments.to(torch.int32),
        "pool_count_tok": visible.to(torch.int32),
        "tail_start": (length_through - tails).to(torch.int32),
        "tail_count": tails.to(torch.int32),
        "kv_len_tok": length_through.to(torch.int32),
        "pool_token_slots": members,
        "pool_slots": destinations,
        "pool_blocks": pool_blocks,
    }


__all__ = [
    "ForwardMetadata",
    "SharedCachePages",
    "build_forward_metadata",
    "build_indexer_chunk_metadata",
    "kpool_metadata",
    "paged_slots",
    "request_ids_from_starts",
    "tp_token_owners",
]
