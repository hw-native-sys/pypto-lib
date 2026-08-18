# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime context geometry for DeepSeek-V4 D-Spark decode (Run 055 Phase A-D).

Single source of truth for the runtime *derivation* of active rows / shards /
leaves / tail from per-request runtime state. The *granularity* (micro-tile /
shard / leaf size) is a compile-time constant defined in :mod:`config`; the
*counts* are derived here at runtime from each request's committed length,
position, allocated capacity and cache validity.

This module is the host reference. The device lowering in
:mod:`decode_metadata` consumes the same names and formulas; contract tests
assert that the two stay in sync. Nothing in this module imports PyPTO, so it
can run as pure Python (static / contract gate) without a device.

Key invariants (enforced at import by :mod:`config`):
  - ``M.max_position_embeddings`` is the only canonical ceiling (1,048,576).
  - 128 / 12K / 16K / 1M are test points, not length profiles or buckets.
  - shard / leaf granularity is fixed at compile time and never appears as a
    runtime ABI input (no ``rows_per_shard`` / ``candidates_per_leaf`` /
    ``pages_per_task`` parameters).
  - pool capacity is a global property and is never derived from ``B``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from config import (
    CACHE_BLOCK_SIZE,
    CSA_CANDIDATES_PER_LEAF,
    CSA_COMPRESS_RATIO,
    CSA_INNER_STATE_BLOCK_SIZE,
    CSA_INNER_STATE_PAGES_PER_REQUEST,
    CSA_INNER_STATE_ROWS_PER_REQUEST,
    CSA_LOGICAL_INDEX_MAX,
    CSA_LOGICAL_INDEX_MIN,
    CSA_MAX_QUERIES,
    CSA_MAX_TOPK_TASKS,
    CSA_MERGE_ARITY,
    CSA_PAIR_BYTES,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_PAGES_PER_REQUEST,
    CSA_STATE_ROWS_PER_REQUEST,
    CSA_TOPK,
    CSA_TOPK_INVALID_TASK_SLOT,
    CSA_TOPK_READY_FRONTIER_W,
    FLASH as M,
    HCA_COMPRESS_RATIO,
    HCA_ROWS_PER_SHARD,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_PAGES_PER_REQUEST,
    HCA_STATE_ROWS_PER_REQUEST,
    MAX_CSA_CANDIDATES,
    MAX_CSA_LEAVES,
    MAX_HCA_ROWS,
    SWA_PERSISTENT_PAGES_PER_REQUEST,
    SWA_PERSISTENT_ROWS_PER_REQUEST,
    SWA_SOURCE_INVALID,
    SWA_SOURCE_INT32_MAX,
    SWA_SOURCE_OVERLAY_BASE,
    SWA_WINDOW_ROWS,
    decode_swa_overlay_source,
    encode_swa_overlay_source,
)


# Absolute-slot integer type. Python ints are arbitrary precision, so 1M-scale
# absolute slot / page ids never overflow in the host reference. Device code
# uses INT64 for the same reason (see decode_metadata slot mappings).
INT_ABSOLUTE = int


def ceil_div(n: int, d: int) -> int:
    """Ceiling division with 0-length safety: ``ceil_div(0, d) == 0``.

    ``d`` must be positive. ``n`` may be negative (treated as 0 work).
    """
    if d <= 0:
        raise ValueError(f"ceil_div divisor must be positive, got {d}")
    if n <= 0:
        return 0
    return (n + d - 1) // d


def admit_ragged_page_counts(
    request_lengths: list[int],
    *,
    block_size: int,
    physical_pool_blocks: int,
) -> list[int]:
    """Validate shared-pool admission and return pages per request."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if physical_pool_blocks <= 0:
        raise ValueError(
            f"physical_pool_blocks must be positive, got {physical_pool_blocks}"
        )
    counts: list[int] = []
    for length in request_lengths:
        if length < 0:
            raise ValueError(f"request length must be >= 0, got {length}")
        counts.append(ceil_div(length, block_size))
    total = sum(counts)
    if total > physical_pool_blocks:
        raise ValueError(
            f"requests need {total} pages in total but the shared pool has "
            f"only {physical_pool_blocks}"
        )
    return counts


def hetero_length_starts_values(lengths: list[int], *, seq: int) -> list[int]:
    """Return decode-chunk starts while accepting an exact 1M final token."""
    if seq <= 0:
        raise ValueError(f"seq must be positive, got {seq}")
    starts: list[int] = []
    for length in lengths:
        if length < 0 or length > M.max_position_embeddings:
            raise ValueError(
                f"request length {length} out of [0, {M.max_position_embeddings}]"
            )
        starts.append(max(0, length - seq))
    return starts


def _intersect(*bounds: int) -> int:
    """Intersect non-negative integer bounds (min of all, floored at 0).

    Used for ``visible_rows = intersect(committed_floor, allocated, valid_range)``.
    """
    return max(0, min(bounds))


# ---------------------------------------------------------------------------
# Per-request runtime descriptor. A request owns a contiguous logical token
# space [0, committed_tokens). ``allocated_*`` is the allocator-granted
# capacity for that cache group; ``valid_*`` is the still-alive range (pages
# may have been evicted/reused with a newer epoch). Both clamp the visible
# count below the committed length.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestGeometry:
    """Per-request runtime geometry inputs (one request's logical history).

    All counts are in *tokens* unless suffixed ``_rows`` / ``_candidates``.
    ``active=False`` collapses every visible count to 0 (inactive lane = 0
    work; the request does not read/write cache and does not submit tasks).
    """

    # Logical history this request has committed (prefill + accepted decode).
    committed_tokens: int
    # Allocator-granted raw/SWA capacity and still-alive absolute-token range.
    raw_allocated_rows: int
    raw_valid_begin: int
    raw_valid_end: int
    # Allocator-granted HCA capacity and still-alive absolute-row range.
    hca_allocated_rows: int
    hca_valid_begin: int
    hca_valid_end: int
    # Allocator-granted CSA capacity and still-alive absolute-candidate range.
    csa_allocated_candidates: int
    csa_valid_begin: int
    csa_valid_end: int
    # Inactive lane flag.
    active: bool = True

    def validate(self) -> "RequestGeometry":
        t = self.committed_tokens
        if t < 0:
            raise ValueError(f"committed_tokens must be >= 0, got {t}")
        if t > M.max_position_embeddings:
            raise ValueError(
                f"committed_tokens {t} exceeds M.max_position_embeddings={M.max_position_embeddings}"
            )
        if self.raw_allocated_rows < 0 or self.raw_allocated_rows > SWA_WINDOW_ROWS:
            raise ValueError(
                f"raw_allocated_rows {self.raw_allocated_rows} out of "
                f"[0, {SWA_WINDOW_ROWS}]"
            )
        if self.raw_valid_begin < 0 or self.raw_valid_end < self.raw_valid_begin:
            raise ValueError(
                f"raw valid range [{self.raw_valid_begin}, {self.raw_valid_end}) is invalid"
            )
        if self.raw_valid_end - self.raw_valid_begin > self.raw_allocated_rows:
            raise ValueError(
                f"raw valid range length {self.raw_valid_end - self.raw_valid_begin} "
                f"exceeds allocated rows {self.raw_allocated_rows}"
            )
        if self.raw_valid_end > t:
            raise ValueError(
                f"raw_valid_end {self.raw_valid_end} exceeds committed_tokens {t}"
            )
        if self.hca_allocated_rows < 0 or self.hca_allocated_rows > MAX_HCA_ROWS:
            raise ValueError(
                f"hca_allocated_rows {self.hca_allocated_rows} out of [0, {MAX_HCA_ROWS}]"
            )
        if self.hca_valid_begin < 0 or self.hca_valid_end < self.hca_valid_begin:
            raise ValueError(
                f"HCA valid range [{self.hca_valid_begin}, {self.hca_valid_end}) is invalid"
            )
        if self.hca_valid_end - self.hca_valid_begin > self.hca_allocated_rows:
            raise ValueError(
                f"HCA valid range length {self.hca_valid_end - self.hca_valid_begin} "
                f"exceeds allocated rows {self.hca_allocated_rows}"
            )
        if self.hca_valid_end > MAX_HCA_ROWS:
            raise ValueError(
                f"hca_valid_end {self.hca_valid_end} exceeds {MAX_HCA_ROWS}"
            )
        if self.csa_allocated_candidates < 0 or self.csa_allocated_candidates > MAX_CSA_CANDIDATES:
            raise ValueError(
                f"csa_allocated_candidates {self.csa_allocated_candidates} out of "
                f"[0, {MAX_CSA_CANDIDATES}]"
            )
        if (
            self.csa_valid_begin < 0
            or self.csa_valid_end < self.csa_valid_begin
        ):
            raise ValueError(
                f"CSA valid range [{self.csa_valid_begin}, {self.csa_valid_end}) is invalid"
            )
        if self.csa_valid_end - self.csa_valid_begin > self.csa_allocated_candidates:
            raise ValueError(
                f"CSA valid range length {self.csa_valid_end - self.csa_valid_begin} "
                f"exceeds allocated candidates {self.csa_allocated_candidates}"
            )
        if self.csa_valid_end > MAX_CSA_CANDIDATES:
            raise ValueError(
                f"csa_valid_end {self.csa_valid_end} exceeds {MAX_CSA_CANDIDATES}"
            )
        return self


# ---------------------------------------------------------------------------
# Per-query geometry. A query is one decode token belonging to one request.
# ``position`` is the token's absolute logical position; ``committed_tokens``
# and the allocator state come from the owning request. ``visible_tokens`` is
# the intersection of committed history, ``position + 1`` (a token cannot see
# the future), allocator capacity and the 1M ceiling.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryGeometry:
    """Per-query runtime geometry inputs."""

    request: RequestGeometry
    position: int  # absolute logical position of this decode token

    def validate(self) -> "QueryGeometry":
        p = self.position
        if p < 0:
            raise ValueError(f"position must be >= 0, got {p}")
        if p >= M.max_position_embeddings:
            raise ValueError(
                f"position {p} >= M.max_position_embeddings={M.max_position_embeddings}"
            )
        self.request.validate()
        # An inactive lane never reads cache and submits 0 work, so its
        # position is not required to fall inside committed_tokens. The
        # visible-derivation helpers still collapse every count to 0.
        if self.request.active and p >= self.request.committed_tokens:
            raise ValueError(
                f"position {p} >= committed_tokens {self.request.committed_tokens}"
            )
        return self


# ---------------------------------------------------------------------------
# Reference formulas (run_055 plan §4.2). The host reference and the device
# lowering use identical names and identical arithmetic.
# ---------------------------------------------------------------------------


def visible_tokens(q: QueryGeometry) -> int:
    """``min(committed, position+1, allocated_via_request, M.max_position_embeddings)``.

    The request's allocated/valid range is applied per cache group (HCA/CSA)
    rather than globally, so here we only clamp to committed, position+1 and
    the ceiling; per-group intersect happens in the row/candidate helpers.
    """
    q.validate()
    if not q.request.active:
        return 0
    return _intersect(
        q.request.committed_tokens,
        q.position + 1,
        M.max_position_embeddings,
    )


def visible_swa_rows(q: QueryGeometry) -> int:
    """``min(visible_tokens, SWA_WINDOW_ROWS)`` — never more than 128 rows."""
    return min(visible_tokens(q), SWA_WINDOW_ROWS)


def visible_hca_rows(q: QueryGeometry) -> int:
    """``intersect(floor(visible_tokens / HCA_COMPRESS_RATIO),
    hca_allocated_rows, hca_valid_range)``.

    For a 1-token request there is no complete compressed row yet, so this is
    0. For a 128-token request exactly 1 row; for 1M exactly 8192 rows.
    """
    vt = visible_tokens(q)
    if vt == 0:
        return 0
    committed_rows = vt // HCA_COMPRESS_RATIO
    live_end = min(committed_rows, q.request.hca_valid_end)
    live_rows = max(0, live_end - q.request.hca_valid_begin)
    return min(live_rows, q.request.hca_allocated_rows)


def num_hca_shards(q: QueryGeometry) -> int:
    """``ceil_div(visible_hca_rows, HCA_ROWS_PER_SHARD)`` — runtime count only."""
    return ceil_div(visible_hca_rows(q), HCA_ROWS_PER_SHARD)


def hca_tail_valid_rows(q: QueryGeometry) -> int:
    """Valid rows in the last active HCA shard; 0 when there is no tail."""
    rows = visible_hca_rows(q)
    if rows == 0:
        return 0
    rem = rows % HCA_ROWS_PER_SHARD
    return rem if rem != 0 else HCA_ROWS_PER_SHARD


def visible_csa_candidates(q: QueryGeometry) -> int:
    """``intersect(floor(visible_tokens / CSA_COMPRESS_RATIO),
    csa_allocated_candidates, csa_valid_range)``."""
    vt = visible_tokens(q)
    if vt == 0:
        return 0
    committed_candidates = vt // CSA_COMPRESS_RATIO
    live_end = min(committed_candidates, q.request.csa_valid_end)
    live_candidates = max(0, live_end - q.request.csa_valid_begin)
    return min(live_candidates, q.request.csa_allocated_candidates)


def num_csa_leaves(q: QueryGeometry) -> int:
    """``ceil_div(visible_csa_candidates, CSA_CANDIDATES_PER_LEAF)``."""
    return ceil_div(visible_csa_candidates(q), CSA_CANDIDATES_PER_LEAF)


def csa_tail_valid_candidates(q: QueryGeometry) -> int:
    """Valid candidates in the last active CSA leaf; 0 when there is no tail."""
    cand = visible_csa_candidates(q)
    if cand == 0:
        return 0
    rem = cand % CSA_CANDIDATES_PER_LEAF
    return rem if rem != 0 else CSA_CANDIDATES_PER_LEAF


# ---------------------------------------------------------------------------
# Convenience: full per-query geometry descriptor. Returned as a plain
# dataclass so contract tests can compare whole structs against the plan's
# §6.1 test matrix in one assertion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryGeometryResult:
    visible_tokens: int
    visible_swa_rows: int
    visible_hca_rows: int
    num_hca_shards: int
    hca_tail_valid_rows: int
    visible_csa_candidates: int
    num_csa_leaves: int
    csa_tail_valid_candidates: int


def derive_query_geometry(q: QueryGeometry) -> QueryGeometryResult:
    """Compute every runtime count for one active query.

    Inactive query -> all-zero result (0 work, no cache access, no tasks).
    """
    q.validate()
    if not q.request.active:
        return QueryGeometryResult(0, 0, 0, 0, 0, 0, 0, 0)
    return QueryGeometryResult(
        visible_tokens=visible_tokens(q),
        visible_swa_rows=visible_swa_rows(q),
        visible_hca_rows=visible_hca_rows(q),
        num_hca_shards=num_hca_shards(q),
        hca_tail_valid_rows=hca_tail_valid_rows(q),
        visible_csa_candidates=visible_csa_candidates(q),
        num_csa_leaves=num_csa_leaves(q),
        csa_tail_valid_candidates=csa_tail_valid_candidates(q),
    )


# ---------------------------------------------------------------------------
# Active-request / active-query batch helpers. A batch is ragged: each
# request contributes its own active queries (e.g. S per active request). The
# host reference never pads inactive lanes to a fixed B; inactive requests
# contribute 0 active queries.
# ---------------------------------------------------------------------------


def active_query_count(requests: list[RequestGeometry]) -> int:
    """Number of active queries across the batch (sum of active requests' S).

    Phase A leaves per-request query count to the metadata layer; here we only
    expose the active-request count and let callers map queries -> requests.
    """
    return sum(1 for r in requests if r.active)


def active_request_count(requests: list[RequestGeometry]) -> int:
    """Number of active requests in the batch."""
    return sum(1 for r in requests if r.active)


# ---------------------------------------------------------------------------
# Absolute logical-slot helpers for the ragged page map. These do NOT use
# modulo to revive stale pages: a slot is valid only if it falls inside the
# request's ``valid_*`` range and the page's epoch matches.
# ---------------------------------------------------------------------------


def logical_to_page(logical_slot: int, *, block_size: int) -> int:
    """Logical absolute slot -> logical page id."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if logical_slot < 0:
        raise ValueError(f"logical_slot must be >= 0, got {logical_slot}")
    return logical_slot // block_size


def logical_to_page_offset(logical_slot: int, *, block_size: int) -> int:
    """Intra-page offset of a logical absolute slot."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if logical_slot < 0:
        raise ValueError(f"logical_slot must be >= 0, got {logical_slot}")
    return logical_slot % block_size


@dataclass(frozen=True)
class PageValidity:
    """Validity window for one logical page of one request.

    A logical slot is live iff ``valid_begin <= slot < valid_end`` AND the
    page's ``epoch`` equals the request's current epoch. Stale pages (older
    epoch) are *not* revived by modulo-arithmetic on the physical pool.
    """

    valid_begin: int
    valid_end: int  # exclusive
    epoch: int


def slot_is_valid(
    logical_slot: int,
    *,
    page_validity: PageValidity,
    request_epoch: int,
) -> bool:
    """``True`` iff the slot is inside the valid window and the page is live."""
    if logical_slot < 0:
        return False
    if page_validity.epoch != request_epoch:
        return False
    return page_validity.valid_begin <= logical_slot < page_validity.valid_end


# ---------------------------------------------------------------------------
# Phase C HCA reference: one recurrent-state ring and packed compressed work.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HcaStateDescriptor:
    """Pre-step state for the current non-overlapping ratio-128 block."""

    page_ids: tuple[int, ...]
    valid_begin: int
    valid_end: int
    page_epochs: tuple[int, ...]
    request_epoch: int
    active: bool = True

    def validate(self) -> "HcaStateDescriptor":
        if not self.active:
            return self
        if len(self.page_ids) != HCA_STATE_PAGES_PER_REQUEST:
            raise ValueError("an active HCA state descriptor requires sixteen pages")
        if len(self.page_epochs) != len(self.page_ids):
            raise ValueError("HCA state page ids and epochs must have equal length")
        if any(page < 0 for page in self.page_ids):
            raise ValueError("active HCA state page ids must be non-negative")
        if any(epoch < 0 for epoch in self.page_epochs) or self.request_epoch < 0:
            raise ValueError("HCA state and request epochs must be non-negative")
        if any(epoch != self.request_epoch for epoch in self.page_epochs):
            raise ValueError("an HCA state page epoch does not match the request epoch")
        if self.valid_begin < 0 or self.valid_end < self.valid_begin:
            raise ValueError(
                f"invalid HCA state range [{self.valid_begin}, {self.valid_end})"
            )
        if self.valid_end > M.max_position_embeddings:
            raise ValueError("HCA state range exceeds the 1M context ceiling")
        if self.valid_end - self.valid_begin >= HCA_STATE_ROWS_PER_REQUEST:
            raise ValueError("HCA pre-step state must contain fewer than 128 rows")
        canonical_begin = (
            self.valid_end // HCA_COMPRESS_RATIO
        ) * HCA_COMPRESS_RATIO
        if self.valid_begin != canonical_begin:
            raise ValueError(
                "HCA state range must be the canonical partial ratio-128 block"
            )
        return self


def hca_boundary_event(position: int, *, active: bool = True) -> Optional[int]:
    """Return the compressed-row id emitted by ``position``, if any."""
    if not active:
        return None
    if position < 0 or position >= M.max_position_embeddings:
        raise ValueError(f"position {position} is outside the 1M context ceiling")
    if (position + 1) % HCA_COMPRESS_RATIO != 0:
        return None
    return position // HCA_COMPRESS_RATIO


def hca_state_write_row(descriptor: HcaStateDescriptor, position: int) -> int:
    """Map a current token into the assigned state page after validation."""
    if not descriptor.active:
        return -1
    descriptor.validate()
    if position < 0 or position >= M.max_position_embeddings:
        return -1
    return (
        descriptor.page_ids[
            (position % HCA_STATE_ROWS_PER_REQUEST) // HCA_STATE_BLOCK_SIZE
        ]
        * HCA_STATE_BLOCK_SIZE
        + position % HCA_STATE_BLOCK_SIZE
    )


def hca_next_state_valid_range(
    last_active_position: Optional[int], *, active: bool = True
) -> tuple[int, int]:
    """Return the post-step partial ratio-block range."""
    if not active or last_active_position is None:
        return 0, 0
    if last_active_position < 0 or last_active_position >= M.max_position_embeddings:
        raise ValueError(
            f"last_active_position {last_active_position} is outside the 1M ceiling"
        )
    step_end = last_active_position + 1
    begin = (step_end // HCA_COMPRESS_RATIO) * HCA_COMPRESS_RATIO
    return begin, step_end


@dataclass(frozen=True)
class HcaPageMap:
    """One request's ragged HCA-main page assignment."""

    page_ids: tuple[int, ...]
    page_epochs: tuple[int, ...]
    valid_begin: int
    valid_end: int
    head: int
    request_epoch: int

    def validate(self) -> "HcaPageMap":
        if len(self.page_ids) != len(self.page_epochs):
            raise ValueError("HCA page ids and epochs must have the same length")
        if self.request_epoch < 0:
            raise ValueError("HCA request epoch must be non-negative")
        if self.valid_begin < 0 or self.valid_end < self.valid_begin:
            raise ValueError(
                f"invalid HCA main range [{self.valid_begin}, {self.valid_end})"
            )
        if self.valid_end > MAX_HCA_ROWS:
            raise ValueError("HCA main range exceeds the 1M row ceiling")
        if not self.page_ids:
            if self.valid_begin != self.valid_end or self.head != 0:
                raise ValueError("an empty HCA page map requires an empty range and head=0")
            return self
        if self.head < 0 or self.head >= len(self.page_ids):
            raise ValueError("HCA page head is outside the ragged page span")
        if any(page < 0 for page in self.page_ids):
            raise ValueError("HCA physical page ids must be non-negative")
        if any(epoch < 0 for epoch in self.page_epochs):
            raise ValueError("HCA page epochs must be non-negative")
        if self.valid_end > self.valid_begin:
            first_page = self.valid_begin // CACHE_BLOCK_SIZE
            last_page = (self.valid_end - 1) // CACHE_BLOCK_SIZE
            if last_page - first_page + 1 > len(self.page_ids):
                raise ValueError("HCA page span cannot cover the declared valid range")
        return self


def hca_physical_row(
    page_map: HcaPageMap,
    logical_row: int,
    *,
    for_write: bool = False,
) -> int:
    """Resolve one HCA logical row after validity and epoch checks."""
    page_map.validate()
    if logical_row < 0 or logical_row >= MAX_HCA_ROWS or not page_map.page_ids:
        return -1
    if not for_write and not (
        page_map.valid_begin <= logical_row < page_map.valid_end
    ):
        return -1
    logical_page_base = page_map.valid_begin // CACHE_BLOCK_SIZE
    relative_page = logical_row // CACHE_BLOCK_SIZE - logical_page_base
    if relative_page < 0 or relative_page >= len(page_map.page_ids):
        return -1
    page_index = (page_map.head + relative_page) % len(page_map.page_ids)
    if page_map.page_epochs[page_index] != page_map.request_epoch:
        return -1
    return (
        page_map.page_ids[page_index] * CACHE_BLOCK_SIZE
        + logical_row % CACHE_BLOCK_SIZE
    )


def visible_hca_row_range_with_events(
    q: QueryGeometry,
    current_event_rows: Sequence[int] = (),
) -> tuple[int, int]:
    """Return the causal HCA range from pre-step rows plus current events."""
    q.validate()
    if not q.request.active:
        return 0, 0
    causal_end = visible_tokens(q) // HCA_COMPRESS_RATIO
    begin = q.request.hca_valid_begin
    allocated_end = min(MAX_HCA_ROWS, begin + q.request.hca_allocated_rows)
    end = min(causal_end, q.request.hca_valid_end, allocated_end)
    for row in sorted(set(int(value) for value in current_event_rows)):
        if row < begin or row >= causal_end or row >= allocated_end:
            continue
        if row > end:
            raise ValueError("current HCA events must append without a logical gap")
        end = max(end, row + 1)
    return begin, max(begin, end)


@dataclass(frozen=True)
class HcaPackedWork:
    """Exact active HCA work; no fixed ``Q * MAX_HCA_SHARDS`` padding."""

    query_work_offsets: tuple[int, ...]
    work_query_ids: tuple[int, ...]
    work_row_begins: tuple[int, ...]
    work_valid_rows: tuple[int, ...]

    @property
    def n_work(self) -> int:
        return len(self.work_query_ids)


def build_hca_packed_work(
    queries: Sequence[QueryGeometry],
    query_request_ids: Sequence[int],
    current_events_by_request: Mapping[int, Sequence[tuple[int, int]]],
) -> HcaPackedWork:
    """Build exact query-major shard work from causal current events."""
    if len(queries) != len(query_request_ids):
        raise ValueError("queries and query_request_ids must have the same length")
    offsets = [0]
    work_query_ids: list[int] = []
    row_begins: list[int] = []
    valid_rows: list[int] = []
    for query_index, (query, request_id) in enumerate(
        zip(queries, query_request_ids)
    ):
        visible_events = [
            row
            for event_position, row in current_events_by_request.get(request_id, ())
            if event_position <= query.position
        ]
        begin, end = visible_hca_row_range_with_events(query, visible_events)
        row = begin
        while row < end:
            count = min(HCA_ROWS_PER_SHARD, end - row)
            work_query_ids.append(query_index)
            row_begins.append(row)
            valid_rows.append(count)
            row += count
        offsets.append(len(work_query_ids))
    return HcaPackedWork(
        query_work_offsets=tuple(offsets),
        work_query_ids=tuple(work_query_ids),
        work_row_begins=tuple(row_begins),
        work_valid_rows=tuple(valid_rows),
    )


# ---------------------------------------------------------------------------
# Phase D CSA reference: paired main/index page maps, two ratio-4 state rings,
# causal event visibility, exact packed leaves, and an odd-carry binary forest.
# All page, state, and forest descriptors are host metadata; no helper below
# introduces a length-dependent micro-tile or a dense max-context table.
# ---------------------------------------------------------------------------


CSA_INVALID_PHYSICAL_ROW = -1
CSA_INVALID_FOREST_SLOT = -1


@dataclass(frozen=True)
class CsaPageMap:
    """One ragged CSA page map for either main KV or index KV.

    ``page_ids`` is ordered by logical page beginning at
    ``valid_begin // CACHE_BLOCK_SIZE``.  ``head`` rotates only that ragged
    span.  Structural validity is checked before a physical page is selected;
    stale epochs are rejected by the row helpers before they rotate the head.
    """

    page_ids: tuple[int, ...]
    page_epochs: tuple[int, ...]
    valid_begin: int
    valid_end: int
    head: int
    request_epoch: int

    @property
    def required_pages(self) -> int:
        """Number of logical pages needed to cover the live candidate range."""
        if self.valid_end <= self.valid_begin:
            return 0
        first = self.valid_begin // CACHE_BLOCK_SIZE
        last = (self.valid_end - 1) // CACHE_BLOCK_SIZE
        return last - first + 1

    @property
    def logical_page_base(self) -> int:
        """First logical page represented by this ragged span."""
        return self.valid_begin // CACHE_BLOCK_SIZE

    @property
    def page_span_end(self) -> int:
        """Exclusive candidate limit addressable by the assigned page span."""
        return (self.logical_page_base + len(self.page_ids)) * CACHE_BLOCK_SIZE

    def validate(self) -> "CsaPageMap":
        if len(self.page_ids) != len(self.page_epochs):
            raise ValueError("CSA page ids and epochs must have the same length")
        if self.request_epoch < 0:
            raise ValueError("CSA request epoch must be non-negative")
        if self.valid_begin < 0 or self.valid_end < self.valid_begin:
            raise ValueError(
                f"invalid CSA candidate range [{self.valid_begin}, {self.valid_end})"
            )
        if self.valid_end > MAX_CSA_CANDIDATES:
            raise ValueError("CSA candidate range exceeds the 1M ceiling")
        if len(self.page_ids) > MAX_CSA_CANDIDATES // CACHE_BLOCK_SIZE:
            raise ValueError("CSA page span exceeds the one-request 1M capacity")
        if not self.page_ids:
            if self.required_pages != 0 or self.head != 0:
                raise ValueError("an empty CSA page map requires an empty range and head=0")
            return self
        if self.head < 0 or self.head >= len(self.page_ids):
            raise ValueError("CSA page head is outside the ragged page span")
        if any(page < 0 for page in self.page_ids):
            raise ValueError("CSA physical page ids must be non-negative")
        if any(epoch < 0 for epoch in self.page_epochs):
            raise ValueError("CSA page epochs must be non-negative")
        if len(self.page_ids) < self.required_pages:
            raise ValueError("CSA page span cannot cover the declared valid range")
        return self


def csa_physical_row(
    page_map: CsaPageMap,
    logical_candidate: int,
    *,
    for_write: bool = False,
) -> int:
    """Resolve one CSA candidate row without reviving stale pages by rotation.

    A write may target the next candidate outside the pre-step valid range,
    but still has to fit the descriptor's already allocated ragged page span.
    """
    page_map.validate()
    if (
        logical_candidate < CSA_LOGICAL_INDEX_MIN
        or logical_candidate > CSA_LOGICAL_INDEX_MAX
        or not page_map.page_ids
    ):
        return CSA_INVALID_PHYSICAL_ROW
    if not for_write and not (
        page_map.valid_begin <= logical_candidate < page_map.valid_end
    ):
        return CSA_INVALID_PHYSICAL_ROW
    if for_write and logical_candidate < page_map.valid_begin:
        return CSA_INVALID_PHYSICAL_ROW
    relative_page = logical_candidate // CACHE_BLOCK_SIZE - page_map.logical_page_base
    if relative_page < 0 or relative_page >= len(page_map.page_ids):
        return CSA_INVALID_PHYSICAL_ROW
    # Epoch validity intentionally precedes head rotation and modulo.
    if any(epoch != page_map.request_epoch for epoch in page_map.page_epochs):
        return CSA_INVALID_PHYSICAL_ROW
    page_index = (page_map.head + relative_page) % len(page_map.page_ids)
    return (
        page_map.page_ids[page_index] * CACHE_BLOCK_SIZE
        + logical_candidate % CACHE_BLOCK_SIZE
    )


@dataclass(frozen=True)
class CsaPairedPageMap:
    """Synchronized main/index CSA maps for one request.

    Physical page ids and heads may differ.  Candidate range, request epoch,
    page-span coverage, and page liveness must agree before a row is exposed.
    """

    main: CsaPageMap
    index: CsaPageMap
    active: bool = True

    def validate(self) -> "CsaPairedPageMap":
        self.main.validate()
        self.index.validate()
        if self.main.request_epoch != self.index.request_epoch:
            raise ValueError("CSA main/index request epochs must match")
        if (
            self.main.valid_begin != self.index.valid_begin
            or self.main.valid_end != self.index.valid_end
        ):
            raise ValueError("CSA main/index candidate valid ranges must match")
        if not self.active:
            return self
        if self.main.required_pages and (
            not self.main.page_ids or not self.index.page_ids
        ):
            raise ValueError("an active CSA paired map requires both page spans")
        if any(epoch != self.main.request_epoch for epoch in self.main.page_epochs):
            raise ValueError("CSA main page epoch is stale")
        if any(epoch != self.index.request_epoch for epoch in self.index.page_epochs):
            raise ValueError("CSA index page epoch is stale")
        return self


def validate_csa_paired_page_maps(
    main: CsaPageMap,
    index: CsaPageMap,
    *,
    active: bool = True,
) -> CsaPairedPageMap:
    """Return a validated paired CSA descriptor instead of intersecting maps."""
    return CsaPairedPageMap(main=main, index=index, active=active).validate()


def csa_paired_physical_rows(
    page_maps: CsaPairedPageMap,
    logical_candidate: int,
    *,
    for_write: bool = False,
) -> tuple[int, int]:
    """Resolve both CSA slots atomically from one logical candidate.

    If either map cannot produce its row, neither row is exposed.  This keeps
    an event from becoming main-only or index-only in the host reference.
    """
    if not page_maps.active:
        return CSA_INVALID_PHYSICAL_ROW, CSA_INVALID_PHYSICAL_ROW
    page_maps.validate()
    main_row = csa_physical_row(
        page_maps.main, logical_candidate, for_write=for_write
    )
    index_row = csa_physical_row(
        page_maps.index, logical_candidate, for_write=for_write
    )
    if main_row < 0 or index_row < 0:
        return CSA_INVALID_PHYSICAL_ROW, CSA_INVALID_PHYSICAL_ROW
    return main_row, index_row


def csa_paired_write_rows(
    page_maps: CsaPairedPageMap,
    logical_candidate: int,
) -> tuple[int, int]:
    """Resolve one delayed main/index event write only when both slots exist."""
    return csa_paired_physical_rows(page_maps, logical_candidate, for_write=True)


@dataclass(frozen=True)
class CsaStateDescriptor:
    """One of the independent eight-row ratio-4 compressor state rings."""

    page_ids: tuple[int, ...]
    valid_begin: int
    valid_end: int
    page_epochs: tuple[int, ...]
    request_epoch: int
    active: bool = True

    def validate(self) -> "CsaStateDescriptor":
        if not self.active:
            return self
        if len(self.page_ids) != CSA_STATE_PAGES_PER_REQUEST:
            raise ValueError("an active CSA state descriptor requires four pages")
        if len(self.page_epochs) != len(self.page_ids):
            raise ValueError("CSA state page ids and epochs must have equal length")
        if any(page < 0 for page in self.page_ids):
            raise ValueError("active CSA state page ids must be non-negative")
        if any(epoch < 0 for epoch in self.page_epochs) or self.request_epoch < 0:
            raise ValueError("CSA state and request epochs must be non-negative")
        if any(epoch != self.request_epoch for epoch in self.page_epochs):
            raise ValueError("a CSA state page epoch does not match the request epoch")
        if self.valid_begin < 0 or self.valid_end < self.valid_begin:
            raise ValueError(
                f"invalid CSA state range [{self.valid_begin}, {self.valid_end})"
            )
        if self.valid_end > M.max_position_embeddings:
            raise ValueError("CSA state range exceeds the 1M context ceiling")
        if self.valid_end - self.valid_begin > CSA_STATE_ROWS_PER_REQUEST:
            raise ValueError("CSA state range must fit the eight-row state ring")
        canonical = (
            csa_next_state_valid_range(self.valid_end - 1, active=True)
            if self.valid_end
            else (0, 0)
        )
        if (self.valid_begin, self.valid_end) != canonical:
            raise ValueError(
                "CSA state range must be the canonical previous/current ratio-4 window"
            )
        return self


def csa_state_write_row(descriptor: CsaStateDescriptor, position: int) -> int:
    """Map one current token to the main state ring after validity checks."""
    if not descriptor.active:
        return CSA_INVALID_PHYSICAL_ROW
    descriptor.validate()
    if position < 0 or position >= M.max_position_embeddings:
        return CSA_INVALID_PHYSICAL_ROW
    return (
        descriptor.page_ids[
            (position % CSA_STATE_ROWS_PER_REQUEST) // CSA_STATE_BLOCK_SIZE
        ]
        * CSA_STATE_BLOCK_SIZE
        + position % CSA_STATE_BLOCK_SIZE
    )


def csa_inner_state_write_row(descriptor: CsaStateDescriptor, position: int) -> int:
    """Map one current token to the independent inner state ring."""
    if CSA_INNER_STATE_ROWS_PER_REQUEST != CSA_STATE_ROWS_PER_REQUEST:
        raise AssertionError("Phase D state rings must use the same eight-row geometry")
    if CSA_INNER_STATE_BLOCK_SIZE != CSA_STATE_BLOCK_SIZE:
        raise AssertionError("Phase D state rings must use the same page geometry")
    if CSA_INNER_STATE_PAGES_PER_REQUEST != CSA_STATE_PAGES_PER_REQUEST:
        raise AssertionError("Phase D state rings must use the same page count")
    return csa_state_write_row(descriptor, position)


def csa_next_state_valid_range(
    last_active_position: Optional[int], *, active: bool = True
) -> tuple[int, int]:
    """Return the post-step previous-block plus current-block state range.

    At a completed boundary the just-finished current block becomes the next
    step's previous block, so the older four rows are released before their
    ring positions are reused.
    """
    if not active or last_active_position is None:
        return 0, 0
    if last_active_position < 0 or last_active_position >= M.max_position_embeddings:
        raise ValueError(
            f"last_active_position {last_active_position} is outside the 1M ceiling"
        )
    step_end = last_active_position + 1
    current_block_begin = (last_active_position // CSA_COMPRESS_RATIO) * CSA_COMPRESS_RATIO
    if step_end % CSA_COMPRESS_RATIO == 0:
        return current_block_begin, step_end
    return max(0, current_block_begin - CSA_COMPRESS_RATIO), step_end


def csa_next_inner_state_valid_range(
    last_active_position: Optional[int], *, active: bool = True
) -> tuple[int, int]:
    """Return the inner ring's next range; it follows the same ratio-4 event."""
    return csa_next_state_valid_range(last_active_position, active=active)


@dataclass(frozen=True)
class CsaRatio4Event:
    """The one main/index event emitted at a ratio-4 boundary."""

    logical_candidate: int
    event_position: int
    compression_start: int


def csa_boundary_event(position: int, *, active: bool = True) -> Optional[int]:
    """Return the CSA candidate emitted by ``position``, if it is a boundary."""
    if not active:
        return None
    if position < 0 or position >= M.max_position_embeddings:
        raise ValueError(f"position {position} is outside the 1M context ceiling")
    if (position + 1) % CSA_COMPRESS_RATIO != 0:
        return None
    return position // CSA_COMPRESS_RATIO


def csa_ratio4_event(position: int, *, active: bool = True) -> Optional[CsaRatio4Event]:
    """Describe a ratio-4 boundary using its shared main/index logical row."""
    logical_candidate = csa_boundary_event(position, active=active)
    if logical_candidate is None:
        return None
    return CsaRatio4Event(
        logical_candidate=logical_candidate,
        event_position=position,
        compression_start=logical_candidate * CSA_COMPRESS_RATIO,
    )


def csa_event_position(logical_candidate: int) -> int:
    """Return the boundary token position for a valid absolute candidate."""
    if (
        logical_candidate < CSA_LOGICAL_INDEX_MIN
        or logical_candidate > CSA_LOGICAL_INDEX_MAX
    ):
        raise ValueError(f"invalid CSA logical candidate {logical_candidate}")
    return logical_candidate * CSA_COMPRESS_RATIO + CSA_COMPRESS_RATIO - 1


def csa_event_compression_start(logical_candidate: int) -> int:
    """Return the event-local RoPE position shared by main and inner paths."""
    return csa_event_position(logical_candidate) - (CSA_COMPRESS_RATIO - 1)


def visible_csa_candidate_range_with_events(
    q: QueryGeometry,
    current_event_rows: Sequence[int] = (),
) -> tuple[int, int]:
    """Return the causal CSA range from pre-step rows plus contiguous events."""
    q.validate()
    if not q.request.active:
        return 0, 0
    causal_end = visible_tokens(q) // CSA_COMPRESS_RATIO
    begin = q.request.csa_valid_begin
    allocated_end = min(
        MAX_CSA_CANDIDATES,
        begin + q.request.csa_allocated_candidates,
    )
    end = min(causal_end, q.request.csa_valid_end, allocated_end)
    for row in sorted(set(int(value) for value in current_event_rows)):
        if row < begin or row >= causal_end or row >= allocated_end:
            continue
        if row > end:
            raise ValueError("current CSA events must append without a logical gap")
        end = max(end, row + 1)
    return begin, max(begin, end)


def visible_csa_candidates_with_events(
    q: QueryGeometry,
    current_event_rows: Sequence[int] = (),
) -> int:
    """Return the exact per-query candidate count after causal event overlay."""
    begin, end = visible_csa_candidate_range_with_events(q, current_event_rows)
    return end - begin


@dataclass(frozen=True)
class CsaLeafDescriptor:
    """One exact packed indexer leaf; no leaf has zero valid candidates."""

    query_id: int
    logical_candidate_begin: int
    valid_candidates: int

    def validate(self) -> "CsaLeafDescriptor":
        if self.query_id < 0:
            raise ValueError("CSA leaf query id must be non-negative")
        if (
            self.logical_candidate_begin < CSA_LOGICAL_INDEX_MIN
            or self.logical_candidate_begin > CSA_LOGICAL_INDEX_MAX
        ):
            raise ValueError("CSA leaf logical candidate begin is outside the 1M range")
        if not 1 <= self.valid_candidates <= CSA_CANDIDATES_PER_LEAF:
            raise ValueError("CSA leaf valid candidates must lie in [1, 2048]")
        if (
            self.logical_candidate_begin + self.valid_candidates
            > MAX_CSA_CANDIDATES
        ):
            raise ValueError("CSA leaf exceeds the candidate ceiling")
        return self


@dataclass(frozen=True)
class CsaPackedWork:
    """Exact query-major CSA leaf descriptors and causal visible ranges."""

    query_leaf_offsets: tuple[int, ...]
    leaf_query_ids: tuple[int, ...]
    leaf_candidate_begins: tuple[int, ...]
    leaf_valid_candidates: tuple[int, ...]
    query_visible_candidate_begins: tuple[int, ...]
    query_visible_candidate_ends: tuple[int, ...]

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_query_ids)

    @property
    def leaf_descriptors(self) -> tuple[CsaLeafDescriptor, ...]:
        return tuple(
            CsaLeafDescriptor(query_id, begin, valid).validate()
            for query_id, begin, valid in zip(
                self.leaf_query_ids,
                self.leaf_candidate_begins,
                self.leaf_valid_candidates,
            )
        )

    def validate(self) -> "CsaPackedWork":
        query_count = len(self.query_visible_candidate_begins)
        if len(self.query_leaf_offsets) != query_count + 1:
            raise ValueError("CSA query leaf offsets must have Q + 1 entries")
        if len(self.query_visible_candidate_ends) != query_count:
            raise ValueError("CSA query visible range arrays must have the same length")
        if (
            len(self.leaf_query_ids) != len(self.leaf_candidate_begins)
            or len(self.leaf_query_ids) != len(self.leaf_valid_candidates)
        ):
            raise ValueError("CSA packed leaf arrays must have the same length")
        if not self.query_leaf_offsets or self.query_leaf_offsets[0] != 0:
            raise ValueError("CSA packed leaf offsets must begin at zero")
        if self.query_leaf_offsets[-1] != self.n_leaves:
            raise ValueError("CSA packed leaf offsets must end at N_leaf")
        if any(
            right < left
            for left, right in zip(
                self.query_leaf_offsets, self.query_leaf_offsets[1:]
            )
        ):
            raise ValueError("CSA packed leaf offsets must be monotonic")
        for begin, end in zip(
            self.query_visible_candidate_begins,
            self.query_visible_candidate_ends,
        ):
            if begin < 0 or end < begin or end > MAX_CSA_CANDIDATES:
                raise ValueError("CSA visible candidate range is invalid")
        for descriptor in self.leaf_descriptors:
            descriptor.validate()
        for query_id, (leaf_begin, leaf_end) in enumerate(
            zip(self.query_leaf_offsets, self.query_leaf_offsets[1:])
        ):
            visible_begin = self.query_visible_candidate_begins[query_id]
            visible_end = self.query_visible_candidate_ends[query_id]
            expected = visible_begin
            for leaf_index in range(leaf_begin, leaf_end):
                if self.leaf_query_ids[leaf_index] != query_id:
                    raise ValueError("CSA packed leaves must stay query-major")
                if self.leaf_candidate_begins[leaf_index] != expected:
                    raise ValueError("CSA packed leaves must be contiguous")
                expected += self.leaf_valid_candidates[leaf_index]
            if expected != visible_end:
                raise ValueError("CSA packed leaves must exactly cover the visible range")
        return self


def build_csa_packed_work(
    queries: Sequence[QueryGeometry],
    query_request_ids: Sequence[int],
    current_events_by_request: Mapping[int, Sequence[tuple[int, int]]],
) -> CsaPackedWork:
    """Build exact packed CSA leaf work from each query's causal event view."""
    if len(queries) != len(query_request_ids):
        raise ValueError("queries and query_request_ids must have the same length")
    offsets = [0]
    leaf_query_ids: list[int] = []
    leaf_begins: list[int] = []
    leaf_valid_candidates: list[int] = []
    visible_begins: list[int] = []
    visible_ends: list[int] = []
    for query_index, (query, request_id) in enumerate(
        zip(queries, query_request_ids)
    ):
        visible_events: list[int] = []
        for event_position, row in current_events_by_request.get(request_id, ()):
            if csa_event_position(int(row)) != int(event_position):
                raise ValueError("CSA event position and logical candidate disagree")
            if event_position <= query.position:
                visible_events.append(row)
        begin, end = visible_csa_candidate_range_with_events(query, visible_events)
        visible_begins.append(begin)
        visible_ends.append(end)
        candidate = begin
        while candidate < end:
            valid_candidates = min(CSA_CANDIDATES_PER_LEAF, end - candidate)
            leaf_query_ids.append(query_index)
            leaf_begins.append(candidate)
            leaf_valid_candidates.append(valid_candidates)
            candidate += valid_candidates
        offsets.append(len(leaf_query_ids))
    return CsaPackedWork(
        query_leaf_offsets=tuple(offsets),
        leaf_query_ids=tuple(leaf_query_ids),
        leaf_candidate_begins=tuple(leaf_begins),
        leaf_valid_candidates=tuple(leaf_valid_candidates),
        query_visible_candidate_begins=tuple(visible_begins),
        query_visible_candidate_ends=tuple(visible_ends),
    ).validate()


build_csa_packed_leaf_work = build_csa_packed_work


@dataclass(frozen=True)
class CsaForestMerge:
    """One binary merge task with real child workspace/task slots only."""

    query_id: int
    level: int
    left_slot: int
    right_slot: int
    output_slot: int


@dataclass(frozen=True)
class CsaTaskWorkspaceCounts:
    """Exact active task and pair-arena counts for one packed CSA batch."""

    n_leaves: int
    n_merges: int
    n_nodes: int
    n_pair_groups: int
    workspace_pair_lists: int
    workspace_bytes: int


@dataclass(frozen=True)
class CsaForestDescriptor:
    """Exact odd-carry forest descriptors for all active queries.

    Node/workspace slots are query-major.  A credit predecessor is either a
    real level-one merge task slot or ``CSA_TOPK_INVALID_TASK_SLOT``; the
    latter indexes the pre-initialized dummy TaskId and never aliases work.
    """

    query_node_offsets: tuple[int, ...]
    query_merge_offsets: tuple[int, ...]
    query_pair_group_offsets: tuple[int, ...]
    leaf_output_slots: tuple[int, ...]
    leaf_credit_predecessors: tuple[int, ...]
    merge_query_ids: tuple[int, ...]
    merge_levels: tuple[int, ...]
    merge_left_slots: tuple[int, ...]
    merge_right_slots: tuple[int, ...]
    merge_output_slots: tuple[int, ...]
    root_slots: tuple[int, ...]
    credit_predecessors: tuple[int, ...]

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_output_slots)

    @property
    def n_merges(self) -> int:
        return len(self.merge_output_slots)

    @property
    def n_nodes(self) -> int:
        return self.n_leaves + self.n_merges

    @property
    def n_pair_groups(self) -> int:
        return len(self.credit_predecessors)

    @property
    def workspace_bytes(self) -> int:
        return self.n_nodes * CSA_PAIR_BYTES

    @property
    def node_credit_predecessors(self) -> tuple[int, ...]:
        """Return a task-slot-aligned credit descriptor for device lowering."""
        credits = [CSA_TOPK_INVALID_TASK_SLOT] * self.n_nodes
        for slot, predecessor in zip(
            self.leaf_output_slots, self.leaf_credit_predecessors
        ):
            credits[slot] = predecessor
        return tuple(credits)

    @property
    def _pair_merge_indices(self) -> tuple[int, ...]:
        """Indices of level-zero merges whose two inputs are real leaves."""
        leaf_slots = set(self.leaf_output_slots)
        return tuple(
            index
            for index, (level, left, right) in enumerate(
                zip(
                    self.merge_levels,
                    self.merge_left_slots,
                    self.merge_right_slots,
                )
            )
            if level == 0 and left in leaf_slots and right in leaf_slots
        )

    @property
    def pair_left_leaf_ids(self) -> tuple[int, ...]:
        """Packed leaf ids for the left member of each branch-free pair group."""
        leaf_id_by_slot = {
            slot: leaf_id for leaf_id, slot in enumerate(self.leaf_output_slots)
        }
        return tuple(
            leaf_id_by_slot[self.merge_left_slots[index]]
            for index in self._pair_merge_indices
        )

    @property
    def pair_right_leaf_ids(self) -> tuple[int, ...]:
        """Packed leaf ids for the right member of each branch-free pair group."""
        leaf_id_by_slot = {
            slot: leaf_id for leaf_id, slot in enumerate(self.leaf_output_slots)
        }
        return tuple(
            leaf_id_by_slot[self.merge_right_slots[index]]
            for index in self._pair_merge_indices
        )

    @property
    def pair_left_slots(self) -> tuple[int, ...]:
        """Task/workspace slots for left leaves in each pair group."""
        return tuple(self.merge_left_slots[index] for index in self._pair_merge_indices)

    @property
    def pair_right_slots(self) -> tuple[int, ...]:
        """Task/workspace slots for right leaves in each pair group."""
        return tuple(self.merge_right_slots[index] for index in self._pair_merge_indices)

    @property
    def pair_output_slots(self) -> tuple[int, ...]:
        """Level-zero merge slots, one per pair group."""
        return tuple(self.merge_output_slots[index] for index in self._pair_merge_indices)

    @property
    def pair_credit_slots(self) -> tuple[int, ...]:
        """Credit predecessors for pair groups, including the dummy-task slot."""
        credit_by_leaf_slot = dict(
            zip(self.leaf_output_slots, self.leaf_credit_predecessors)
        )
        return tuple(
            credit_by_leaf_slot[self.merge_left_slots[index]]
            for index in self._pair_merge_indices
        )

    @property
    def singleton_leaf_ids(self) -> tuple[int, ...]:
        """Packed leaf ids carried past level zero without a fake partner."""
        paired_slots = set(self.pair_left_slots) | set(self.pair_right_slots)
        return tuple(
            leaf_id
            for leaf_id, slot in enumerate(self.leaf_output_slots)
            if slot not in paired_slots
        )

    @property
    def singleton_slots(self) -> tuple[int, ...]:
        """Task/workspace slots for odd carried leaves."""
        paired_slots = set(self.pair_left_slots) | set(self.pair_right_slots)
        return tuple(slot for slot in self.leaf_output_slots if slot not in paired_slots)

    @property
    def singleton_credit_slots(self) -> tuple[int, ...]:
        """Credit predecessors for odd carried leaves."""
        credit_by_leaf_slot = dict(
            zip(self.leaf_output_slots, self.leaf_credit_predecessors)
        )
        return tuple(credit_by_leaf_slot[slot] for slot in self.singleton_slots)

    @property
    def upper_left_slots(self) -> tuple[int, ...]:
        """Child slots for merges above the branch-free pair-group level."""
        return tuple(
            left
            for level, left in zip(self.merge_levels, self.merge_left_slots)
            if level > 0
        )

    @property
    def upper_right_slots(self) -> tuple[int, ...]:
        """Right child slots for merges above the pair-group level."""
        return tuple(
            right
            for level, right in zip(self.merge_levels, self.merge_right_slots)
            if level > 0
        )

    @property
    def upper_output_slots(self) -> tuple[int, ...]:
        """Output task/workspace slots for upper-level merges."""
        return tuple(
            output
            for level, output in zip(self.merge_levels, self.merge_output_slots)
            if level > 0
        )

    @property
    def root_dependency_slots(self) -> tuple[int, ...]:
        """TaskId slots used by root materialization, including zero-leaf roots."""
        return tuple(
            root if root != CSA_INVALID_FOREST_SLOT else CSA_TOPK_INVALID_TASK_SLOT
            for root in self.root_slots
        )

    @property
    def merges(self) -> tuple[CsaForestMerge, ...]:
        return tuple(
            CsaForestMerge(query_id, level, left, right, output)
            for query_id, level, left, right, output in zip(
                self.merge_query_ids,
                self.merge_levels,
                self.merge_left_slots,
                self.merge_right_slots,
                self.merge_output_slots,
            )
        )

    def task_workspace_counts(self) -> CsaTaskWorkspaceCounts:
        return CsaTaskWorkspaceCounts(
            n_leaves=self.n_leaves,
            n_merges=self.n_merges,
            n_nodes=self.n_nodes,
            n_pair_groups=self.n_pair_groups,
            workspace_pair_lists=self.n_nodes,
            workspace_bytes=self.workspace_bytes,
        )

    def validate(self) -> "CsaForestDescriptor":
        query_count = len(self.root_slots)
        if len(self.query_node_offsets) != query_count + 1:
            raise ValueError("CSA forest node offsets must have Q + 1 entries")
        if len(self.query_merge_offsets) != query_count + 1:
            raise ValueError("CSA forest merge offsets must have Q + 1 entries")
        if len(self.query_pair_group_offsets) != query_count + 1:
            raise ValueError("CSA forest credit offsets must have Q + 1 entries")
        if (
            len(self.merge_query_ids) != self.n_merges
            or len(self.merge_levels) != self.n_merges
            or len(self.merge_left_slots) != self.n_merges
            or len(self.merge_right_slots) != self.n_merges
        ):
            raise ValueError("CSA forest merge arrays must have the same length")
        if len(self.leaf_credit_predecessors) != self.n_leaves:
            raise ValueError("CSA forest leaf credit array must match leaf slots")
        if self.query_node_offsets[-1] != self.n_nodes:
            raise ValueError("CSA forest node offsets must end at N_node")
        if self.query_merge_offsets[-1] != self.n_merges:
            raise ValueError("CSA forest merge offsets must end at N_merge")
        if self.query_pair_group_offsets[-1] != self.n_pair_groups:
            raise ValueError("CSA forest credit offsets must end at N_pair_groups")
        for offsets, label, limit in (
            (self.query_node_offsets, "node", self.n_nodes),
            (self.query_merge_offsets, "merge", self.n_merges),
            (self.query_pair_group_offsets, "credit", self.n_pair_groups),
        ):
            if not offsets or offsets[0] != 0 or offsets[-1] != limit:
                raise ValueError(f"CSA forest {label} offsets are malformed")
            if any(right < left for left, right in zip(offsets, offsets[1:])):
                raise ValueError(f"CSA forest {label} offsets must be monotonic")
        if self.n_nodes > CSA_MAX_TOPK_TASKS:
            raise ValueError("CSA forest exceeds the bounded local task capacity")
        all_slots = self.leaf_output_slots + self.merge_output_slots
        if len(set(all_slots)) != self.n_nodes or set(all_slots) != set(range(self.n_nodes)):
            raise ValueError("CSA forest task/workspace slots must be a dense active arena")
        if len(self.pair_output_slots) != self.n_pair_groups:
            raise ValueError("CSA forest pair-group count must match level-zero merges")
        if self.pair_credit_slots != self.credit_predecessors:
            raise ValueError("CSA forest pair credits must preserve pair-group order")
        for query_id, (node_begin, node_end, merge_begin, merge_end, root) in enumerate(
            zip(
                self.query_node_offsets,
                self.query_node_offsets[1:],
                self.query_merge_offsets,
                self.query_merge_offsets[1:],
                self.root_slots,
            )
        ):
            if root == CSA_INVALID_FOREST_SLOT:
                if node_begin != node_end:
                    raise ValueError("only a zero-leaf query may have an invalid CSA root")
            elif not node_begin <= root < node_end:
                raise ValueError("CSA forest root slot is outside its query arena")
            for merge_index in range(merge_begin, merge_end):
                if self.merge_query_ids[merge_index] != query_id:
                    raise ValueError("CSA forest merge offsets must be query-major")
                left = self.merge_left_slots[merge_index]
                right = self.merge_right_slots[merge_index]
                output = self.merge_output_slots[merge_index]
                if left == right or not (
                    node_begin <= left < node_end
                    and node_begin <= right < node_end
                    and node_begin <= output < node_end
                ):
                    raise ValueError(
                        "CSA merge must reference two real children in its query arena"
                    )
        pair_output_slots = set(self.pair_output_slots)
        for slot in self.credit_predecessors:
            if slot == CSA_TOPK_INVALID_TASK_SLOT:
                continue
            if slot not in pair_output_slots:
                raise ValueError("CSA credit predecessor is outside the active task slots")
        for slot in self.leaf_credit_predecessors:
            if slot == CSA_TOPK_INVALID_TASK_SLOT:
                continue
            if slot not in pair_output_slots:
                raise ValueError("CSA leaf credit predecessor is outside active tasks")
        return self


def _validate_csa_leaf_counts(leaf_counts: Sequence[int]) -> tuple[int, ...]:
    """Validate exact per-query leaf counts for the bounded local forest."""
    if len(leaf_counts) > CSA_MAX_QUERIES:
        raise ValueError("CSA forest exceeds the compile-time local query bound")
    result = tuple(int(count) for count in leaf_counts)
    if any(count < 0 or count > MAX_CSA_LEAVES for count in result):
        raise ValueError("CSA per-query leaf count is outside [0, 128]")
    return result


def csa_task_workspace_counts(leaf_counts: Sequence[int]) -> CsaTaskWorkspaceCounts:
    """Return exact active leaf/merge/node/workspace counts without padding."""
    counts = _validate_csa_leaf_counts(leaf_counts)
    n_leaves = sum(counts)
    n_merges = sum(max(count - 1, 0) for count in counts)
    n_nodes = n_leaves + n_merges
    return CsaTaskWorkspaceCounts(
        n_leaves=n_leaves,
        n_merges=n_merges,
        n_nodes=n_nodes,
        n_pair_groups=sum(count // CSA_MERGE_ARITY for count in counts),
        workspace_pair_lists=n_nodes,
        workspace_bytes=n_nodes * CSA_PAIR_BYTES,
    )


def build_csa_binary_forest(leaf_counts: Sequence[int]) -> CsaForestDescriptor:
    """Build an exact query-major forest, carrying odd nodes without fake work."""
    counts = _validate_csa_leaf_counts(leaf_counts)
    node_offsets = [0]
    merge_offsets = [0]
    pair_group_offsets = [0]
    leaf_slots: list[int] = []
    leaf_credit_predecessors: list[int] = []
    merge_query_ids: list[int] = []
    merge_levels: list[int] = []
    merge_left_slots: list[int] = []
    merge_right_slots: list[int] = []
    merge_output_slots: list[int] = []
    root_slots: list[int] = []
    credit_predecessors: list[int] = []
    node_base = 0
    for query_id, leaf_count in enumerate(counts):
        current: list[int] = []
        query_leaf_slots = [CSA_INVALID_FOREST_SLOT] * leaf_count
        query_leaf_credits = [CSA_TOPK_INVALID_TASK_SLOT] * leaf_count
        next_output_slot = node_base
        level_one_outputs: list[int] = []
        for pair_group in range(leaf_count // CSA_MERGE_ARITY):
            left_leaf_id = pair_group * CSA_MERGE_ARITY
            right_leaf_id = left_leaf_id + 1
            left_slot = next_output_slot
            next_output_slot += 1
            right_slot = next_output_slot
            next_output_slot += 1
            output_slot = next_output_slot
            next_output_slot += 1
            credit_predecessor = (
                CSA_TOPK_INVALID_TASK_SLOT
                if pair_group < CSA_TOPK_READY_FRONTIER_W
                else level_one_outputs[pair_group - CSA_TOPK_READY_FRONTIER_W]
            )
            query_leaf_slots[left_leaf_id] = left_slot
            query_leaf_slots[right_leaf_id] = right_slot
            query_leaf_credits[left_leaf_id] = credit_predecessor
            query_leaf_credits[right_leaf_id] = credit_predecessor
            merge_query_ids.append(query_id)
            merge_levels.append(0)
            merge_left_slots.append(left_slot)
            merge_right_slots.append(right_slot)
            merge_output_slots.append(output_slot)
            credit_predecessors.append(credit_predecessor)
            level_one_outputs.append(output_slot)
            current.append(output_slot)
        if leaf_count % CSA_MERGE_ARITY:
            singleton_leaf_id = leaf_count - 1
            singleton_slot = next_output_slot
            next_output_slot += 1
            singleton_credit = (
                CSA_TOPK_INVALID_TASK_SLOT
                if len(level_one_outputs) < CSA_TOPK_READY_FRONTIER_W
                else level_one_outputs[-CSA_TOPK_READY_FRONTIER_W]
            )
            query_leaf_slots[singleton_leaf_id] = singleton_slot
            query_leaf_credits[singleton_leaf_id] = singleton_credit
            current.append(singleton_slot)

        level = 1
        while len(current) > 1:
            next_level: list[int] = []
            for pair_group in range(len(current) // CSA_MERGE_ARITY):
                child = pair_group * CSA_MERGE_ARITY
                left_slot = current[child]
                right_slot = current[child + 1]
                output_slot = next_output_slot
                next_output_slot += 1
                merge_query_ids.append(query_id)
                merge_levels.append(level)
                merge_left_slots.append(left_slot)
                merge_right_slots.append(right_slot)
                merge_output_slots.append(output_slot)
                next_level.append(output_slot)
            if len(current) % CSA_MERGE_ARITY:
                next_level.append(current[-1])
            current = next_level
            level += 1
        root_slots.append(current[0] if current else CSA_INVALID_FOREST_SLOT)
        leaf_slots.extend(query_leaf_slots)
        leaf_credit_predecessors.extend(query_leaf_credits)
        node_base = next_output_slot
        node_offsets.append(node_base)
        merge_offsets.append(len(merge_output_slots))
        pair_group_offsets.append(len(credit_predecessors))
    return CsaForestDescriptor(
        query_node_offsets=tuple(node_offsets),
        query_merge_offsets=tuple(merge_offsets),
        query_pair_group_offsets=tuple(pair_group_offsets),
        leaf_output_slots=tuple(leaf_slots),
        leaf_credit_predecessors=tuple(leaf_credit_predecessors),
        merge_query_ids=tuple(merge_query_ids),
        merge_levels=tuple(merge_levels),
        merge_left_slots=tuple(merge_left_slots),
        merge_right_slots=tuple(merge_right_slots),
        merge_output_slots=tuple(merge_output_slots),
        root_slots=tuple(root_slots),
        credit_predecessors=tuple(credit_predecessors),
    ).validate()


def build_csa_forest(packed_work: CsaPackedWork) -> CsaForestDescriptor:
    """Build the forest matching one exact packed CSA leaf descriptor batch."""
    packed_work.validate()
    leaf_counts = tuple(
        right - left
        for left, right in zip(
            packed_work.query_leaf_offsets,
            packed_work.query_leaf_offsets[1:],
        )
    )
    forest = build_csa_binary_forest(leaf_counts)
    if forest.n_leaves != packed_work.n_leaves:
        raise AssertionError("CSA forest leaves must exactly match packed leaf work")
    return forest


# ---------------------------------------------------------------------------
# Phase B SWA raw-ring reference. Unlike the generic paged cache groups, SWA
# owns exactly one physical page whose rows are addressed by the absolute
# token-position modulo 128. Validity and epoch checks always precede modulo.
# ---------------------------------------------------------------------------


def _swa_raw_range_is_well_formed(raw_valid_begin: int, raw_valid_end: int) -> bool:
    """Whether a raw/SWA live range fits one persistent ring page."""
    return (
        0 <= raw_valid_begin <= raw_valid_end <= M.max_position_embeddings
        and raw_valid_end - raw_valid_begin <= SWA_PERSISTENT_ROWS_PER_REQUEST
    )


def validate_swa_raw_descriptor(
    raw_page_ids: Sequence[int],
    raw_valid_begin: int,
    raw_valid_end: int,
    page_epochs: Sequence[int],
    request_epoch: int,
    *,
    active: bool = True,
    raw_page_count: int = SWA_PERSISTENT_PAGES_PER_REQUEST,
    raw_head: int = 0,
) -> None:
    """Validate the structural four-page raw/SWA descriptor contract.

    Epoch equality is intentionally checked by the row/write helpers rather
    than here: a stale descriptor is a valid input to rejection tests and must
    lower to an invalid source instead of being revived by ring arithmetic.
    """
    if raw_page_count < 0:
        raise ValueError(f"raw_page_count must be non-negative, got {raw_page_count}")
    if raw_head != 0:
        raise ValueError(f"raw/SWA head is reserved and must be 0, got {raw_head}")
    if not _swa_raw_range_is_well_formed(raw_valid_begin, raw_valid_end):
        raise ValueError(
            "raw/SWA valid range must lie in the context ceiling and contain "
            f"at most {SWA_PERSISTENT_ROWS_PER_REQUEST} rows, got "
            f"[{raw_valid_begin}, {raw_valid_end})"
        )
    if len(raw_page_ids) != raw_page_count or len(page_epochs) != raw_page_count:
        raise ValueError("raw/SWA page ids, epochs, and page count must agree")
    if any(epoch < 0 for epoch in page_epochs) or request_epoch < 0:
        raise ValueError(
            "raw/SWA page and request epochs must be non-negative, got "
            f"{tuple(page_epochs)} and {request_epoch}"
        )
    if active:
        if raw_page_count != SWA_PERSISTENT_PAGES_PER_REQUEST:
            raise ValueError(
                "an active raw/SWA request must own exactly four physical pages, "
                f"got {raw_page_count}"
            )
        if any(page < 0 for page in raw_page_ids):
            raise ValueError("an active raw/SWA request requires non-negative page ids")
        if any(
            page > SWA_SOURCE_INT32_MAX // CACHE_BLOCK_SIZE
            for page in raw_page_ids
        ):
            raise ValueError(
                "raw/SWA page id cannot address an INT32 persistent source row, "
                f"got {tuple(raw_page_ids)}"
            )
    elif raw_page_count > SWA_PERSISTENT_PAGES_PER_REQUEST:
        raise ValueError(
            "an inactive raw/SWA descriptor may retain at most four pages, got "
            f"{raw_page_count}"
        )


@dataclass(frozen=True)
class SwaRawDescriptor:
    """One request's pre-step persistent SWA ring descriptor."""

    raw_page_ids: tuple[int, ...]
    raw_valid_begin: int
    raw_valid_end: int
    page_epochs: tuple[int, ...]
    request_epoch: int
    active: bool = True
    raw_page_count: int = SWA_PERSISTENT_PAGES_PER_REQUEST
    raw_head: int = 0

    def validate(self) -> "SwaRawDescriptor":
        """Validate four-page ring structure while preserving stale epochs."""
        validate_swa_raw_descriptor(
            self.raw_page_ids,
            self.raw_valid_begin,
            self.raw_valid_end,
            self.page_epochs,
            self.request_epoch,
            active=self.active,
            raw_page_count=self.raw_page_count,
            raw_head=self.raw_head,
        )
        return self


def swa_window_end(committed_tokens: int, position: int, *, active: bool = True) -> int:
    """Return the causal visible end for one SWA query, exclusive."""
    if committed_tokens < 0 or committed_tokens > M.max_position_embeddings:
        raise ValueError(
            f"committed_tokens {committed_tokens} out of [0, {M.max_position_embeddings}]"
        )
    if position < 0 or position >= M.max_position_embeddings:
        raise ValueError(f"position {position} out of [0, {M.max_position_embeddings})")
    if not active:
        return 0
    return min(committed_tokens, position + 1, M.max_position_embeddings)


def swa_window_begin(visible_end: int) -> int:
    """Return the inclusive oldest position in a fixed-width SWA window."""
    if visible_end < 0 or visible_end > M.max_position_embeddings:
        raise ValueError(f"visible_end {visible_end} out of [0, {M.max_position_embeddings}]")
    return max(0, visible_end - SWA_WINDOW_ROWS)


def swa_window_range(
    committed_tokens: int,
    position: int,
    *,
    active: bool = True,
) -> tuple[int, int]:
    """Return the causal SWA logical window as ``[begin, end)``."""
    end = swa_window_end(committed_tokens, position, active=active)
    return swa_window_begin(end), end


def swa_ring_row(
    raw_page_ids: Sequence[int],
    logical_position: int,
    raw_valid_begin: int,
    raw_valid_end: int,
    page_epochs: Sequence[int],
    request_epoch: int,
) -> int:
    """Map a live pre-step logical position to a flattened persistent row.

    Invalid, missing, malformed, or stale descriptors return
    ``SWA_SOURCE_INVALID``. The modulo operation occurs only after validity
    and epoch checks have proved that the row is live.
    """
    if len(raw_page_ids) != SWA_PERSISTENT_PAGES_PER_REQUEST or logical_position < 0:
        return SWA_SOURCE_INVALID
    if not _swa_raw_range_is_well_formed(raw_valid_begin, raw_valid_end):
        return SWA_SOURCE_INVALID
    if logical_position < raw_valid_begin or logical_position >= raw_valid_end:
        return SWA_SOURCE_INVALID
    if len(page_epochs) != len(raw_page_ids):
        return SWA_SOURCE_INVALID
    if any(epoch != request_epoch for epoch in page_epochs):
        return SWA_SOURCE_INVALID
    if any(page_id < 0 for page_id in raw_page_ids):
        return SWA_SOURCE_INVALID
    ring_offset = logical_position % SWA_PERSISTENT_ROWS_PER_REQUEST
    relative_page = ring_offset // CACHE_BLOCK_SIZE
    raw_page_id = raw_page_ids[relative_page]
    physical_row = raw_page_id * CACHE_BLOCK_SIZE + ring_offset % CACHE_BLOCK_SIZE
    if physical_row > SWA_SOURCE_INT32_MAX:
        return SWA_SOURCE_INVALID
    return physical_row


def swa_write_row(
    raw_page_ids: Sequence[int],
    logical_position: int,
    active: bool,
    page_epochs: Sequence[int],
    request_epoch: int,
) -> int:
    """Return a delayed-commit row without requiring pre-step range membership."""
    if not active or len(raw_page_ids) != SWA_PERSISTENT_PAGES_PER_REQUEST:
        return SWA_SOURCE_INVALID
    if logical_position < 0 or logical_position >= M.max_position_embeddings:
        return SWA_SOURCE_INVALID
    if len(page_epochs) != len(raw_page_ids):
        return SWA_SOURCE_INVALID
    if any(epoch != request_epoch for epoch in page_epochs):
        return SWA_SOURCE_INVALID
    if any(page_id < 0 for page_id in raw_page_ids):
        return SWA_SOURCE_INVALID
    ring_offset = logical_position % SWA_PERSISTENT_ROWS_PER_REQUEST
    relative_page = ring_offset // CACHE_BLOCK_SIZE
    raw_page_id = raw_page_ids[relative_page]
    physical_row = raw_page_id * CACHE_BLOCK_SIZE + ring_offset % CACHE_BLOCK_SIZE
    if physical_row > SWA_SOURCE_INT32_MAX:
        return SWA_SOURCE_INVALID
    return physical_row


def swa_next_valid_range(
    raw_valid_end: int,
    last_active_position: Optional[int],
    *,
    active: bool = True,
) -> tuple[int, int]:
    """Return the post-commit raw/SWA live range for one request."""
    if raw_valid_end < 0 or raw_valid_end > M.max_position_embeddings:
        raise ValueError(
            f"raw_valid_end {raw_valid_end} out of [0, {M.max_position_embeddings}]"
        )
    if not active:
        return 0, 0
    if last_active_position is None:
        raise ValueError("an active request requires its last active position")
    if last_active_position < 0 or last_active_position >= M.max_position_embeddings:
        raise ValueError(
            "last_active_position must lie in the context ceiling, got "
            f"{last_active_position}"
        )
    step_end = max(raw_valid_end, last_active_position + 1)
    return max(0, step_end - SWA_PERSISTENT_ROWS_PER_REQUEST), step_end


def swa_source_reference(
    logical_position: int,
    *,
    target_position: int,
    raw_page_ids: Sequence[int],
    raw_valid_begin: int,
    raw_valid_end: int,
    page_epochs: Sequence[int],
    request_epoch: int,
    overlay_query_by_position: Optional[Mapping[int, int]] = None,
) -> int:
    """Choose the causal overlay or persistent source for one logical row.

    ``overlay_query_by_position`` must contain only current-step queries from
    the target request. Metadata builds it from that request's contiguous query
    range, so no device-side scan over unrelated queries is required.
    """
    if logical_position < 0 or logical_position > target_position:
        return SWA_SOURCE_INVALID
    if overlay_query_by_position is not None:
        overlay_query = overlay_query_by_position.get(logical_position)
        if overlay_query is not None:
            return encode_swa_overlay_source(overlay_query)
    return swa_ring_row(
        raw_page_ids,
        logical_position,
        raw_valid_begin,
        raw_valid_end,
        page_epochs,
        request_epoch,
    )


def swa_window_sources_reference(
    *,
    committed_tokens: int,
    target_position: int,
    raw_page_ids: Sequence[int],
    raw_valid_begin: int,
    raw_valid_end: int,
    page_epochs: Sequence[int],
    request_epoch: int,
    overlay_query_by_position: Optional[Mapping[int, int]] = None,
    active: bool = True,
) -> tuple[list[int], int]:
    """Build fixed-width oldest-to-newest SWA sources and the valid prefix."""
    sources = [SWA_SOURCE_INVALID] * SWA_WINDOW_ROWS
    if not active:
        return sources, 0
    begin, end = swa_window_range(committed_tokens, target_position, active=True)
    valid_rows = 0
    for logical_position in range(begin, end):
        source = swa_source_reference(
            logical_position,
            target_position=target_position,
            raw_page_ids=raw_page_ids,
            raw_valid_begin=raw_valid_begin,
            raw_valid_end=raw_valid_end,
            page_epochs=page_epochs,
            request_epoch=request_epoch,
            overlay_query_by_position=overlay_query_by_position,
        )
        if source != SWA_SOURCE_INVALID:
            sources[valid_rows] = source
            valid_rows += 1
    return sources, valid_rows
