# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 host-side Engram hash/history contract.

This module owns the part of Engram that runs before the PyPTO BF16 table
lookup: compressed token ids, request lookback precedence, rolling hash and
bounded host-table gathers.  The arithmetic mirrors vLLM's
``vllm.models.deepseek_v41.common.engram`` and the Ascend
``_hash_ids_kernel``.  It deliberately does not pretend to implement the
384,006,168-row host UVM table or a cross-rank collective; those remain
explicit integration boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


DEAD_ID = -1


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % p == 0:
            return value == p
    d, r = value - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(r - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while candidate in seen or not _is_prime(candidate):
        candidate += 1
    return candidate


@dataclass(frozen=True)
class HashLayout:
    """Frozen V4.1 hash parameters for the configured Engram layers."""

    layer_ids: tuple[int, ...]
    multipliers: torch.Tensor  # [layers, max_ngram]
    primes: torch.Tensor  # [layers, (max_ngram - 1) * heads]
    offsets: torch.Tensor  # [layers, (max_ngram - 1) * heads]
    max_ngram_size: int
    n_heads: int
    compressed_vocab_size: int
    pad_token_id: int
    vocab_size: int

    @property
    def lookback_depth(self) -> int:
        return self.max_ngram_size - 1

    @property
    def n_hash_cols(self) -> int:
        return self.lookback_depth * self.n_heads


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Build the V4.1 normalized token map used by the hash state.

    The tokenizer object is deliberately supplied by the host runner.  This
    keeps tokenizer I/O out of PyPTO graphs while preserving the exact
    normalization contract (including one-space and partial UTF-8 tokens).
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def make_layout(
    layer_ids: Sequence[int] = (1, 14),
    *,
    max_ngram_size: int = 4,
    n_heads: int = 8,
    compressed_vocab_size: int = 99092,
    vocab_size: int = 16_000_000,
    pad_token_id: int = 2,
) -> HashLayout:
    """Construct the same deterministic layout as upstream vLLM."""
    layers = tuple(int(x) for x in layer_ids)
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab_size) // 2)
    multipliers = []
    primes = []
    offsets = []
    seen: set[int] = set()
    for layer_id in layers:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        multipliers.append(torch.from_numpy(values * 2 + 1))
        layer_primes: list[int] = []
        for _ in range(max_ngram_size - 1):
            for _ in range(n_heads):
                prime = _next_prime(vocab_size - 1, seen)
                seen.add(prime)
                layer_primes.append(prime)
        primes.append(torch.tensor(layer_primes, dtype=torch.int64))
        running = 0
        layer_offsets = []
        for prime in layer_primes:
            layer_offsets.append(running)
            running += prime
        offsets.append(torch.tensor(layer_offsets, dtype=torch.int64))
    return HashLayout(
        layers,
        torch.stack(multipliers),
        torch.stack(primes),
        torch.stack(offsets),
        max_ngram_size,
        n_heads,
        compressed_vocab_size,
        pad_token_id,
        vocab_size,
    )


def update_slot_history(
    input_ids: torch.Tensor,
    token_map: torch.Tensor,
    dead_mask: torch.Tensor,
    slot_mapping: torch.Tensor,
    hash_cache: torch.Tensor,
) -> None:
    """Publish compressed ids to physical slots, matching Ascend's writer."""
    if input_ids.ndim != 1 or token_map.ndim != 1 or dead_mask.ndim != 1:
        raise ValueError("input_ids, token_map and dead_mask must be vectors")
    if slot_mapping.shape != input_ids.shape:
        raise ValueError("slot_mapping shape mismatch")
    valid = (slot_mapping >= 0) & (slot_mapping < hash_cache.numel())
    values = token_map[input_ids.clamp(0, token_map.numel() - 1).long()].to(torch.int32)
    values = torch.where(dead_mask.bool(), torch.full_like(values, DEAD_ID), values)
    hash_cache[slot_mapping[valid].long()] = values[valid]


def hash_ids(
    input_ids: torch.Tensor,
    token_map: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    layout: HashLayout,
    dead_mask: torch.Tensor | None = None,
    lookback_token_ids: torch.Tensor | None = None,
    lookback_dead_mask: torch.Tensor | None = None,
    hash_cache: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    cache_block_size: int = 128,
) -> torch.Tensor:
    """Generate ``[tokens, layers, 24]`` hash ids with vLLM precedence.

    Current chunk tokens win over explicit request lookback, which wins over
    the physical slot cache.  A dead token or a missing/negative history
    position blocks the rolling hash and emits ``pad_token_id`` for all later
    n-gram sizes at that position.
    """
    input_ids = input_ids.to(torch.int64).flatten()
    token_map = token_map.to(torch.int64).flatten()
    positions = positions.to(torch.int64).flatten()
    query_start_loc = query_start_loc.to(torch.int64).flatten()
    n_tokens = int(input_ids.numel())
    if positions.numel() != n_tokens or dead_mask is not None and dead_mask.numel() != n_tokens:
        raise ValueError("input/position/dead-mask lengths differ")
    dead = torch.zeros(n_tokens, dtype=torch.bool) if dead_mask is None else dead_mask.bool().flatten()
    if query_start_loc.numel() < 2 or int(query_start_loc[-1]) != n_tokens:
        raise ValueError("query_start_loc must bound every input token")
    lookback_depth = layout.lookback_depth
    if lookback_token_ids is None:
        lookback_token_ids = torch.full((query_start_loc.numel() - 1, lookback_depth), -1, dtype=torch.int64)
    else:
        lookback_token_ids = lookback_token_ids.to(torch.int64)
    if lookback_dead_mask is None:
        lookback_dead_mask = torch.zeros_like(lookback_token_ids, dtype=torch.bool)
    else:
        lookback_dead_mask = lookback_dead_mask.bool()
    if lookback_token_ids.shape != lookback_dead_mask.shape:
        raise ValueError("lookback token and mask shapes differ")
    output = torch.full((n_tokens, len(layout.layer_ids), layout.n_hash_cols), layout.pad_token_id, dtype=torch.int32)
    if n_tokens == 0:
        return output
    req_ends = query_start_loc[1:]
    for token in range(n_tokens):
        # The native binary search advances when token >= end, i.e. upper
        # bound.  A token exactly at a request boundary belongs to the next
        # request rather than the preceding chunk.
        req = int(torch.searchsorted(req_ends, torch.tensor(token), right=True))
        req = min(req, query_start_loc.numel() - 2)
        chunk_start = int(positions[int(query_start_loc[req])])
        rolling = [0] * len(layout.layer_ids)
        blocked = False
        for shift in range(layout.max_ngram_size):
            lookback = int(positions[token]) - shift
            in_batch = lookback >= chunk_start
            batch_idx = max(token - shift, 0)
            source: int
            source_dead = False
            if in_batch:
                source = int(token_map[int(input_ids[batch_idx])])
                source_dead = bool(dead[batch_idx])
            else:
                col = chunk_start - 1 - lookback
                in_window = 0 <= col < lookback_depth
                known = in_window and int(lookback_token_ids[req, col]) >= 0
                if known:
                    source_token = int(lookback_token_ids[req, col])
                    source = int(token_map[source_token])
                    source_dead = bool(lookback_dead_mask[req, col])
                elif hash_cache is not None and block_table is not None:
                    clamped = min(max(lookback, 0), int(block_table.shape[1]) * cache_block_size - 1)
                    block = int(block_table[min(req, block_table.shape[0] - 1), clamped // cache_block_size])
                    slot = min(max(block * cache_block_size + clamped % cache_block_size, 0), hash_cache.numel() - 1)
                    source = int(hash_cache[slot])
                else:
                    source = layout.pad_token_id
            blocked = blocked or lookback < 0 or source_dead or source == DEAD_ID
            value = layout.pad_token_id if blocked else source
            for li in range(len(layout.layer_ids)):
                rolling[li] ^= int(value) * int(layout.multipliers[li, shift])
                if shift:
                    base = (shift - 1) * layout.n_heads
                    for head in range(layout.n_heads):
                        col = base + head
                        output[token, li, col] = int(rolling[li] % int(layout.primes[li, col]) + int(layout.offsets[li, col]))
    return output


def bounded_host_gather(
    table: torch.Tensor,
    row_ids: torch.Tensor,
    *,
    full_rows: int,
    chunk_rows: int = 32,
) -> torch.Tensor:
    """Gather a bounded host table window with explicit range checks.

    ``table`` is a resident or mmap-backed host tensor.  The function records
    the full checkpoint row count separately so callers cannot accidentally
    treat a 4,096-row audit window as the 384,006,168-row production table.
    """
    if table.ndim != 2 or row_ids.ndim != 2:
        raise ValueError("table and row_ids must be rank-2")
    if full_rows < table.shape[0] or row_ids.numel() and (int(row_ids.min()) < 0 or int(row_ids.max()) >= full_rows):
        raise ValueError("row id outside declared full table range")
    if int(row_ids.max()) >= table.shape[0]:
        raise LookupOutsideResidentWindow(int(row_ids.max()), int(table.shape[0]), full_rows)
    pieces = []
    for start in range(0, row_ids.shape[0], chunk_rows):
        pieces.append(table.index_select(0, row_ids[start : start + chunk_rows].reshape(-1).long()).reshape(-1, row_ids.shape[1], table.shape[1]))
    return torch.cat(pieces, dim=0) if pieces else table.new_empty((0, row_ids.shape[1], table.shape[1]))


class LookupOutsideResidentWindow(IndexError):
    """A valid full-table row was not present in the bounded audit window."""

    def __init__(self, row: int, resident_rows: int, full_rows: int):
        super().__init__(f"row {row} needs host offload (resident={resident_rows}, full={full_rows})")
        self.row, self.resident_rows, self.full_rows = row, resident_rows, full_rows


__all__ = [
    "DEAD_ID",
    "HashLayout",
    "LookupOutsideResidentWindow",
    "build_compressed_token_map",
    "make_layout",
    "hash_ids",
    "update_slot_history",
    "bounded_host_gather",
]
