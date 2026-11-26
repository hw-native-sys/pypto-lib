# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare embeddings, positions, and paged cache metadata for A3 forwards."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

from . import config as C
from .config import FLASH as M
from .rope_tables import prepare_forward_rope_tables
from .utils import paged_slots, window_metadata
from .weights import ENGRAM_K, ResidentWeights

if TYPE_CHECKING:
    from .engram import HostEngram


def prepare_prefill_inputs(
    embedding: torch.Tensor,
    capacity: int,
    *,
    start_pos: int = 0,
    max_seq_len: int | None = None,
    cache_state: Mapping[str, torch.Tensor] | None = None,
    request_slots: int = 1,
    moe_slab_rows: int = 64,
    rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare one TP group member's next embedding chunk and native cache metadata.

    ``capacity`` bounds current rows, independently of ``max_seq_len``, which
    sizes the retained history. Reuse rope_tables across calls; their full
    table capacity may exceed capacity. Pass the previous call's cache tensors after
    its completion and advance ``start_pos`` by its valid count. An empty
    sequence has zero work and preserves every cache byte. This convenience
    helper owns one request at physical slot zero per DP rank; serving may
    supply its own paged metadata directly to the forward entry instead.
    """
    if embedding.ndim != 2 or embedding.shape[1] != M.hidden_size:
        raise ValueError("embedding must have shape [current tokens, hidden size]")
    if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity < 16 or capacity % 16:
        raise ValueError("prefill capacity must be an integer multiple of 16")
    count = embedding.shape[0]
    if not 0 <= count <= capacity <= C.PREFILL_MAX_TOKENS:
        raise ValueError("current rows must fit the configured prefill capacity")
    if isinstance(start_pos, bool) or not isinstance(start_pos, Integral) or start_pos < 0:
        raise ValueError("start_pos must be a non-negative integer")
    max_seq_len = capacity if max_seq_len is None else max_seq_len
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, Integral):
        raise ValueError("max_seq_len must be a positive integer")
    if not 1 <= max_seq_len <= M.max_position_embeddings or start_pos + count > max_seq_len:
        raise ValueError("current rows exceed the retained history capacity")
    if start_pos and cache_state is None:
        raise ValueError("continuation requires the preceding call's retained cache state")

    data_pages = (max_seq_len + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    pages = data_pages * request_slots + 1
    table = torch.arange(data_pages, dtype=torch.int32).reshape(1, -1)
    c2a_pages = max(1, (max_seq_len // 2 + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE)
    c2a_table = torch.arange(c2a_pages, dtype=torch.int32).reshape(1, -1)
    table = table.expand(request_slots, -1).contiguous()
    c2a_table = c2a_table.expand(request_slots, -1).contiguous()
    if request_slots > 1:
        if count:
            raise ValueError("Use packed metadata for multi-slot inputs")
        table.fill_(-1)
        c2a_table.fill_(-1)
    compressed_tables = {
        source: c2a_table if M.compress_ratios[source] == 2 else table
        for source in M.kv_source_layer_ids
    }
    state_table = torch.arange(request_slots, dtype=torch.int32).reshape(-1, 1)
    state_tables = {
        source: state_table
        for source in M.kv_source_layer_ids
        if M.compress_ratios[source] == 2
    }
    query_starts = torch.tensor([0] + [count] * request_slots, dtype=torch.int32)
    request_ids = torch.zeros(count, dtype=torch.int32)
    positions = torch.arange(start_pos, start_pos + count, dtype=torch.int32)
    window_slots, window_indices, _ = window_metadata(positions, request_ids, table)
    c2a_slots = paged_slots(positions, request_ids, c2a_table, C.BLOCK_SIZE, 2, True)
    c1a_slots = window_slots
    freqs_cos, freqs_sin = prepare_forward_rope_tables(max_seq_len, rope_tables)

    def pad(value, fill=0):
        result = torch.full((capacity, *value.shape[1:]), fill, dtype=value.dtype)
        result[: value.shape[0]] = value
        return result

    initial_pre = torch.zeros((capacity, M.hc_mult), dtype=torch.float32)
    initial_pre[:, 0] = 1.0
    logit_rows = torch.full((16,), -1, dtype=torch.int32)
    logit_rows[0] = count - 1
    window_shape = (M.num_hidden_layers, pages, C.BLOCK_SIZE, 1)
    source_shape = (len(M.kv_source_layer_ids), pages, C.BLOCK_SIZE, 1)

    def native_cache(name, shape, dtype, fill=0):
        if cache_state is not None:
            value = cache_state[name]
            if value.device.type != "cpu" or value.dtype != dtype:
                raise ValueError(f"retained {name} must be a CPU tensor with the native cache dtype")
            if tuple(value.shape) != shape or not value.is_contiguous():
                raise ValueError(f"retained {name} must preserve the history layout and capacity")
            return value
        if dtype in (torch.float32, torch.bfloat16):
            return torch.full(shape, float(fill), dtype=dtype)
        return torch.full(shape, fill, dtype=torch.uint8).view(dtype)

    round_rows = C.TP_SIZE * moe_slab_rows
    rounds = (capacity + round_rows - 1) // round_rows
    local_moe_rows = rounds * moe_slab_rows
    values = {
        "layer_range": torch.tensor([0, M.num_hidden_layers], dtype=torch.int32),
        "topk_indices": torch.full((capacity, M.index_topk), -1, dtype=torch.int32),
        "pool_block_ids": torch.full((capacity, min(M.candidate_topk_blocks, ((data_pages + 1) * 16 + 31) // 32 * 32)), -1, dtype=torch.int32),
        "pool_block_counts": torch.zeros(capacity, dtype=torch.int32),
        "moe_statistics": torch.zeros((M.num_hidden_layers, rounds, 3), dtype=torch.int32),
        "moe_gate_inputs": torch.zeros((local_moe_rows, M.hidden_size), dtype=torch.bfloat16),
        "moe_route_indices": torch.zeros((M.num_hidden_layers, local_moe_rows, 6), dtype=torch.int32),
        "moe_route_weights": torch.zeros((M.num_hidden_layers, local_moe_rows, 6), dtype=torch.float32),
        "embedding": pad(embedding),
        "engram_lookup": torch.zeros((2, capacity, ENGRAM_K), dtype=torch.bfloat16),
        "embedding_ids": torch.arange(capacity, dtype=torch.int64),
        "initial_pre": initial_pre,
        "identity_rows": pad(torch.arange(count, dtype=torch.int32), -1),
        "counts": torch.tensor([count, count], dtype=torch.int32),
        "replay_rows": pad(torch.arange(count, dtype=torch.int32), -1),
        "position_ids": pad(positions, -1).repeat(2, 1),
        "freqs_cos": freqs_cos,
        "freqs_sin": freqs_sin,
        "window_slots": pad(window_slots, -1).repeat(2, 1),
        "window_indices": pad(window_indices, -1).repeat(2, 1, 1),
        "c2a_compressed_slots": pad(c2a_slots, -1),
        "c1a_compressed_slots": pad(c1a_slots, -1),
        "token_to_req_indices": pad(request_ids, -1).repeat(2, 1),
        "query_start_loc": query_starts.repeat(2, 1),
        "c2a_index_block_table": c2a_table,
        "c1a_index_block_table": table,
        "state_block_table": state_table,
        "window_cache": native_cache("window_cache", (*window_shape, M.head_dim), torch.uint8),
        "window_scales": native_cache(
            "window_scales", (*window_shape, M.head_dim // C.WINDOW_CACHE_GROUP), torch.bfloat16, 1.0,
        ),
        "compressed_cache": native_cache("compressed_cache", (*source_shape, M.head_dim // 2), torch.uint8),
        "compressed_scales": native_cache(
            "compressed_scales", (*source_shape, M.head_dim // C.COMPRESSED_CACHE_GROUP), torch.bfloat16, 1.0,
        ),
        "index_cache": native_cache("index_cache", (*source_shape, M.index_head_dim // 2), torch.uint8),
        "index_scales": native_cache(
            "index_scales", (*source_shape, M.index_head_dim // C.INDEX_CACHE_GROUP), torch.uint8, 127,
        ),
        "state_cache": native_cache(
            "state_cache", (len(state_tables), request_slots, C.STATE_CAPACITY, 2 * M.head_dim), torch.float32,
        ),
        "logit_rows": logit_rows,
    }
    validate_prefill_caches(values, table, compressed_tables, state_tables)
    return values

def prepare_resident_inputs(
    weights: ResidentWeights,
    token_ids: torch.Tensor | Sequence[int],
    capacity: int,
    *,
    start_pos: int = 0,
    max_seq_len: int | None = None,
    cache_state: Mapping[str, torch.Tensor] | None = None,
    request_slots: int = 1,
    moe_slab_rows: int = 64,
    rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
    engram: HostEngram,
    request_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Prepare checkpoint embeddings and causal replay for one request."""
    ids = torch.as_tensor(token_ids, device="cpu")
    if ids.ndim != 1 or ids.numel() and ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be a one-dimensional integer sequence")
    if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity % 16:
        raise ValueError("prefill capacity must be an integer multiple of 16")
    count = ids.numel()
    if not 0 <= count <= capacity <= C.PREFILL_MAX_TOKENS:
        raise ValueError("current rows must fit the configured prefill capacity")
    if bool(((ids < 0) | (ids >= M.vocab_size)).any()):
        raise ValueError("token id is outside the checkpoint vocabulary")
    ids = ids.long()
    embedding = weights.lookup_rows(ids)

    values = prepare_prefill_inputs(
        embedding, capacity, start_pos=start_pos, max_seq_len=max_seq_len,
        cache_state=cache_state, rope_tables=rope_tables, request_slots=request_slots, moe_slab_rows=moe_slab_rows,
    )
    values["engram_lookup"] = engram.lookup(ids, capacity, start_pos, request_id=request_id)
    return values

def validate_prefill_caches(
    cache_state: Mapping[str, torch.Tensor],
    window_block_table: torch.Tensor,
    compressed_block_tables: Mapping[int, torch.Tensor],
    state_block_tables: Mapping[int, torch.Tensor],
) -> None:
    """Validate native prefill cache layouts and source-local page/state ownership."""
    cache_names = (
        "window_cache", "window_scales", "compressed_cache", "compressed_scales", "index_cache", "index_scales",
    )
    for name in cache_names:
        value = cache_state[name]
        if value.ndim not in (5, 6) or not value.is_contiguous():
            raise ValueError(f"{name} must preserve the contiguous native source/page layout")
        sources = M.num_hidden_layers if name.startswith("window") else len(M.kv_source_layer_ids)
        if value.shape[-5] != sources or value.shape[-4] < 1 or tuple(value.shape[-3:-1]) != (C.BLOCK_SIZE, 1):
            raise ValueError(f"{name} has an incompatible source or page geometry")
        expected_dtype = {
            "window_cache": torch.uint8, "window_scales": torch.bfloat16,
            "compressed_cache": torch.uint8, "compressed_scales": torch.bfloat16,
            "index_cache": torch.uint8, "index_scales": torch.uint8,
        }[name]
        if value.dtype != expected_dtype:
            raise ValueError(f"{name} must retain its native payload or scale dtype")
        expected_width = {
            "window_cache": M.head_dim, "window_scales": M.head_dim // C.WINDOW_CACHE_GROUP,
            "compressed_cache": M.head_dim // 2,
            "compressed_scales": M.head_dim // C.COMPRESSED_CACHE_GROUP,
            "index_cache": M.index_head_dim // 2, "index_scales": M.index_head_dim // C.INDEX_CACHE_GROUP,
        }[name]
        if value.shape[-1] != expected_width:
            raise ValueError(f"{name} has an incompatible native row width")
        reference = cache_state["window_cache"]
        if value.shape[:-5] != reference.shape[:-5] or value.shape[-4] != reference.shape[-4]:
            raise ValueError("all prefill cache pools must preserve the same rank and page extents")
        if value.device != reference.device:
            raise ValueError("all cache payloads and scales must reside on the same device")
    state = cache_state["state_cache"]
    if state.ndim not in (4, 5) or not state.is_contiguous():
        raise ValueError("state_cache must preserve its contiguous source/request layout")
    if state.shape[-4] != 3 or state.shape[-3] < 1 or tuple(state.shape[-2:]) != (C.STATE_CAPACITY, 2 * M.head_dim):
        raise ValueError("state_cache has an incompatible C2A source or state geometry")
    if state.shape[:-4] != cache_state["window_cache"].shape[:-5]:
        raise ValueError("state_cache must preserve the cache pools' rank extent")
    if state.dtype != torch.float32:
        raise ValueError("state_cache must retain FP32 compressor state")
    if state.device != cache_state["window_cache"].device:
        raise ValueError("state_cache must reside on the cache pools' device")

    if cache_state["window_cache"].ndim != 5:
        raise ValueError("validate_prefill_caches requires one rank's native cache tensors")
    if window_block_table.ndim != 2:
        raise ValueError("window block table must be a matrix")
    requests = window_block_table.shape[0]
    pages = cache_state["window_cache"].shape[1]
    state_blocks = cache_state["state_cache"].shape[1]

    def validate_table(name, table, capacity, state=False):
        if table.device.type != "cpu" or table.dtype != torch.int32:
            raise ValueError(f"{name} must be a CPU INT32 tensor")
        if table.ndim != 2 or table.shape[0] != requests or (state and table.shape[1] != 1):
            raise ValueError(f"{name} has incompatible request or table dimensions")
        if bool(((table < -1) | (table >= capacity)).any()):
            raise ValueError(f"{name} IDs must be -1 or within the source-local capacity {capacity}")
        if state:
            allocated = table[table >= 0]
            if allocated.unique().numel() != allocated.numel():
                raise ValueError("live requests must own distinct state blocks")

    validate_table("window block table", window_block_table, pages)
    family_tables = {}
    shared_state = None
    for source in M.kv_source_layer_ids:
        if source not in compressed_block_tables:
            raise ValueError(f"missing compressed block table for source layer {source}")
        table = compressed_block_tables[source]
        validate_table(f"source {source} compressed block table", table, pages)
        ratio = M.compress_ratios[source]
        family_table = family_tables.setdefault(ratio, table)
        if not torch.equal(table, family_table):
            raise ValueError("sources in one family must share local physical page IDs")
        if ratio > 1:
            if source not in state_block_tables:
                raise ValueError(f"missing state block table for source layer {source}")
            state_table = state_block_tables[source]
            validate_table(f"source {source} state block table", state_table, state_blocks, state=True)
            if shared_state is None:
                shared_state = state_table
            elif not torch.equal(state_table, shared_state):
                raise ValueError("C2A sources must share request slots within their separate state pools")
