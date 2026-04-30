# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .types import KvAllocation, ModelConfig, RuntimeConfig


@dataclass
class _CachePool:
    page_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    key_pages: torch.Tensor
    value_pages: torch.Tensor
    free_pages: list[int]


class KvCacheManager:
    def __init__(self) -> None:
        self._pools: dict[str, _CachePool] = {}

    def register_model(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> None:
        if model_id in self._pools:
            return
        num_pages = runtime.total_kv_pages
        if num_pages is None:
            max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
            num_pages = runtime.max_batch_size * max_blocks_per_seq
        kv_dtype = getattr(torch, runtime.kv_dtype)
        key_pages = torch.zeros(
            config.num_hidden_layers,
            num_pages,
            config.num_key_value_heads,
            runtime.page_size,
            config.head_dim,
            dtype=kv_dtype,
            device=runtime.device,
        )
        value_pages = torch.zeros_like(key_pages)
        self._pools[model_id] = _CachePool(
            page_size=runtime.page_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            key_pages=key_pages,
            value_pages=value_pages,
            free_pages=list(range(num_pages - 1, -1, -1)),
        )

    def allocate_for_prompt(self, model_id: str, request_id: str, prompt_len: int) -> KvAllocation:
        pool = self._pool(model_id)
        num_pages = max(1, math.ceil(prompt_len / pool.page_size))
        page_ids = self._take_pages(pool, num_pages)
        return KvAllocation(
            request_id=request_id,
            model_id=model_id,
            page_ids=page_ids,
            tokens_capacity=len(page_ids) * pool.page_size,
            tokens_used=0,
        )

    def ensure_one_more_slot(self, alloc: KvAllocation) -> int:
        pool = self._pool(alloc.model_id)
        if alloc.tokens_used >= alloc.tokens_capacity:
            alloc.page_ids.extend(self._take_pages(pool, 1))
            alloc.tokens_capacity = len(alloc.page_ids) * pool.page_size
        return self.slot_mapping_for_request(alloc, alloc.tokens_used)

    def block_table_for_request(self, alloc: KvAllocation) -> torch.Tensor:
        return torch.tensor(alloc.page_ids, dtype=torch.int32)

    def block_table_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        max_blocks = max((len(alloc.page_ids) for alloc in allocations), default=0)
        table = torch.full((len(allocations), max_blocks), -1, dtype=torch.int32)
        for row, alloc in enumerate(allocations):
            if alloc.page_ids:
                table[row, : len(alloc.page_ids)] = torch.tensor(alloc.page_ids, dtype=torch.int32)
        return table

    def slot_mapping_for_request(self, alloc: KvAllocation, token_index: int | None = None) -> int:
        pool = self._pool(alloc.model_id)
        logical_index = alloc.tokens_used if token_index is None else token_index
        page_idx = logical_index // pool.page_size
        offset = logical_index % pool.page_size
        return alloc.page_ids[page_idx] * pool.page_size + offset

    def slot_mapping_for_batch(self, allocations: list[KvAllocation]) -> torch.Tensor:
        return torch.tensor(
            [self.slot_mapping_for_request(alloc) for alloc in allocations],
            dtype=torch.int32,
        )

    def slot_mapping_for_positions(self, alloc: KvAllocation, num_tokens: int, *, max_tokens: int | None = None) -> torch.Tensor:
        size = num_tokens if max_tokens is None else max_tokens
        mapping = torch.full((size,), -1, dtype=torch.int32)
        for token_index in range(num_tokens):
            mapping[token_index] = self.slot_mapping_for_request(alloc, token_index)
        return mapping

    def write_tokens(
        self,
        layer_idx: int,
        alloc: KvAllocation,
        start_token_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        pool = self._pool(alloc.model_id)
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        for row in range(keys.shape[0]):
            token_index = start_token_index + row
            page_idx = token_index // pool.page_size
            offset = token_index % pool.page_size
            physical_page = alloc.page_ids[page_idx]
            pool.key_pages[layer_idx, physical_page, :, offset, :] = keys[row]
            pool.value_pages[layer_idx, physical_page, :, offset, :] = values[row]
        alloc.tokens_used = max(alloc.tokens_used, start_token_index + keys.shape[0])

    def ingest_prefill_cache(
        self,
        layer_idx: int,
        alloc: KvAllocation,
        keys_flat: torch.Tensor,
        values_flat: torch.Tensor,
        *,
        max_seq: int,
        seq_len: int,
    ) -> None:
        pool = self._pool(alloc.model_id)
        keys = keys_flat.view(pool.num_kv_heads, max_seq, pool.head_dim)[:, :seq_len, :].permute(1, 0, 2).contiguous()
        values = values_flat.view(pool.num_kv_heads, max_seq, pool.head_dim)[:, :seq_len, :].permute(1, 0, 2).contiguous()
        self.write_tokens(layer_idx, alloc, 0, keys, values)

    def read_context(self, layer_idx: int, alloc: KvAllocation, upto_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        pool = self._pool(alloc.model_id)
        token_count = alloc.tokens_used if upto_tokens is None else upto_tokens
        keys = torch.empty(
            token_count,
            pool.num_kv_heads,
            pool.head_dim,
            dtype=pool.key_pages.dtype,
            device=pool.key_pages.device,
        )
        values = torch.empty_like(keys)
        for token_index in range(token_count):
            page_idx = token_index // pool.page_size
            offset = token_index % pool.page_size
            physical_page = alloc.page_ids[page_idx]
            keys[token_index] = pool.key_pages[layer_idx, physical_page, :, offset, :]
            values[token_index] = pool.value_pages[layer_idx, physical_page, :, offset, :]
        return keys, values

    def materialize_decode_cache(self, model_id: str, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pool = self._pool(model_id)
        return (
            pool.key_pages[layer_idx].reshape(-1, pool.head_dim),
            pool.value_pages[layer_idx].reshape(-1, pool.head_dim),
        )

    def materialize_decode_cache_all_layers(self, model_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        pool = self._pool(model_id)
        return (
            pool.key_pages.reshape(-1, pool.head_dim),
            pool.value_pages.reshape(-1, pool.head_dim),
        )

    def free(self, alloc: KvAllocation) -> None:
        pool = self._pool(alloc.model_id)
        pool.free_pages.extend(alloc.page_ids)
        alloc.page_ids.clear()
        alloc.tokens_capacity = 0
        alloc.tokens_used = 0

    def _pool(self, model_id: str) -> _CachePool:
        if model_id not in self._pools:
            raise KeyError(f"Model {model_id} is not registered with the KV cache manager.")
        return self._pools[model_id]

    @staticmethod
    def _take_pages(pool: _CachePool, num_pages: int) -> list[int]:
        if len(pool.free_pages) < num_pages:
            raise RuntimeError(
                f"Insufficient KV cache capacity: requested {num_pages} pages, only {len(pool.free_pages)} available."
            )
        page_ids = pool.free_pages[-num_pages:]
        del pool.free_pages[-num_pages:]
        return page_ids
