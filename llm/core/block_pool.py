# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field


NONE_HASH = hash(("__none__",))


def hash_block_tokens(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    return hash((parent_hash, token_ids))


@dataclass(slots=True)
class KVBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: int | None = None
    prev_free: KVBlock | None = field(default=None, repr=False)
    next_free: KVBlock | None = field(default=None, repr=False)


class FreeBlockQueue:
    """Doubly-linked list of free blocks in LRU order (head=oldest, tail=newest)."""

    def __init__(self) -> None:
        self.head: KVBlock | None = None
        self.tail: KVBlock | None = None
        self.count: int = 0

    def append(self, block: KVBlock) -> None:
        block.prev_free = self.tail
        block.next_free = None
        if self.tail is not None:
            self.tail.next_free = block
        else:
            self.head = block
        self.tail = block
        self.count += 1

    def popleft(self) -> KVBlock | None:
        if self.head is None:
            return None
        block = self.head
        self._remove(block)
        return block

    def remove(self, block: KVBlock) -> None:
        self._remove(block)

    def _remove(self, block: KVBlock) -> None:
        prev_b = block.prev_free
        next_b = block.next_free
        if prev_b is not None:
            prev_b.next_free = next_b
        else:
            self.head = next_b
        if next_b is not None:
            next_b.prev_free = prev_b
        else:
            self.tail = prev_b
        block.prev_free = None
        block.next_free = None
        self.count -= 1

    def __len__(self) -> int:
        return self.count


class BlockPool:
    """Manages physical KV cache blocks with LRU eviction and hash-based prefix caching."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [KVBlock(block_id=i) for i in range(num_blocks)]
        self.free_queue = FreeBlockQueue()
        self.hash_to_block: dict[int, KVBlock] = {}

        for block in self.blocks:
            self.free_queue.append(block)

    @property
    def num_free_blocks(self) -> int:
        return self.free_queue.count

    def get_cached_block(self, block_hash: int) -> KVBlock | None:
        block = self.hash_to_block.get(block_hash)
        if block is None:
            return None
        if block.ref_cnt == 0:
            self.free_queue.remove(block)
        block.ref_cnt += 1
        return block

    def cache_block(self, block: KVBlock, block_hash: int) -> None:
        if block.block_hash is not None and block.block_hash in self.hash_to_block:
            del self.hash_to_block[block.block_hash]
        block.block_hash = block_hash
        self.hash_to_block[block_hash] = block

    def allocate(self) -> KVBlock | None:
        block = self.free_queue.popleft()
        if block is None:
            return None
        if block.block_hash is not None:
            del self.hash_to_block[block.block_hash]
            block.block_hash = None
        block.ref_cnt = 1
        return block

    def release(self, block: KVBlock) -> None:
        block.ref_cnt -= 1
        if block.ref_cnt <= 0:
            block.ref_cnt = 0
            self.free_queue.append(block)

    def get_computed_blocks(self, token_ids: list[int]) -> list[KVBlock]:
        """Find the longest prefix of cached blocks for the given token sequence."""
        hit_blocks: list[KVBlock] = []
        parent_hash = NONE_HASH
        num_full_blocks = len(token_ids) // self.block_size

        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            block_tokens = tuple(token_ids[start:end])
            block_hash = hash_block_tokens(parent_hash, block_tokens)
            block = self.get_cached_block(block_hash)
            if block is None:
                break
            hit_blocks.append(block)
            parent_hash = block_hash

        return hit_blocks

    def compute_block_hashes(self, token_ids: list[int]) -> list[int]:
        """Compute chained hashes for all full blocks in the token sequence."""
        hashes: list[int] = []
        parent_hash = NONE_HASH
        num_full_blocks = len(token_ids) // self.block_size

        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            block_tokens = tuple(token_ids[start:end])
            block_hash = hash_block_tokens(parent_hash, block_tokens)
            hashes.append(block_hash)
            parent_hash = block_hash

        return hashes
