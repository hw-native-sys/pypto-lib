# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-owned pending-pair storage for sequential C2A execution."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

from models.deepseek_v4_1_flash.config import FLASH, HEAD_DIM, MAX_BATCH_PER_DP, STATE_HEADS


@dataclass(frozen=True)
class CompressorStateSnapshot:
    """Detached per-source pending pairs at a completed prefix boundary."""

    prefix_length: int
    pending: Mapping[int, torch.Tensor]


class CompressorStateCache:
    """Own stable slots and per-source FP32 kernel buffers on one TP rank.

    Use request keys including their generation. The engine must serialize
    lifecycle operations with kernel completion on the same stream (or wait
    explicitly), and snapshot only after all source layers finish. This pool
    retains one pending pair, not historical states for speculative rollback.
    """

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self._buffers = {
            source: torch.zeros((MAX_BATCH_PER_DP, STATE_HEADS, HEAD_DIM), dtype=torch.float32, device=device)
            for source in FLASH.kv_source_layer_ids
            if FLASH.compress_ratios[source] == 2
        }
        self._owners: dict[Hashable, int] = {}
        self._free = list(reversed(range(MAX_BATCH_PER_DP)))

    @property
    def buffers(self) -> Mapping[int, torch.Tensor]:
        """Pass buffers[source] unchanged to that source's compressor kernel."""
        return MappingProxyType(self._buffers)

    @staticmethod
    def _validate_length(prefix_length: int) -> None:
        if type(prefix_length) is not int or prefix_length < 0:
            raise ValueError("prefix_length must be a non-negative integer")

    def allocate(
        self,
        request_key: Hashable,
        *,
        prefix_length: int = 0,
        snapshot: CompressorStateSnapshot | None = None,
    ) -> int:
        """Allocate or restore a request; odd prefixes require their snapshot.

        Even prefixes may resume with cleared state because their next token
        starts a new pair. The engine restores matching KV caches separately.
        """
        self._validate_length(prefix_length)
        if request_key in self._owners:
            raise ValueError("request already owns compressor state")
        if not self._free:
            raise RuntimeError("compressor state pool is full; release or snapshot a request first")
        if prefix_length % 2 and snapshot is None:
            raise ValueError("an odd prefix requires its pending compressor state")
        if snapshot is not None:
            self._validate_length(snapshot.prefix_length)
            if snapshot.prefix_length != prefix_length or set(snapshot.pending) != set(self._buffers):
                raise ValueError("snapshot must match the prefix length and all compressor sources")
            for pending in snapshot.pending.values():
                if pending.shape != (STATE_HEADS, HEAD_DIM) or pending.dtype != torch.float32:
                    raise ValueError("snapshot pending pairs must be FP32 [STATE_HEADS, HEAD_DIM]")
        slot = self._free[-1]
        # Publish ownership only after every source is initialized successfully.
        for source, buffer in self._buffers.items():
            if snapshot is None:
                buffer[slot].zero_()
            else:
                buffer[slot].copy_(snapshot.pending[source])
        self._free.pop()
        self._owners[request_key] = slot
        return slot

    def slots(self, request_keys: Sequence[Hashable]) -> Mapping[int, torch.Tensor]:
        """Build batch-ordered stable slots for build_forward_metadata.

        Omitted/paused requests retain ownership. Duplicate and unknown keys
        are rejected, rather than assigning state from current batch order.
        """
        if len(set(request_keys)) != len(request_keys):
            raise ValueError("a batch must not contain duplicate request keys")
        rows = [self._owners[key] for key in request_keys]
        return {
            source: torch.tensor(rows, dtype=torch.int64, device=buffer.device)
            for source, buffer in self._buffers.items()
        }

    def snapshot(self, request_key: Hashable, *, prefix_length: int) -> CompressorStateSnapshot:
        """Copy completed state before eviction; the engine supplies the committed length.

        Snapshots stay on the pool device and share no storage with live rows.
        Save the matching KV-cache prefix separately. Never snapshot in-flight
        or partially completed layers, or label current state with an old length.
        """
        self._validate_length(prefix_length)
        slot = self._owners[request_key]
        return CompressorStateSnapshot(
            prefix_length,
            {source: buffer[slot].detach().clone() for source, buffer in self._buffers.items()},
        )

    def release(self, request_key: Hashable) -> None:
        """Clear all pending pairs before making a completed request's slot reusable."""
        slot = self._owners[request_key]
        for buffer in self._buffers.values():
            buffer[slot].zero_()
        del self._owners[request_key]
        self._free.append(slot)
