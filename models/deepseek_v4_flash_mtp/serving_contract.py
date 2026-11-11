# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lightweight serving capabilities owned by the DeepSeek-V4 Flash kernels.

This module exposes shape constraints and CPU request planning. Device completion
and communication metadata remain owned by the kernels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV4FlashServingContract:
    """Kernel limits and scheduling constraints consumed by serving runtimes."""

    schema_version: str
    prefill_tile_tokens: int
    max_prefill_tokens_per_request: int
    max_prefill_requests_per_partition: int
    requires_homogeneous_prefill_decode: bool

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if self.prefill_tile_tokens <= 0:
            raise ValueError("prefill_tile_tokens must be positive")
        if self.max_prefill_tokens_per_request < self.prefill_tile_tokens:
            raise ValueError(
                "max_prefill_tokens_per_request must cover one prefill tile"
            )
        if self.max_prefill_tokens_per_request % self.prefill_tile_tokens:
            raise ValueError(
                "max_prefill_tokens_per_request must be a multiple of prefill_tile_tokens"
            )
        if self.max_prefill_requests_per_partition <= 0:
            raise ValueError("max_prefill_requests_per_partition must be positive")

    def padded_prefill_tokens(self, active_tokens: int) -> int:
        """Round one active dispatch extent to the kernel's internal tile."""
        if isinstance(active_tokens, bool) or not isinstance(active_tokens, int):
            raise TypeError(
                f"active prefill tokens must be an int, got {active_tokens!r}"
            )
        if active_tokens <= 0:
            raise ValueError("active prefill tokens must be positive")
        if active_tokens > self.max_prefill_tokens_per_request:
            raise ValueError(
                f"active prefill tokens must not exceed "
                f"{self.max_prefill_tokens_per_request}, got {active_tokens}"
            )
        tile = self.prefill_tile_tokens
        return ((active_tokens + tile - 1) // tile) * tile

    def prefill_request_plan(self, counts, position_ids, *, max_segment_tiles: int = 4):
        """Build CPU-only [mode, owner, length, span, base] rows for CP dispatch.

        Each row uses the request owner's position, including on ranks that
        receive that owner's payload. Device counts remain a separate input.
        """
        import torch

        if counts.device.type != "cpu" or position_ids.device.type != "cpu":
            raise ValueError("prefill request planning requires CPU inputs")
        if counts.dtype != torch.int32 or position_ids.dtype != torch.int32:
            raise TypeError("prefill request planning requires INT32 inputs")
        if counts.ndim != 1 or position_ids.ndim != 2:
            raise ValueError("prefill request planning requires [owners] counts and [owners, rows] positions")
        cp_size = counts.numel()
        if cp_size not in (1, 2, 4, 8, 16) or position_ids.shape[0] != cp_size or position_ids.shape[1] < 1:
            raise ValueError("prefill request planning requires matching CP owners and nonempty position rows")
        if type(max_segment_tiles) is not int or not 1 <= max_segment_tiles <= 4:
            raise ValueError("max_segment_tiles must be an int in [1, 4]")
        segments = 2 * cp_size
        capacity = segments * max_segment_tiles * self.prefill_tile_tokens
        plan = torch.zeros((cp_size, 5), dtype=torch.int32)
        for owner, length in enumerate(counts.tolist()):
            base = int(position_ids[owner, 0])
            span = max(self.prefill_tile_tokens, (length + segments - 1) // segments)
            mode = int(0 < length <= capacity and base >= 0)
            plan[owner] = torch.tensor((mode, owner, length, span, base), dtype=torch.int32)
        return plan

    def prefill_history_tail_slots(self, request_plan, ori_block_table, *, block_size: int = 128):
        """Map each active owner's preceding tail row using ready CPU page tables."""
        import torch

        if request_plan.device.type != "cpu" or ori_block_table.device.type != "cpu":
            raise ValueError("prefill history planning requires CPU inputs")
        if request_plan.dtype != torch.int32 or ori_block_table.dtype != torch.int32:
            raise TypeError("prefill history planning requires INT32 inputs")
        if request_plan.ndim != 2 or request_plan.shape[1] != 5 or ori_block_table.ndim != 2:
            raise ValueError("prefill history planning requires [owners, 5] plan and [owners, pages] table")
        owners = request_plan.shape[0]
        if ori_block_table.shape[0] != owners:
            raise ValueError("prefill history planning requires matching owner counts")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("history block_size must be a positive int")
        result = torch.full((owners,), -1, dtype=torch.int32)
        if ori_block_table.shape[1] == 0:
            return result
        previous = request_plan[:, 4].to(torch.int64) - 1
        logical = previous // block_size
        indices = logical.clamp(0, ori_block_table.shape[1] - 1)
        pages = ori_block_table.gather(1, indices[:, None]).flatten().to(torch.int64)
        valid = (request_plan[:, 0] == 1) & (previous >= 0)
        valid = valid & (logical < ori_block_table.shape[1]) & (pages >= 0)
        physical = pages * block_size + previous % block_size
        return torch.where(valid, physical, result).to(torch.int32)


DEEPSEEK_V4_FLASH_SERVING_CONTRACT = DeepSeekV4FlashServingContract(
    schema_version="1",
    prefill_tile_tokens=128,
    max_prefill_tokens_per_request=8192,
    max_prefill_requests_per_partition=1,
    requires_homogeneous_prefill_decode=True,
)
