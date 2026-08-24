# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lightweight serving capabilities owned by the DeepSeek-V4 Flash kernels.

This manifest exposes shape and scheduling constraints only.  The full external
kernel ABI remains outside the scope of this module.
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


DEEPSEEK_V4_FLASH_SERVING_CONTRACT = DeepSeekV4FlashServingContract(
    schema_version="1",
    prefill_tile_tokens=128,
    max_prefill_tokens_per_request=8192,
    max_prefill_requests_per_partition=1,
    requires_homogeneous_prefill_decode=True,
)
