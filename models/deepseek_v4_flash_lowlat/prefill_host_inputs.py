# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reusable CPU prompt storage for explicit per-rank device uploads.

Create the storage before Worker preparation and register it as mutable inherited
host storage. Fill it inside the request and retire all copies before refilling.
The returned rank-broadcast view is not a direct shared-CPU dispatch argument.
"""

import torch


class SharedInput:
    def __init__(self, max_rows=8192):
        if type(max_rows) is not int or max_rows < 8 or max_rows > 8192 or max_rows % 8:
            raise ValueError("Aligned1..8192-row buffer required")
        self.storage = torch.zeros((max_rows, 4, 4096), dtype=torch.float32).share_memory_()
        self.copying = False
        self.generation = 0

    def fill(self, embeddings):
        if self.copying:
            raise RuntimeError("Do not overwrite shared input while copies are active")
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.bfloat16
            or embeddings.ndim != 2
            or embeddings.shape[1] != 4096
        ):
            raise ValueError("Actual CPU BF16 embeddings required")
        rows = embeddings.shape[0]
        padded = (rows + 7) // 8 * 8
        if rows < 1 or padded > self.storage.shape[0]:
            raise ValueError("Input exceeds the real shared allocation")
        hidden = self.storage[:padded]
        hidden.zero_()
        hidden[:rows] = embeddings.float().unsqueeze(1)
        self.generation += 1
        return hidden.unsqueeze(0).expand(8, *hidden.shape)

    def begin_copy(self):
        if self.copying:
            raise RuntimeError("A copy is already active")
        self.copying = True

    def end_copy(self):
        if not self.copying:
            raise RuntimeError("No active copy to retire")
        self.copying = False


class CompactInput(SharedInput):
    """Register real BF16 prompt rows; HC expansion is a separate device op."""

    def __init__(self, max_rows=8192):
        if type(max_rows) is not int or max_rows < 8 or max_rows > 8192 or max_rows % 8:
            raise ValueError("Aligned1..8192-row buffer required")
        self.storage = torch.zeros((max_rows, 4096), dtype=torch.bfloat16).share_memory_()
        self.copying = False
        self.generation = 0

    def fill(self, embeddings):
        if self.copying:
            raise RuntimeError("Do not overwrite shared input while copies are active")
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.bfloat16
            or embeddings.ndim != 2
            or embeddings.shape[1] != 4096
        ):
            raise ValueError("Actual CPU BF16 embeddings required")
        rows = embeddings.shape[0]
        padded = (rows + 7) // 8 * 8
        if rows < 1 or padded > self.storage.shape[0]:
            raise ValueError("Input exceeds the real shared allocation")
        result = self.storage[:padded]
        result[:rows].copy_(embeddings)
        result[rows:].zero_()
        self.generation += 1
        return result.unsqueeze(0).expand(8, *result.shape)
