# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared metadata layout for the mapped expert directory and LRU list."""


def expert_cache_shape(capacity: int, layers: int, experts: int) -> tuple[int, int]:
    return capacity + 1 + (layers * experts + 5) // 6, 6


def initial_expert_cache(capacity: int, layers: int, experts: int):
    import torch

    state = torch.zeros(expert_cache_shape(capacity, layers, experts), dtype=torch.int32)
    # Row zero is the LRU sentinel; slot s occupies row s + 1. The trailing
    # words hold directory[layer, expert], using -1 for an absent expert.
    state[0, 0] = capacity
    state[0, 2:4] = torch.tensor([capacity, 1])
    state[1:capacity + 1, :2] = -1
    state[1:capacity + 1, 2] = torch.arange(capacity)
    state[1:capacity + 1, 3] = torch.arange(2, capacity + 2)
    state[capacity, 3] = 0
    state[capacity + 1:] = -1
    return state
