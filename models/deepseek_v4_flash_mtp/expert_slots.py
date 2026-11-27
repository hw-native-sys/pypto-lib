# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Optional expert bank parameters shared by prefill and ordinary decode."""

import pypto.language as pl
from models.common.expert_cache import expert_cache_shape

from config import EP_WORLD_SIZE, EXPERT_SLOTS, FLASH

if EXPERT_SLOTS and EXPERT_SLOTS < FLASH.n_routed_experts // EP_WORLD_SIZE:
    raise ValueError("V4 mapped expert slots must hold one complete local expert wave")

MAPPED = EXPERT_SLOTS > 0
BANK_DIRECTIONS = {"inout_params": (
    "routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale",
)} if MAPPED else {}
CACHE_ROWS, CACHE_COLS = expert_cache_shape(EXPERT_SLOTS, FLASH.num_hidden_layers, FLASH.n_routed_experts // EP_WORLD_SIZE)
RankState = pl.Tensor[[CACHE_ROWS, CACHE_COLS], pl.INT32] if MAPPED else pl.constexpr
RankDescriptors = pl.Tensor[[FLASH.num_hidden_layers, 6, 4], pl.INT64] if MAPPED else pl.constexpr
WorkerState = pl.Tensor[[EP_WORLD_SIZE, CACHE_ROWS, CACHE_COLS], pl.INT32] if MAPPED else pl.constexpr
WorkerDescriptors = pl.Tensor[[EP_WORLD_SIZE, FLASH.num_hidden_layers, 6, 4], pl.INT64] if MAPPED else pl.constexpr
