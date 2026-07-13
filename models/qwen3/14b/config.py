# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B model and decode-kernel configuration."""

from dataclasses import dataclass
from typing import Any

import pypto.language as pl

from constants import QWEN3_14B as QWEN3_14B
from constants import QWEN3_14B_TILING as QWEN3_14B_TILING


@dataclass(frozen=True)
class Qwen3DynamicDims:
    """Dynamic dimensions used by the JIT/program signatures."""

    user_batch: Any
    kv_cache_rows: Any
    block_table_flat: Any
    rope_seq: Any
    layer: Any
    layer_hidden_rows: Any
    layer_inter_rows: Any

QWEN3_14B_DIMS = Qwen3DynamicDims(
    user_batch=pl.dynamic("USER_BATCH_DYN"),
    kv_cache_rows=pl.dynamic("KV_CACHE_ROWS_DYN"),
    block_table_flat=pl.dynamic("BLOCK_TABLE_FLAT_DYN"),
    rope_seq=pl.dynamic("ROPE_SEQ_DYN"),
    layer=pl.dynamic("LAYER_DYN"),
    layer_hidden_rows=pl.dynamic("LAYER_HIDDEN_ROWS_DYN"),
    layer_inter_rows=pl.dynamic("LAYER_INTER_ROWS_DYN"),
)
