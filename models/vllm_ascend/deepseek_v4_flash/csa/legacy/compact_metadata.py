# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inline addressing of Native compact metadata; no token-sized GM expansion."""

import pypto.language as pl

from ..common.config import DECODE_SEQ
from ..common.config import FLASH as M
from ..common.layout import QUERY_BOUNDS_DYN

REQUESTS = pl.dynamic("CSA_COMPACT_REQUESTS")
TOKENS = pl.dynamic("CSA_COMPACT_TOKENS")
COMPACT_ROWS = pl.dynamic("CSA_COMPACT_ROWS")
COMPRESS_RATIO = 4
ROPE_DIM = M.qk_rope_head_dim
ROPE_TILE_ROWS = 16


@pl.jit.inline(auto_scope=False)
def build_compact_row_offsets(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS], pl.INT32],
    offsets: pl.Out[pl.Tensor[[REQUESTS], pl.INT32]],
):
    """Run in an existing single-owner task; one offset per Native request."""
    prefix = pl.cast(0, pl.INDEX)
    for request in pl.range(pl.tensor.dim(lengths, 0)):
        begin = pl.read(bounds, [request])
        end = pl.read(bounds, [request + 1])
        length = pl.read(lengths, [request])
        start = length - (end - begin)
        pl.write(offsets, [request], pl.cast(prefix - start // COMPRESS_RATIO - 1, pl.INT32))
        prefix = prefix + length // COMPRESS_RATIO - start // COMPRESS_RATIO
    return offsets


@pl.jit.inline(auto_scope=False)
def load_compact_rope_rows(
    cos: pl.Tensor[[COMPACT_ROWS, ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[COMPACT_ROWS, ROPE_DIM], pl.FP32],
    positions: pl.Tensor[[TOKENS], pl.INT64],
    offsets: pl.Tensor[[REQUESTS], pl.INT32],
    begin: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
):
    """Gather only closing rows into the consumer's existing 16-row UB tile."""
    cosine = pl.full([ROPE_TILE_ROWS, ROPE_DIM], dtype=pl.FP32, value=1.0)
    sine = pl.full([ROPE_TILE_ROWS, ROPE_DIM], dtype=pl.FP32, value=0.0)
    for row in pl.range(rows):
        token = begin + row
        position = pl.read(positions, [token])
        if (position + 1) % COMPRESS_RATIO == 0:
            compact_row = pl.cast(pl.read(offsets, [token // DECODE_SEQ]), pl.INDEX) + pl.cast(
                (position + 1) // COMPRESS_RATIO, pl.INDEX
            )
            # 这两张 RoPE 表和 compact slot mapping 同高，只有 Native 算好的
            # num_compressed_tokens 行。图捕获的 dummy run 把 position 填成 127，
            # 推出的行号会远超该档行数，必须按真实行数兜住。
            if compact_row < pl.tensor.dim(cos, 0):
                cosine = pl.gather_row(cosine, cos, [row, 0], [compact_row, 0], [1, ROPE_DIM])
                sine = pl.gather_row(sine, sin, [row, 0], [compact_row, 0], [1, ROPE_DIM])
    return cosine, sine
