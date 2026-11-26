# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pack a 16 by 1024 BF16 tile into FRACTAL_NZ stripes in vector memory."""

import pypto.language as pl


NZ_PACK_ROWS = 16
NZ_PACK_COLS = 1024
NZ_C0 = 16
NZ_STRIPE_WORDS = NZ_PACK_ROWS * NZ_C0 // 2
NZ_PACK_WORDS = NZ_PACK_ROWS * NZ_PACK_COLS // 2


@pl.jit.inline
def bf16_nz_pack_indices() -> pl.Tile[[1, NZ_PACK_WORDS], pl.INT32]:
    lane = pl.tile.arange(0, [1, NZ_C0 // 2], dtype=pl.INT32)
    rows = pl.tile.arange(0, [1, NZ_PACK_ROWS], dtype=pl.INT32)
    row_start = pl.tile.mul(rows, NZ_PACK_COLS // 2)
    zeros = pl.tile.full([NZ_PACK_ROWS, NZ_C0 // 2], dtype=pl.INT32, value=0)
    lanes = pl.tile.col_expand_add(zeros, lane)
    row_major = pl.tile.row_expand_add(lanes, pl.tile.reshape(row_start, [NZ_PACK_ROWS, 1]))
    one_stripe = pl.tile.reshape(row_major, [1, NZ_STRIPE_WORDS])
    strips = pl.tile.arange(0, [1, NZ_PACK_COLS // NZ_C0], dtype=pl.INT32)
    strip_offsets = pl.tile.mul(strips, NZ_C0 // 2)
    stripe_zeros = pl.tile.full([NZ_PACK_COLS // NZ_C0, NZ_STRIPE_WORDS], dtype=pl.INT32, value=0)
    repeated = pl.tile.col_expand_add(stripe_zeros, one_stripe)
    offsets = pl.tile.reshape(strip_offsets, [NZ_PACK_COLS // NZ_C0, 1])
    lookup = pl.tile.row_expand_add(repeated, offsets)
    return pl.tile.reshape(lookup, [1, NZ_PACK_WORDS])


@pl.jit.inline
def pack_bf16_nz_chunk(
    nd: pl.Tile[[NZ_PACK_ROWS, NZ_PACK_COLS], pl.BF16],
    indices: pl.Tile[[1, NZ_PACK_WORDS], pl.INT32],
    tmp: pl.Tile[[1, NZ_PACK_WORDS], pl.INT32],
) -> pl.Tile[[NZ_PACK_COLS // NZ_C0, NZ_PACK_ROWS * NZ_C0], pl.BF16]:
    words = pl.tile.reinterpret_view(nd, pl.INT32, shape=[1, NZ_PACK_WORDS])
    ordered = pl.tile.gather(words, indices, tmp)
    return pl.tile.reinterpret_view(
        ordered, pl.BF16, shape=[NZ_PACK_COLS // NZ_C0, NZ_PACK_ROWS * NZ_C0],
    )
