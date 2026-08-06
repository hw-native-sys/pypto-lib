# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared RoPE cos/sin interleave-duplication for the DeepSeek-V4 Flash indexer.

The A3 interleaved rotation needs, per rope column ``j``:

    cos_il[j]     = cos_half[j >> 1]
    sin_signed[j] = sin_half[j >> 1] * sign[j],  sign = [-1, +1, -1, +1, ...]

so the forward rotation is ``out[j] = x[j]*cos_il[j] + x[j^1]*sin_signed[j]`` and the
inverse (conjugate) rotation is the same expression with ``pl.sub``.

Building this per consumer block means re-running the ``j >> 1`` dup-gather on every
block. ``pl.gather`` lowers to a per-row ``TGATHER`` loop, so the cost scales with
(blocks x rows): the indexer's ``qr_rope`` alone spent 16 blocks x 32 rows x 2 tables
= 1024 row-gathers per layer rebuilding one small position-invariant table, plus 32
more in its compressor. This runs it once per layer over ``B`` rows instead.

Folding the sign into sin here rather than at each consumer is exact: multiplying by
+/-1 only flips the sign bit, so ``(x*sign)*sin`` and ``x*(sin*sign)`` are bit-identical.
"""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH


B = DECODE_BATCH
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2


@pl.jit.inline
def rope_interleave(
    cos_half: pl.Tensor[[B, HALF_ROPE], pl.FP32],
    sin_half: pl.Tensor[[B, HALF_ROPE], pl.FP32],
    cos_il: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
    sin_signed: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
):
    """Expand half-width cos/sin rows to the interleaved, sign-folded rope layout."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_interleave", allow_early_resolve=True):
        il_ones = pl.full([B, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(
            il_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)                                    # j>>1
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))                                         # j%2
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)                                             # [-1,+1,...]
        cos_il[0:B, 0:ROPE_HEAD_DIM] = pl.gather(
            cos_half[0:B, 0:HALF_ROPE], dim=-1, index=il_dup_idx)
        sin_signed[0:B, 0:ROPE_HEAD_DIM] = pl.mul(
            pl.gather(sin_half[0:B, 0:HALF_ROPE], dim=-1, index=il_dup_idx), il_sign)
