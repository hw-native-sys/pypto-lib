# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B final RMSNorm program.

Applies the standard pre-LM-head RMSNorm:

    y = x / sqrt(mean(x ** 2) + eps) * gamma

Shapes:
    x:     [BATCH, HIDDEN] BF16
    gamma: [1,     HIDDEN] FP32
    out:   [BATCH, HIDDEN] BF16
"""

import pypto.language as pl

BATCH = 16
HIDDEN = 5120
EPS = 1e-6

K_CHUNK = 128
BATCH_TILE = 16


def build_qwen3_final_rms_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    eps: float = EPS,
):
    if batch % BATCH_TILE != 0:
        raise ValueError(
            f"batch ({batch}) must be a multiple of BATCH_TILE ({BATCH_TILE}); "
            "pad inputs at the call site."
        )
    if hidden_size % K_CHUNK != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be a multiple of K_CHUNK ({K_CHUNK})."
        )
    BATCH_CFG = batch
    HIDDEN_CFG = hidden_size

    HIDDEN_BLOCKS = HIDDEN_CFG // K_CHUNK
    hidden_inv = 1.0 / HIDDEN_CFG

    @pl.program
    class Qwen3FinalRMS:
        @pl.function(type=pl.FunctionType.Opaque)
        def final_rms(
            self,
            x: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            gamma: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            with pl.at(level=pl.Level.CORE_GROUP):
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(x, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(
                                pl.row_sum(pl.mul(x_chunk, x_chunk)),
                                [1, BATCH_TILE],
                            ),
                        )
                    inv_rms = pl.reshape(
                        pl.rsqrt(pl.add(pl.mul(sq_sum, hidden_inv), eps)),
                        [BATCH_TILE, 1],
                    )

                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(x, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        g = pl.slice(gamma, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(x_chunk, inv_rms),
                            g,
                        )
                        out = pl.assemble(
                            out, pl.cast(normed, target_type=pl.BF16), [b0, k0]
                        )
            return out

    return Qwen3FinalRMS
