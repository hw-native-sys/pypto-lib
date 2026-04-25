# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B LM head program.

Projects the final hidden state to vocabulary logits:

    logits = hidden @ lm_head_weight.T

Shapes:
    hidden:         [BATCH, HIDDEN] BF16
    lm_head_weight: [VOCAB, HIDDEN] BF16   (HuggingFace nn.Linear layout)
    out:            [BATCH, VOCAB]  FP32

VOCAB is expected padded to a multiple of VOCAB_CHUNK (default 64).

Weight is stored as [VOCAB, HIDDEN] and the matmul uses ``b_trans=True``
instead of [HIDDEN, VOCAB]. On a2a3 the ND->NZ MatTile load path requires
dim-3 stride to fit in uint16_t (<= 65535); a [HIDDEN, VOCAB] layout would
have row stride VOCAB = 152064 and silently truncate. Using [VOCAB, HIDDEN]
makes the row stride HIDDEN = 5120 — within the limit — and matches the
HuggingFace ``nn.Linear`` checkpoint layout.
"""

import pypto.language as pl

BATCH = 16
HIDDEN = 5120
VOCAB = 152064

K_CHUNK = 128
VOCAB_CHUNK = 64
BATCH_TILE = 16


def build_qwen3_lm_head_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    vocab_size: int = VOCAB,
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
    if vocab_size % VOCAB_CHUNK != 0:
        raise ValueError(
            f"vocab_size ({vocab_size}) must be a multiple of VOCAB_CHUNK ({VOCAB_CHUNK}); "
            "pad vocab at the call site."
        )
    BATCH_CFG = batch
    HIDDEN_CFG = hidden_size
    VOCAB_CFG = vocab_size

    HIDDEN_BLOCKS = HIDDEN_CFG // K_CHUNK
    VOCAB_BLOCKS = VOCAB_CFG // VOCAB_CHUNK

    @pl.program
    class Qwen3LMHead:
        @pl.function(type=pl.FunctionType.Opaque)
        def lm_head(
            self,
            hidden: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            lm_head_weight: pl.Tensor[[VOCAB_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, VOCAB_CFG], pl.FP32],
        ) -> pl.Tensor[[BATCH_CFG, VOCAB_CFG], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    for ob in pl.parallel(VOCAB_BLOCKS, chunk=8):
                        o0 = ob * VOCAB_CHUNK
                        h0 = pl.slice(hidden, [BATCH_TILE, K_CHUNK], [b0, 0])
                        w0 = pl.slice(
                            lm_head_weight, [VOCAB_CHUNK, K_CHUNK], [o0, 0]
                        )
                        acc = pl.matmul(h0, w0, out_dtype=pl.FP32, b_trans=True)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            h_chunk = pl.slice(
                                hidden, [BATCH_TILE, K_CHUNK], [b0, k0]
                            )
                            w_chunk = pl.slice(
                                lm_head_weight, [VOCAB_CHUNK, K_CHUNK], [o0, k0]
                            )
                            acc = pl.matmul_acc(acc, h_chunk, w_chunk, b_trans=True)
                        out = pl.assemble(out, acc, [b0, o0])
            return out

    return Qwen3LMHead
