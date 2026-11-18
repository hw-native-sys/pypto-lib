# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

T_DYN = pl.dynamic("NATIVE_M_MARKOV_HEAD_T_DYN")

VOCAB_DYN = pl.dynamic("NATIVE_M_MARKOV_HEAD_VOCAB_DYN")

MARKOV_RANK = 256

T_TILE = 16

VOCAB_TILE = 128

SPMD_BLOCKS = 48


@pl.jit.inline
def markov_head(
    token_ids: pl.Tensor[[T_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    logits_bias: pl.Tensor[[T_DYN, VOCAB_DYN], pl.FP32],
    markov_embed: pl.Tensor[[T_DYN, MARKOV_RANK], pl.BF16],
    token_ids_ready_tid: pl.Scalar[pl.TASK_ID] = None,
):
    t_dim = pl.tensor.dim(token_ids, 0)
    t_linear = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE

    with pl.spmd(
        t_dim,
        name_hint="markov_embedding",
        deps=[token_ids_ready_tid],
        allow_early_resolve=True,
    ) as embedding_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(markov_w1, pl.CachePolicy.BYPASS)
        token_idx = pl.tile.get_block_idx()
        token_id = pl.read(token_ids, [token_idx])
        token_row = pl.cast(token_id, target_type=pl.INDEX)
        markov_embed[token_idx : token_idx + 1, 0:MARKOV_RANK] = markov_w1[
            token_row : token_row + 1, 0:MARKOV_RANK
        ]

    vocab_dim = pl.tensor.dim(markov_w2, 0)
    work_items = (t_linear // T_TILE) * (vocab_dim // VOCAB_TILE)
    with pl.spmd(
        SPMD_BLOCKS,
        name_hint="markov_logits",
        deps=[embedding_tid],
    ) as logits_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(markov_w2, pl.CachePolicy.BYPASS)
        block = pl.tile.get_block_idx()
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            t0 = (work_idx // (vocab_dim // VOCAB_TILE)) * T_TILE
            vocab0 = (work_idx % (vocab_dim // VOCAB_TILE)) * VOCAB_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            embed_bf16 = pl.slice(
                markov_embed, [T_TILE, MARKOV_RANK], [t0, 0], valid_shape=[valid_rows, MARKOV_RANK]
            )
            weight_bf16 = markov_w2[vocab0 : vocab0 + VOCAB_TILE, 0:MARKOV_RANK]
            logits_tile = pl.matmul(embed_bf16, weight_bf16, b_trans=True, out_dtype=pl.FP32)
            logits_valid = pl.set_validshape(logits_tile, valid_rows, VOCAB_TILE)
            logits_bias[t0 : t0 + T_TILE, vocab0 : vocab0 + VOCAB_TILE] = logits_valid

    return logits_bias, markov_embed, embedding_tid, logits_tid
