# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Flash token embedding lookup for packed prefill and decode IDs.

Emits both the plain hidden row and the Hyper-Connections view: every block
entry point consumes ``[T, HC_MULT, D]`` FP32, and layer 0 feeds it HC_MULT
identical copies of the embedding row.
"""

import pypto.language as pl

from config import FLASH as M


# Dynamic shape variables.
T_DYN = pl.dynamic("LOOKUP_EMBEDDING_T_DYN")
VOCAB_DYN = pl.dynamic("LOOKUP_EMBEDDING_VOCAB_DYN")

# model config
D = M.hidden_size
HC_MULT = M.hc_mult

# tiling
HIDDEN_TILE = 1024
SPMD_BLOCKS = 48


def _lookup_embedding(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    hidden_states: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_hc: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    input_ids.bind_dynamic(0, T_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    hidden_states.bind_dynamic(0, T_DYN)
    x_hc.bind_dynamic(0, T_DYN)
    token_count = pl.tensor.dim(input_ids, 0)
    x_hc_flat = pl.reshape(x_hc, [token_count * HC_MULT, D])
    work_items = token_count * (D // HIDDEN_TILE)
    for block in pl.spmd(SPMD_BLOCKS, name_hint="lookup_embedding"):
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            token_idx = work_idx // (D // HIDDEN_TILE)
            hidden_block = work_idx % (D // HIDDEN_TILE)
            hidden_offset = hidden_block * HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            hidden_chunk = embed_weight[token_row : token_row + 1, hidden_offset : hidden_offset + HIDDEN_TILE]
            hidden_states[token_idx : token_idx + 1, hidden_offset : hidden_offset + HIDDEN_TILE] = hidden_chunk
            hc_chunk = pl.cast(hidden_chunk, target_type=pl.FP32, mode="none")
            for hc_idx in pl.range(HC_MULT):
                hc_row = token_idx * HC_MULT + hc_idx
                x_hc_flat[hc_row : hc_row + 1, hidden_offset : hidden_offset + HIDDEN_TILE] = hc_chunk

    return hidden_states, x_hc


lookup_embedding = pl.jit.inline(_lookup_embedding)
