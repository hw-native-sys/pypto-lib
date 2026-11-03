# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Look up one vocabulary shard and replicate BF16 embeddings into mHC streams."""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import HC_MULT, LOCAL_VOCAB, D

T_DYN = pl.dynamic('GLM53_T_DYN')

D_TILE = 1024


@pl.jit.inline
def lookup_embedding(
    token_ids: pl.Tensor[[T_DYN], pl.INT32],
    table: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
    vocab_start: pl.Scalar[pl.INT32],
    hidden_streams: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    """Write local-shard rows or zeros; the caller reduces across vocabulary shards."""
    token_count = pl.tensor.dim(token_ids, 0)
    vocab_stop = vocab_start + LOCAL_VOCAB
    streams_flat = pl.reshape(hidden_streams, [token_count * HC_MULT, D])
    for block in pl.spmd(48, name_hint="lookup_embedding"):
        for work in pl.range(block, token_count * (D // D_TILE), 48):
            token = work // (D // D_TILE)
            hidden_offset = (work % (D // D_TILE)) * D_TILE
            token_id = pl.read(token_ids, [token])
            chunk = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            if token_id >= vocab_start:
                if token_id < vocab_stop:
                    local_id = pl.cast(token_id - vocab_start, pl.INDEX)
                    chunk = table[local_id : local_id + 1, hidden_offset : hidden_offset + D_TILE]
            for stream in pl.range(HC_MULT):
                row = token * HC_MULT + stream
                streams_flat[row : row + 1, hidden_offset : hidden_offset + D_TILE] = chunk
    return hidden_streams


def golden_lookup_embedding(
    token_ids: torch.Tensor,
    table: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    """Return replicated local-shard rows with exact zeros for nonlocal IDs."""
    local_ids = token_ids.to(torch.int64) - int(vocab_start)
    local = (local_ids >= 0) & (local_ids < table.shape[0])
    output = torch.zeros(token_ids.numel(), HC_MULT, D, dtype=table.dtype, device=table.device)
    rows = table.index_select(0, local_ids[local])
    output[local] = rows.unsqueeze(1).expand(-1, HC_MULT, -1)
    return output
