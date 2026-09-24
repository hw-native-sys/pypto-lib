# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The indexer's own paged state cache: 256 values per token per DSA layer.

``Glm5NextTextIndexer.forward`` packs ``[key(128), gate_scores(128), valid(1)]``
into one row and pushes it through ``past_key_values.update_indexer``. The valid
channel is the cached form of the attention mask, and it exists only to let pooling
start at the first real token of a **left-padded dense** batch. A paged cache has no
left padding, so this ABI drops it and derives validity from the request length —
which is also what vLLM Ascend does (``AscendIndexerKPoolStateSpec`` has
``head_size = 2 * 128``). A kernel that ever sees a padded dense layout must put it
back.

vLLM Ascend splits this into two specs — an ``AscendMLAAttentionSpec`` with
``head_size=128`` for the keys and an ``AscendIndexerKPoolStateSpec`` with
``block_size=4``, ``head_size=256`` and ``dtype=float32`` for the pooled state — and
excludes both from the generic page-size alignment so they form their own page
class. Deciding whether to keep that split or store the raw 257-wide row is the
first thing the assignee has to settle with the cache manager.

The ABI below stores FP32, which is what vLLM Ascend enforces for the pooled state.
The a2a3 sibling port quantizes its own indexer cache to INT8 with a per-row FP32
scale (``docs/models/deepseek_v4_flash_mtp/index.md``), which would take this from
about 3 KB to 0.8 KB per token per rank. Treat that as a follow-up, not a default:
the pooled key feeds a ``relu``-gated score whose sensitivity to INT8 has not been
measured for this checkpoint.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, INDEX_DIM, INDEX_STATE_WIDTH
from models.glm5_3_flash.config import TABLE_DYN, T_DYN


def golden_indexer_cache_write(
    cache: torch.Tensor,
    index_k: torch.Tensor,
    gate_scores: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Scatter one step's packed indexer state, skipping ``-1`` slots.

    A slot of ``-1`` marks a row that owns no cache position — a padded row in a
    packed decode batch — and is skipped, exactly as
    :func:`models.glm5_3_flash.mla_cache.golden_mla_cache_write` skips it for the
    latent pool. ``index_slots`` and ``mla_slots`` are the same kind of quantity, both
    built by :func:`models.glm5_3_flash.metadata.paged_slots`, so without the guard a
    negative slot would be a wrapped index into the tail of the pool. The kernel body
    owes the same guard.
    """
    packed = torch.cat([index_k, gate_scores], dim=-1)
    updated = cache.clone()
    written = slots >= 0
    updated[slots.to(torch.long)[written]] = packed.to(cache.dtype)[written]
    return updated


@pl.jit.inline
def indexer_cache_write(
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
):
    raise NotImplementedError("indexer cache write kernel body is assigned independently")


__all__ = [
    "golden_indexer_cache_write",
    "indexer_cache_write",
]
