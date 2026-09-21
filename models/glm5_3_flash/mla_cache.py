# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The paged 512-wide latent cache the 12 MLA layers share.

One row per token per layer: ``kv_a_layernorm(kv_a_proj_with_mqa(x))``, width
``kv_lora_rank = 512``, with ``num_kv_heads = 1``, so the cache is **not** TP
sharded — every rank holds the same latent. Pages are ``BLOCK_SIZE = 128`` tokens.

The write is a scatter by ``ForwardMetadata.mla_slots``; the read is a paged gather
driven by the indexer's selected rows, which is why the two live in separate files.

At BF16 this is ``12 layers x 512 x 2`` = 12.3 KB per token per rank, and it is the
larger half of the hybrid cache budget.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, KV_LORA, TABLE_DYN, T_DYN


def golden_mla_cache_write(
    cache: torch.Tensor,
    latent: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    updated = cache.clone()
    updated[slots.to(torch.long)] = latent.to(cache.dtype)
    return updated


@pl.jit.inline
def mla_cache_write(
    latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
):
    raise NotImplementedError("MLA cache write kernel body is assigned independently")


__all__ = [
    "golden_mla_cache_write",
    "mla_cache_write",
]
