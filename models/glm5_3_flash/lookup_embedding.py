# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vocab-parallel embedding lookup, and the mHC stream seed.

``vocab_size = 154880`` over TP16 gives 9680 rows per rank; a token whose id falls
outside the rank's shard contributes zero and the all-reduce completes it. The
embedding table is never quantized.

The result is immediately replicated into the four hyper-connection streams
(``inputs_embeds.unsqueeze(2).expand(-1, -1, 4, -1)``), so this kernel can emit the
``[T, 4, D]`` stream directly and save a copy — but note the stream is BF16, so the
replication is a view-shaped broadcast, not a cast.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, HC_MULT, LOCAL_VOCAB, T_DYN


def golden_lookup_embedding(
    token_ids: torch.Tensor,
    table: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    raise NotImplementedError("embedding golden is assigned with the kernel")


@pl.jit.inline
def lookup_embedding(
    token_ids: pl.Tensor[[T_DYN], pl.INT32],
    table: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
    vocab_start: pl.Scalar[pl.INT32],
    hidden_streams: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    raise NotImplementedError("embedding kernel body is assigned independently")


__all__ = [
    "golden_lookup_embedding",
    "lookup_embedding",
]
