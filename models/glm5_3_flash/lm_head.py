# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vocab-parallel logits and the cross-rank argmax.

``lm_head`` is untied from the embedding (``tie_word_embeddings`` is false) and is
one of the modules the checkpoint never quantizes. At TP16 each rank produces 9680
of the 154880 logits, so greedy sampling is a local argmax followed by an
all-reduce over ``(value, global_index)`` pairs rather than a full logit gather —
gathering 154880 BF16 logits per row costs 310 KB of collective per token.

Only the rows in ``ForwardMetadata.logit_row_indices`` need logits in prefill; in
decode every row does, including the ``MTP_SPEC_TOKENS`` draft rows.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, LOCAL_VOCAB, LOGIT_ROWS_DYN, T_DYN


def golden_lm_head(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("lm_head golden is assigned with the kernel")


@pl.jit.inline
def lm_head(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    row_indices: pl.Tensor[[LOGIT_ROWS_DYN], pl.INT32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
    local_max_value: pl.Tensor[[LOGIT_ROWS_DYN], pl.FP32],
    local_argmax: pl.Tensor[[LOGIT_ROWS_DYN], pl.INT32],
):
    raise NotImplementedError("lm_head kernel body is assigned independently")


__all__ = [
    "golden_lm_head",
    "lm_head",
]
