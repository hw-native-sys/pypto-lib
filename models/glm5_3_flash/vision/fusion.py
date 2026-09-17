# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scatter the merged vision embeddings into the text embedding stream.

The processor expands each image into ``image_token_id = 154854`` placeholders
(``154855`` for video) between ``image_start_token_id = 154830`` and
``image_end_token_id = 154831``. This kernel finds those rows and overwrites them
with the merged vision tokens, in order, leaving every other row untouched.

The count must match exactly: one placeholder per merged vision token. A mismatch
means the host-side ``smart_resize`` and ``get_number_of_image_patches`` disagree
with the tower's merge arithmetic, which is a load-time error, not a runtime one.
The vision tower is **staged after the text backbone**. The A3 recipe serves this
checkpoint text-only unless ``--limit-mm-per-prompt`` is set, the tower is 0.53 G
parameters against the text model's 320 G, and vLLM Ascend builds it with
``quant_config=None`` because the checkpoint ships it BF16 and inheriting the
global quant config yields NaN image features. It runs in prefill only, once per
request, and never in decode.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, HC_MULT, T_DYN


def golden_vision_fusion(
    inputs_embeds: torch.Tensor,
    vision_embeds: torch.Tensor,
    placeholder_mask: torch.Tensor,
) -> torch.Tensor:
    if int(placeholder_mask.sum()) != vision_embeds.shape[0]:
        raise ValueError("placeholder count does not match the merged vision token count")
    fused = inputs_embeds.clone()
    fused[placeholder_mask] = vision_embeds.to(inputs_embeds.dtype)
    return fused


@pl.jit.inline
def vision_fusion(
    vision_embeds: pl.Tensor[[T_DYN, D], pl.BF16],
    placeholder_rows: pl.Tensor[[T_DYN], pl.INT32],
    hidden_streams: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    raise NotImplementedError("vision fusion kernel body is assigned independently")


__all__ = [
    "golden_vision_fusion",
    "vision_fusion",
]
