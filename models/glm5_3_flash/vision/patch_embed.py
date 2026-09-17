# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vision patch embedding: a degenerate Conv3d that is really one GEMM.

``patch_size = 14``, ``temporal_patch_size = 2``, ``in_channels = 3``, so the
convolution kernel equals its stride and covers a whole patch. The checkpoint
stores ``visual.patch_embed.proj.weight`` as ``[1024, 3, 2, 14, 14]``; flattened to
``[1024, 1176]`` it is a plain linear projection of each flattened patch, plus the
bias.

The real work here is the weight-loader reshape and agreeing the patch ordering
with the host-side processor, not the arithmetic.
The vision tower is **staged after the text backbone**. The A3 recipe serves this
checkpoint text-only unless ``--limit-mm-per-prompt`` is set, the tower is 0.53 G
parameters against the text model's 320 G, and vLLM Ascend builds it with
``quant_config=None`` because the checkpoint ships it BF16 and inheriting the
global quant config yields NaN image features. It runs in prefill only, once per
request, and never in decode.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import T_DYN
from models.glm5_3_flash.vision.config import VISION_HIDDEN, VISION_PATCH_IN


def golden_vision_patch_embed(
    patches: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.linear(patches, weight, bias)


@pl.jit.inline
def vision_patch_embed(
    patches: pl.Tensor[[T_DYN, VISION_PATCH_IN], pl.BF16],
    weight: pl.Tensor[[VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
):
    raise NotImplementedError("vision patch embed kernel body is assigned independently")


__all__ = [
    "golden_vision_patch_embed",
    "vision_patch_embed",
]
