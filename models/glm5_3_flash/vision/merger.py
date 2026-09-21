# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Spatial merge and the projection into the text width.

``spatial_merge_size = 2``, so four neighbouring patches become one token. The
checkpoint has a ``visual.downsample`` convolution before the merger and then
``visual.merger`` = ``proj`` + a clamped-SwiGLU pair (``gate_proj`` / ``up_proj`` /
``down_proj``) at ``projection_intermediate_size = 10240`` with a
``post_projection_norm`` that carries a **bias**, i.e. a LayerNorm rather than the
RMSNorm used everywhere else.

The output width is ``out_hidden_size = 4096``, matching the text model, so the
merged tokens can be scattered straight into the embedding stream.
The vision tower is **staged after the text backbone**. The A3 recipe serves this
checkpoint text-only unless ``--limit-mm-per-prompt`` is set, the tower is 0.53 G
parameters against the text model's 320 G, and vLLM Ascend builds it with
``quant_config=None`` because the checkpoint ships it BF16 and inheriting the
global quant config yields NaN image features. It runs in prefill only, once per
request, and never in decode.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, T_DYN
from models.glm5_3_flash.vision.config import VISION_HIDDEN, VISION_MERGE, VISION_PROJ_INTER


def golden_vision_merger(
    x: torch.Tensor,
    downsample_weight: torch.Tensor,
    downsample_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    post_norm_bias: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("vision merger golden is assigned with the kernel")


@pl.jit.inline
def vision_merger(
    x: pl.Tensor[[T_DYN, VISION_MERGE * VISION_MERGE, VISION_HIDDEN], pl.BF16],
    proj_weight: pl.Tensor[[D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    post_norm_weight: pl.Tensor[[D], pl.BF16],
    post_norm_bias: pl.Tensor[[D], pl.BF16],
    w_gate: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_up: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_down: pl.Tensor[[D, VISION_PROJ_INTER], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    raise NotImplementedError("vision merger kernel body is assigned independently")


__all__ = [
    "golden_vision_merger",
    "vision_merger",
]
