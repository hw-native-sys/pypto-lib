# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One vision encoder block's MLP and its two RMSNorms.

Clamped SwiGLU at ``1024 -> 4096 -> 1024`` with the same ``swiglu_limit = 10.0`` as
the text model, but **with biases** on all three projections. ``norm1`` and
``norm2`` are the block's pre-attention and pre-MLP RMSNorms.
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
from models.glm5_3_flash.golden import swiglu
from models.glm5_3_flash.vision.config import VISION_HIDDEN, VISION_INTER


def golden_vision_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    b_gate: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
) -> torch.Tensor:
    hidden = swiglu(
        torch.nn.functional.linear(x, w_gate, b_gate),
        torch.nn.functional.linear(x, w_up, b_up),
    )
    return torch.nn.functional.linear(hidden.to(x.dtype), w_down, b_down)


@pl.jit.inline
def vision_mlp(
    x: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_gate: pl.Tensor[[VISION_INTER], pl.BF16],
    w_up: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_up: pl.Tensor[[VISION_INTER], pl.BF16],
    w_down: pl.Tensor[[VISION_HIDDEN, VISION_INTER], pl.BF16],
    b_down: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
):
    raise NotImplementedError("vision MLP kernel body is assigned independently")


__all__ = [
    "golden_vision_mlp",
    "vision_mlp",
]
