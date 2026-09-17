# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One vision encoder block's attention: qkv with bias, per-head norms, axial rope.

16 heads of 64 over width 1024. ``attention_bias`` is true, so ``qkv`` and ``proj``
both carry biases — unusual for this codebase, where every text projection is
bias-free. ``q_norm`` and ``k_norm`` are per-head RMSNorms applied before the 2D
axial rope.

The attention itself is **non-causal and variable-length**: images arrive packed as
segments and every patch attends to every other patch of its own image, with no
mask between a patch and a later one. Nothing in ``models/deepseek_v4_*`` or
``models/qwen3_14b`` is a drop-in — this repo's entire attention inventory is causal
or paged-decode.
The vision tower is **staged after the text backbone**. The A3 recipe serves this
checkpoint text-only unless ``--limit-mm-per-prompt`` is set, the tower is 0.53 G
parameters against the text model's 320 G, and vLLM Ascend builds it with
``quant_config=None`` because the checkpoint ships it BF16 and inheriting the
global quant config yields NaN image features. It runs in prefill only, once per
request, and never in decode.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import B_DYN, T_DYN
from models.glm5_3_flash.vision.config import VISION_HEAD_DIM, VISION_HIDDEN


def golden_vision_attention(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    segment_start_loc: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("vision attention golden is assigned with the kernel")


@pl.jit.inline
def vision_attention(
    x: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[3 * VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    qkv_bias: pl.Tensor[[3 * VISION_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[VISION_HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[VISION_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    proj_weight: pl.Tensor[[VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    proj_bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    segment_start_loc: pl.Tensor[[B_DYN + 1], pl.INT32],
    output: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
):
    raise NotImplementedError("vision attention kernel body is assigned independently")


@pl.jit.inline
def vision_axial_rope_table(
    position_ids: pl.Tensor[[T_DYN, 2], pl.INT32],
    inv_freq: pl.Tensor[[VISION_HEAD_DIM // 4], pl.FP32],
    cos: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
):
    raise NotImplementedError("vision rope table kernel body is assigned independently")


__all__ = [
    "golden_vision_attention",
    "vision_attention",
    "vision_axial_rope_table",
]
