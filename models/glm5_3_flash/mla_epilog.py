# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The MLA output projection and its TP16 reduction.

Prefill projects ``[T, LOCAL_H * V_DIM]`` through the plain ``o_proj``; decode
projects ``[T, LOCAL_H * KV_LORA]`` through the value-absorbed ``o_proj`` produced
by :mod:`models.glm5_3_flash.mla_absorb`. Both are row-parallel over heads, so the
result is a partial sum that needs an all-reduce across the 16 ranks before mHC
folds it back into the residual stream.

``o_proj`` carries a ``weight_scale_inv`` in the released FP8 checkpoint but is
**BF16** in the deployment W8A8 one (``[4096, 8192]``, marked ``FLOAT`` in
``quant_model_description.json``), so neither path quantizes here.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, KV_LORA, LOCAL_H, T_DYN, V_DIM


def golden_mla_epilog_prefill(
    attn_out: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    """Head-space projection. ``o_proj`` is BF16 in the deployment checkpoint."""
    flattened = attn_out.reshape(*attn_out.shape[:-2], -1)
    return torch.nn.functional.linear(flattened.float(), w_o.float())


def golden_mla_epilog_decode(
    attn_out: torch.Tensor,
    w_o_absorbed: torch.Tensor,
) -> torch.Tensor:
    """The absorbed path: ``kv_b_proj``'s value half is folded in, so BF16."""
    flattened = attn_out.reshape(*attn_out.shape[:-2], -1)
    return torch.nn.functional.linear(flattened.float(), w_o_absorbed.float())


@pl.jit.inline
def mla_epilog_prefill(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_H * V_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    raise NotImplementedError("MLA prefill epilog kernel body is assigned independently")


@pl.jit.inline
def mla_epilog_decode(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    w_o_absorbed: pl.Tensor[[D, LOCAL_H * KV_LORA], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    raise NotImplementedError("MLA decode epilog kernel body is assigned independently")


__all__ = [
    "golden_mla_epilog_decode",
    "golden_mla_epilog_prefill",
    "mla_epilog_decode",
    "mla_epilog_prefill",
]
