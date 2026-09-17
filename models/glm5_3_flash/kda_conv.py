# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The KDA short depthwise causal convolution, prefill and decode.

``short_conv_kernel_size = 4``, depthwise over every q/k/v channel, followed by the
SiLU activation. The reference concatenates q, k and v and runs one grouped
``Conv1d`` over ``3 * 8192`` channels, but **the checkpoint stores three separate
convs** — ``q_conv1d.weight``, ``k_conv1d.weight``, ``v_conv1d.weight`` — so the
weight loader either stacks them or the kernel takes three.

Prefill consumes the request's 3-token conv state, convolves the packed sequence
and writes back the last 3 tokens. Decode is a pure state update: shift the window,
insert the new row, and take a 4-tap dot product. Both paths address the state by
the request row, not by position, so the conv state never grows with context.

Both goldens take ``state_rows`` as well, because the state pools are addressed
by request row rather than by batch position: without it a golden cannot model a
reordered or paged batch, which is exactly the case a state bug hides in.

Both entries emit **q, k and v separately**, already viewed as
``[T, LOCAL_KDA_H, KDA_DIM]``, because that is what ``prefill_kda`` and
``decode_kda`` consume. Splitting inside the kernel avoids materialising the
merged ``[T, 3 * LOCAL_KDA_QKV_DIM]`` row twice, and it keeps the q/k L2
normalisation — which the delta-rule kernels apply in FP32 on their own input —
from having to re-view the tensor.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import B_DYN, KDA_CONV_K, KDA_DIM, LOCAL_KDA_H
from models.glm5_3_flash.config import LOCAL_KDA_QKV_DIM, T_DYN


CONV_DIM = 3 * LOCAL_KDA_QKV_DIM


def golden_kda_conv_prefill(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("KDA prefill conv golden is assigned with the kernel")


def golden_kda_conv_decode(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("KDA decode conv golden is assigned with the kernel")


@pl.jit.inline
def kda_conv_prefill(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[CONV_DIM, KDA_CONV_K], pl.BF16],
    conv_state: pl.Tensor[[B_DYN, CONV_DIM, KDA_CONV_K - 1], pl.BF16],
    query_start_loc: pl.Tensor[[B_DYN + 1], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    raise NotImplementedError("KDA prefill conv kernel body is assigned independently")


@pl.jit.inline
def kda_conv_decode(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[CONV_DIM, KDA_CONV_K], pl.BF16],
    conv_state: pl.Tensor[[B_DYN, CONV_DIM, KDA_CONV_K - 1], pl.BF16],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    raise NotImplementedError("KDA decode conv kernel body is assigned independently")


__all__ = [
    "CONV_DIM",
    "golden_kda_conv_decode",
    "golden_kda_conv_prefill",
    "kda_conv_decode",
    "kda_conv_prefill",
]
