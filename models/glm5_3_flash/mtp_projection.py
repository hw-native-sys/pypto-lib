# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The MTP layer's embedding/hidden fusion (checkpoint layer 45).

    h = eh_proj(concat(enorm(embedding), hnorm(hidden)))        [T, 2D] -> [T, D]

``enorm``, ``hnorm`` and ``eh_proj`` are unique to the MTP layer; everything after
the fusion is an ordinary MLA + indexer + sparse-MoE layer. The layer also ships its
**own** ``embed_tokens.weight`` and ``shared_head.head.weight`` as distinct index
entries rather than reusing the target's — whether the values are tied is
UNVERIFIED, but the loader has to handle two more tensors than the name
"shared_head" suggests.

Two things separate this layer from the backbone: it has **no** ``hc_*`` weights, so
it uses a plain residual rather than the four-stream mHC, and
``index_share_for_mtp_iteration`` lets it reuse the target step's top-k indices
instead of running its own selection.

``num_speculative_tokens`` is 3 in the A3 recipe, and that recipe also sets
``enforce_eager: true`` because GLM-5.3-Flash does not support graph-mode
speculative decoding.

**Unresolved: ``rot.weight``.** The deployment checkpoint sets ``is_rot_used: true``
and ships a top-level ``rot.weight``, BF16 ``[4096, 4096]``, the only tensor in the
weight index with no entry in ``quant_model_description.json``. vLLM Ascend's
``AscendDeepSeekMTP`` consumes exactly this name and flag, applying it to the
previous hidden state **before** ``hnorm``; but GLM routes to ``Glm5NextMTP``
instead, which has no ``rot`` and silently drops the tensor. So either this layer
owes a ``[D, D]`` GEMM that the reference port is missing, or ``eh_proj`` subsumes
it. Settle that before implementing, because it cannot be folded away: ``hnorm`` is
a non-linear RMSNorm sitting between ``rot`` and ``eh_proj``, so only the constant
factor would absorb.

Measured, if it is needed: ``rot.weight`` is not a rotation. It is symmetric with
``R[i,j] = g(i XOR j)`` — verified bit-exact on 11 rows spanning the full index
range — i.e. a per-channel diagonal scaling in the Hadamard basis, determined
entirely by its 4096-element first row. A Walsh-Hadamard transform, a scale, and a
second transform costs about 340x fewer operations than the dense GEMM.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, T_DYN


def golden_mtp_projection(
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    w_eh: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("MTP projection golden is assigned with the kernel")


@pl.jit.inline
def mtp_projection(
    embedding: pl.Tensor[[T_DYN, D], pl.BF16],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    enorm_weight: pl.Tensor[[D], pl.BF16],
    hnorm_weight: pl.Tensor[[D], pl.BF16],
    w_eh: pl.Tensor[[D, 2 * D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    raise NotImplementedError("MTP projection kernel body is assigned independently")


__all__ = [
    "golden_mtp_projection",
    "mtp_projection",
]
