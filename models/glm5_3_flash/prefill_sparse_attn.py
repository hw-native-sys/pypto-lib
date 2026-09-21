# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sparse NoPE MLA attention over the indexer's selected rows, prefill path.

Each query row carries up to ``TOPK_INDEX_WIDTH = 2051`` int32 cache rows, padded
with ``-1``. The reference turns them into a dense boolean mask and runs SDPA, which
is only tractable because the mask is built with ``scatter_add``; on device the
kernel gathers the selected latent rows instead and runs an online-softmax
attention over them.

Scale is ``qk_head_dim ** -0.5`` = ``256 ** -0.5``; ``v_head_dim`` is 256, so the
per-head output is 256 wide and the heads flatten to ``[T, LOCAL_H * 256]``.

Causality and padding are already folded into the index list by the indexer — a row
that should not be visible is ``-1`` — so this kernel must not re-apply a causal
mask, only honour the ``-1`` sentinel.

Like the decode path, the indexer emits **logical per-request positions**, so the
block table is part of this kernel's ABI: resolving it host-side would mean a
per-layer index translation outside the kernel.

**Prior art.** ``ops/pypto_python/impl/sparse_compress_flash_attention_pypto.py`` in
cann-recipes-infer implements sparse flash attention over selected rows on this
hardware generation. It targets the older ``pypto.Tensor`` / ``pypto_impl``
frontend, which this repo's pinned pypto does not export, so treat it as an
algorithm and tiling reference rather than code.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, B_DYN, KV_LORA, LOCAL_H, QK_DIM
from models.glm5_3_flash.config import BLOCK_TABLE_DYN, TABLE_DYN, TOPK_INDEX_WIDTH, T_DYN, V_DIM


def golden_prefill_sparse_attn(
    query: torch.Tensor,
    latent_cache: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("prefill sparse attention golden is assigned with the kernel")


@pl.jit.inline
def prefill_sparse_attn(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    w_v: pl.Tensor[[LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
):
    raise NotImplementedError("prefill sparse attention kernel body is assigned independently")


__all__ = [
    "golden_prefill_sparse_attn",
    "prefill_sparse_attn",
]
