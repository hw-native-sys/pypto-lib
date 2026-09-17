# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sparse NoPE MLA attention, decode path, over the absorbed latent query.

Decode attends in **latent space**: the query has already had the key half of
``kv_b_proj`` folded in, so it is ``[LOCAL_H, KV_LORA]`` and multiplies the cached
512-wide rows directly, with no per-token key expansion. The value half comes back
out through the absorbed ``o_proj``, so this kernel's output is ``[T, LOCAL_H,
KV_LORA]``, not ``[T, LOCAL_H, V_DIM]``.

Each of the ``1 + MTP_SPEC_TOKENS`` rows per request carries its own index list, so
the gather is ragged across rows even within one request. FULL_DECODE_ONLY graph
capture needs the shapes static, so the index list stays padded to
``TOPK_INDEX_WIDTH`` and the ``-1`` sentinel does the masking.

**The indexer emits logical per-request positions, not physical cache rows**, so the
block table is part of this kernel's ABI rather than something the host resolves.
Resolving it host-side would push a per-layer index translation into the captured
decode graph, which is exactly what FULL_DECODE_ONLY cannot absorb.

Donor: ``models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py`` (827 lines,
a2a3-tuned — its comments call out the AIC-count dispatch lanes and the L0C wall).
Its KV row width is 512, the same as ``kv_lora_rank``, so the 128x512 L1 tile, the
per-row block-table gather, the additive ``NEG_INF`` validity bias, the flash
partials and the online-softmax merge all transfer. Delete from it: the
sliding-window half of the gather, the ratio-4 compressed-slot rewrite (GLM's
indices are raw positions, already causality- and padding-filtered by the indexer),
the attention sink, the whole inverse-RoPE block, and the fused ``o_proj`` tail —
that last one belongs to :mod:`models.glm5_3_flash.mla_epilog`.

Watch the plan-stage compile: the donor unrolls ``T x SPARSE_BLOCKS`` work items,
which is 40 iterations there and 2176 here.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, B_DYN, KV_LORA, LOCAL_H
from models.glm5_3_flash.config import TABLE_DYN, TOPK_INDEX_WIDTH, T_DYN


def golden_decode_sparse_attn(
    absorbed_query: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("decode sparse attention golden is assigned with the kernel")


@pl.jit.inline
def decode_sparse_attn(
    absorbed_query: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
):
    raise NotImplementedError("decode sparse attention kernel body is assigned independently")


__all__ = [
    "decode_sparse_attn",
    "golden_decode_sparse_attn",
]
