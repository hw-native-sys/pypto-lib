# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The kpool DSA indexer, decode path: projections, pooling, scoring, selection.

One file owns the whole selection pipeline for this phase, the way
``models/deepseek_v4_flash_mtp/decode_indexer.py`` does. The stages are:

1. **Projections.** ``q = wq_b(q_resid)`` [T, 32, 128]; ``k = k_norm(wk(x))``
   [T, 128], and note ``k_norm`` carries a **bias**, unlike every other norm in
   this model; ``head_weights = weights_proj(x) * 32 ** -0.5`` [T, 32]; and
   ``gate_scores = index_kpool_compress_gate @ x`` [T, 128]. Every one of these
   weights is BF16 in the checkpoint.
2. **Pooling.** Group four consecutive cached tokens and take a learned weighted
   average: ``p = softmax(gate_scores + index_kpool_compress_ape)`` over the four,
   ``pool_key = sum(p * key)``. A pool is a candidate only when all four of its
   tokens are valid.
3. **Scoring.** ``scores = relu(q . pool_keys^T * 128 ** -0.5)`` then a weighted sum
   over the 32 heads. The ``relu`` before the head reduction is what makes this a
   lightning indexer and not a second attention: a head that disagrees contributes
   zero rather than a negative.
4. **Selection.** Top ``index_topk / index_kpool`` = 512 pools out of
   ``P = ceil(kv_len / 4)`` — 32768 at 128k context, 262144 at the 1M limit.
5. **Expansion.** Each selected pool becomes 4 raw cache rows; the incomplete tail
   pool is always appended (``index_kpool_always_select_tail``); the result is
   padded with ``-1`` to a fixed ``TOPK_INDEX_WIDTH`` = 2051 so FULL_DECODE_ONLY
   graph capture sees a static shape.

All 32 indexer heads live on **every** rank: the score sums over heads before the
top-k, so head-sharding would force a cross-rank reduction of partial scores on
every sparse layer, and this kernel is small enough that replication is cheaper.

**There is no rope here.** ``indexer_rope_interleave`` is set in ``config.json`` but
is a vestigial field inherited from the GLM-MoE-DSA base: ``Glm5NextTextConfig``
sets ``rope_parameters = AttributeError()``, the model forward passes
``position_embeddings=None``, and ``Glm5NextTextIndexer.forward`` never touches a
cos/sin table. Combined with ``qk_rope_head_dim = 0``, this model has no rope
anywhere, so none of the ``rope_tables.py`` machinery that ``deepseek_v4_pro`` and
``deepseek_v4_flash_mtp`` carry is needed.

**Donors, all a2a3 and all already run by the daily sweep.**
``models/deepseek_v4_flash_mtp/decode_indexer.py`` is the shape to port. Its
``_cp_topk512_query`` (prefill_indexer.py:358-418) is an exact top-512 over the same
262144-candidate cap as GLM's, because DeepSeek-V4-Flash's indexer is itself a
ratio-4 compressed selector. Two device facts recorded there cost real debugging
time: a narrow (256) sort **faults with 507018**, so the leaf stays wide (2048); and
the merge-stage list must match the leaf, because a 4096 stage on a 2048-score row
"lowers to an illegal AIV config".

What to delete from the donor: the INT8/Hadamard half (GLM feeds BF16 q against a
BF16 pooled cache, so the score is one ``pl.matmul(..., out_dtype=pl.FP32,
b_trans=True)`` with no scale plumbing) and the compressor. What to add: the pooling
stage, which has no donor anywhere, and a ``selected_valid`` output so the expansion
can invalidate whole 4-wide groups.

vLLM Ascend is no help: ``sparse_attn_indexer_kpool.py`` raises
``NotImplementedError`` and says the upstream is a set of CUDA kernels with "no NPU
equivalent yet".
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import B_DYN, BLOCK_SIZE, D, INDEX_DIM, INDEX_H
from models.glm5_3_flash.config import INDEX_KPOOL, INDEX_STATE_WIDTH, KPOOL_SELECT_K
from models.glm5_3_flash.config import POOLS_DYN, Q_LORA, TABLE_DYN, TOPK_INDEX_WIDTH, T_DYN


LEAF = 2048  # the donor's confirmed fault-free sort width on a2a3; 8192 also works
             # but needs the extra 4096 merge stage (see the module docstring)
PAIR_WIDTH = 2 * KPOOL_SELECT_K


def golden_indexer_proj(
    x: torch.Tensor,
    q_resid: torch.Tensor,
    w_q_b: torch.Tensor,
    w_k: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    w_weights: torch.Tensor,
    w_compress_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("indexer projection golden is assigned with the kernel")


@pl.jit.inline
def indexer_proj(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[INDEX_H * INDEX_DIM, Q_LORA], pl.BF16],
    w_k: pl.Tensor[[INDEX_DIM, D], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[INDEX_H, D], pl.BF16],
    w_compress_gate: pl.Tensor[[INDEX_DIM, D], pl.BF16],
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
):
    raise NotImplementedError("indexer projection kernel body is assigned independently")


def golden_indexer_kpool(
    packed_states: torch.Tensor,
    compress_ape: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the pooled keys, their raw token indices and their validity."""
    raise NotImplementedError("kpool compress golden is assigned with the kernel")



@pl.jit.inline
def indexer_kpool(
    packed_states: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.FP32],
    pool_count: pl.Tensor[[B_DYN], pl.INT32],
    tail_count: pl.Tensor[[B_DYN], pl.INT32],
    pool_keys: pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.BF16],
    pool_valid: pl.Tensor[[POOLS_DYN], pl.INT32],
):
    raise NotImplementedError("kpool decode compress kernel body is assigned independently")


def golden_indexer_score(
    index_q: torch.Tensor,
    pool_keys: torch.Tensor,
    head_weights: torch.Tensor,
    pool_visible: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("indexer score golden is assigned with the kernel")


@pl.jit.inline
def indexer_score(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    pool_keys: pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    pool_valid: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_last_position: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_position: pl.Tensor[[T_DYN], pl.INT32],
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
):
    raise NotImplementedError("indexer score kernel body is assigned independently")


PAIR_WIDTH = 2 * KPOOL_SELECT_K
LEAF = 2048  # the donor's confirmed fault-free sort width on a2a3; 8192 also works
             # but needs the extra 4096 merge stage (see the module docstring)


def golden_indexer_topk(
    index_scores: torch.Tensor,
    pool_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected pool ids and, beside them, which of them are real.

    A query whose visible pool count is below ``KPOOL_SELECT_K`` still fills the
    full width, so the selection carries padding. ``indexer_expand`` needs to
    know which entries are padding to blank whole four-wide groups, and the
    donor kernel emits indices only — this second output is the addition.
    """
    raise NotImplementedError("indexer top-k golden is assigned with the kernel")


@pl.jit.inline
def indexer_topk(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
):
    raise NotImplementedError("indexer top-k kernel body is assigned independently")


def golden_indexer_expand(
    selected_pools: torch.Tensor,
    pool_valid: torch.Tensor,
    tail_start: torch.Tensor,
    tail_count: torch.Tensor,
    kv_len: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("indexer expand golden is assigned with the kernel")


@pl.jit.inline
def indexer_expand(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    pool_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
):
    raise NotImplementedError("indexer expand kernel body is assigned independently")


__all__ = [
    "LEAF",
    "PAIR_WIDTH",
    "golden_indexer_expand",
    "golden_indexer_kpool",
    "golden_indexer_proj",
    "golden_indexer_score",
    "golden_indexer_topk",
    "indexer_expand",
    "indexer_kpool",
    "indexer_proj",
    "indexer_score",
    "indexer_topk",
]
