# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The kpool DSA indexer, prefill path: projections, scoring, selection.

The pooled-table half of the pipeline lives in
:mod:`models.glm5_3_flash.indexer_cache` under the scheme-B layout decided for
this module: the raw support table is scattered per token, closed pools are
compressed once into an INT8 pooled key table (plus a per-row FP32 dequant
scale) by pool-close events, and the scorer reads that table directly. What
remains here is the per-query half:

1. **Projections.** ``q = wq_b(q_resid)`` [T, 32, 128]; ``k = k_norm(wk(x))``
   [T, 128], where ``k_norm`` is a full **LayerNorm with eps 1e-6** carrying a
   bias — the reference ``Glm5NextTextIndexer`` builds ``nn.LayerNorm(head_dim,
   eps=1e-6)``, so unlike every other norm in this model it subtracts a mean;
   ``head_weights = weights_proj(x) * 32 ** -0.5`` [T, 32]; and
   ``gate_scores = compress_gate @ x`` [T, 128]. Every weight is BF16 in the
   checkpoint. All projection weights are declared K-major ``[in, out]`` — the
   loader transposes the checkpoint's ``nn.Linear`` layout once at load time,
   the way the a2a3 sibling port does, so no kernel slice needs a transposed
   matmul.
2. **Scoring.** ``relu(q . pool_key * 128 ** -0.5)`` per head, then a weighted
   sum over the 32 heads. The query head is Hadamard-128 rotated and quantized
   to INT8 first, and the pooled key rows arrive already Hadamard-rotated and
   INT8-quantized from the pool-close write, so the dot product runs as an
   exact INT32 ``mmad`` and both scales re-enter in FP32
   — the a2a3 retarget of the upstream FP8 numerics (see below). Each query row
   scores only its own request's pools: ``seg_start`` locates the request's
   segment in the compacted pool id space and ``pool_count`` is the causally
   visible pool count, ``(position + 1) // index_kpool`` clamped to the
   request's pool count. Lanes outside a query's visible window are
   **undefined scratch**: prefill group tiles store their whole 128-lane span
   and no init or post-mask pass cleans the rest, so only :func:`indexer_topk`
   — whose leaf loads clamp through their valid shape and ``PadValue.min``
   fill — may read this matrix.
3. **Selection.** Exact top ``index_topk / index_kpool`` = 512 pools per query
   over the same 262144-candidate cap as the donor, with a ``selected_valid``
   output so the expansion can tell padding from a real pick. Selections are
   emitted as **segment-local** pool ids — request-relative, so pool ``j`` maps
   to raw positions ``4j .. 4j+3`` without knowing the batch layout.
4. **Expansion.** Each selected pool becomes 4 raw logical positions; the
   incomplete tail pool is always appended (``index_kpool_always_select_tail``);
   the row is padded with ``-1`` to the fixed ``TOPK_INDEX_WIDTH`` = 2051.
   **The live positions are front packed**: every valid position precedes every
   ``-1``, so a row's padding is one suffix. This is an ABI guarantee both
   sparse attention kernels rely on — they test one lane to decide a whole
   128-wide block, or a whole row, carries no selection. The reference
   implementation leaves holes (it places the tail at a fixed column even for
   short requests); this kernel compacts, matching what the consumers require.

All 32 indexer heads live on **every** rank: the score sums over heads before
the top-k, so head-sharding would force a cross-rank reduction of partial
scores on every sparse layer, and this kernel is small enough that replication
is cheaper.

**There is no rope here.** ``indexer_rope_interleave`` is set in ``config.json``
but is a vestigial field inherited from the GLM-MoE-DSA base: the model forward
passes ``position_embeddings=None`` and ``Glm5NextTextIndexer.forward`` never
touches a cos/sin table. Combined with ``qk_rope_head_dim = 0``, this model has
no rope anywhere.

**Donors, all a2a3 and all already run by the daily sweep.**
``models/deepseek_v4_flash_mtp/prefill_indexer.py`` is the shape to port. Its
``_cp_topk512_query`` is an exact top-512 over the same 262144-candidate cap,
because DeepSeek-V4-Flash's indexer is itself a ratio-4 compressed selector, and
it is reused here almost verbatim. Two device facts recorded there cost real
debugging time: a narrow (256) sort **faults with 507018**, so the leaf stays
wide (2048); and the merge-stage list must match the leaf, because a 4096 stage
on a 2048-score row "lowers to an illegal AIV config".

What was deleted from the donor: the compressor with its ratio-4 slot rewrite
(superseded by the pooled-table writers) and rope. What was **kept**: the
Hadamard-128 rotation and the INT8 quantization of the indexer query — that
half is GLM's own numerics, not a donor quirk. The upstream kernel rotates each
128-wide query head by a Hadamard-128 and quantizes (FP8 e4m3 upstream, INT8
here because the a2a3 cube has no fp8 ``mmad``). The checkpoint ships no
Hadamard matrix; it is generated at load time, and the fixtures build the
Sylvester form scaled by ``128 ** -0.5``.

vLLM Ascend is no help beyond semantics: ``sparse_attn_indexer_kpool.py``
orchestrates a Triton scorer whose top-k is a fused ``aclnn`` call, so only its
contract was ported, not its kernels.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, FP32_NEG_INF
from models.glm5_3_flash.config import INDEX_BLOCKS_DYN, INDEX_DIM, INDEX_H
from models.glm5_3_flash.config import INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE
from models.glm5_3_flash.config import KPOOL_SELECT_K, Q_LORA, T_DYN
from models.glm5_3_flash.config import TOPK_INDEX_WIDTH
from models.glm5_3_flash.config import POOLS_DYN, POOL_PAGES_DYN
from models.glm5_3_flash.indexer_cache import POOL_SCALE_WIDTH
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, quantize_per_token_int8

# The sort tree: 2048-wide leaves are the donor's confirmed fault-free width, and
# the 64/256/1024 merge stages fully order one leaf. A 4096 stage on this row
# lowers to an illegal AIV config, so the running merge across leaves uses the
# two-tensor mrgsort instead of a wider stage.
LEAF = 2048
PAIR_WIDTH = 2 * KPOOL_SELECT_K

# k_norm is nn.LayerNorm(head_dim, eps=1e-6) in the reference — mean-subtracted,
# biased, and at a different eps than the backbone's RMSNorm.
K_NORM_EPS = 1e-6
SOFTMAX_SCALE = INDEX_DIM**-0.5
WEIGHTS_SCALE = INDEX_H**-0.5

PROJ_T_TILE = 16
PROJ_K_TILE = 128
Q_OUT_TILE = 256
QH_MM_TILE = 64
SCORE_C_TILE = 64
# The pooled table is addressed in INDEX_STATE_BLOCK_SIZE-row pages: the host
# hands the page table down (one physical base per page, the granularity
# :mod:`models.glm5_3_flash.metadata` maps pool slots at), so the gather walks
# whole pages instead of single scattered rows. If a future allocator can
# guarantee wider contiguous runs, widening the gather block is a purely local
# optimization here.
EXPAND_W_TILE = 256
EXPAND_TAIL_CHUNK = 8  # covers the last lanes in one aligned tile
# pl.const accepts literals only; assert keeps the hard-coded start honest.
assert TOPK_INDEX_WIDTH - EXPAND_TAIL_CHUNK == 2043
# pl.full takes float constants even for integer tensors; precompute because
# the tracer does not allow calling float() inside kernel bodies.


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
    """The four indexer projections in FP32, rounded once per output dtype.

    Weights arrive K-major ``[in, out]``; the reference's ``nn.Linear`` layout
    is the transpose, applied once at load time.
    """
    tokens = x.shape[0]
    heads = w_weights.shape[1]
    dim = w_k.shape[1]
    index_q = torch.nn.functional.linear(q_resid.float(), w_q_b.t().float())
    index_q = index_q.reshape(tokens, heads, dim).to(torch.bfloat16)

    projected = torch.nn.functional.linear(x.float(), w_k.t().float())
    mean = projected.mean(dim=-1, keepdim=True)
    centered = projected - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + K_NORM_EPS)
    index_k = (normalized * k_norm_weight.float() + k_norm_bias.float()).to(torch.bfloat16)

    head_weights = torch.nn.functional.linear(x.float(), w_weights.t().float()) * WEIGHTS_SCALE
    gate_scores = torch.nn.functional.linear(x.float(), w_compress_gate.t().float())
    return index_q, index_k, head_weights, gate_scores


@pl.jit.inline
def indexer_proj(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
):
    """Project one dispatch's rows: queries, biased LayerNorm keys, both gates.

    ``T`` must be a multiple of ``PROJ_T_TILE`` = 16; the scheduler pads a
    packed batch up to it and feeds the padding rows ordinary metadata. Four
    independent scopes, one cube accumulator each, so the compiler can lay them
    out concurrently.
    """
    t_dim = pl.tensor.dim(x, 0)
    q_flat = pl.reshape(index_q, [t_dim, INDEX_H * INDEX_DIM])
    q_blocks = (INDEX_H * INDEX_DIM) // Q_OUT_TILE

    for idx in pl.spmd((t_dim // PROJ_T_TILE) * q_blocks, name_hint="indexer_q_proj"):
        t_block = idx // q_blocks
        t0 = t_block * PROJ_T_TILE
        o0 = (idx - t_block * q_blocks) * Q_OUT_TILE
        q_acc = pl.create_tensor([PROJ_T_TILE, Q_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, Q_LORA // PROJ_K_TILE, stage=2):
            k0 = kb * PROJ_K_TILE
            q_acc = pl.matmul_acc(
                q_acc,
                q_resid[t0 : t0 + PROJ_T_TILE, k0 : k0 + PROJ_K_TILE],
                w_q_b[k0 : k0 + PROJ_K_TILE, o0 : o0 + Q_OUT_TILE],
                init_cond=(k0 == 0),
            )
        q_flat[t0 : t0 + PROJ_T_TILE, o0 : o0 + Q_OUT_TILE] = pl.cast(q_acc, pl.BF16, mode="rint")

    for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_k_proj"):
        t0 = idx * PROJ_T_TILE
        k_acc = pl.create_tensor([PROJ_T_TILE, INDEX_DIM], dtype=pl.FP32)
        for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            d0 = db * PROJ_K_TILE
            k_acc = pl.matmul_acc(
                k_acc,
                x[t0 : t0 + PROJ_T_TILE, d0 : d0 + PROJ_K_TILE],
                w_k[d0 : d0 + PROJ_K_TILE, 0:INDEX_DIM],
                init_cond=(d0 == 0),
            )
        mean = pl.mul(pl.row_sum(k_acc), 1.0 / INDEX_DIM)
        centered = pl.row_expand_sub(k_acc, mean)
        variance = pl.mul(pl.row_sum(pl.mul(centered, centered)), 1.0 / INDEX_DIM)
        rstd = pl.recip(pl.sqrt(pl.add(variance, K_NORM_EPS)))
        normed = pl.row_expand_mul(centered, rstd)
        weight_row = pl.cast(
            pl.reshape(k_norm_weight, [1, INDEX_DIM])[0:1, 0:INDEX_DIM], pl.FP32, mode="none"
        )
        bias_row = pl.cast(pl.reshape(k_norm_bias, [1, INDEX_DIM])[0:1, 0:INDEX_DIM], pl.FP32, mode="none")
        index_k[t0 : t0 + PROJ_T_TILE, 0:INDEX_DIM] = pl.cast(
            pl.col_expand_add(pl.col_expand_mul(normed, weight_row), bias_row),
            pl.BF16,
            mode="rint",
        )

    for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_weights_proj"):
        t0 = idx * PROJ_T_TILE
        w_acc = pl.create_tensor([PROJ_T_TILE, INDEX_H], dtype=pl.FP32)
        for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            d0 = db * PROJ_K_TILE
            w_acc = pl.matmul_acc(
                w_acc,
                x[t0 : t0 + PROJ_T_TILE, d0 : d0 + PROJ_K_TILE],
                w_weights[d0 : d0 + PROJ_K_TILE, 0:INDEX_H],
                init_cond=(d0 == 0),
            )
        head_weights[t0 : t0 + PROJ_T_TILE, 0:INDEX_H] = pl.mul(w_acc, WEIGHTS_SCALE)

    for idx in pl.spmd(t_dim // PROJ_T_TILE, name_hint="indexer_gate_proj"):
        t0 = idx * PROJ_T_TILE
        g_acc = pl.create_tensor([PROJ_T_TILE, INDEX_DIM], dtype=pl.FP32)
        for db in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            d0 = db * PROJ_K_TILE
            g_acc = pl.matmul_acc(
                g_acc,
                x[t0 : t0 + PROJ_T_TILE, d0 : d0 + PROJ_K_TILE],
                w_compress_gate[d0 : d0 + PROJ_K_TILE, 0:INDEX_DIM],
                init_cond=(d0 == 0),
            )
        gate_scores[t0 : t0 + PROJ_T_TILE, 0:INDEX_DIM] = g_acc


def golden_indexer_score(
    index_q: torch.Tensor,
    hadamard: torch.Tensor,
    pool_cache: torch.Tensor,
    pool_scale: torch.Tensor,
    pool_blocks: torch.Tensor,
    head_weights: torch.Tensor,
    seg_start: torch.Tensor,
    pool_count: torch.Tensor,
) -> torch.Tensor:
    """Head-weighted ReLU scores of every query against the pooled table.

    The pooled table arrives already quantized — INT8 keys with their per-row
    dequant scales, rotated into this Hadamard basis and quantized once at pool
    close — and the query-side Hadamard rotation and INT8 quantization mirror
    the kernel op for op. ``pool_blocks`` holds one physical base row per
    compacted INDEX_STATE_BLOCK_SIZE-row page, the same page granularity the
    metadata module maps pool slots at. The reference
    ``Glm5NextTextIndexer`` scores in plain FP32, and the quantized path is
    the a2a3 deployment numerics, so the golden carries the quantization
    rather than approximating around it.
    """
    tokens = index_q.shape[0]
    pools = pool_blocks.shape[0] * INDEX_STATE_BLOCK_SIZE
    scores = torch.full((tokens, pools), FP32_NEG_INF, dtype=torch.float32)
    if pools == 0 or tokens == 0:
        return scores
    rotated = index_q.float().reshape(tokens * INDEX_H, INDEX_DIM) @ hadamard.float()
    q_i8, q_scale = quantize_per_token_int8(rotated)
    gathered = pool_blocks.to(torch.long).repeat_interleave(INDEX_STATE_BLOCK_SIZE) + torch.arange(
        INDEX_STATE_BLOCK_SIZE
    ).repeat(pool_blocks.shape[0])
    k_i8 = pool_cache[gathered]
    k_scale = pool_scale[gathered, 0:1]
    for token in range(tokens):
        visible = int(pool_count[token])
        if visible <= 0:
            continue
        seg0 = int(seg_start[token])
        q_rows = slice(token * INDEX_H, (token + 1) * INDEX_H)
        dots = torch.matmul(
            k_i8[seg0 : seg0 + visible].to(torch.int32),
            q_i8[q_rows].to(torch.int32).t(),
        ).float()
        dots = dots * k_scale[seg0 : seg0 + visible] * q_scale[q_rows].t()
        dots = torch.relu(dots * SOFTMAX_SCALE)
        scores[token, seg0 : seg0 + visible] = (dots * head_weights[token]).sum(dim=1)
    return scores


@pl.jit.inline
def _indexer_score_prepare(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_blocks: pl.Tensor[[POOL_PAGES_DYN], pl.INT32],
):
    """Shared scorer prologue: query rotation/quant and the pool gather.

    Returns the INT8 query rows with their dequant scales and the contiguous
    INT8/scale scratch the score loops read. ``pool_blocks`` holds one physical
    base row per INDEX_STATE_BLOCK_SIZE-row page; the gather walks it page by
    page, which is the granularity the metadata module lowers the pool page
    table at — the host lowers paging, the kernel gathers. The score matrix
    itself needs no preparation: prefill tiles store their whole lane span and
    the top-k clamps through its valid-shaped leaf loads, so there is no init
    fill.
    """
    t_dim = pl.tensor.dim(index_q, 0)
    page_count = pl.tensor.dim(pool_blocks, 0)
    pools = page_count * INDEX_STATE_BLOCK_SIZE
    q_flat = pl.reshape(index_q, [t_dim * INDEX_H, INDEX_DIM])

    qh_acc = pl.create_tensor([t_dim * INDEX_H, INDEX_DIM], dtype=pl.FP32)
    for idx in pl.spmd(
        t_dim * INDEX_H // QH_MM_TILE, name_hint="indexer_q_hadamard", allow_early_resolve=True
    ):
        r0 = idx * QH_MM_TILE
        qh_acc[r0 : r0 + QH_MM_TILE, :] = pl.matmul(
            q_flat[r0 : r0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32
        )

    q_i8 = pl.create_tensor([t_dim * INDEX_H, INDEX_DIM], dtype=pl.INT8)
    q_scale_dq = pl.create_tensor([t_dim * INDEX_H, 1], dtype=pl.FP32)
    for idx in pl.spmd(t_dim * INDEX_H // QH_MM_TILE, name_hint="indexer_q_quant", allow_early_resolve=True):
        r0 = idx * QH_MM_TILE
        qh_tile = qh_acc[r0 : r0 + QH_MM_TILE, :]
        qh_amax = pl.full([1, QH_MM_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qh_abs = pl.maximum(qh_tile, pl.neg(qh_tile))
        qh_amax = pl.maximum(qh_amax, pl.reshape(pl.row_max(qh_abs), [1, QH_MM_TILE]))
        scale_quant_row = pl.div(pl.full([1, QH_MM_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
        q_scale_dq[r0 : r0 + QH_MM_TILE, :] = pl.reshape(pl.recip(scale_quant_row), [QH_MM_TILE, 1])
        qh_scaled = pl.row_expand_mul(qh_tile, pl.reshape(scale_quant_row, [QH_MM_TILE, 1]))
        qh_i32 = pl.cast(qh_scaled, target_type=pl.INT32, mode="rint")
        qh_half = pl.cast(qh_i32, target_type=pl.FP16, mode="round")
        q_i8[r0 : r0 + QH_MM_TILE, :] = pl.cast(qh_half, target_type=pl.INT8, mode="trunc")

    # One pass over the pooled table gathers the scattered INT8 rows and their
    # scales into contiguous GM scratch so the per-token score loops read plain
    # GM slices, one INDEX_STATE_BLOCK_SIZE-row page per entry. The table is
    # already quantized — nothing here re-quantizes. Row moves are Tensor
    # slice-assignments: a tile-level TLOAD of an INT8 row hits an unsupported
    # layout, and the donor reads its C8 cache the same way.
    # The scratch carries SCORE_C_TILE of slack, the widest tile any score
    # loop reads: the production page table is not padded to a tile boundary,
    # so a score loop's final tile always starts inside the live rows and its
    # full-width read must stay in bounds.
    pool_i8 = pl.create_tensor([pools + SCORE_C_TILE, INDEX_DIM], dtype=pl.INT8)
    pool_scale_g = pl.create_tensor([pools + SCORE_C_TILE, POOL_SCALE_WIDTH], dtype=pl.FP32)
    for page in pl.spmd(page_count, name_hint="indexer_pool_gather"):
        p0_idx = pl.cast(page, pl.INDEX) * INDEX_STATE_BLOCK_SIZE
        base = pl.cast(pl.read(pool_blocks, [page]), pl.INDEX)
        pool_i8[p0_idx : p0_idx + INDEX_STATE_BLOCK_SIZE, :] = pool_cache[
            base : base + INDEX_STATE_BLOCK_SIZE, :
        ]
        pool_scale_g[p0_idx : p0_idx + INDEX_STATE_BLOCK_SIZE, :] = pool_scale[
            base : base + INDEX_STATE_BLOCK_SIZE, :
        ]
    # Collapse the replicated scale lanes once, width-proportional work outside
    # the per-token loop: a strided lane-0 column read is not a legal TLOAD, and
    # reducing inside the score loop costs ~40-70 ns per (token, tile) step.
    # The softmax scale rides along here (it is per-row constant), so the score
    # loops never multiply it per tile.
    pool_scale_col = pl.create_tensor([pools + SCORE_C_TILE, 1], dtype=pl.FP32)
    # pools is the page-table length times INDEX_STATE_BLOCK_SIZE and is NOT
    # padded to SCORE_C_TILE — the production page table ends at the last live
    # page — so the dispatch is ceiled: the final partial tile still packs its
    # real rows. Lanes at or beyond pools stay undefined under the same
    # clamped-by-valid-shape contract as the score matrix, and the scratch
    # slack above covers every reader's widest tile, so the tail reads in
    # bounds.
    for c_block in pl.spmd((pools + SCORE_C_TILE - 1) // SCORE_C_TILE, name_hint="indexer_pool_scale_pack"):
        c0 = pl.cast(c_block, pl.INT32) * SCORE_C_TILE
        c0_idx = pl.cast(c0, pl.INDEX)
        sc8 = pool_scale_g[c0_idx : c0_idx + SCORE_C_TILE, :]
        summed = pl.reshape(pl.row_sum(sc8), [SCORE_C_TILE, 1])
        pool_scale_col[c0_idx : c0_idx + SCORE_C_TILE, :] = pl.mul(summed, 0.125 * SOFTMAX_SCALE)
    return q_i8, q_scale_dq, pool_i8, pool_scale_col


@pl.jit.inline
def indexer_score(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_blocks: pl.Tensor[[POOL_PAGES_DYN], pl.INT32],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
):
    """Score each query's own pooled-key segment for prefill and decode.

    One work item per query starts at its request's segment and skips an empty
    window. Sparse batches and separated request segments therefore use the
    same per-query schedule. Hadamard rotation, INT8 quantization and FP32
    dequantization/reduction share the unchanged preparation and golden.
    """
    t_dim = pl.tensor.dim(index_q, 0)
    q_i8, q_scale_dq, pool_i8, pool_scale_col = _indexer_score_prepare(
        index_q, hadamard, pool_cache, pool_scale, pool_blocks
    )

    for token in pl.spmd(t_dim, name_hint="indexer_score"):
        visible = pl.max(pl.read(pool_count, [token]), 0)
        if visible > 0:
            seg0 = pl.read(seg_start, [token])
            q_tile_i8 = q_i8[token * INDEX_H : token * INDEX_H + INDEX_H, :]
            q_scale_row = pl.reshape(q_scale_dq[token * INDEX_H : token * INDEX_H + INDEX_H, :], [1, INDEX_H])
            # Fused per-head multiplier; the softmax scale already rides on the
            # packed key scale, and the positive scale commutes with the relu.
            qsw = pl.mul(q_scale_row, head_weights[token : token + 1, 0:INDEX_H])
            for c0 in pl.range(0, visible, SCORE_C_TILE):
                kv_q_i8 = pool_i8[seg0 + c0 : seg0 + c0 + SCORE_C_TILE, :]
                kv_cache_scale_dq = pool_scale_col[seg0 + c0 : seg0 + c0 + SCORE_C_TILE, :]
                dots_i32 = pl.matmul(kv_q_i8, q_tile_i8, out_dtype=pl.INT32, b_trans=True)
                dots = pl.cast(dots_i32, target_type=pl.FP32, mode="none")
                dots = pl.row_expand_mul(dots, kv_cache_scale_dq)
                relu = pl.maximum(dots, 0.0)
                weighted = pl.col_expand_mul(relu, qsw)
                row_score = pl.reshape(pl.row_sum(weighted), [1, SCORE_C_TILE])
                valid_len = pl.min(SCORE_C_TILE, visible - c0)
                masked = pl.fillpad(pl.set_validshape(row_score, 1, valid_len), pad_value=pl.PadValue.min)
                masked = pl.maximum(masked, pl.full([1, SCORE_C_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
                index_scores[token : token + 1, seg0 + c0 : seg0 + c0 + SCORE_C_TILE] = masked



def golden_indexer_topk(
    index_scores: torch.Tensor,
    seg_start: torch.Tensor,
    pool_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected segment-local pool ids and which of them are real.

    A query whose visible pool count is below ``KPOOL_SELECT_K`` still fills the
    full width, so the selection carries padding. ``indexer_expand`` needs to
    know which entries are padding to blank whole four-wide groups, and the
    donor kernel emits indices only — the validity output is the addition here.
    """
    tokens = index_scores.shape[0]
    selected = torch.full((tokens, KPOOL_SELECT_K), -1, dtype=torch.int32)
    valid = torch.zeros(tokens, KPOOL_SELECT_K, dtype=torch.int32)
    for token in range(tokens):
        visible = max(int(pool_count[token]), 0)
        if visible > 0:
            seg0 = int(seg_start[token])
            keep = min(KPOOL_SELECT_K, visible)
            picked = torch.topk(index_scores[token, seg0 : seg0 + visible], keep).indices
            selected[token, :keep] = picked.to(torch.int32)
            valid[token, :keep] = 1
    return selected, valid


@pl.jit.incore
def _indexer_topk_query(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    running_pairs: pl.Tensor[[T_DYN, PAIR_WIDTH], pl.FP32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
) -> None:
    """Select one query's top visible pools, one sort leaf at a time."""
    query = pl.tile.get_block_idx()
    pl.store(pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=-1), [query, 0], selected_pools)
    pl.store(pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=0), [query, 0], selected_valid)
    visible_count = pl.max(pl.read(pool_count, [query]), 0)
    if visible_count > 0:
        seg0 = pl.read(seg_start, [query])
        # The running top pairs round-trip through the GM scratch, the donor's
        # way of keeping one SSA type across the leaf loop.
        leaf_count = (visible_count + LEAF - 1) // LEAF
        for leaf in pl.range(leaf_count):
            leaf0 = leaf * LEAF
            leaf_valid = pl.min(LEAF, visible_count - leaf0)
            row_raw = pl.load(index_scores, [query, seg0 + leaf0], [1, LEAF], valid_shape=[1, leaf_valid])
            row = pl.tile.fillpad(row_raw, pad_value=pl.PadValue.min)
            row = pl.maximum(row, pl.tile.full([1, LEAF], dtype=pl.FP32, value=FP32_NEG_INF))
            index_ramp = pl.tile.arange(0, [1, LEAF], dtype=pl.INT32)
            leaf_indices = pl.tile.add(index_ramp, pl.cast(leaf0, pl.INT32))
            leaf_pairs = pl.tile.sort32(row, pl.reinterpret_view(leaf_indices, pl.UINT32))
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
            leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=1024)
            leaf_pairs = pl.tile.slice(leaf_pairs, [1, PAIR_WIDTH], [0, 0])
            if leaf == 0:
                pl.store(leaf_pairs, [query, 0], running_pairs)
            else:
                pairs = pl.load(running_pairs, [query, 0], [1, PAIR_WIDTH])
                merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
                merged = pl.tile.mrgsort(pairs, leaf_pairs, tmp=merge_tmp)
                pl.store(pl.tile.slice(merged, [1, PAIR_WIDTH], [0, 0]), [query, 0], running_pairs)
        pairs = pl.load(running_pairs, [query, 0], [1, PAIR_WIDTH])
        picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        valid_topk = pl.min(visible_count, KPOOL_SELECT_K)
        indices_out = pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=-1)
        validity_out = pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=0)
        for lane in pl.range(valid_topk):
            pl.tile.write(indices_out, [0, lane], pl.tile.read(picked, [0, lane]))
            pl.tile.write(validity_out, [0, lane], pl.const(1, pl.INT32))
        pl.store(indices_out, [query, 0], selected_pools)
        pl.store(validity_out, [query, 0], selected_valid)


@pl.jit.inline
def indexer_topk(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
):
    """Exact top-512 over each query's visible window, one block per query.

    Selections are segment-local pool ids: the sort payload carries the offset
    from ``seg_start``, so ``indexer_expand`` can map pool ``j`` to raw
    positions ``4j .. 4j+3`` without any batch knowledge.

    **Sharing ABI** (``index_share_for_mtp_iteration``, settled in issue
    #1267): ``selected_pools``/``selected_valid`` are the shareable selection
    surface. A request's MTP draft steps 1+ reuse the selection their
    proposal computed on step 0 — per request, not per row — with the query
    projection, scoring and this selection skipped on those steps. The
    expansion is *not* shareable and is rebuilt per row by
    :func:`indexer_expand`. The sharing path (``indexer_share_mtp``) is out
    of scope for this phase: every row here is selected independently.

    Returns the region's TaskId so the expansion can order against it.
    """
    t_dim = pl.tensor.dim(index_scores, 0)
    running_pairs = pl.create_tensor([t_dim, PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(t_dim, name_hint="indexer_topk") as topk_tid:
        _indexer_topk_query(
            index_scores, seg_start, pool_count, running_pairs, selected_pools, selected_valid
        )
    return topk_tid


def golden_indexer_expand(
    selected_pools: torch.Tensor,
    selected_valid: torch.Tensor,
    tail_start: torch.Tensor,
    tail_count: torch.Tensor,
    kv_len: torch.Tensor,
) -> torch.Tensor:
    """Expand the selections into front-packed raw positions, one row per query."""
    tokens = selected_pools.shape[0]
    expanded = torch.full((tokens, TOPK_INDEX_WIDTH), -1, dtype=torch.int32)
    lanes = torch.arange(TOPK_INDEX_WIDTH)
    pool_raw = lanes // INDEX_KPOOL
    pool_id = pool_raw.clamp(max=selected_pools.shape[1] - 1)
    rem = lanes - pool_raw * INDEX_KPOOL
    for token in range(tokens):
        groups = min(int(kv_len[token]) // INDEX_KPOOL, KPOOL_SELECT_K)
        hist_len = groups * INDEX_KPOOL
        base = int(tail_start[token])
        tail = int(tail_count[token])
        hist_pos = selected_pools[token].to(torch.int64)[pool_id] * INDEX_KPOOL + rem
        tail_offset = lanes - hist_len
        hist_active = ((lanes < hist_len) & (selected_valid[token].to(torch.int64)[pool_id] > 0)).to(
            torch.int64
        )
        tail_active = ((tail_offset >= 0) & (tail_offset < tail)).to(torch.int64)
        positions = hist_active * (hist_pos + 1) + tail_active * (base + tail_offset + 1) - 1
        expanded[token] = positions.to(torch.int32)
    return expanded


@pl.jit.inline
def indexer_expand(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
):
    """Expand the selected pools into raw cache rows, front packed per row.

    **ABI**: ``topk_indices`` is front packed. Each row holds its valid logical
    per-request positions in its leading lanes and pads the remaining suffix
    with ``-1``; a ``-1`` never sits between two valid entries. Both sparse
    attention kernels read this as a contract rather than a convention: they
    test one lane to decide that a 128-wide block, or a whole row, carries no
    selection, so an interleaved ``-1`` would silently drop the live entries
    behind it. See :mod:`models.glm5_3_flash.decode_sparse_attn` and
    :mod:`models.glm5_3_flash.prefill_sparse_attn`.

    The prefix length is ``min(KPOOL_SELECT_K, kv_len // 4) * 4`` — selection
    order within it is free because the attention is a permutation-invariant
    softmax, so this constrains only where the padding goes. The tail pool is
    always appended right behind the selected groups, which is what keeps the
    packing tight for short contexts; the reference leaves holes there instead.

    **Sharing ABI**: only the selection (``selected_pools``/``selected_valid``)
    may be shared across a request's rows; the expansion is rebuilt per row
    because ``tail_start``/``tail_count``/``kv_len`` differ per row — a decode
    request's four rows (one live plus three MTP spec tokens) carry different
    tails, and the front-packed row bakes its own tail in.
    """
    t_dim = pl.tensor.dim(selected_pools, 0)
    with pl.spmd(t_dim, name_hint="indexer_expand") as expand_tid:
        token = pl.tile.get_block_idx()
        # Scalar bookkeeping stays INT32: pl.read preserves the tensor dtype but
        # pl.min promotes to index, which scalar instructions reject.
        groups = pl.cast(pl.min(pl.read(kv_len, [token]) // INDEX_KPOOL, KPOOL_SELECT_K), pl.INT32)
        hist_len = pl.cast(groups * INDEX_KPOOL, pl.INT32)
        base = pl.cast(pl.read(tail_start, [token]), pl.INT32)
        tail = pl.cast(pl.read(tail_count, [token]), pl.INT32)
        for w0 in pl.unroll(0, TOPK_INDEX_WIDTH - (INDEX_KPOOL - 1), EXPAND_W_TILE):
            lane = pl.arange(pl.cast(w0, pl.INT32), [1, EXPAND_W_TILE], dtype=pl.INT32)
            pool_raw = pl.cast(
                pl.mul(pl.cast(lane, pl.FP32, mode="none"), 1.0 / INDEX_KPOOL),
                pl.INT32,
                mode="trunc",
            )
            pool_id = pl.minimum(pool_raw, KPOOL_SELECT_K - 1)
            sel_row = pl.gather(selected_pools[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_id)
            sel_validity = pl.gather(
                selected_valid[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_id
            )
            hist_slack = pl.neg(pl.sub(lane, hist_len))
            hist_mask = pl.minimum(pl.maximum(hist_slack, 0), 1)
            hist_active = pl.mul(hist_mask, pl.minimum(sel_validity, 1))
            tail_offset = pl.sub(lane, hist_len)
            tail_in_range = pl.minimum(pl.maximum(pl.neg(pl.sub(tail_offset, tail)), 0), 1)
            tail_ge_zero = pl.minimum(pl.maximum(pl.add(tail_offset, 1), 0), 1)
            tail_active = pl.mul(tail_in_range, tail_ge_zero)
            # Position math in FP32 for the float-constant multiplies, rounded
            # once at the store.
            lane_f = pl.cast(lane, pl.FP32, mode="none")
            pool_f = pl.cast(pool_raw, pl.FP32, mode="none")
            rem_f = pl.sub(lane_f, pl.mul(pool_f, 4.0))
            hist_pos_f = pl.add(pl.mul(pl.cast(sel_row, pl.FP32, mode="none"), 4.0), rem_f)
            tail_pos_f = pl.cast(pl.add(pl.add(tail_offset, base), 1), pl.FP32, mode="none")
            positions_f = pl.sub(
                pl.add(
                    pl.mul(pl.cast(hist_active, pl.FP32, mode="none"), pl.add(hist_pos_f, 1.0)),
                    pl.mul(pl.cast(tail_active, pl.FP32, mode="none"), tail_pos_f),
                ),
                1.0,
            )
            topk_indices[token : token + 1, w0 : w0 + EXPAND_W_TILE] = pl.cast(
                positions_f, pl.INT32, mode="none"
            )
        # The odd 2051-wide row cannot end on an aligned tile, so the last chunk
        # is an eight-lane tile starting five lanes early; the overlap is written
        # twice with identical values, which keeps every store on the MTE3 path.
        lane_t = pl.arange(pl.const(2043, pl.INT32), [1, EXPAND_TAIL_CHUNK], dtype=pl.INT32)
        pool_t = pl.cast(
            pl.mul(pl.cast(lane_t, pl.FP32, mode="none"), 1.0 / INDEX_KPOOL),
            pl.INT32,
            mode="trunc",
        )
        pool_t = pl.minimum(pool_t, KPOOL_SELECT_K - 1)
        sel_t = pl.gather(selected_pools[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_t)
        valid_t = pl.gather(selected_valid[token : token + 1, 0:KPOOL_SELECT_K], dim=-1, index=pool_t)
        hslack_t = pl.neg(pl.sub(lane_t, hist_len))
        hmask_t = pl.minimum(pl.maximum(hslack_t, 0), 1)
        hactive_t = pl.mul(hmask_t, pl.minimum(valid_t, 1))
        offset_t = pl.sub(lane_t, hist_len)
        trange_t = pl.minimum(pl.maximum(pl.neg(pl.sub(offset_t, tail)), 0), 1)
        tge_t = pl.minimum(pl.maximum(pl.add(offset_t, 1), 0), 1)
        tactive_t = pl.mul(trange_t, tge_t)
        lane_tf = pl.cast(lane_t, pl.FP32, mode="none")
        pool_tf = pl.cast(pool_t, pl.FP32, mode="none")
        rem_tf = pl.sub(lane_tf, pl.mul(pool_tf, 4.0))
        hpos_tf = pl.add(pl.mul(pl.cast(sel_t, pl.FP32, mode="none"), 4.0), rem_tf)
        tpos_tf = pl.cast(pl.add(pl.add(offset_t, base), 1), pl.FP32, mode="none")
        pos_tf = pl.sub(
            pl.add(
                pl.mul(pl.cast(hactive_t, pl.FP32, mode="none"), pl.add(hpos_tf, 1.0)),
                pl.mul(pl.cast(tactive_t, pl.FP32, mode="none"), tpos_tf),
            ),
            1.0,
        )
        topk_indices[
            token : token + 1,
            TOPK_INDEX_WIDTH - EXPAND_TAIL_CHUNK : TOPK_INDEX_WIDTH,
        ] = pl.cast(pos_tf, pl.INT32, mode="none")
    return expand_tid
