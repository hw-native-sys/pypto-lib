# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode — FUSED attn + dense balance + gp-loop pl.pipeline.

Same as the blocklevel variant but expresses the cube/vector software pipeline
DECLARATIVELY: the per-head ``gp`` loop is a ``pl.pipeline(GP_SIZE, stage=2)``
instead of a hand-unrolled QK0,QK1 -> softmax0,softmax1 -> SV0,SV1. The compiler
should overlap iteration i+1's QK (AIC) with iteration i's softmax (AIV), the
same implicit cube/vector overlap the manual unroll produced — but without the
user having to write the unrolled program.


EXPECTED / INTENT program (the dense block-level load balancer). NOTE: this does
NOT compile on the current toolchain — the data-dependent ``pl.read`` scalar that
feeds the store offset (``g_base + sb * Q_HEAD_PAD``) trips a PTO codegen
limitation (``GetOrCreateTensorView`` / ptoas ``index vs i64``; see
``KNOWN_ISSUES.md``). It is written to capture the desired structure; the
affine fallback that DOES compile lives in
``qwen3_manual_scope_fused_kvsplit_static.py`` (coprime-stride, ~1.9x balance).

Derived from the static file; Scope 1 (RMSNorm + Q/K/V proj) and Scope 3
(out_proj + MLP) are unchanged. Scope 2 uses BLOCK-LEVEL balancing:

  1. ``fa_work_build`` (AIV prep task) compacts only the REAL seq-blocks of the
     ragged batch into a gap-free work table — ``fa_work_table[w] = b*MCB + p``
     for w in ``[0, fa_total)`` (prefix-sum cursor over per-batch block counts),
     and writes ``fa_total`` (= total real blocks).
  2. ``fa_fused`` (ONE mixed cube+vec root) grid-strides ``w`` over
     ``[0, fa_total)`` with ``core = w % NUM_CORES``. Each step decodes one real
     block ``(b, p)`` from the table and runs QK→softmax→SV. Because the table is
     dense (no per-batch gaps, only real blocks), the ``fa_total / NUM_CORES``
     equal-cost blocks distribute as evenly as integer packing allows (≈1.25x of
     ideal) — independent of per-batch length skew, and free of the
     index-stride/NUM_CORES resonance the affine layout suffered (8x→2x there;
     this targets ~1.25x).
  3. ``online_softmax`` is UNCHANGED — it reduces the per-block partials
     (``all_oi_tmp`` / ``all_cur_mi`` / ``all_cur_li``) across ALL blocks of a
     ``(b, kvh)`` lane; auto-dep on the shared scratch serializes it after the
     contributing ``fa_fused`` blocks.

A2/A3 NOTE: the fused root's C2V/V2C boundary still routes through a GM pipe
buffer on 910B (``InjectGMPipeBuffer`` is backend-gated), so fusing does NOT
avoid the AIC↔AIV GM round-trip on A2/A3 — the win here is load balance, not GM
savings. (On A5 the L2-swimlane path keeps it on-chip.)

Usage::

    python qwen3_manual_scope_fused_kvsplit_blocklevel.py --smoke         # parser/passes
    python qwen3_manual_scope_fused_kvsplit_blocklevel.py --platform a2a3 # device (expected to fail codegen)
"""

import argparse
import os
from pathlib import Path

import pypto.language as pl
import torch
from pypto.backend import BackendType, set_backend_type
from pypto.runtime import RunConfig

from config import VOCAB  # vocab size for the fused decode_fwd LM head / logits
from rms_lm_head import rms_lm_head  # LM head for the fused multi-layer decode_fwd

# ══════════════════════════════════════════════════════════════════════════════
# Functional config — model architecture + workload.
# ══════════════════════════════════════════════════════════════════════════════

# ── Model architecture (Qwen3-14B, fixed by the checkpoint) ──
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 17408
# MAX_SEQ is env-overridable for the e2e generate harness: it sizes the
# (non-paged) KV cache rows and the RoPE tables, so a 512-token run can use a
# 512-row cache instead of the 4096 micro-benchmark default (8x less KV memory).
MAX_SEQ = int(os.environ.get("PTO2_MANUAL_MAX_SEQ", "4096"))
EPS = 1e-6  # RMSNorm epsilon

# ── Workload (decode batch) ──
BATCH = 16

# ── Derived shapes — recomputed from the above, don't edit ──
HIDDEN = NUM_HEADS * HEAD_DIM  # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 1024
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 5 (GQA ratio)
HALF_DIM = HEAD_DIM // 2  # 64 (RoPE rotates lo/hi halves)
ATTN_SCALE = 1.0 / (HEAD_DIM**0.5)
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM  # per-head QK-norm RMSNorm denominator

# Attention head grouping. Q_HEAD_BATCH = Q_PER_KV (one attn lane per KV head).
Q_HEAD_BATCH = Q_PER_KV  # 5: Q heads bundled per attention task
Q_HEAD_PAD = ((Q_HEAD_BATCH + 15) // 16) * 16  # 16: rounded up to matmul L0 inner-dim multiple of 16
# Real-vs-padded handling for the fused-attn scratch: tiles stay PHYSICALLY
# Q_HEAD_PAD (16) rows — the minimal cube M and a 32B-aligned size for the
# col-major cur_mi/cur_li [.,1] FP32 tiles — while set_validshape marks only the
# Q_HEAD_BATCH (5) real rows, so each tile.load / tile.store transfers 5 rows from
# GM (rest auto-padded). Never shrink the tile (that trips alloc_tile alignment).


# ══════════════════════════════════════════════════════════════════════════════
# Optimization config — per-stage tile sizes, K/N splits, inner-pipe widths.
# ══════════════════════════════════════════════════════════════════════════════

# ── Scope 1a · input RMSNorm ──
RMSNORM_K_CHUNK = 256
# x*gamma is pure elementwise along HIDDEN — split it across XG_BLOCKS SPMD
# vector blocks (grid-stride over the HIDDEN//RMSNORM_K_CHUNK = 20 chunks; each
# block writes disjoint columns, so no atomic). On the QKV critical path.
# 5 divides 20 evenly → 4 chunks/block (same as residual_rms_cast), which
# amortizes the stage=2 pipeline fill better than 8 blocks (only 2-3 chunks each).
XG_BLOCKS = 5

# ── Scope 1b · Q / K / V projections — SPLIT-K + inner N/K tiling, SPMD style ──
# Tiling: TM=16 (M = full batch, OM=1), TN=256 inner N sub-tile, TK=256 inner
# K chunk. Outer: ON = 10 (Q) / 2 (K) / 2 (V) N-tiles of QKV_N_TILE=512 each
# (= N_SUB=2 inner TN subtiles); OK=4 split-K slices of QKV_K_SLICE=1280 each
# (= QKV_K_CHUNKS=5 inner TK chunks), atomic-added. Each projection is ONE
# pl.spmd dispatch of ON*OK blocks; each block does N_SUB N-subtiles x
# QKV_K_CHUNKS chunks and atomic-adds into a zero-seeded output. SPMD (not
# pl.parallel + per-iter pl.at) keeps the split-K atomic-adds inside a SINGLE
# orchestration task so they accumulate in parallel via hardware atomic; auto-dep
# orders seed -> spmd -> rope_qkv, so NO explicit deps are needed.
TM = 16  # M tile = BATCH (OM = 1; M is not split)
TN = 256  # inner N sub-tile
TK = 256  # inner K chunk
QKV_N_TILE = 512  # outer N-tile width (one ON unit) = N_SUB inner TN subtiles
N_SUB = QKV_N_TILE // TN  # 2 inner N-subtiles per outer N-tile
Q_ON = HIDDEN // QKV_N_TILE  # 10 outer N-tiles (Q)
KV_ON = KV_HIDDEN // QKV_N_TILE  # 2 outer N-tiles (K, V)
QKV_OK = 4  # split-K slices (atomic-add)
QKV_K_SLICE = HIDDEN // QKV_OK  # 1280 K per split
QKV_K_CHUNKS = QKV_K_SLICE // TK  # 5 inner TK chunks per split

# ── Scope 2 · grouped-query attention (fused, PAGED) ──
# SEQ_TILE = seq length per KV block. PINNED to the serving paged page_size (128):
# decode now reads KV from the PAGED pool via block_table, so one logical block
# must map to exactly one physical page — i.e. SEQ_TILE == page_size — or a block
# would straddle two pages whose physical ids are unrelated. prefill_fwd uses the
# same 128 page; 128 is also the known-good L0B double-buffer cap (256 only became
# legal after the AutoTileMatmulL0 fix #1595, and is not needed once tiling tracks
# the page size). BLOCK_SIZE is an alias used by the paged cache-row arithmetic.
SEQ_TILE = 128
BLOCK_SIZE = SEQ_TILE  # paged page size (== serving runtime.page_size)
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE  # logical blocks per seq (32 @ 4096)
# Worst-case physical page count for the standalone golden/smoke pool (one band
# per (batch, block)); the kernel itself reads the pool size from k_cache's dynamic
# dim, so this only sizes the test fixtures.
MAX_BLOCKS_PER_SEQ = MAX_CTX_BLOCKS
NUM_PAGES = BATCH * MAX_BLOCKS_PER_SEQ
CACHE_ROWS = NUM_PAGES * NUM_KV_HEADS * BLOCK_SIZE  # paged k_cache / v_cache rows (one layer)

# ── Scope 2 · KV-block split (flash-decoding) for ragged load balance ──
# Each fa_fused lane (a Q-group pair) is split into contiguous KV partitions of
# TOKENS_PER_SPLIT tokens. Smaller TOKENS_PER_SPLIT = finer load balance for
# ragged seq_lens, at the cost of more fa_fused tasks
# (BATCH * (NUM_KV_HEADS // 2) * KV_SPLITS).
# Dispatch unit = ONE seq block (TOKENS_PER_SPLIT == SEQ_TILE). Every fa_fused
# work item is then a single SEQ_TILE block (×2 heads) — equal cost regardless of
# which batch/sequence it belongs to. The grid-stride over FA_WORK items thus
# spreads equal-cost blocks across cores (cross-batch load balance), instead of
# assigning whole variable-length partitions per (batch) lane. Larger
# TOKENS_PER_SPLIT (256/512/…) = coarser units, less loop overhead, more skew.
TOKENS_PER_SPLIT = SEQ_TILE  # 128: one seq block per work item
BLOCKS_PER_SPLIT = TOKENS_PER_SPLIT // SEQ_TILE  # 1 SEQ_TILE block per work item
KV_SPLITS = MAX_CTX_BLOCKS // BLOCKS_PER_SPLIT  # 32 = MAX_CTX_BLOCKS (one item per block)

# GP_SIZE = number of KV heads bundled per fa_fused work item (the `gp` loop).
# All bundled heads process the SAME block index, so an item stays equal-cost
# (GP_SIZE head-blocks) and the per-item scalar setup — notably
# pl.read(seq_lens, ...) — is amortized across GP_SIZE heads. Larger GP_SIZE =
# fewer work items / fewer seq_lens reads, at a slightly coarser balance
# granularity. Must divide NUM_KV_HEADS. GP_SIZE = NUM_KV_HEADS (8) bundles all
# heads → one head-group per (batch, block).
GP_SIZE = 8
HEAD_GROUPS = NUM_KV_HEADS // GP_SIZE  # head-groups per (batch, block) (8//8 = 1)

# ── Scope 2 · STATIC SPMD dispatch + BLOCK-LEVEL (dense) load balance ──
# Launch a FIXED grid of NUM_CORES persistent blocks and grid-stride over the
# work items inside each kernel (collapses per-item dispatch to ~NUM_CORES).
#
# Block-level balancing: the unit of work is ONE real seq-block (×GP_SIZE heads).
# A separate AIV prep task (`fa_work_build`) compacts only the REAL blocks of the
# ragged batch into a gap-free work table `fa_work_table[w] = b*MAX_CTX_BLOCKS + p`
# (one entry per real block, w in [0, fa_total)), and writes `fa_total`. fa_fused
# then grid-strides w in [0, fa_total) with core = w % NUM_CORES, so the
# `total_real_blocks / NUM_CORES` equal-cost blocks distribute as evenly as
# integer packing allows (≈1.25x vs ideal) — independent of per-batch length skew
# and free of the index-stride/NUM_CORES resonance the affine layout suffered.
NUM_CORES = 24
FA_TABLE_CAP = BATCH * MAX_CTX_BLOCKS  # upper bound on real blocks (all seqs full)
OS_WORK = BATCH * NUM_KV_HEADS  # online_softmax work items (128)

# ── Scope 3a · out_proj (split-K × split-N, atomic-add into attn_proj_fp32) ──
K_SPLITS_OUT = 5
N_SPLITS_OUT = 10
OUT_INNER_TK = 64
OUT_TN = HIDDEN // N_SPLITS_OUT  # 512 output N per task
OUT_TK = HIDDEN // K_SPLITS_OUT  # 1024 K per task
OUT_N_SUB_K = OUT_TK // OUT_INNER_TK  # 16 inner K iters per task

# ── Scope 3b · residual + BF16 cast + RMS reduce ──
K_CHUNK = 256  # inner pipe width for residual_rms_cast and post_rms_reduce

# ── Scope 3b · MLP gate / up (split-K, atomic-add into per-batch FP32) ──
MLP_TN = 1024  # output N-tile per task (= silu task N-width = DOWN_TN)
K_SPLITS_MLP = 5
MLP_INNER_TK = 64
MLP_K_SLICE = HIDDEN // K_SPLITS_MLP  # 1024 K per task
MLP_N_SUB_K = MLP_K_SLICE // MLP_INNER_TK  # 16 inner K iters per task
MLP_ON = INTERMEDIATE // MLP_TN  # 17 output N-blocks (= silu task count)

# ── Scope 3b · silu (MLP_TN-wide tasks, inner pipe over MLP_OUT_CHUNK sub-tiles) ──
MLP_OUT_CHUNK = 256  # silu inner-pipe sub-tile width
SILU_INNER_CHUNKS = MLP_TN // MLP_OUT_CHUNK  # 4 sub-tiles per silu task

# ── Scope 3b · down (split-K, atomic-add into down_acc_all) ──
DOWN_TN = 1024  # output N-tile per task (must equal MLP_TN, see assert)
DOWN_TK = 64  # inner K iter (keeps L0 W tile within Mat buffer at DOWN_TN)
DOWN_ON = HIDDEN // DOWN_TN  # 5 output N-blocks
K_SPLITS = INTERMEDIATE // DOWN_TN  # 17 K-slices per N-block
N_SUB_K = DOWN_TN // DOWN_TK  # 16 inner K iters per task

# ── Cross-stage wiring constraints ──
N_PER_CAST_K = MLP_K_SLICE // OUT_TN  # 2

# Geometry assertions — keep at the bottom so all constants are defined first.
assert QKV_N_TILE % TN == 0, "TN must divide the outer N-tile"
assert HIDDEN % QKV_N_TILE == 0 and KV_HIDDEN % QKV_N_TILE == 0, "QKV_N_TILE must divide Q and KV widths"
assert HIDDEN % QKV_OK == 0, "OK must divide HIDDEN (K dim)"
assert QKV_K_SLICE % TK == 0, "TK must divide the split-K slice"
assert Q_ON == 10 and KV_ON == 2, "expected ON = 10 (Q) + 2 (K) + 2 (V)"
assert TM == BATCH and N_SUB == 2 and QKV_K_CHUNKS == 5
assert TOKENS_PER_SPLIT % SEQ_TILE == 0, "TOKENS_PER_SPLIT must be a multiple of SEQ_TILE"
assert MAX_CTX_BLOCKS % BLOCKS_PER_SPLIT == 0, "BLOCKS_PER_SPLIT must divide MAX_CTX_BLOCKS"
assert KV_SPLITS * BLOCKS_PER_SPLIT == MAX_CTX_BLOCKS
assert NUM_KV_HEADS % GP_SIZE == 0, "GP_SIZE must divide NUM_KV_HEADS"
assert HEAD_GROUPS * GP_SIZE == NUM_KV_HEADS
assert GP_SIZE == NUM_KV_HEADS, "block-level work table encodes (b, block); needs HEAD_GROUPS == 1"
assert FA_TABLE_CAP == BATCH * MAX_CTX_BLOCKS
assert DOWN_TN % MLP_OUT_CHUNK == 0, "DOWN_TN must be a multiple of MLP_OUT_CHUNK"
assert DOWN_ON * DOWN_TN == HIDDEN
assert K_SPLITS * DOWN_TN == INTERMEDIATE
assert MLP_ON * MLP_TN == INTERMEDIATE
assert K_SPLITS_MLP * MLP_K_SLICE == HIDDEN
assert MLP_TN % MLP_OUT_CHUNK == 0
assert MLP_TN == DOWN_TN, "silu/down K-slice alignment requires MLP_TN == DOWN_TN"
assert N_SPLITS_OUT * OUT_TN == HIDDEN
assert K_SPLITS_OUT * OUT_TK == HIDDEN
assert OUT_N_SUB_K * OUT_INNER_TK == OUT_TK
assert N_PER_CAST_K * OUT_TN == MLP_K_SLICE


# ──────────────────────────────────────────────────────────────────────────────
# Monolithic JIT entry.
# ──────────────────────────────────────────────────────────────────────────────


@pl.jit.inline
def _decode_layer(  # noqa: PLR0913 — model signature is intrinsic
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    # Per-layer offsets into the STACKED weights / PAGED KV cache. For the
    # single-layer entry (qwen3_decode_mpmd) layer_idx==0 so every base is 0 and
    # single-layer tensors work unchanged; decode_fwd passes the running loop index.
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    # Paged KV: rows are runtime-dynamic (the paged pool sizes them). Derive the
    # per-layer stride and the block-table row stride from the tensor dims, exactly
    # as prefill_fwd does, so decode reads the SAME pool prefill wrote.
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    layer_cache_base = layer_idx * layer_cache_rows
    user_batch = pl.tensor.dim(seq_lens, 0)
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // user_batch
    q_norm_w = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
    k_norm_w = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])

    # Scope 1
    normed_states = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    out_partial = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # pre-consolidation layer output
    dcr_tids = pl.array.create(DOWN_ON, pl.TASK_ID)  # terminal writers of out_partial (filled in manual_scope)
    inv_rms_states = pl.create_tensor([BATCH, 1], dtype=pl.FP32)  # deferred 1/rms denominator
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    # ── Scope 1: input RMSNorm, SPLIT into two INDEPENDENT steps. ──
    # RMSNorm(x) = (x * inv_rms) * gamma, where inv_rms[b] = 1/sqrt(mean_k x^2 +
    # eps) is a per-row SCALAR. Q/K/V proj and RoPE are linear, so the 1/rms factor
    # commutes through them: q = inv_rms * ((x*gamma) @ Wq). We therefore DEFER the
    # 1/rms division past the projections and fold it into rope_qkv (one scalar mul
    # per batch row). This decouples the sum-of-squares reduction from the gamma
    # scaling: `x_gamma` (which feeds QKV) no longer waits on the reduction, and
    # `rms_recip` overlaps the QKV proj. normed_states is consumed ONLY by QKV; the
    # residual / post_rms path reads raw hidden_states, so it is unaffected.
    for xg_core in pl.spmd(XG_BLOCKS, name_hint="x_gamma"):
        for kb in pl.pipeline(xg_core, HIDDEN // RMSNORM_K_CHUNK, XG_BLOCKS, stage=2):
            k0 = kb * RMSNORM_K_CHUNK
            x_chunk = pl.cast(hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK], target_type=pl.FP32)
            gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [layer_idx, k0])
            xg = pl.col_expand_mul(x_chunk, gamma)
            normed_states = pl.assemble(normed_states, pl.cast(xg, target_type=pl.BF16), [0, k0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip"):
        partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):
            k0 = kb * RMSNORM_K_CHUNK
            x_chunk = pl.cast(hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK], target_type=pl.FP32)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]),
            )
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        inv_rms_states = pl.assemble(inv_rms_states, inv_rms, [0, 0])

    # ── Scope 1: Q projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed"):
        for snb in pl.pipeline(Q_ON, stage=2):
            q_proj = pl.assemble(
                q_proj, pl.full([BATCH, QKV_N_TILE], dtype=pl.FP32, value=0.0), [0, snb * QKV_N_TILE]
            )
    for q_core in pl.spmd(Q_ON * QKV_OK, name_hint="q_proj"):
        q_n_region = (q_core // QKV_OK) * QKV_N_TILE
        q_k_base = (q_core % QKV_OK) * QKV_K_SLICE
        for n_sub in pl.range(N_SUB):
            n0 = q_n_region + n_sub * TN
            q_acc = pl.matmul(
                normed_states[:, q_k_base : q_k_base + TK],
                wq[layer_hidden_base + q_k_base : layer_hidden_base + q_k_base + TK, n0 : n0 + TN],
                out_dtype=pl.FP32,
            )
            for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                kk = q_k_base + kc * TK
                q_acc = pl.matmul_acc(
                    q_acc, normed_states[:, kk : kk + TK], wq[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                )
            q_proj = pl.assemble(q_proj, q_acc, [0, n0], atomic=pl.AtomicType.Add)

    # ── Scope 1: K projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_seed"):
        k_proj = pl.assemble(k_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])
    for k_core in pl.spmd(KV_ON * QKV_OK, name_hint="k_proj"):
        k_n_region = (k_core // QKV_OK) * QKV_N_TILE
        k_k_base = (k_core % QKV_OK) * QKV_K_SLICE
        for n_sub in pl.range(N_SUB):
            n0 = k_n_region + n_sub * TN
            k_acc = pl.matmul(
                normed_states[:, k_k_base : k_k_base + TK],
                wk[layer_hidden_base + k_k_base : layer_hidden_base + k_k_base + TK, n0 : n0 + TN],
                out_dtype=pl.FP32,
            )
            for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                kk = k_k_base + kc * TK
                k_acc = pl.matmul_acc(
                    k_acc, normed_states[:, kk : kk + TK], wk[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                )
            k_proj = pl.assemble(k_proj, k_acc, [0, n0], atomic=pl.AtomicType.Add)

    # ── Scope 1: V projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_seed"):
        v_proj = pl.assemble(v_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])
    for v_core in pl.spmd(KV_ON * QKV_OK, name_hint="v_proj"):
        v_n_region = (v_core // QKV_OK) * QKV_N_TILE
        v_k_base = (v_core % QKV_OK) * QKV_K_SLICE
        for n_sub in pl.range(N_SUB):
            n0 = v_n_region + n_sub * TN
            v_acc = pl.matmul(
                normed_states[:, v_k_base : v_k_base + TK],
                wv[layer_hidden_base + v_k_base : layer_hidden_base + v_k_base + TK, n0 : n0 + TN],
                out_dtype=pl.FP32,
            )
            for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                kk = v_k_base + kc * TK
                v_acc = pl.matmul_acc(
                    v_acc, normed_states[:, kk : kk + TK], wv[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                )
            v_proj = pl.assemble(v_proj, v_acc, [0, n0], atomic=pl.AtomicType.Add)

    # ── Scope 2: RoPE + KV cache update + fused flash-attention. ──
    all_q_padded = pl.create_tensor([BATCH * NUM_KV_HEADS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    all_oi_tmp = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_CTX_BLOCKS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32
    )
    all_cur_mi = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_CTX_BLOCKS * Q_HEAD_PAD, 1], dtype=pl.FP32
    )
    all_cur_li = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_CTX_BLOCKS * Q_HEAD_PAD, 1], dtype=pl.FP32
    )

    # Block-level (dense) work list. `fa_work_table[w]` encodes the w-th REAL
    # seq-block as `b * MAX_CTX_BLOCKS + p`; `fa_total[0]` holds the number of
    # real blocks. Built once by the `fa_work_build` AIV task below, consumed by
    # fa_fused's grid-stride. Sized to the worst case (every sequence full).
    fa_work_table = pl.create_tensor([FA_TABLE_CAP, 1], dtype=pl.INT32)
    fa_total = pl.create_tensor([1, 1], dtype=pl.INT32)

    # ── Scope 2 prep: build the dense block-level work list on an AIV task. ──
    # Walk batches in order; for each, append its real blocks [0, ctx_blocks[b])
    # to the table at a running cursor (prefix-sum), so the table is gap-free and
    # `core = w % NUM_CORES` over [0, fa_total) load-balances equal-cost blocks
    # across cores regardless of per-batch length skew.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_work_build"):
        cursor = pl.read(seq_lens, [0]) * 0  # scalar 0 (INDEX)
        for wb in pl.unroll(BATCH):
            wb_ctx = (pl.read(seq_lens, [wb]) + (SEQ_TILE - 1)) // SEQ_TILE
            for wp in pl.range(wb_ctx):
                # fa_work_table is INT32 (fixed-width for the host↔ptoas ABI; see
                # KNOWN_ISSUES "pl.INDEX GM tensor width mismatch") — cast the
                # index-typed encoding to match the tensor dtype.
                pl.tensor.write(
                    fa_work_table, [cursor + wp, 0], pl.cast(wb * MAX_CTX_BLOCKS + wp, target_type=pl.INT32)
                )
            cursor = cursor + wb_ctx
        pl.tensor.write(fa_total, [0, 0], pl.cast(cursor, target_type=pl.INT32))

    # ── Scope 2: per-head Qwen3 QK-norm, SPLIT into two INDEPENDENT steps. ──
    # Same trick as the input RMSNorm above. QKnorm(x)_head = (x * qk_inv_head) *
    # gamma, where qk_inv_head[b,h] = 1/sqrt(mean_d x^2 + eps) is a per-(row, head)
    # SCALAR. RoPE is linear within a head, so qk_inv commutes through it. We
    # therefore (a) apply ONLY the elementwise `* gamma` here — q_proj_norm /
    # k_proj_norm no longer wait on the reduction — and (b) DEFER the per-head
    # qk_inv reciprocal past RoPE, folding it into rope_qkv as one scalar mul per
    # head-row. The two steps read q_proj / k_proj independently, so `qk_recip`
    # (the sum-of-squares reduction) overlaps `qk_gamma` instead of serializing
    # after it.
    #
    # CONTROL EXPERIMENT: the deferred input-RMSNorm inv_rms is a POSITIVE per-row
    # scalar and QK-norm is scale-invariant, so it CANCELS inside this QK-norm and
    # the optimized path omits it on Q/K. Here we instead APPLY it explicitly — we
    # scale q_proj / k_proj by inv_rms[b] BEFORE the QK-norm in BOTH sub-steps
    # (qk_gamma AND qk_recip). Because the reciprocal step sees inv_rms*x its
    # denominator picks up a 1/inv_rms factor that exactly undoes the inv_rms in
    # the gamma step, so q_out / k_out are bit-for-bit the SAME as the optimized
    # path (RoPE folds the two together). This makes the full mathematical chain
    # input-RMSNorm -> proj -> QK-norm visible in the code, at the cost of two
    # redundant row-scales. Must be in BOTH steps; applying it to only one would
    # NOT cancel and would change the result.
    #
    # qk_inv layout is (head, batch[, q-in-group]) so rope_qkv can slice a
    # contiguous per-(KV head, batch) column. Per head h, the q reduction yields
    # [BATCH * Q_PER_KV, 1] rows ordered (b, j); we stack the NUM_KV_HEADS blocks,
    # so global row = h*BATCH*Q_PER_KV + b*Q_PER_KV + j.
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_inv_states = pl.create_tensor([NUM_KV_HEADS * BATCH * Q_PER_KV, 1], dtype=pl.FP32)
    k_inv_states = pl.create_tensor([NUM_KV_HEADS * BATCH, 1], dtype=pl.FP32)

    # inv_rms[b] as a [BATCH, 1] column — applied row-wise to q_proj / k_proj
    # BEFORE QK-norm in both sub-steps below (the control-experiment scale).
    inv_rms_col = inv_rms_states[:, 0:1]

    # Step (a): input inv_rms (control scale) + `* gamma` — elementwise along
    # HEAD_DIM, no reduction dep.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_gamma"):
        for h in pl.range(NUM_KV_HEADS):
            q0 = h * Q_PER_KV * HEAD_DIM
            # Scale by inv_rms[b] per row while still [BATCH, Q_PER_KV*HEAD_DIM]
            # (one column hits all Q_PER_KV heads of a batch row), then reshape.
            q_slice = pl.row_expand_mul(
                pl.slice(q_proj, [BATCH, Q_PER_KV * HEAD_DIM], [0, q0]), inv_rms_col
            )
            q_chunk = pl.reshape(q_slice, [BATCH * Q_PER_KV, HEAD_DIM])
            q_g = pl.col_expand_mul(q_chunk, q_norm_w)
            q_proj_norm = pl.assemble(
                q_proj_norm, pl.reshape(q_g, [BATCH, Q_PER_KV * HEAD_DIM]), [0, q0]
            )
            k0 = h * HEAD_DIM
            k_chunk = pl.row_expand_mul(pl.slice(k_proj, [BATCH, HEAD_DIM], [0, k0]), inv_rms_col)
            k_g = pl.col_expand_mul(k_chunk, k_norm_w)
            k_proj_norm = pl.assemble(k_proj_norm, k_g, [0, k0])

    # Step (b): per-head 1/rms reciprocal — DEFERRED, folded into rope_qkv below.
    # Same inv_rms[b] control scale on the input; its 1/inv_rms in the denominator
    # cancels the inv_rms in step (a) (see CONTROL EXPERIMENT note above).
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_recip"):
        for h in pl.range(NUM_KV_HEADS):
            q0 = h * Q_PER_KV * HEAD_DIM
            q_slice = pl.row_expand_mul(
                pl.slice(q_proj, [BATCH, Q_PER_KV * HEAD_DIM], [0, q0]), inv_rms_col
            )
            q_chunk = pl.reshape(q_slice, [BATCH * Q_PER_KV, HEAD_DIM])
            q_ss = pl.row_sum(pl.mul(q_chunk, q_chunk))
            q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
            q_inv_states = pl.assemble(q_inv_states, q_inv, [h * BATCH * Q_PER_KV, 0])
            k0 = h * HEAD_DIM
            k_chunk = pl.row_expand_mul(pl.slice(k_proj, [BATCH, HEAD_DIM], [0, k0]), inv_rms_col)
            k_ss = pl.row_sum(pl.mul(k_chunk, k_chunk))
            k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS)))
            k_inv_states = pl.assemble(k_inv_states, k_inv, [h * BATCH, 0])

    # Per-batch rope_kv_cache: K RoPE + K/V cache write + Q RoPE + Q-pad.
    for b in pl.spmd(BATCH, name_hint="rope_qkv"):
        ctx_len = pl.read(seq_lens, [b])
        # Deferred RMSNorm denominator: scale this row's q/k/v by inv_rms[b] (a
        # per-row scalar) here, instead of normalizing before the projection.
        # Read on-device so it auto-deps on the rms_recip producer.
        inv_rms_b = pl.read(inv_rms_states, [b, 0])
        pos = ctx_len - 1  # absolute position -> RoPE cos/sin row (NOT the cache row)
        # Paged write target for this row's current token: slot_mapping[b] decomposes
        # into (physical page, in-page offset). Same scheme prefill_fwd uses.
        wr_slot = pl.cast(pl.tensor.read(slot_mapping, [b]), pl.INDEX)
        wr_slot_block = wr_slot // BLOCK_SIZE
        wr_slot_offset = wr_slot - wr_slot_block * BLOCK_SIZE
        cos_lo = rope_cos[pos : pos + 1, 0:HALF_DIM]
        cos_hi = rope_cos[pos : pos + 1, HALF_DIM:HEAD_DIM]
        sin_lo = rope_sin[pos : pos + 1, 0:HALF_DIM]
        sin_hi = rope_sin[pos : pos + 1, HALF_DIM:HEAD_DIM]

        for ki in pl.pipeline(NUM_KV_HEADS, stage=2):
            kv_col = ki * HEAD_DIM
            # K carries the qk_norm gamma (applied batch-wide in qk_gamma above);
            # fold in the DEFERRED per-head qk_inv scalar here (one scalar mul before
            # the linear RoPE — read on-device so it auto-deps on qk_recip). The
            # input-RMSNorm inv_rms cancels inside qk_norm, so no extra inv_rms factor
            # is needed on K. Then split lo/hi for RoPE.
            k_inv_b = pl.read(k_inv_states, [ki * BATCH + b, 0])
            k_full = pl.mul(k_proj_norm[b : b + 1, kv_col : kv_col + HEAD_DIM], k_inv_b)
            k_lo = k_full[:, 0:HALF_DIM]
            k_hi = k_full[:, HALF_DIM:HEAD_DIM]
            rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
            rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
            cache_row = layer_cache_base + (wr_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + wr_slot_offset
            k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_row, 0])
            k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_row, HALF_DIM])
            v_row_bf16 = pl.cast(
                pl.mul(v_proj[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM], inv_rms_b),
                target_type=pl.BF16,
            )
            v_cache = pl.assemble(v_cache, v_row_bf16, [cache_row, 0])

            q_base = ki * Q_PER_KV
            q_pad_row0 = b * NUM_KV_HEADS * Q_HEAD_PAD + ki * Q_HEAD_PAD
            # Q carries the qk_norm gamma (applied batch-wide in qk_gamma above);
            # fold in the DEFERRED per-head qk_inv scalar here. The Q_PER_KV Q heads
            # bundled under this KV head each have their OWN qk_inv scalar, so we
            # scale + rotate one head at a time (a bundled [Q_PER_KV, 1] column is
            # too small for the 32B col-major tile-alloc constraint; per-head
            # pl.read scalars sidestep it). inv_rms cancels inside qk_norm, so no
            # inv_rms factor on Q either.
            q_inv_base = ki * BATCH * Q_PER_KV + b * Q_PER_KV
            for qj in pl.range(Q_PER_KV):
                q_inv_bj = pl.read(q_inv_states, [q_inv_base + qj, 0])
                q_head = pl.mul(
                    q_proj_norm[
                        b : b + 1, (q_base + qj) * HEAD_DIM : (q_base + qj + 1) * HEAD_DIM
                    ],
                    q_inv_bj,
                )
                q_lo = q_head[:, 0:HALF_DIM]
                q_hi = q_head[:, HALF_DIM:HEAD_DIM]
                q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                all_q_padded = pl.assemble(
                    all_q_padded, pl.cast(q_rot_lo, target_type=pl.BF16), [q_pad_row0 + qj, 0]
                )
                all_q_padded = pl.assemble(
                    all_q_padded, pl.cast(q_rot_hi, target_type=pl.BF16), [q_pad_row0 + qj, HALF_DIM]
                )
            q_pad_zero = pl.cast(
                pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),
                target_type=pl.BF16,
            )
            all_q_padded = pl.assemble(all_q_padded, q_pad_zero, [q_pad_row0 + Q_HEAD_BATCH, 0])

    # ── Scope 3b MLP-accumulator seeds, HOISTED between rope and attn. ──
    # gate/up/down accumulators are zeroed for the later split-K atomic-add
    # tasks; the seeds have NO data dependency on rope or attention, so placing
    # them here lets the scheduler overlap the (vector) zero-fills with the
    # fa_fused cube/vector work instead of serializing them at the head of the
    # MLP manual_scope. Their TASK_IDs (seed_tid / gate_seed_tid / up_seed_tid)
    # cross into the manual_scope below as explicit deps — the same pattern as
    # attn_fence's attn_done_tid. The accumulator create_tensor calls move up
    # with them (a seed must follow its tensor's allocation).
    down_acc_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    gate_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)
    up_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_seed") as seed_tid:
        for nb in pl.pipeline(DOWN_ON, stage=2):
            n0 = nb * DOWN_TN
            zero = pl.full([BATCH, DOWN_TN], dtype=pl.FP32, value=0.0)
            down_acc_all = pl.assemble(down_acc_all, zero, [0, n0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_seed") as gate_seed_tid:
        for nb in pl.pipeline(MLP_ON, stage=2):
            n0 = nb * MLP_TN
            zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
            gate_acc_all = pl.assemble(gate_acc_all, zero, [0, n0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_seed") as up_seed_tid:
        for nb in pl.pipeline(MLP_ON, stage=2):
            n0 = nb * MLP_TN
            zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
            up_acc_all = pl.assemble(up_acc_all, zero, [0, n0])

    # fa_fused: ONE mixed cube+vec root (QK -> softmax -> SV), BLOCK-LEVEL dense
    # static dispatch. Each grid-stride step processes exactly ONE real seq-block
    # (×GP_SIZE heads), decoded from the dense work table. Because the table holds
    # only real blocks, core = w % NUM_CORES distributes total_real_blocks evenly
    # across cores (≈1.25x of ideal), independent of per-batch length skew.
    for fa_core in pl.spmd(
        NUM_CORES,
        name_hint="fa_fused",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        # Read the device-computed block count ON-DEVICE (inside the kernel) so
        # `fa_total` enters fa_fused as an INPUT and the normal task auto-dep
        # orders it after the `fa_work_build` producer. Reading it at
        # orchestration scope instead lowers to a host get_tensor_data that does
        # NOT wait for the producer (stale ~0 read → empty dispatch; see
        # KNOWN_ISSUES). This mirrors the existing on-device fa_work_table read.
        fa_total_blocks = pl.cast(pl.read(fa_total, [0, 0]), target_type=pl.INDEX)
        # Grid-stride over the dense real-block list: core fa_core owns table
        # entries fa_core, fa_core+NUM_CORES, … < fa_total_blocks.
        for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):
            fa_enc = pl.cast(pl.read(fa_work_table, [fa_w, 0]), target_type=pl.INDEX)
            fa_b = fa_enc // MAX_CTX_BLOCKS
            fa_p = fa_enc % MAX_CTX_BLOCKS
            fa_hg = 0  # HEAD_GROUPS == 1 (GP_SIZE == NUM_KV_HEADS)
            fa_ctx_len = pl.read(seq_lens, [fa_b])
            # Table holds only real blocks → exactly one block per entry, at fa_p.
            sb = fa_p  # logical KV block index (no inner loop — old p_blocks was 1)
            s0 = sb * SEQ_TILE
            valid_len = pl.min(SEQ_TILE, fa_ctx_len - s0)
            # Paged read: map logical block sb -> physical page via this request's
            # block_table row. SEQ_TILE == page_size, so one page is exactly one
            # contiguous SEQ_TILE-row slice of the pool (shared with the kvh below).
            fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)

            # Declarative software pipeline over the per-head gp loop: instead of
            # manually unrolling and reordering (QK0,QK1 -> softmax0,softmax1 ->
            # SV0,SV1), let pl.pipeline stage=2 overlap iteration i+1's QK (AIC)
            # with iteration i's softmax (AIV) through the C2V/V2C pipe.
            for gp in pl.pipeline(GP_SIZE, stage=2):
                gi = fa_hg * GP_SIZE + gp
                kvh = gi  # Q_GROUPS=1
                q_pad_row_g = fa_b * NUM_KV_HEADS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]
                g_base = (fa_b * NUM_KV_HEADS + gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD
                cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE

                # QK matmul (cube) -> C2V boundary move (first vec consumer).
                k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]
                raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                scores_scaled = pl.mul(raw_scores, ATTN_SCALE)
                # Mark only the Q_HEAD_BATCH (5) real rows valid (tile stays 16):
                # the vec softmax and the cur_mi / cur_li GM stores then touch 5
                # rows, not the padded 16. NOTE: this cannot shrink the AIC<->AIV
                # C2V/V2C transfer — tpush/tpop always carry the full 16-row tile
                # box (matmul M fractal).
                scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_BATCH, valid_len)
                scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                cur_mi = pl.row_max(scores)
                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                cur_li = pl.row_sum(exp_scores_fp32)

                # SV matmul (cube) reads exp directly -> V2C boundary move.
                v_tile = v_cache[cache_row : cache_row + SEQ_TILE, :]
                oi_tmp = pl.matmul(exp_scores_bf16, v_tile, out_dtype=pl.FP32)
                # The SV-matmul output spans the full padded 16 rows but only
                # the first Q_HEAD_BATCH are real — mark 5 valid (tile stays 16)
                # so the (dominant) all_oi_tmp GM write stores 5 rows instead of 16.
                oi_tmp = pl.set_validshape(oi_tmp, Q_HEAD_BATCH, HEAD_DIM)

                all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [g_base + sb * Q_HEAD_PAD, 0])
                all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + sb * Q_HEAD_PAD, 0])
                all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + sb * Q_HEAD_PAD, 0])

    # online_softmax: flat top-level spmd, writes attn_out directly. Unchanged —
    # reduces per-block partials across ALL blocks of a lane (hence across the
    # KV partitions that produced them); auto-dep on the scratch serializes it
    # after every contributing fa_fused partition.
    for os_core in pl.spmd(NUM_CORES * 2, name_hint="online_softmax"):
        # Grid-stride over online_softmax work items (one per (b, kvh) lane).
        for os_spmd_idx in pl.range(os_core, OS_WORK, NUM_CORES * 2):
            os_b = os_spmd_idx // NUM_KV_HEADS
            os_gi = os_spmd_idx % NUM_KV_HEADS
            os_ctx_len = pl.read(seq_lens, [os_b])
            os_ctx_blocks = (os_ctx_len + SEQ_TILE - 1) // SEQ_TILE
            os_kvh = os_gi  # Q_GROUPS=1
            os_q_base = os_kvh * Q_PER_KV
            os_g_base = (os_b * NUM_KV_HEADS + os_gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD

            # Loop-carried accumulators: full Q_HEAD_PAD (16) rows. They can't be
            # valid-shaped because the recurrence (add / maximum) drops valid_shape,
            # which would break loop-carry type consistency. Only the per-block GM
            # reads below are trimmed to the 5 real rows via pl.slice valid_shape.
            oi = all_oi_tmp[os_g_base : os_g_base + Q_HEAD_PAD, :]
            mi = all_cur_mi[os_g_base : os_g_base + Q_HEAD_PAD, :]
            li = all_cur_li[os_g_base : os_g_base + Q_HEAD_PAD, :]
            for sb in pl.pipeline(1, os_ctx_blocks, stage=2):
                rec = os_g_base + sb * Q_HEAD_PAD
                # Partial load: tile stays Q_HEAD_PAD (16) rows, GM transfer = the 5
                # real rows (valid_shape on the slice → tile.load valid_shapes).
                oi_tmp_valid = pl.slice(
                    all_oi_tmp, [Q_HEAD_PAD, HEAD_DIM], [rec, 0], valid_shape=[Q_HEAD_BATCH, HEAD_DIM]
                )
                online_cur_mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [rec, 0], valid_shape=[Q_HEAD_BATCH, 1])
                online_cur_li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [rec, 0], valid_shape=[Q_HEAD_BATCH, 1])
                mi_new = pl.maximum(mi, online_cur_mi)
                alpha = pl.exp(pl.sub(mi, mi_new))
                beta = pl.exp(pl.sub(online_cur_mi, mi_new))
                li = pl.add(pl.mul(alpha, li), pl.mul(beta, online_cur_li))
                oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp_valid, beta))
                mi = mi_new

            ctx = pl.row_expand_div(oi, li)
            ctx_valid = ctx[0:Q_HEAD_BATCH, :]
            ctx_flat_bf16 = pl.cast(
                pl.reshape(ctx_valid, [1, Q_HEAD_BATCH * HEAD_DIM]), target_type=pl.BF16
            )
            attn_out = pl.assemble(attn_out, ctx_flat_bf16, [os_b, os_q_base * HEAD_DIM])

    # Scope-3 allocations. (down_acc_all / gate_acc_all / up_acc_all are created
    # earlier, alongside their hoisted seed tasks between rope and attn.)
    attn_proj_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    post_norm_partial = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # raw residual h1 (add-back)
    mlp_norm_in = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # h1 * post_gamma (gate/up input)
    inv_rms_tile = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
    mlp_tile = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)

    # Out_seed zeros attn_proj_fp32 in OUT_TN-wide chunks so out_proj
    # split-K tasks can atomic-add into it.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_seed") as out_seed_tid:
        for nb in pl.pipeline(N_SPLITS_OUT, stage=2):
            out_seed_n0 = nb * OUT_TN
            out_zero = pl.full([BATCH, OUT_TN], dtype=pl.FP32, value=0.0)
            attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_zero, [0, out_seed_n0])

    # ── Scope 3a: attn_fence — bridges attn→out_proj across the manual_scope
    # boundary. Reads attn_out (so it auto-deps on the attn loop writers). ──
    attn_fence_dummy = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_fence") as attn_done_tid:
        attn_touch_data = pl.cast(attn_out[0:BATCH, 0:K_CHUNK], target_type=pl.FP32)
        attn_touch_sum = pl.row_sum(attn_touch_data)
        attn_fence_dummy = pl.assemble(attn_fence_dummy, attn_touch_sum, [0, 0])

    # ── Scope 3b: manual_scope MLP block. ──
    with pl.manual_scope():
        silu_tids = pl.array.create(MLP_ON, pl.TASK_ID)
        down_tids = pl.array.create(DOWN_ON * K_SPLITS, pl.TASK_ID)
        gate_tids = pl.array.create(MLP_ON * K_SPLITS_MLP, pl.TASK_ID)
        up_tids = pl.array.create(MLP_ON * K_SPLITS_MLP, pl.TASK_ID)
        cast_tids = pl.array.create(K_SPLITS_MLP, pl.TASK_ID)
        out_tids = pl.array.create(N_SPLITS_OUT * K_SPLITS_OUT, pl.TASK_ID)

        # Split-K split-N out_proj: 10 × 5 = 50 atomic-add tasks.
        for n_out_proj in pl.parallel(N_SPLITS_OUT):
            n_op = n_out_proj * OUT_TN
            for k_split_out in pl.range(K_SPLITS_OUT):
                k_op = k_split_out * OUT_TK
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="out_proj",
                    deps=[out_seed_tid, attn_done_tid],
                ) as out_tid:
                    out_a0 = attn_out[:, k_op : k_op + OUT_INNER_TK]
                    out_w0 = wo[layer_hidden_base + k_op : layer_hidden_base + k_op + OUT_INNER_TK, n_op : n_op + OUT_TN]
                    out_c_acc = pl.matmul(out_a0, out_w0, out_dtype=pl.FP32)
                    for out_lk in pl.pipeline(1, OUT_N_SUB_K, stage=2):
                        out_ks_off = out_lk * OUT_INNER_TK
                        out_a_k = attn_out[:, k_op + out_ks_off : k_op + out_ks_off + OUT_INNER_TK]
                        out_w_k = wo[
                            layer_hidden_base + k_op + out_ks_off : layer_hidden_base + k_op + out_ks_off + OUT_INNER_TK,
                            n_op : n_op + OUT_TN,
                        ]
                        out_c_acc = pl.matmul_acc(out_c_acc, out_a_k, out_w_k)
                    attn_proj_fp32 = pl.assemble(
                        attn_proj_fp32, out_c_acc, [0, n_op], atomic=pl.AtomicType.Add
                    )
                out_tids[n_out_proj * K_SPLITS_OUT + k_split_out] = out_tid

        # Tiled residual + BF16 cast.
        for k_slice in pl.unroll(K_SPLITS_MLP):
            k_base = k_slice * MLP_K_SLICE
            n_split_base = k_slice * N_PER_CAST_K
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="residual_rms_cast",
                deps=[
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 0],
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 1],
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 2],
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 3],
                    out_tids[(n_split_base + 0) * K_SPLITS_OUT + 4],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 0],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 1],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 2],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 3],
                    out_tids[(n_split_base + 1) * K_SPLITS_OUT + 4],
                ],
            ) as cast_tid_k:
                for kb in pl.pipeline(MLP_K_SLICE // K_CHUNK, stage=2):
                    k0 = k_base + kb * K_CHUNK
                    attn_chunk = attn_proj_fp32[:, k0 : k0 + K_CHUNK]
                    hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]
                    resid_fp32 = pl.add(attn_chunk, pl.cast(hidden_chunk, target_type=pl.FP32))
                    # Raw residual h1 — added back after down_proj (must NOT be gamma-scaled).
                    post_norm_partial = pl.assemble(
                        post_norm_partial, pl.cast(resid_fp32, target_type=pl.BF16), [0, k0]
                    )
                    # Explicit post-RMS gamma: gate/up input = h1 * post_gamma. gamma is
                    # per-K (the matmul contraction dim) so it canNOT defer past the matmul
                    # like inv_rms does — it scales the input here (with raw w_gate/w_up).
                    post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                    mlp_norm_in = pl.assemble(
                        mlp_norm_in,
                        pl.cast(pl.col_expand_mul(resid_fp32, post_gamma), target_type=pl.BF16),
                        [0, k0],
                    )
            cast_tids[k_slice] = cast_tid_k

        # RMS reduction reads all of attn_proj_fp32.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rms_reduce", deps=[out_tids]) as reduce_tid:
            sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=2):
                k0 = kb * K_CHUNK
                attn_chunk = attn_proj_fp32[:, k0 : k0 + K_CHUNK]
                hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]
                resid_chunk = pl.add(attn_chunk, pl.cast(hidden_chunk, target_type=pl.FP32))
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH]),
                )
            post_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
            post_inv_rms_col = pl.reshape(post_inv_rms, [BATCH, 1])
            inv_rms_tile = pl.assemble(inv_rms_tile, post_inv_rms_col, [0, 0])

        # Split-K gate + up interleaved.
        for n_out in pl.parallel(MLP_ON):
            n0 = n_out * MLP_TN
            for k_split in pl.range(K_SPLITS_MLP):
                k0 = k_split * MLP_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="gate_proj",
                    deps=[cast_tids[k_split], gate_seed_tid],
                ) as gate_tid:
                    a0 = mlp_norm_in[:, k0 : k0 + MLP_INNER_TK]
                    w0 = w_gate[layer_hidden_base + k0 : layer_hidden_base + k0 + MLP_INNER_TK, n0 : n0 + MLP_TN]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):
                        ks_off = lk * MLP_INNER_TK
                        a_k = mlp_norm_in[:, k0 + ks_off : k0 + ks_off + MLP_INNER_TK]
                        w_k = w_gate[layer_hidden_base + k0 + ks_off : layer_hidden_base + k0 + ks_off + MLP_INNER_TK, n0 : n0 + MLP_TN]
                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)
                    gate_acc_all = pl.assemble(gate_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)
                gate_tids[n_out * K_SPLITS_MLP + k_split] = gate_tid

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="up_proj",
                    deps=[cast_tids[k_split], up_seed_tid],
                ) as up_tid:
                    a0 = mlp_norm_in[:, k0 : k0 + MLP_INNER_TK]
                    w0 = w_up[layer_hidden_base + k0 : layer_hidden_base + k0 + MLP_INNER_TK, n0 : n0 + MLP_TN]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):
                        ks_off = lk * MLP_INNER_TK
                        a_k = mlp_norm_in[:, k0 + ks_off : k0 + ks_off + MLP_INNER_TK]
                        w_k = w_up[layer_hidden_base + k0 + ks_off : layer_hidden_base + k0 + ks_off + MLP_INNER_TK, n0 : n0 + MLP_TN]
                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)
                    up_acc_all = pl.assemble(up_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)
                up_tids[n_out * K_SPLITS_MLP + k_split] = up_tid

        # silu.
        for n_out in pl.parallel(MLP_ON):
            n0 = n_out * MLP_TN
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="silu",
                deps=[
                    reduce_tid,
                    gate_tids[n_out * K_SPLITS_MLP + 0],
                    gate_tids[n_out * K_SPLITS_MLP + 1],
                    gate_tids[n_out * K_SPLITS_MLP + 2],
                    gate_tids[n_out * K_SPLITS_MLP + 3],
                    gate_tids[n_out * K_SPLITS_MLP + 4],
                    up_tids[n_out * K_SPLITS_MLP + 0],
                    up_tids[n_out * K_SPLITS_MLP + 1],
                    up_tids[n_out * K_SPLITS_MLP + 2],
                    up_tids[n_out * K_SPLITS_MLP + 3],
                    up_tids[n_out * K_SPLITS_MLP + 4],
                ],
            ) as silu_tid:
                inv_rms_chunk = inv_rms_tile[:, 0:1]
                for sub in pl.pipeline(SILU_INNER_CHUNKS, stage=2):
                    silu_off = n0 + sub * MLP_OUT_CHUNK
                    gate_chunk = gate_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]
                    up_chunk = up_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]
                    scaled_gate = pl.row_expand_mul(gate_chunk, inv_rms_chunk)
                    scaled_up = pl.row_expand_mul(up_chunk, inv_rms_chunk)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(scaled_gate)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(scaled_gate, sigmoid), scaled_up)
                    mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, silu_off])
            silu_tids[n_out] = silu_tid

        for n_out in pl.parallel(DOWN_ON):
            n0 = n_out * DOWN_TN
            for k_split in pl.range(K_SPLITS):
                k0 = k_split * DOWN_TN
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="down_proj",
                    deps=[seed_tid, silu_tids[k_split]],
                ) as down_tid:
                    a0 = mlp_tile[:, k0 : k0 + DOWN_TK]
                    w0 = w_down[layer_inter_base + k0 : layer_inter_base + k0 + DOWN_TK, n0 : n0 + DOWN_TN]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, N_SUB_K, stage=2):
                        ks_off = lk * DOWN_TK
                        a_k = mlp_tile[:, k0 + ks_off : k0 + ks_off + DOWN_TK]
                        w_k = w_down[layer_inter_base + k0 + ks_off : layer_inter_base + k0 + ks_off + DOWN_TK, n0 : n0 + DOWN_TN]
                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)
                    down_acc_all = pl.assemble(down_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)
                down_tids[n_out * K_SPLITS + k_split] = down_tid

        for n_out in pl.parallel(DOWN_ON):
            n0 = n_out * DOWN_TN
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="down_cast_residual",
                deps=[down_tids[n_out * K_SPLITS + k] for k in range(K_SPLITS)],
            ) as dcr_tid:
                resid_block_bf16 = post_norm_partial[:, n0 : n0 + DOWN_TN]
                resid_block = pl.cast(resid_block_bf16, target_type=pl.FP32)
                acc_chunk = down_acc_all[:, n0 : n0 + DOWN_TN]
                out_chunk = pl.add(acc_chunk, resid_block)
                out_partial = pl.assemble(out_partial, pl.cast(out_chunk, target_type=pl.BF16), [0, n0])
            dcr_tids[n_out] = dcr_tid

    # ── out_consolidate (OUTSIDE manual_scope): copy out_partial -> out in a SINGLE
    # pl.at task in the AUTO-DEP region. Inside manual_scope, auto-dep (tensormap)
    # registration is suppressed (explicit deps= only), so a write to `out` there is
    # invisible to caller-scope readers. The body's terminal write is otherwise the
    # DOWN_ON-way `down_cast_residual` pl.parallel fan-out inside manual_scope, whose
    # partial writers register NO tensormap edge to a downstream caller reader (proven
    # on device: next_hidden's writers had no edge to copy_out). Re-emitting `out` here
    # as a single full-tensor writer in the auto-dep region — exactly the copy_hidden
    # -> x_gamma path that works — makes the layer output visible to the next layer /
    # copy_out / LM head, fixing the fused multi-layer decode_fwd with no caller-side
    # plumbing. Gated on all down_cast_residual writers via deps=.
    for oc_b in pl.parallel(0, BATCH, BATCH):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="out_consolidate",
            deps=[dcr_tids[k] for k in range(DOWN_ON)],
        ):
            for oc_kb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                oc_k0 = oc_kb * RMSNORM_K_CHUNK
                out = pl.assemble(
                    out, pl.slice(out_partial, [BATCH, RMSNORM_K_CHUNK], [oc_b, oc_k0]), [oc_b, oc_k0]
                )
    return out


NUM_LAYERS = 40  # full Qwen3-14B depth, for the fused decode_fwd loop
_FWD_NLAYERS = NUM_LAYERS  # decode_fwd loop bound; overridable for layer-count tests


@pl.jit
def qwen3_decode_mpmd(  # noqa: PLR0913 — single-layer entry (external API unchanged)
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    res = _decode_layer(
        hidden_states, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight,
        seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache, wo, w_gate, w_up, w_down,
        post_rms_weight, out, 0,
    )
    return res


@pl.jit
def decode_fwd(  # noqa: PLR0913 — device-side fused NUM_LAYERS decode + LM head
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    final_norm_weight: pl.Tensor,
    lm_head_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    # Device-side fused decode: loop the inline body over all _FWD_NLAYERS layers,
    # slicing each layer's weights / KV-cache region by layer_idx, then run the LM
    # head. Weights are STACKED [_FWD_NLAYERS*HIDDEN, ...] / [_FWD_NLAYERS*INTERMEDIATE,
    # ...]; k_cache / v_cache cover [_FWD_NLAYERS*BATCH*NUM_KV_HEADS*MAX_SEQ, ...]; out
    # is logits [BATCH, VOCAB]. _FWD_NLAYERS defaults to NUM_LAYERS (40) and is settable
    # for layer-count tests.
    #
    # The loop-carried `cur` is seeded from hidden_states via a create_tensor + tiled
    # copy (copy_hidden, a single full-tensor pl.at writer). Each layer's output is made
    # visible to the next layer / the LM head by _decode_layer's `out_consolidate` scope
    # (a single full-tensor writer in the auto-dep region) — without it, the inline body's
    # `down_cast_residual` pl.parallel writes do not register a tensormap edge to the
    # downstream reader and the fused output is garbage. See decode-fwd-dep-fix notes.
    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for cb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                cur = pl.assemble(
                    cur, pl.slice(hidden_states, [BATCH, RMSNORM_K_CHUNK], [cb0, ck0]), [cb0, ck0]
                )
    for layer_idx in pl.range(_FWD_NLAYERS):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        cur = _decode_layer(
            cur, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache, wo, w_gate, w_up, w_down,
            post_rms_weight, next_hidden, layer_idx,
        )
    out = rms_lm_head(cur, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


_CHUNK_NLAYERS = 8  # layers per decode_fwd_layers dispatch (chunked fused decode)


@pl.jit
def decode_fwd_layers(  # noqa: PLR0913 — fused decode of a CONTIGUOUS layer CHUNK, no LM head
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    # Fused decode of _CHUNK_NLAYERS consecutive layers, output = hidden (NO LM head).
    # Used to run all 40 layers in a few dispatches instead of one — a single 40-layer
    # dispatch exceeds the device AICPU stream-sync timeout (PLATFORM_STREAM_SYNC_TIMEOUT
    # _MS=2000ms). Each layer's output is made visible to the next by _decode_layer's
    # out_consolidate (the fused-decode dependency fix). The caller passes the weight /
    # KV-cache SLICES for the chunk's layers (stacked [_CHUNK_NLAYERS*dim, ...]); the body
    # indexes them with layer_idx 0.._CHUNK_NLAYERS-1.
    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for cb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                cur = pl.assemble(
                    cur, pl.slice(hidden_states, [BATCH, RMSNORM_K_CHUNK], [cb0, ck0]), [cb0, ck0]
                )
    for i in pl.range(_CHUNK_NLAYERS):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        cur = _decode_layer(
            cur, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache, wo, w_gate, w_up, w_down,
            post_rms_weight, next_hidden, i,
        )
    for ob0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_out"):
            for okb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ok0 = okb * RMSNORM_K_CHUNK
                out = pl.assemble(
                    out, pl.slice(cur, [BATCH, RMSNORM_K_CHUNK], [ob0, ok0]), [ob0, ok0]
                )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Inputs / golden / driver — same data dir layout as qwen3_v4.py.
# ──────────────────────────────────────────────────────────────────────────────

INPUT_NAMES = (
    "hidden_states",
    "input_rms_weight",
    "wq",
    "wk",
    "wv",
    "q_norm_weight",
    "k_norm_weight",
    "seq_lens",
    "block_table",
    "slot_mapping",
    "rope_cos",
    "rope_sin",
    "k_cache",
    "v_cache",
    "wo",
    "w_gate",
    "w_up",
    "w_down",
    "post_rms_weight",
)


def load_inputs(data_in_dir: Path) -> list[torch.Tensor]:
    return [torch.load(data_in_dir / f"{name}.pt", weights_only=True) for name in INPUT_NAMES]


def _paged_block_table_slot_mapping(seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity paging for the standalone harness: batch b's logical blocks map to
    physical pages [b*MAX_BLOCKS_PER_SEQ .. ]; slot_mapping[b] is the current
    token's (pos=seq_len-1) physical row. Mirrors the serving runner's layout."""
    block_table = torch.arange(BATCH * MAX_BLOCKS_PER_SEQ, dtype=torch.int32)
    slot_mapping = torch.empty(BATCH, dtype=torch.int32)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        logical_block = pos // BLOCK_SIZE
        phys_page = b * MAX_BLOCKS_PER_SEQ + logical_block
        slot_mapping[b] = phys_page * BLOCK_SIZE + (pos % BLOCK_SIZE)
    return block_table, slot_mapping


def _smoke_inputs() -> list[torch.Tensor]:
    def randn(shape, dtype):
        return torch.empty(shape, dtype=dtype).normal_()

    seq_lens = torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)
    block_table, slot_mapping = _paged_block_table_slot_mapping(seq_lens)
    return [
        randn([BATCH, HIDDEN], torch.bfloat16),
        randn([1, HIDDEN], torch.float32),
        randn([HIDDEN, HIDDEN], torch.bfloat16),
        randn([HIDDEN, KV_HIDDEN], torch.bfloat16),
        randn([HIDDEN, KV_HIDDEN], torch.bfloat16),
        torch.ones([1, HEAD_DIM], dtype=torch.float32),  # q_norm_weight
        torch.ones([1, HEAD_DIM], dtype=torch.float32),  # k_norm_weight
        seq_lens,
        block_table,
        slot_mapping,
        randn([MAX_SEQ, HEAD_DIM], torch.float32),
        randn([MAX_SEQ, HEAD_DIM], torch.float32),
        randn([CACHE_ROWS, HEAD_DIM], torch.bfloat16),
        randn([CACHE_ROWS, HEAD_DIM], torch.bfloat16),
        randn([HIDDEN, HIDDEN], torch.bfloat16),
        randn([HIDDEN, INTERMEDIATE], torch.bfloat16),
        randn([HIDDEN, INTERMEDIATE], torch.bfloat16),
        randn([INTERMEDIATE, HIDDEN], torch.bfloat16),
        torch.ones([1, HIDDEN], dtype=torch.float32),  # post_rms_weight
    ]


def _backend_type(platform: str) -> BackendType:
    return BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B


# ──────────────────────────────────────────────────────────────────────────────
# On-the-fly random fixture + torch golden for the single-layer unit test.
#
# The default `-p a2a3 -d N` run builds a deterministic RANDOM fixture, computes
# the golden with `golden_decode_layer` (a torch reference mirroring the kernel's
# math AND its bf16 cast points), runs qwen3_decode_mpmd on device through the
# `golden/` harness (golden.run_jit), and validates the device output against the
# golden — no pre-generated data files needed.
#
# Fixture scales are chosen so the (unnormalized) residual-stream output stays
# O(1) and attention is well conditioned: large output magnitudes make the bf16
# output fail rtol=3e-3 (one bf16 ULP then exceeds the relative tolerance), and a
# zero-mean / near-uniform attention drives the kernel's bf16-exp / bf16-matmul
# attention away from an fp32 reference. So past KV is small (current qk-normed
# token dominates the softmax) and V carries a nonzero mean (stable attention
# average), while wo / w_down are extra-small (modest residual perturbations).
# ──────────────────────────────────────────────────────────────────────────────

_REF_ATTN_SCALE = 1.0 / (HEAD_DIM**0.5)


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """Round through bf16 then back to fp32 — emulates a kernel ``pl.cast(BF16)``."""
    return t.to(torch.bfloat16).to(torch.float32)


def _rmsnorm_inv(x: torch.Tensor) -> torch.Tensor:
    """1/sqrt(mean(x^2, last) + eps), keepdim — the kernel's deferred denominator."""
    return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)


def _rope_half(vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Kernel RoPE (NeoX half-split): rot_lo = lo*cos_lo - hi*sin_lo ; rot_hi = hi*cos_hi + lo*sin_hi."""
    lo, hi = vec[..., :HALF_DIM], vec[..., HALF_DIM:]
    cos_lo, cos_hi = cos[..., :HALF_DIM], cos[..., HALF_DIM:]
    sin_lo, sin_hi = sin[..., :HALF_DIM], sin[..., HALF_DIM:]
    return torch.cat([lo * cos_lo - hi * sin_lo, hi * cos_hi + lo * sin_hi], dim=-1)


def random_inputs(full_seq: bool = False, seed: int = 1234) -> dict[str, torch.Tensor]:
    """Deterministic random fixture (name -> tensor) for qwen3_decode_mpmd.

    full_seq: set every sequence length to MAX_SEQ (full KV cache) for a stable,
    maximum-load performance run; otherwise sample varied lengths in [1, MAX_SEQ].
    """
    g = torch.Generator().manual_seed(seed)

    def rn(shape, std=1.0, bias=0.0):
        return torch.empty(shape).normal_(0.0, std, generator=g) + bias

    if full_seq:
        seq_lens = torch.full([BATCH], MAX_SEQ, dtype=torch.int32)
    else:
        seq_lens = torch.randint(1, MAX_SEQ + 1, (BATCH,), generator=g, dtype=torch.int32)

    # Paged block_table / slot_mapping (identity paging) for the PAGED KV pool the
    # kernel reads — k_cache/v_cache below are sized as the paged pool (CACHE_ROWS).
    block_table, slot_mapping = _paged_block_table_slot_mapping(seq_lens)

    # Proper NeoX half-split RoPE tables (cols [0:64] and [64:128] duplicated).
    posv = torch.arange(MAX_SEQ).float().unsqueeze(1)
    inv_freq = 1.0 / (1.0e4 ** (torch.arange(0, HALF_DIM).float() / HALF_DIM))
    ang = posv * inv_freq.unsqueeze(0)
    rope_cos = torch.cat([ang.cos(), ang.cos()], dim=1).float()
    rope_sin = torch.cat([ang.sin(), ang.sin()], dim=1).float()

    return {
        "hidden_states": rn([BATCH, HIDDEN], 1.0).to(torch.bfloat16),
        "input_rms_weight": rn([1, HIDDEN], 0.1, 1.0).float(),
        "wq": rn([HIDDEN, HIDDEN], 0.02).to(torch.bfloat16),
        "wk": rn([HIDDEN, KV_HIDDEN], 0.02).to(torch.bfloat16),
        "wv": rn([HIDDEN, KV_HIDDEN], 0.02).to(torch.bfloat16),
        "q_norm_weight": rn([1, HEAD_DIM], 0.1, 1.0).float(),
        "k_norm_weight": rn([1, HEAD_DIM], 0.1, 1.0).float(),
        "seq_lens": seq_lens,
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "k_cache": rn([CACHE_ROWS, HEAD_DIM], 0.01).to(torch.bfloat16),
        "v_cache": rn([CACHE_ROWS, HEAD_DIM], 0.02, 0.3).to(torch.bfloat16),
        "wo": rn([HIDDEN, HIDDEN], 0.0006).to(torch.bfloat16),
        "w_gate": rn([HIDDEN, INTERMEDIATE], 0.02).to(torch.bfloat16),
        "w_up": rn([HIDDEN, INTERMEDIATE], 0.02).to(torch.bfloat16),
        "w_down": rn([INTERMEDIATE, HIDDEN], 0.0004).to(torch.bfloat16),
        "post_rms_weight": rn([1, HIDDEN], 0.1, 1.0).float(),
    }


def golden_decode_layer(values: dict) -> None:
    """Torch reference for ONE Qwen3 decode layer; fills ``values['out']`` in place.

    Mirrors qwen3_decode_mpmd / _decode_layer at layer_idx 0: RMSNorm -> Q/K/V proj
    -> per-head QK-norm -> RoPE -> KV-cache write at pos=seq_len-1 -> GQA flash
    attention over [0, seq_len) -> out_proj -> residual -> post-RMSNorm -> SwiGLU
    MLP -> residual. The deferred input-RMSNorm inv_rms and the QK-norm control
    scale cancel to the standard math (QK-norm is scale-invariant).
    """
    x = values["hidden_states"].float()                          # [B,H], residual source
    gamma_in = values["input_rms_weight"].float()[0]             # [H]
    inv_rms = _rmsnorm_inv(x)                                    # [B,1] (deferred)

    normed = _bf16(x * gamma_in)                                 # bf16 normed (no inv_rms yet)
    q_proj = normed @ values["wq"].float()                      # [B,H]
    k_proj = normed @ values["wk"].float()                      # [B,KVH]
    v_proj = normed @ values["wv"].float()                      # [B,KVH]

    qn = values["q_norm_weight"].float()[0]
    kn = values["k_norm_weight"].float()[0]
    qh = (q_proj * inv_rms).reshape(BATCH, NUM_HEADS, HEAD_DIM)
    qh = qh * _rmsnorm_inv(qh) * qn                              # per-head QK-norm
    kh = (k_proj * inv_rms).reshape(BATCH, NUM_KV_HEADS, HEAD_DIM)
    kh = kh * _rmsnorm_inv(kh) * kn
    v_heads = (v_proj * inv_rms).reshape(BATCH, NUM_KV_HEADS, HEAD_DIM)

    seq_lens = values["seq_lens"]
    block_table = values["block_table"]
    rope_cos = values["rope_cos"].float()
    rope_sin = values["rope_sin"].float()
    k_cache = values["k_cache"].float()
    v_cache = values["v_cache"].float()

    attn_out = torch.zeros(BATCH, HIDDEN)
    for b in range(BATCH):
        slen = int(seq_lens[b].item())
        p = slen - 1
        cos_p, sin_p = rope_cos[p], rope_sin[p]
        q_b = _bf16(_rope_half(qh[b], cos_p, sin_p))             # [40,128] current Q (bf16)
        k_cur = _bf16(_rope_half(kh[b], cos_p, sin_p))           # [8,128] current K (bf16)
        v_cur = _bf16(v_heads[b])                                # [8,128]
        n_blocks = (slen + BLOCK_SIZE - 1) // BLOCK_SIZE
        for kvh in range(NUM_KV_HEADS):
            # Gather this (b, kvh) lane's past KV from the PAGED pool exactly as the
            # kernel does: logical block sb -> physical page block_table[b*MBPS + sb],
            # whose (page, kvh) tile starts at (pbid*NUM_KV_HEADS + kvh)*BLOCK_SIZE.
            k_lane = torch.empty(slen, HEAD_DIM)
            v_lane = torch.empty(slen, HEAD_DIM)
            for sb in range(n_blocks):
                pbid = int(block_table[b * MAX_BLOCKS_PER_SEQ + sb].item())
                row = (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                lo = sb * BLOCK_SIZE
                blk = min(BLOCK_SIZE, slen - lo)
                k_lane[lo : lo + blk] = k_cache[row : row + blk]
                v_lane[lo : lo + blk] = v_cache[row : row + blk]
            k_lane[p] = k_cur[kvh]                               # current token (kernel writes it first)
            v_lane[p] = v_cur[kvh]
            for j in range(Q_PER_KV):
                hq = kvh * Q_PER_KV + j
                scores = (q_b[hq].unsqueeze(0) * k_lane).sum(-1) * _REF_ATTN_SCALE
                w = torch.softmax(scores, dim=-1)
                attn_out[b, hq * HEAD_DIM : (hq + 1) * HEAD_DIM] = (w.unsqueeze(-1) * v_lane).sum(0)
    attn_out = _bf16(attn_out)

    attn_proj = attn_out @ values["wo"].float()                  # out_proj
    h1 = x + attn_proj                                           # raw residual
    h1_bf16 = _bf16(h1)
    post_gamma = values["post_rms_weight"].float()[0]
    post_inv = _rmsnorm_inv(h1)                                  # deferred into silu
    mlp_in = _bf16(h1 * post_gamma)                              # gamma before inv_rms
    gate = mlp_in @ values["w_gate"].float()
    up = mlp_in @ values["w_up"].float()
    sg = gate * post_inv
    su = up * post_inv
    mlp = _bf16(sg * torch.sigmoid(sg) * su)                     # SwiGLU
    down = mlp @ values["w_down"].float()
    values["out"] = _bf16(down + h1_bf16).to(torch.bfloat16)     # residual is bf16(h1)


def _build_specs(inputs: dict) -> list:
    """TensorSpec list in qwen3_decode_mpmd parameter order, from a fixture dict."""
    from golden import TensorSpec  # repo root added to sys.path in __main__

    specs = [
        TensorSpec(name, list(inputs[name].shape), inputs[name].dtype, init_value=inputs[name])
        for name in INPUT_NAMES
    ]
    specs.append(TensorSpec("out", [BATCH, HIDDEN], torch.bfloat16, is_output=True))
    return specs


if __name__ == "__main__":
    import sys

    # The `golden/` harness lives at the repo root, which is not on sys.path when
    # this script is launched directly (only its own dir is). CI sets PYTHONPATH to
    # the repo root, so this insert is the standalone-run fallback.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False,
                        help="set EVERY sequence length to MAX_SEQ (full KV cache) for a stable, "
                             "maximum-load performance run; default samples varied random lengths.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="RNG seed for the random fixture (reproducible inputs + golden).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "build_output" / "data",
        help="only used by --validate-fwd (pre-generated stacked-fwd inputs).",
    )
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="compile-only (no device); also the implicit behavior on *sim platforms.")
    parser.add_argument("--no-dep-gen", action="store_true", default=False,
                        help="disable dep_gen (avoids 'register failed: 8' overflow on big graphs)")
    parser.add_argument("--validate-fwd", action="store_true", default=False,
                        help="validate the fused decode_fwd (N stacked layers + on-device LM head "
                             "-> logits) against a host chain reference, instead of the default "
                             "single-layer golden test.")
    parser.add_argument("--fwd-layers", type=int, default=4, help="layer count N for --validate-fwd")
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="persist inputs + golden for replay (off: large fixtures)")
    args = parser.parse_args()

    set_backend_type(_backend_type(args.platform))

    # Compile-only smoke: explicit --smoke, or any *sim platform (the CI
    # `python decode_layer.py -p a2a3sim` sweep — codegen regressions are still
    # caught without needing a device or the heavy full-graph simulation).
    if args.smoke or args.platform.endswith("sim"):
        smoke_out = torch.empty([BATCH, HIDDEN], dtype=torch.bfloat16)
        post_pass = qwen3_decode_mpmd.compile_for_test(*_smoke_inputs(), smoke_out)
        print(f"Compiled program has {len(post_pass.functions)} function(s):")
        for fn in post_pass.functions.values():
            print(f"  {fn.name}: {fn.func_type}")
        raise SystemExit(0)

    # ── Default single-layer unit test: RANDOM inputs, on-the-fly torch golden,
    # on-device run + compare, all through the golden/ harness. ──
    if not args.validate_fwd:
        from golden import ratio_allclose, run_jit

        inputs = random_inputs(full_seq=args.max_seq, seed=args.seed)
        specs = _build_specs(inputs)
        print(f"[decode_layer] single-layer golden unit test | platform={args.platform} "
              f"device={args.device} seq={'MAX' if args.max_seq else 'varied'} seed={args.seed} "
              f"seq_lens={inputs['seq_lens'].tolist()}")
        # Ratio tolerance: bf16 outputs cannot satisfy a strict 100% allclose at
        # rtol/atol=3e-3 (one bf16 ULP at value 1 is 2**-8 ≈ 0.0039 > 3e-3), so allow
        # up to 2% outliers — the codebase's ratio_allclose convention. Remaining
        # mismatches are 1-2 ULP bf16 quantization, not errors.
        result = run_jit(
            fn=qwen3_decode_mpmd,
            specs=specs,
            golden_fn=golden_decode_layer,
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
                enable_dep_gen=not args.no_dep_gen,
            ),
            rtol=3e-3,
            atol=3e-3,
            compare_fn={"out": ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)},
            save_data=args.save_data,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
        raise SystemExit(0)

    # ── --validate-fwd: pre-generated stacked-fwd inputs from --data-dir. ──
    inputs = load_inputs(args.data_dir / "in")
    run_cfg = RunConfig(
        platform=args.platform,
        device_id=args.device,
        backend_type=_backend_type(args.platform),
        enable_l2_swimlane=args.enable_l2_swimlane,
        enable_dep_gen=not args.no_dep_gen,
        dump_passes=True,
    )

    # Full fused decode_fwd validation: N stacked layers + on-device LM head -> logits,
    # vs host (chain N hidden -> final RMSNorm + lm_head matmul). Builds N-layer stacks by
    # replicating the single-layer weights (every layer computes layer 0) and exercises the
    # runtime layer_idx slicing, the layer->layer out_consolidate dependency, and the LM
    # head reading the final layer's consolidated output. The host chain feeds each output
    # as the next hidden (KV past[0:pos] untouched, current pos overwritten each layer, so
    # it reproduces the in-kernel const-layer-0 chain).
    if args.validate_fwd:
        N = args.fwd_layers
        _FWD_NLAYERS = N
        def stack0(t, reps):  # replicate along dim 0
            return torch.cat([t] * reps, dim=0).contiguous()
        hs, irw, wq_, wk_, wv_, qn, kn, sl, bt, sm, rc, rs, kc, vc, wo_, wg, wu, wd, prw = inputs
        torch.manual_seed(1234)
        final_norm_w = torch.empty([1, HIDDEN], dtype=torch.float32).normal_() * 0.1 + 1.0
        lm_head_w = (torch.empty([VOCAB, HIDDEN], dtype=torch.bfloat16).normal_() * 0.02)
        # block_table / slot_mapping are shared across layers (NOT per-layer stacked);
        # the paged pool kc/vc IS stacked N times (one paged pool per layer).
        stacked = [
            hs, stack0(irw, N), stack0(wq_, N), stack0(wk_, N), stack0(wv_, N),
            stack0(qn, N), stack0(kn, N), sl, bt, sm, rc, rs, stack0(kc, N), stack0(vc, N),
            stack0(wo_, N), stack0(wg, N), stack0(wu, N), stack0(wd, N), stack0(prw, N),
            final_norm_w, lm_head_w,
        ]
        logits = torch.zeros(BATCH, VOCAB, dtype=torch.float32)
        decode_fwd(*stacked, logits, config=run_cfg)
        # host ref: chain N -> final RMSNorm -> lm_head
        ref_hidden = inputs[0]; ref_out = None
        for _step in range(N):
            ci = load_inputs(args.data_dir / "in"); ci[0] = ref_hidden
            ref_out = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
            qwen3_decode_mpmd(*ci, ref_out, config=run_cfg)
            ref_hidden = ref_out.clone()
        hn = ref_out.float()
        inv = torch.rsqrt(hn.pow(2).mean(-1, keepdim=True) + EPS)
        ref_normed = (hn * inv) * final_norm_w.float()
        ref_logits = ref_normed @ lm_head_w.float().t()  # [BATCH, VOCAB]
        a = logits.cpu(); e = ref_logits.cpu()
        # compare argmax (the actual generation signal) + value closeness
        amax_k = a.argmax(-1); amax_r = e.argmax(-1)
        argmax_match = int((amax_k == amax_r).sum())
        close = torch.isclose(a, e, rtol=5e-2, atol=5e-2)
        print(f"[stacked-fwd {N}L+LMhead] argmax match {argmax_match}/{BATCH} | "
              f"logits {int(close.sum())/a.numel():.4%} within 5e-2 | "
              f"max_abs_err={(a-e).abs().max():.4f} | kernel_argmax={amax_k.tolist()} ref_argmax={amax_r.tolist()}")
        raise SystemExit(0 if argmax_match == BATCH else 1)
