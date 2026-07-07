# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode kernel — FP32 inter-layer carry + fused layer output.

The inter-layer hidden (hidden_states / out / cur / post_norm_partial) is carried
as FP32. The layer output and the next layer's x*gamma are produced together by a
single fused `dcr_xgamma` task (DOWN_ON-way), emitting `out` (the residual, for
rms_recip + the residual stream) and `normed_out` (x*gamma, for the next layer's
QKV) from the same in-register chunk — no GM round-trip for x_gamma.

Why fuse: inside manual_scope, tensormap / auto-dep registration is suppressed —
only explicit `deps=[tid]` edges apply — so the old DOWN_ON-way
`down_cast_residual` PARTIAL writers (to `out_partial`) registered NO tensormap
edge to the next layer's reader. `out_consolidate` re-emitted `out` as a single
full-tensor writer in the auto-dep region to restore that edge. Carrying the
residual-add out of manual_scope directly into that single consolidated writer
drops both the `out_partial` scratch tensor and the extra copy task: the residual
add (down_acc_all + post_norm_partial, both FP32) now writes `out` straight, as a
single full-tensor auto-dep writer gated on the 85 down_proj TaskIds.

FP32 boundaries:
  * FIRST layer  — `copy_hidden` casts the external BF16 embed input -> FP32 once.
  * LAST layer   — `cast_lmhead_in` casts the final FP32 hidden -> BF16 once for the
                   (unchanged, BF16-input) rms_lm_head.
`decode_fwd` (single-dispatch, with LM head) and `decode_fwd_layers` (chunked, no
LM head) share the one FP32-carry `_decode_layer`.
Numerics differ from the BF16 baseline (more precise — no per-layer BF16 rounding),
so the saved golden's argmax is the sanity check, not bit-equality.

-- original header --
Qwen3-14B decode — FUSED attn + dense balance + gp-loop pl.pipeline.

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

from config import KV_CACHE_ROWS_DYN, REAL_VOCAB, VOCAB  # vocab size for the fused decode_fwd LM head / logits
from rms_lm_head import rms_lm_head  # LM head for the fused multi-layer decode_fwd

# ══════════════════════════════════════════════════════════════════════════════
# Functional config — model architecture + workload.
# ══════════════════════════════════════════════════════════════════════════════

# ── Model architecture (Qwen3-14B, fixed by the checkpoint) ──
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 17408
# MAX_SEQ is env-overridable for the e2e generate harness: it sizes the standalone
# paged KV pool (CACHE_ROWS) and the RoPE tables, so a 512-token run can use a much
# smaller pool than the 4096 micro-benchmark default (less KV memory).
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

EMBED_HIDDEN_CHUNK = 256
SAMPLE_VOCAB_CHUNK = 512
SAMPLE_CHUNK_PAD = 512
SAMPLE_TOPK = 16
SAMPLED_IDS_PAD = 8
SAMPLE_NUM_VOCAB_CHUNKS = VOCAB // SAMPLE_VOCAB_CHUNK
SAMPLE_REAL_NUM_FULL_VOCAB_CHUNKS = REAL_VOCAB // SAMPLE_VOCAB_CHUNK
SAMPLE_REAL_VOCAB_TAIL = REAL_VOCAB % SAMPLE_VOCAB_CHUNK
SAMPLE_REAL_NUM_VOCAB_CHUNKS = SAMPLE_REAL_NUM_FULL_VOCAB_CHUNKS + (1 if SAMPLE_REAL_VOCAB_TAIL != 0 else 0)

assert HIDDEN % EMBED_HIDDEN_CHUNK == 0
assert VOCAB % SAMPLE_VOCAB_CHUNK == 0
assert SAMPLE_NUM_VOCAB_CHUNKS <= SAMPLE_CHUNK_PAD
assert REAL_VOCAB <= VOCAB

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
QKV_OK = 5  # split-K slices (atomic-add)  # 5 -> QKV_K_SLICE=1024 = normed slab (1:1 partial-Q)
QKV_K_SLICE = HIDDEN // QKV_OK  # 1280 K per split
QKV_K_CHUNKS = QKV_K_SLICE // TK  # 5 inner TK chunks per split

# ── Scope 2 · grouped-query attention (fused, PAGED) ──
# SEQ_TILE = seq length per KV block. PINNED to the serving paged page_size (128):
# decode reads KV from the PAGED pool via block_table, so one logical block must
# map to exactly one physical page — i.e. SEQ_TILE == page_size — or a block would
# straddle two pages whose physical ids are unrelated. prefill_fwd uses the same
# 128 page; 128 is also the known-good L0B double-buffer cap. BLOCK_SIZE is an
# alias used by the paged cache-row arithmetic.
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

# RoPE GROUPED by HEADS_PER_ROPE heads/grid (re-test @ swimlane level 2).
HEADS_PER_ROPE = 4
NUM_ROPE_GROUPS = NUM_KV_HEADS // HEADS_PER_ROPE
ROPE_GROUP_CORES = 16
ROPE_GROUP_ITEMS_PER_CORE = (HEADS_PER_ROPE * BATCH) // ROPE_GROUP_CORES
# RoPE SINGLE spmd grid (fork): all NUM_KV_HEADS*BATCH items in ONE launch,
# replacing the 2 grouped grids (per-grid scheduler overhead). 32 cores @ 4/core.
ROPE_CORES = 32
ROPE_ITEMS_PER_CORE = (NUM_KV_HEADS * BATCH) // ROPE_CORES
assert (NUM_KV_HEADS * BATCH) % ROPE_CORES == 0

# Fused rope_qkv dependency layout: rope (QK-norm + RoPE) reads ALL of q/k/v_proj +
# inv_rms_states, so it gates on every q/k/v projection tile writer plus rms_tid,
# gathered into one TASK_ID array (rope_dep_tids). Offsets must be module-level Python
# ints (in-function int locals get traced as Scalars, which pl.array.create rejects).
K_TID_BASE = Q_ON * QKV_OK  # 50 — q tiles occupy [0, K_TID_BASE)
V_TID_BASE = K_TID_BASE + KV_ON * QKV_OK  # 60 — k tiles in [K_TID_BASE, V_TID_BASE)
RMS_TID_IDX = V_TID_BASE + KV_ON * QKV_OK  # 70 — v tiles in [V_TID_BASE, RMS_TID_IDX); rms_tid last
ROPE_NDEPS = RMS_TID_IDX + 1  # 71
FA_NDEPS = ROPE_NDEPS + 1  # rope deps + work_tid (fa_fused folds in rope as phase 0)

# Fused QK-norm reduction alignment. The per-(KV head, batch) sum-of-squares emits a
# col-major [rows, 1] tile; ptoas requires its column byte size (rows * sizeof(FP32))
# to be 32B-aligned, i.e. rows a multiple of 8. Q already pads to Q_HEAD_PAD (16) rows;
# K's single real row is zero-padded to K_RED_ROWS for the reduction (then row 0 kept).
K_RED_ROWS = 8
assert (Q_HEAD_PAD * 4) % 32 == 0, "Q QK-norm reduction rows (Q_HEAD_PAD) must be 32B-aligned"
assert (K_RED_ROWS * 4) % 32 == 0, "K QK-norm reduction rows (K_RED_ROWS) must be 32B-aligned"

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
assert TM == BATCH and N_SUB == 2 and QKV_K_CHUNKS == 4
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
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.FP32],  # FP32: inter-layer carry (was BF16)
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
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Tensor[[BATCH, HIDDEN], pl.FP32],  # FP32: inter-layer carry (was BF16)
    # normed_in: THIS layer's x*gamma (BF16), produced by the previous layer's fused
    # dcr_xgamma (x_gamma_0 for layer 0). Consumed by QKV — replaces the local x_gamma.
    normed_in: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    # normed_out: NEXT layer's x*gamma, written by THIS layer's dcr_xgamma. Escapes via
    # the inline alias (verified: inline out-param tensor writes are visible to caller).
    normed_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
    next_gamma_idx: pl.Scalar[pl.INT32],  # clamped min(layer_idx+1, N-1) for dcr_xgamma's gamma
    # Cross-iteration carries: prev_out_tids = DOWN_ON slab writers of `hidden_states`
    # (prev dcr / copy_hidden) — gate rms_recip / residual / seeds. prev_normed_tids =
    # DOWN_ON slab writers of `normed_in` (prev dcr_xgamma / x_gamma_0) — gate QKV. Both
    # refilled by this layer's dcr_xgamma (same task writes out + normed_out).
    prev_out_tids: pl.Array[DOWN_ON, pl.TASK_ID],
    prev_normed_tids: pl.Array[DOWN_ON, pl.TASK_ID],
) -> pl.Tensor[[BATCH, HIDDEN], pl.FP32]:
    # Per-layer offsets into the STACKED weights / PAGED KV cache. decode_fwd passes
    # the running loop index 0.._FWD_NLAYERS-1; decode_fwd_layers passes
    # 0.._CHUNK_NLAYERS-1 (per-chunk weight slices).
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
    # down_proj TaskIds — HOISTED to orchestration scope (declared before
    # manual_scope) so the consolidated `down_cast_residual` writer that runs
    # AFTER the manual_scope can gate on them via deps=. Filled inside the
    # manual_scope down_proj loop; the consolidated writer reads them per-index
    # (deps=[down_tids[k] for k in range(DOWN_ON * K_SPLITS)] — list-comprehension
    # per-index fence works for large N; whole-array deps=[down_tids] does not).
    down_tids = pl.array.create(DOWN_ON * K_SPLITS, pl.TASK_ID)
    # The fused rope_qkv (QK-norm + RoPE) gates on ALL q/k/v projection tile writers
    # plus rms_tid, collected in ONE TASK_ID array so the spmd deps= can be a single
    # comprehension (the frontend requires deps to be one list/tuple/comprehension, not
    # a concatenation). Layout (offsets module-level): [q tiles | k tiles | v tiles | rms_tid].
    rope_dep_tids = pl.array.create(FA_NDEPS, pl.TASK_ID)
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
    # `rms_recip` overlaps the QKV proj. normed_in is consumed ONLY by QKV; the
    # residual / post_rms path reads raw hidden_states, so it is unaffected.
    #
    # WHOLE-LAYER manual scope (x_gamma .. rope): tensormap registration is
    # suppressed; every cross-task edge below is an explicit deps=. x_gamma slab
    # n deps ONLY on prev_out_tids[n] (the previous layer's dcr writer of that
    # carry slab, copy_hidden for layer 0) — the fine-grained dcr->x_gamma edge
    # the auto path cannot express without whole-parent WAW serialization.
    # normed_in is now the `normed_in` PARAM (produced by prev dcr_xgamma / x_gamma_0).

    # ── Scope 2 allocations (hoisted before the manual scope). ──
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
    # real blocks. Built once by the `fa_work_build` AIV task, consumed by
    # fa_fused's grid-stride. Sized to the worst case (every sequence full).
    fa_work_table = pl.create_tensor([FA_TABLE_CAP, 1], dtype=pl.INT32)
    fa_total = pl.create_tensor([1, 1], dtype=pl.INT32)

    with pl.manual_scope():
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="rms_recip",
            deps=[prev_out_tids[i] for i in range(DOWN_ON)],
        ) as rms_tid:
            partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):
                k0 = kb * RMSNORM_K_CHUNK
                x_chunk = hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK]  # FP32 already (was cast from BF16)
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]),
                )
            variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            inv_rms_states = pl.assemble(inv_rms_states, inv_rms, [0, 0])
        rope_dep_tids[RMS_TID_IDX] = rms_tid  # fused rope reads inv_rms_states

        # ── Scope 1: Q projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed") as q_seed_tid:  # no explicit dep: runtime q_proj WAR hazard orders it after prev rope_qkv (now the q_proj reader)
            for snb in pl.pipeline(Q_ON, stage=2):
                q_proj = pl.assemble(
                    q_proj, pl.full([BATCH, QKV_N_TILE], dtype=pl.FP32, value=0.0), [0, snb * QKV_N_TILE]
                )
        for q_nt in pl.parallel(Q_ON):
            q_n_region = q_nt * QKV_N_TILE
            for q_ks in pl.range(QKV_OK):
                q_k_base = q_ks * QKV_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="q_proj",
                    deps=[prev_normed_tids[q_ks], q_seed_tid],  # 1:1 with normed slab (QKV_K_SLICE=DOWN_TN)
                ) as q_tid:
                    for n_sub in pl.range(N_SUB):
                        n0 = q_n_region + n_sub * TN
                        q_acc = pl.matmul(
                            normed_in[:, q_k_base : q_k_base + TK],
                            wq[layer_hidden_base + q_k_base : layer_hidden_base + q_k_base + TK, n0 : n0 + TN],
                            out_dtype=pl.FP32,
                        )
                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                            kk = q_k_base + kc * TK
                            q_acc = pl.matmul_acc(
                                q_acc, normed_in[:, kk : kk + TK], wq[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                            )
                        q_proj = pl.assemble(q_proj, q_acc, [0, n0], atomic=pl.AtomicType.Add)
                rope_dep_tids[q_nt * QKV_OK + q_ks] = q_tid

        # ── Scope 1: K projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as k_seed_tid:
            k_proj = pl.assemble(k_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])
        for k_nt in pl.parallel(KV_ON):
            k_n_region = k_nt * QKV_N_TILE
            for k_ks in pl.range(QKV_OK):
                k_k_base = k_ks * QKV_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="k_proj",
                    deps=[prev_normed_tids[0], prev_normed_tids[1], prev_normed_tids[2], prev_normed_tids[3], prev_normed_tids[4], k_seed_tid],
                ) as k_tid:
                    for n_sub in pl.range(N_SUB):
                        n0 = k_n_region + n_sub * TN
                        k_acc = pl.matmul(
                            normed_in[:, k_k_base : k_k_base + TK],
                            wk[layer_hidden_base + k_k_base : layer_hidden_base + k_k_base + TK, n0 : n0 + TN],
                            out_dtype=pl.FP32,
                        )
                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                            kk = k_k_base + kc * TK
                            k_acc = pl.matmul_acc(
                                k_acc, normed_in[:, kk : kk + TK], wk[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                            )
                        k_proj = pl.assemble(k_proj, k_acc, [0, n0], atomic=pl.AtomicType.Add)
                rope_dep_tids[K_TID_BASE + k_nt * QKV_OK + k_ks] = k_tid

        # ── Scope 1: V projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as v_seed_tid:
            v_proj = pl.assemble(v_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])
        for v_nt in pl.parallel(KV_ON):
            v_n_region = v_nt * QKV_N_TILE
            for v_ks in pl.range(QKV_OK):
                v_k_base = v_ks * QKV_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="v_proj",
                    deps=[prev_normed_tids[0], prev_normed_tids[1], prev_normed_tids[2], prev_normed_tids[3], prev_normed_tids[4], v_seed_tid],
                ) as v_tid:
                    for n_sub in pl.range(N_SUB):
                        n0 = v_n_region + n_sub * TN
                        v_acc = pl.matmul(
                            normed_in[:, v_k_base : v_k_base + TK],
                            wv[layer_hidden_base + v_k_base : layer_hidden_base + v_k_base + TK, n0 : n0 + TN],
                            out_dtype=pl.FP32,
                        )
                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                            kk = v_k_base + kc * TK
                            v_acc = pl.matmul_acc(
                                v_acc, normed_in[:, kk : kk + TK], wv[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN]
                            )
                        v_proj = pl.assemble(v_proj, v_acc, [0, n0], atomic=pl.AtomicType.Add)
                rope_dep_tids[V_TID_BASE + v_nt * QKV_OK + v_ks] = v_tid

        # ── Scope 2 prep: build the dense block-level work list on an AIV task. ──
        # Inputs are external (seq_lens) — no deps.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_work_build") as work_tid:
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

        rope_dep_tids[ROPE_NDEPS] = work_tid  # fa_fused (now folds in rope) also waits on fa_work_build

        # ── Scope 3b MLP-accumulator seeds, HOISTED between rope and attn. ──
        # gate/up/down accumulators are zeroed for the later split-K atomic-add
        # tasks; the seeds have NO data dependency on rope or attention, so placing
        # them here lets the scheduler overlap the (vector) zero-fills with the
        # fa_fused cube/vector work instead of serializing them at the head of the
        # MLP manual_scope. Their TASK_IDs (seed_tid / gate_seed_tid / up_seed_tid)
        # cross into the manual_scope below as explicit deps — the same pattern as
        # online_softmax's captured attn_done_tid. The accumulator create_tensor
        # calls move up with them (a seed must follow its tensor's allocation).
        down_acc_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        gate_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)
        up_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as seed_tid:
            for nb in pl.pipeline(DOWN_ON, stage=2):
                n0 = nb * DOWN_TN
                zero = pl.full([BATCH, DOWN_TN], dtype=pl.FP32, value=0.0)
                down_acc_all = pl.assemble(down_acc_all, zero, [0, n0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as gate_seed_tid:
            for nb in pl.pipeline(MLP_ON, stage=2):
                n0 = nb * MLP_TN
                zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                gate_acc_all = pl.assemble(gate_acc_all, zero, [0, n0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as up_seed_tid:
            for nb in pl.pipeline(MLP_ON, stage=2):
                n0 = nb * MLP_TN
                zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                up_acc_all = pl.assemble(up_acc_all, zero, [0, n0])

        # fa_fused: ONE mixed cube+vec root (QK -> softmax -> SV), BLOCK-LEVEL dense
        # static dispatch. Each grid-stride step processes exactly ONE real seq-block
        # (×GP_SIZE heads), decoded from the dense work table. Because the table holds
        # only real blocks, core = w % NUM_CORES distributes total_real_blocks evenly
        # across cores (≈1.25x of ideal), independent of per-batch length skew.
        with pl.spmd(
            NUM_CORES,
            name_hint="fa_fused",
            # NOTE: no function-level UP_DOWN split here. Phase-1 (attn) and phase-2
            # (online-softmax) are split PER-REGION instead (pl.split_aiv): phase-1 is a
            # data-parallel UP_DOWN region (the compiler inserts aiv_shard / aic_gather,
            # halving the 16-row softmax across both AIV lanes), phase-2 a task-parallel
            # NONE region (both lanes, disjoint (b, kvh) work). A function-level pl.split
            # cannot express this — it would also try to halve phase-2's un-halvable
            # [5, 128] / [1, 640] reduction tiles ("even split dimension").
            deps=[rope_dep_tids[i] for i in range(FA_NDEPS)],  # rope deps + work_tid (rope folded into fa_fused phase 0)
        ) as attn_done_tid:  # now also runs phase-2 softmax (writes attn_out) after the syncall
            fa_core = pl.get_block_idx()
            # ── Phase 0: QK-norm + RoPE, folded in (was the standalone rope_qkv grid).
            # Pure in-kernel HBM sync: the syncall below publishes k_cache / v_cache /
            # all_q_padded to every core before phase-1 reads them, replacing the old
            # rope->fa dispatch + dep edge (one fewer grid dispatch -> less schedule
            # overhead). 32 of the 48 AIV lanes run the UNCHANGED per-(KV head, batch)
            # rope pipeline (identical tiles + pl.pipeline stage=2); lanes 32..47 idle.
            # The conditional GM write relies on pypto's cross-half phi base-param repoint
            # (ExpandMixedKernel, branch fix/split-aiv-gm-cross-half-view / PR #1894).
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                rope_lane = fa_core * 2 + aiv_id  # 0..47
                if rope_lane < ROPE_CORES:        # lanes 0..31 carry the 4-item rope pipeline
                    # Zero column-pads that grow one (KV head, batch) q/k row-slab up to a
                    # 32B-aligned reduction row count (see K_RED_ROWS comment). Constant per item.
                    q_red_pad = pl.full([1, (Q_HEAD_PAD - Q_PER_KV) * HEAD_DIM], dtype=pl.FP32, value=0.0)
                    k_red_pad = pl.full([1, (K_RED_ROWS - 1) * HEAD_DIM], dtype=pl.FP32, value=0.0)
                    for it in pl.pipeline(ROPE_ITEMS_PER_CORE, stage=2):
                        g_idx = rope_lane * ROPE_ITEMS_PER_CORE + it
                        ki = g_idx // BATCH
                        b = g_idx % BATCH
                        ctx_len = pl.read(seq_lens, [b])
                        inv_rms_b = pl.read(inv_rms_states, [b, 0])
                        pos = ctx_len - 1  # absolute position -> RoPE cos/sin row (NOT the cache row)
                        # Paged write target for this row's current token: slot_mapping[b]
                        # decomposes into (physical page, in-page offset). Same scheme prefill
                        # uses; one physical page == one SEQ_TILE/BLOCK_SIZE band of the pool.
                        wr_slot = pl.cast(pl.tensor.read(slot_mapping, [b]), pl.INDEX)
                        wr_slot_block = wr_slot // BLOCK_SIZE
                        wr_slot_offset = wr_slot - wr_slot_block * BLOCK_SIZE
                        cos_lo = rope_cos[pos : pos + 1, 0:HALF_DIM]
                        cos_hi = rope_cos[pos : pos + 1, HALF_DIM:HEAD_DIM]
                        sin_lo = rope_sin[pos : pos + 1, 0:HALF_DIM]
                        sin_hi = rope_sin[pos : pos + 1, HALF_DIM:HEAD_DIM]

                        kv_col = ki * HEAD_DIM
                        # FUSED QK-norm (K): read the raw k_proj row, scale by inv_rms[b], apply
                        # gamma, fold in the per-row qk_inv reciprocal — all in-register — so
                        # k_full is fully normed without a q/k_proj_norm GM round-trip. inv_rms
                        # cancels (control experiment): it scales k_raw before BOTH the gamma mul
                        # and the sum-of-squares, so the recip undoes it. RoPE then just rotates.
                        # The single real row is column-padded to K_RED_ROWS so the col-major
                        # sum-of-squares tile is 32B-aligned; the zero rows reduce to 0 (finite
                        # recip, 0 after gamma) and are dropped — only row 0 (the head) is kept.
                        k_raw = pl.mul(
                            pl.reshape(
                                pl.concat(k_proj[b : b + 1, kv_col : kv_col + HEAD_DIM], k_red_pad),
                                [K_RED_ROWS, HEAD_DIM],
                            ),
                            inv_rms_b,
                        )
                        k_ss = pl.row_sum(pl.mul(k_raw, k_raw))
                        k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS)))
                        k_normed = pl.row_expand_mul(pl.col_expand_mul(k_raw, k_norm_w), k_inv)
                        k_full = k_normed[0:1, :]
                        k_lo = k_full[:, 0:HALF_DIM]
                        k_hi = k_full[:, HALF_DIM:HEAD_DIM]
                        rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                        rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                        cache_row = layer_cache_base + (wr_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + wr_slot_offset
                        # Coalesce lo|hi into one [1, HEAD_DIM] row -> a single paged k_cache MTE3
                        # write instead of two half-width ones (the store tail bounds makespan).
                        rot = pl.concat(rot_lo, rot_hi)
                        k_cache = pl.assemble(k_cache, pl.cast(rot, target_type=pl.BF16), [cache_row, 0])
                        v_row_bf16 = pl.cast(
                            pl.mul(v_proj[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM], inv_rms_b),
                            target_type=pl.BF16,
                        )
                        v_cache = pl.assemble(v_cache, v_row_bf16, [cache_row, 0])

                        q_base = ki * Q_PER_KV
                        q_pad_row0 = b * NUM_KV_HEADS * Q_HEAD_PAD + ki * Q_HEAD_PAD
                        # FUSED QK-norm (Q) + BATCHED heads + free zero-pad. The Q_PER_KV heads of
                        # this (KV head, batch) are one contiguous q_proj row-slab; column-pad it
                        # with zeros up to the Q_HEAD_PAD physical rows of all_q_padded and reshape
                        # to [Q_HEAD_PAD, HEAD_DIM]. Running the QK-norm (inv_rms scale + gamma +
                        # the per-row qk_inv reciprocal) and RoPE over ALL Q_HEAD_PAD rows then:
                        #   * keeps the col-major sum-of-squares tile 32B-aligned (Q_HEAD_PAD*4B);
                        #     a [Q_PER_KV, 1]=20B reduction is not 32B-aligned and ptoas rejects it;
                        #   * leaves rows Q_PER_KV.. as 0 (0 -> finite recip -> 0 after gamma; RoPE
                        #     of 0 = 0), so ONE [Q_HEAD_PAD, HEAD_DIM] store replaces the old real +
                        #     separate q_pad_zero stores.
                        # qk_inv is a per-row reduction (each head row normed independently);
                        # cos/sin [1, HALF_DIM] broadcast over rows.
                        q_raw = pl.mul(
                            pl.reshape(
                                pl.concat(
                                    q_proj[b : b + 1, q_base * HEAD_DIM : (q_base + Q_PER_KV) * HEAD_DIM],
                                    q_red_pad,
                                ),
                                [Q_HEAD_PAD, HEAD_DIM],
                            ),
                            inv_rms_b,
                        )
                        q_ss = pl.row_sum(pl.mul(q_raw, q_raw))
                        q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
                        q_heads = pl.row_expand_mul(pl.col_expand_mul(q_raw, q_norm_w), q_inv)
                        q_lo = q_heads[:, 0:HALF_DIM]
                        q_hi = q_heads[:, HALF_DIM:HEAD_DIM]
                        q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                        # Coalesce lo|hi -> one [Q_HEAD_PAD, HEAD_DIM] store (real rows + zero pad).
                        q_rot = pl.concat(q_rot_lo, q_rot_hi)
                        all_q_padded = pl.assemble(
                            all_q_padded, pl.cast(q_rot, target_type=pl.BF16), [q_pad_row0, 0]
                        )
            pl.system.syncall(core_type="mix")
            # Read the device-computed block count ON-DEVICE (inside the kernel) so
            # `fa_total` enters fa_fused as an INPUT and the normal task auto-dep
            # orders it after the `fa_work_build` producer. Reading it at
            # orchestration scope instead lowers to a host get_tensor_data that does
            # NOT wait for the producer (stale ~0 read → empty dispatch; see
            # KNOWN_ISSUES). This mirrors the existing on-device fa_work_table read.
            fa_total_blocks = pl.cast(pl.read(fa_total, [0, 0]), target_type=pl.INDEX)
            # Grid-stride over the dense real-block list: core fa_core owns table
            # entries fa_core, fa_core+NUM_CORES, … < fa_total_blocks.
            # Phase-1 attn: cube QK/SV OUTSIDE a PER-HEAD split_aiv region; vector softmax
            # on the UP_DOWN half via EXPLICIT aiv_shard (C->V, shards the QK Acc tile
            # directly) + aic_gather (V->C, halves -> full for the SV cube). Region sits
            # INSIDE the gp pipeline so each head's shard->softmax->gather overlaps cube work.
            for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):
                fa_enc = pl.cast(pl.read(fa_work_table, [fa_w, 0]), target_type=pl.INDEX)
                fa_b = fa_enc // MAX_CTX_BLOCKS
                fa_p = fa_enc % MAX_CTX_BLOCKS
                fa_hg = 0  # HEAD_GROUPS == 1 (GP_SIZE == NUM_KV_HEADS)
                fa_ctx_len = pl.read(seq_lens, [fa_b])
                sb = fa_p
                s0 = sb * SEQ_TILE
                valid_len = pl.min(SEQ_TILE, fa_ctx_len - s0)
                fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)

                for gp in pl.pipeline(GP_SIZE, stage=3):
                    gi = fa_hg * GP_SIZE + gp
                    kvh = gi
                    q_pad_row_g = fa_b * NUM_KV_HEADS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                    q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]
                    g_base = (fa_b * NUM_KV_HEADS + gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD
                    cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE

                    # QK matmul (cube) — OUTSIDE the split_aiv region.
                    k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]
                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)

                    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        scores_h = pl.aiv_shard(raw_scores)  # tensor-level C->V half
                        scores_scaled = pl.mul(scores_h, ATTN_SCALE)
                        scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_BATCH, valid_len)
                        scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                        cur_mi = pl.row_max(scores)
                        exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                        cur_li = pl.row_sum(exp_scores)
                        exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                        mi_row = g_base + sb * Q_HEAD_PAD + aiv_id * (Q_HEAD_PAD // 2)
                        all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [mi_row, 0])
                        all_cur_li = pl.assemble(all_cur_li, cur_li, [mi_row, 0])
                        exp_full = pl.aic_gather(exp_scores_bf16)  # tensor-level V->C full

                    v_tile = v_cache[cache_row : cache_row + SEQ_TILE, :]
                    oi_tmp = pl.matmul(exp_full, v_tile, out_dtype=pl.FP32)
                    oi_tmp = pl.set_validshape(oi_tmp, Q_HEAD_BATCH, HEAD_DIM)
                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [g_base + sb * Q_HEAD_PAD, 0])

            # ── Phase 2: online-softmax cross-block reduction, FUSED in-kernel ──
            # The per-block partials (all_oi_tmp / all_cur_mi / all_cur_li) of a
            # (b, kvh) lane are produced by DIFFERENT cores — the dense work table
            # spreads a lane's blocks across cores for load balance — so reducing a
            # lane needs every core's phase-1 writes visible first. A hard cross-core
            # barrier (pto::SYNCALL) provides exactly that: every participant must
            # arrive before any proceeds.
            #
            # FULL-OCCUPANCY CONTRACT: the hard/FFTS form waits for ALL physical cores
            # of the participant set. NUM_CORES (24) group blocks == 24 AIC + 48 AIV ==
            # every 910B core, so the launch is full-occupancy; a partial launch (or a
            # retune of NUM_CORES off the physical AIC count) leaves cores unreached and
            # the wait never completes (AICore timeout 507018). core_type="mix" syncs
            # both the AIC SV output (oi) and the AIV softmax output (mi/li).
            #
            # A2/A3 NOTE: partials still transit GM (cross-core data has no on-chip path
            # on 910B), so this only removes the separate online_softmax dispatch + the
            # fa->os dep edge — it does NOT save the partials' round-trip (that needs the
            # A5 L2-resident path).
            pl.system.syncall(core_type="mix")
            # Phase-2 online_softmax as a TASK-PARALLEL dual-AIV region. pl.split_aiv
            # with mode=NONE dispatches BOTH AIV lanes of every group block (no halving,
            # no aiv_shard / aic_gather): each lane owns disjoint (b, kvh) work items via
            # os_aiv = fa_core*2 + aiv_id, restoring the pre-fusion 48-way reduction
            # parallelism (24 group blocks x 2 AIV) that the fused 24-way single-spmd loop
            # gave up — without UP_DOWN's un-halvable [5,128] / [1,640] reduction tiles.
            # The hard syncall above fences phase-1 (non-split attn) from phase-2.
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                os_aiv = fa_core * 2 + aiv_id  # 0..47 global AIV-lane id
                for os_spmd_idx in pl.range(os_aiv, OS_WORK, NUM_CORES * 2):
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
        post_norm_partial = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)  # raw residual h1 (add-back); FP32 (was BF16)
        mlp_norm_in = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # h1 * post_gamma (gate/up input)
        inv_rms_tile = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
        mlp_tile = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)

        # Out_seed zeros attn_proj_fp32 in OUT_TN-wide chunks so out_proj
        # split-K tasks can atomic-add into it.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_seed", deps=[prev_out_tids[_si] for _si in range(DOWN_ON)]) as out_seed_tid:
            for nb in pl.pipeline(N_SPLITS_OUT, stage=2):
                out_seed_n0 = nb * OUT_TN
                out_zero = pl.full([BATCH, OUT_TN], dtype=pl.FP32, value=0.0)
                attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_zero, [0, out_seed_n0])

        # ── Scope 3b: manual_scope MLP block. ──
        silu_tids = pl.array.create(MLP_ON, pl.TASK_ID)
        # down_tids is hoisted to orchestration scope (declared before this
        # manual_scope) so the post-scope consolidated writer can gate on it;
        # it is FILLED here in the down_proj loop below.
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
                    hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]  # FP32 already
                    resid_fp32 = pl.add(attn_chunk, hidden_chunk)
                    # Raw residual h1 — added back after down_proj (must NOT be gamma-scaled).
                    post_norm_partial = pl.assemble(post_norm_partial, resid_fp32, [0, k0])  # FP32 (no BF16 cast)
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
                hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]  # FP32 already
                resid_chunk = pl.add(attn_chunk, hidden_chunk)
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

    # ── down_cast_residual (DOWN_ON-way, OUTSIDE manual_scope): the residual
    # add (down_acc_all + post_norm_partial, both FP32) is the layer output,
    # emitted as DOWN_ON sliced writers of `out` in the AUTO-DEP region —
    # restoring the baseline's 5-way parallelism while keeping the fusion
    # (no scratch out_partial, no out_consolidate copy).
    #
    # Why outside manual_scope: inside manual_scope, auto-dep (tensormap)
    # registration is suppressed (explicit deps= only), so partial writers
    # register NO tensormap edge to downstream caller readers (proven on
    # device: next_hidden's writers had no edge to copy_out → garbage). In the
    # auto-dep region each sliced writer registers `out`; OptimizeOrchTensors
    # Pattern-5 narrows the footprint to a window, and the runtime tensormap
    # check is region-precise per-dim, so the 5 writers do not serialize
    # against each other while the next layer's x_gamma still waits on all 5.
    #
    # Deps: each block needs only its own K_SPLITS (=17) down_proj atomic-add
    # tids (per-index list comprehension), so block n starts as soon as its
    # column slab is accumulated. Transitivity covers post_norm_partial: every
    # down_proj K-slice deps on a silu task, which deps on ALL cast_tids =
    # residual_rms_cast — the full producer of post_norm_partial.
    # dcr_xgamma as a SINGLE pl.spmd(DOWN_ON) dispatch (was DOWN_ON separate pl.parallel
    # pl.at tasks). PERF: the separate-task form WAW-serialized the DOWN_ON sliced writers
    # of `out` / `normed_out` on this runtime — the OptimizeOrchTensors region-narrowing
    # that was meant to keep them parallel did NOT fire (measured: the 5 dcr ran SERIAL on
    # one core, ~35us, both at the chunk tail AND every layer boundary). A single spmd
    # dispatch's blocks are inherently parallel (exactly like x_gamma's disjoint
    # normed_states writes), so the disjoint-slice writes run on DOWN_ON cores. Trade-offs:
    # the dispatch deps on ALL down_proj tids (not per-column), but down_proj finishes
    # ~together; and the carry is ONE dispatch tid for all DOWN_ON slabs (the next layer's
    # rms_recip/QKV/seeds wait on the whole dcr dispatch — fine once it is ~3us not ~35us).
    with pl.spmd(
        DOWN_ON,
        name_hint="dcr_xgamma",
        deps=[down_tids[i] for i in range(DOWN_ON * K_SPLITS)],
    ) as dcr_tid:
        n_out = pl.tile.get_block_idx()
        n0 = n_out * DOWN_TN
        # OUTPUT 1: layer residual (down_acc + post_norm, both FP32) -> `out` (cur).
        out_chunk = pl.add(
            down_acc_all[:, n0 : n0 + DOWN_TN], post_norm_partial[:, n0 : n0 + DOWN_TN]
        )
        out = pl.assemble(out, out_chunk, [0, n0])
        # OUTPUT 2: NEXT layer's x*gamma from the same in-register FP32 chunk (no GM
        # re-read of `out`). gamma row clamped via next_gamma_idx (last layer unused).
        gamma_next = pl.slice(input_rms_weight, [1, DOWN_TN], [next_gamma_idx, n0])
        xg = pl.col_expand_mul(out_chunk, gamma_next)
        normed_out = pl.assemble(normed_out, pl.cast(xg, target_type=pl.BF16), [0, n0])
    # One spmd dispatch tid carries BOTH out + normed_out for all DOWN_ON slabs (the
    # per-block tids of the old pl.parallel form are gone): the next layer's rms_recip
    # / seeds (read prev_out_tids[*]) and QKV (read prev_normed_tids[*]) all wait on
    # the single dcr dispatch. Fill every slab slot so the DOWN_ON-wide carry (sized
    # for copy_hidden's layer-0 seed) stays valid.
    for _slab in pl.unroll(DOWN_ON):
        prev_out_tids[_slab] = dcr_tid
        prev_normed_tids[_slab] = dcr_tid
    return out


@pl.jit.inline
def _token_embed_inline(
    sampled_ids: pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    for b in pl.parallel(0, BATCH, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_embed"):
            token_id = pl.read(sampled_ids, [b, 0])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            for k0 in pl.range(0, HIDDEN, EMBED_HIDDEN_CHUNK):
                hidden_chunk = pl.slice(embed_weight, [1, EMBED_HIDDEN_CHUNK], [token_row, k0])
                next_hidden = pl.assemble(next_hidden, hidden_chunk, [b, k0])
    return next_hidden


@pl.jit.inline
def _greedy_sample_inline(
    logits: pl.Tensor[[BATCH, VOCAB], pl.FP32],
    sampled_ids: pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32],
) -> pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32]:
    for b in pl.parallel(BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="greedy_sample"):
            idx_init = pl.arange(0, [1, SAMPLE_VOCAB_CHUNK], dtype=pl.UINT32)
            chunk_vals = pl.create_tensor([1, SAMPLE_CHUNK_PAD], dtype=pl.FP32)
            chunk_vals[:, :] = pl.full([1, SAMPLE_CHUNK_PAD], dtype=pl.FP32, value=-3.402823e38)
            for c in pl.range(SAMPLE_REAL_NUM_VOCAB_CHUNKS):
                c0 = c * SAMPLE_VOCAB_CHUNK
                local_scores = logits[b : b + 1, c0 : c0 + SAMPLE_VOCAB_CHUNK]
                if SAMPLE_REAL_VOCAB_TAIL != 0:
                    if c == SAMPLE_REAL_NUM_FULL_VOCAB_CHUNKS:
                        local_scores_valid = pl.set_validshape(local_scores, 1, SAMPLE_REAL_VOCAB_TAIL)
                        local_scores_padded = pl.fillpad(local_scores_valid, pad_value=pl.PadValue.min)
                        sorted_pairs = pl.sort32(local_scores_padded, idx_init)
                    else:
                        sorted_pairs = pl.sort32(local_scores, idx_init)
                else:
                    sorted_pairs = pl.sort32(local_scores, idx_init)
                sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
                sorted_pairs = pl.mrgsort(sorted_pairs, block_len=256)
                top_pairs = sorted_pairs[:, 0 : 2 * SAMPLE_TOPK]
                top_vals = pl.gather(top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                best_val = pl.read(top_vals, [0, 0])
                pl.write(chunk_vals, [0, c], best_val)

            chunk_sorted = pl.sort32(chunk_vals, idx_init)
            chunk_sorted = pl.mrgsort(chunk_sorted, block_len=64)
            chunk_sorted = pl.mrgsort(chunk_sorted, block_len=256)
            chunk_top_pairs = chunk_sorted[:, 0 : 2 * SAMPLE_TOPK]
            chunk_top_vals = pl.gather(chunk_top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
            best_val = pl.read(chunk_top_vals, [0, 0])
            chunk_i32 = pl.cast(0, pl.INT32)
            for c in pl.range(SAMPLE_REAL_NUM_VOCAB_CHUNKS):
                scan_c = (SAMPLE_REAL_NUM_VOCAB_CHUNKS - 1) - c
                val = pl.read(chunk_vals, [0, scan_c])
                if val == best_val:
                    chunk_i32 = pl.cast(scan_c, pl.INT32)

            local_token = pl.cast(0, pl.INT32)
            chunk_base = chunk_i32 * pl.cast(SAMPLE_VOCAB_CHUNK, target_type=pl.INT32)
            chunk_base_idx = pl.cast(chunk_base, target_type=pl.INDEX)
            winning_logits = pl.slice(logits, [1, SAMPLE_VOCAB_CHUNK], [pl.cast(b, pl.INDEX), chunk_base_idx])
            if SAMPLE_REAL_VOCAB_TAIL != 0:
                if chunk_i32 == pl.cast(SAMPLE_REAL_NUM_FULL_VOCAB_CHUNKS, target_type=pl.INT32):
                    winning_logits_valid = pl.set_validshape(winning_logits, 1, SAMPLE_REAL_VOCAB_TAIL)
                    winning_logits_padded = pl.fillpad(winning_logits_valid, pad_value=pl.PadValue.min)
                    for t in pl.range(SAMPLE_VOCAB_CHUNK):
                        scan_t = (SAMPLE_VOCAB_CHUNK - 1) - t
                        val = pl.read(winning_logits_padded, [0, pl.cast(scan_t, pl.INDEX)])
                        if val == best_val:
                            local_token = pl.cast(scan_t, pl.INT32)
                else:
                    for t in pl.range(SAMPLE_VOCAB_CHUNK):
                        scan_t = (SAMPLE_VOCAB_CHUNK - 1) - t
                        val = pl.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)])
                        if val == best_val:
                            local_token = pl.cast(scan_t, pl.INT32)
            else:
                for t in pl.range(SAMPLE_VOCAB_CHUNK):
                    scan_t = (SAMPLE_VOCAB_CHUNK - 1) - t
                    val = pl.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)])
                    if val == best_val:
                        local_token = pl.cast(scan_t, pl.INT32)
            token_id = chunk_base + local_token
            if token_id >= pl.cast(REAL_VOCAB, target_type=pl.INT32):
                token_id = pl.cast(0, pl.INT32)
            token_out = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
            token_out[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
            pl.write(token_out, [0, 0], token_id)
            sampled_ids[b : b + 1, :] = token_out
    return sampled_ids


NUM_LAYERS = 40  # full Qwen3-14B depth, for the fused decode_fwd loop
_FWD_NLAYERS = NUM_LAYERS  # decode_fwd loop bound; overridable for layer-count tests


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
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    final_norm_weight: pl.Tensor,
    lm_head_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
    embed_weight: pl.Tensor,
    sampled_ids_in: pl.Tensor,
    sampled_ids_out: pl.Out[pl.Tensor],
    next_hidden: pl.Out[pl.Tensor],
):
    # Device-side fused decode: embed the previous sampled token id, loop the inline
    # body over all _FWD_NLAYERS layers, run the LM head, then sample the next token
    # id. Weights are STACKED [_FWD_NLAYERS*HIDDEN, ...] /
    # [_FWD_NLAYERS*INTERMEDIATE, ...]; k_cache / v_cache cover
    # [_FWD_NLAYERS*BATCH*NUM_KV_HEADS*MAX_SEQ, ...]; out is logits [BATCH, VOCAB].
    # _FWD_NLAYERS defaults to NUM_LAYERS (40) and is settable for layer-count tests.
    #
    # The loop-carried `cur` is seeded from next_hidden after embedding the previous
    # sampled token id. Each layer's output is made
    # visible to the next layer / the LM head by _decode_layer's CONSOLIDATED
    # `down_cast_residual` writer (a single full-tensor writer in the auto-dep region,
    # placed after the MLP manual_scope and gated on the down_proj TaskIds) — without it,
    # the inline body's manual_scope partial writes do not register a tensormap edge to
    # the downstream reader and the fused output is garbage. See decode-fwd-dep-fix notes.
    #
    # The paged KV pool (k_cache / v_cache) is runtime-dynamic — its row count is the
    # actual num_pages * layers * kv_heads * page_size, which varies with the device-side
    # KV cache size (and the 1-page warm-up scratch). Bind dim-0 dynamic (mirroring
    # prefill_fwd) so the compiled program does not bake a fixed shape and reject any
    # non-batching-shaped pool.
    k_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    next_hidden = _token_embed_inline(sampled_ids_in, embed_weight, next_hidden)

    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)  # FP32 inter-layer carry (was BF16)
    # Cross-iteration carry-writer tids (see _decode_layer's prev_out_tids):
    # seeded with copy_hidden for layer 0, refilled per layer by the dcr writers.
    carry_tids = pl.array.create(DOWN_ON, pl.TASK_ID)
    for cb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden") as ch_tid:
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                # FIRST-layer boundary: cast the external BF16 embed input -> FP32 once,
                # so every layer consumes FP32 hidden with no per-boundary round-trip.
                cur = pl.assemble(
                    cur,
                    pl.cast(
                        pl.slice(next_hidden, [BATCH, RMSNORM_K_CHUNK], [cb0, ck0]),
                        target_type=pl.FP32,
                    ),
                    [cb0, ck0],
                )
        for cseed in pl.range(DOWN_ON):
            carry_tids[cseed] = ch_tid

    # ── Pre-loop x_gamma_0: layer 0's normed = cur_0 * gamma_0 (BF16). For layers 1+,
    # the per-layer normed is produced by the PREVIOUS layer's fused dcr_xgamma; only
    # layer 0 (whose cur comes from copy_hidden, not a dcr) needs this standalone task.
    normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    carry_normed_tids = pl.array.create(DOWN_ON, pl.TASK_ID)
    with pl.manual_scope():
        for xg_n in pl.range(XG_BLOCKS):
            xg_k0 = xg_n * (HIDDEN // XG_BLOCKS)
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="x_gamma0", deps=[carry_tids[xg_n]]
            ) as xg0_tid:
                for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK // XG_BLOCKS, stage=2):
                    k0 = xg_k0 + kb * RMSNORM_K_CHUNK
                    x_chunk = cur[:, k0 : k0 + RMSNORM_K_CHUNK]  # FP32 carry
                    gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [0, k0])
                    xg = pl.col_expand_mul(x_chunk, gamma)
                    normed = pl.assemble(normed, pl.cast(xg, target_type=pl.BF16), [0, k0])
            carry_normed_tids[xg_n] = xg0_tid

    for layer_idx in pl.range(_FWD_NLAYERS):
        layer_next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)  # FP32 layer output
        next_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # next layer's x*gamma
        next_gamma_idx = pl.min(layer_idx + 1, _FWD_NLAYERS - 1)  # clamp: last layer's normed unused
        cur = _decode_layer(
            cur, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache, wo, w_gate, w_up, w_down,
            post_rms_weight, layer_next_hidden, normed, next_normed, layer_idx, next_gamma_idx,
            carry_tids, carry_normed_tids,
        )
        normed = next_normed
    # LAST-layer boundary: cast the final FP32 hidden -> BF16 once for the (unchanged,
    # BF16-input) rms_lm_head. This single cast replaces the per-layer round-trips.
    cur_bf16 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for lb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cast_lmhead_in"):
            for lkb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                lk0 = lkb * RMSNORM_K_CHUNK
                cur_bf16 = pl.assemble(
                    cur_bf16,
                    pl.cast(
                        pl.slice(cur, [BATCH, RMSNORM_K_CHUNK], [lb0, lk0]),
                        target_type=pl.BF16,
                    ),
                    [lb0, lk0],
                )
    out = rms_lm_head(cur_bf16, final_norm_weight, lm_head_weight, seq_lens, out)
    sampled_ids_out = _greedy_sample_inline(out, sampled_ids_out)
    return out, sampled_ids_out, next_hidden


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
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
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
    # FP32 inter-layer carry (matches _decode_layer's FP32-carry signature). The
    # chunk input/output hidden are BF16 (host passes BF16 between chunks), so cast
    # BF16->FP32 at the chunk head (copy_hidden) and FP32->BF16 at the tail (copy_out),
    # mirroring decode_fwd's embed-in / lm-head-in casts. (Without these the BF16 cur
    # hits _decode_layer's FP32 x_gamma/rms_recip -> ptoas bfloat16 type error.)
    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    carry_tids = pl.array.create(DOWN_ON, pl.TASK_ID)
    for cb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden") as ch_tid:
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                cur = pl.assemble(
                    cur,
                    pl.cast(
                        pl.slice(hidden_states, [BATCH, RMSNORM_K_CHUNK], [cb0, ck0]),
                        target_type=pl.FP32,
                    ),
                    [cb0, ck0],
                )
        for cseed in pl.range(DOWN_ON):
            carry_tids[cseed] = ch_tid

    # Pre-loop x_gamma_0: chunk layer 0's normed = cur * gamma_0 (mirrors decode_fwd).
    normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    carry_normed_tids = pl.array.create(DOWN_ON, pl.TASK_ID)
    with pl.manual_scope():
        for xg_n in pl.range(XG_BLOCKS):
            xg_k0 = xg_n * (HIDDEN // XG_BLOCKS)
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="x_gamma0", deps=[carry_tids[xg_n]]
            ) as xg0_tid:
                for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK // XG_BLOCKS, stage=2):
                    k0 = xg_k0 + kb * RMSNORM_K_CHUNK
                    x_chunk = cur[:, k0 : k0 + RMSNORM_K_CHUNK]
                    gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [0, k0])
                    xg = pl.col_expand_mul(x_chunk, gamma)
                    normed = pl.assemble(normed, pl.cast(xg, target_type=pl.BF16), [0, k0])
            carry_normed_tids[xg_n] = xg0_tid

    for i in pl.range(_CHUNK_NLAYERS):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        next_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        next_gamma_idx = pl.min(i + 1, _CHUNK_NLAYERS - 1)
        cur = _decode_layer(
            cur, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache, wo, w_gate, w_up, w_down,
            post_rms_weight, next_hidden, normed, next_normed, i, next_gamma_idx,
            carry_tids, carry_normed_tids,
        )
        normed = next_normed
    for ob0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_out"):
            for okb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ok0 = okb * RMSNORM_K_CHUNK
                out = pl.assemble(
                    out,
                    pl.cast(
                        pl.slice(cur, [BATCH, RMSNORM_K_CHUNK], [ob0, ok0]), target_type=pl.BF16
                    ),
                    [ob0, ok0],
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
    """Single-layer (_CHUNK_NLAYERS == 1) input list for the compile-only smoke,
    in decode_fwd_layers parameter order (PAGED KV via block_table + slot_mapping)."""
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
# math AND its bf16 cast points), runs decode_fwd_layers with _CHUNK_NLAYERS == 1
# (a single fused decode layer, hidden -> hidden, no LM head) on device through the
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
    """Deterministic random fixture (name -> tensor) for decode_fwd_layers (N==1).

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

    # k_cache / v_cache are the PAGED pool (rows = NUM_PAGES * NUM_KV_HEADS *
    # BLOCK_SIZE) the kernel reads/writes via block_table / slot_mapping.
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

    Mirrors decode_fwd_layers (N==1) / _decode_layer at layer_idx 0: RMSNorm ->
    Q/K/V proj -> per-head QK-norm -> RoPE -> KV-cache write at pos=seq_len-1 -> GQA
    flash attention over [0, seq_len) -> out_proj -> residual -> post-RMSNorm ->
    SwiGLU MLP -> residual. The deferred input-RMSNorm inv_rms and the QK-norm
    control scale cancel to the standard math (QK-norm is scale-invariant).

    The inter-layer hidden is carried in FP32 now: the residual stream (h1, out)
    stays FP32 with NO intermediate bf16 rounding; only the chunk boundary casts
    (BF16 embed-in at copy_hidden, FP32->BF16 hidden-out at copy_out) round. So the
    only bf16 cast points are normed, the QKV inputs, q/k/v cache, attn_out, the
    gate/up input, the SwiGLU output, and the final copy_out — NOT the residual add.
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

    attn_proj = attn_out @ values["wo"].float()                  # out_proj (FP32)
    h1 = x + attn_proj                                           # raw residual (FP32, NOT bf16-rounded)
    post_gamma = values["post_rms_weight"].float()[0]
    post_inv = _rmsnorm_inv(h1)                                  # deferred into silu (FP32 h1)
    mlp_in = _bf16(h1 * post_gamma)                              # gamma before inv_rms
    gate = mlp_in @ values["w_gate"].float()
    up = mlp_in @ values["w_up"].float()
    sg = gate * post_inv
    su = up * post_inv
    mlp = _bf16(sg * torch.sigmoid(sg) * su)                     # SwiGLU
    down = mlp @ values["w_down"].float()
    # FP32-carry residual: out = down + h1 in FP32 (no per-layer bf16(h1) rounding);
    # the single FP32->BF16 round is decode_fwd_layers' copy_out at the chunk tail.
    values["out"] = (down + h1).to(torch.bfloat16)


def _build_specs(inputs: dict) -> list:
    """TensorSpec list in decode_fwd_layers parameter order, from a fixture dict."""
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
    parser.add_argument(
        "--enable-l2-swimlane",
        nargs="?",
        const=4,
        default=0,
        type=int,
        metavar="PERF_LEVEL",
        help="Enable L2 swimlane perf capture at the given granularity level. Bare flag "
             "= level 4 (full). Levels: 1=AICore timing, 2=+dispatch/fanout, 3=+sched "
             "phases, 4=+orch phases; 0 (default) disables.",
    )
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
    parser.add_argument("--validate-fwd", action="store_true", default=False,
                        help="validate the fused decode_fwd (N stacked layers + on-device LM head "
                             "-> logits) against a host chain reference, instead of the default "
                             "single-layer golden test.")
    parser.add_argument("--fwd-layers", type=int, default=4, help="layer count N for --validate-fwd")
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="persist inputs + golden for replay (off: large fixtures)")
    args = parser.parse_args()

    set_backend_type(_backend_type(args.platform))

    # The single-layer golden test and the smoke run both drive decode_fwd_layers
    # with a 1-layer chunk (hidden -> hidden, no LM head). Force the chunk size to 1
    # so the single-layer (un-stacked) weights / KV pool line up; decode_fwd_layers
    # reads _CHUNK_NLAYERS at trace time, so the rebind must precede any call.
    _CHUNK_NLAYERS = 1

    # Compile-only smoke: explicit --smoke, or any *sim platform (the CI
    # `python decode_layer.py -p a2a3sim` sweep — codegen regressions are still
    # caught without needing a device or the heavy full-graph simulation).
    if args.smoke or args.platform.endswith("sim"):
        smoke_out = torch.empty([BATCH, HIDDEN], dtype=torch.bfloat16)
        post_pass = decode_fwd_layers.compile_for_test(*_smoke_inputs(), smoke_out)
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
            fn=decode_fwd_layers,
            specs=specs,
            golden_fn=golden_decode_layer,
            compile_cfg=dict(dump_passes=False),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
                # dep_gen forced OFF: this kernel's full-occupancy pl.system.syncall
                # is incompatible with dep_gen — the DFX instrumentation perturbs core
                # occupancy and trips AICore timeout 507018 (pypto#1931).
                enable_dep_gen=False,
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
    data_input_dir = args.data_dir / "in"
    if data_input_dir.is_dir():
        inputs = load_inputs(data_input_dir)
    else:
        random_values = random_inputs(full_seq=args.max_seq, seed=args.seed)
        inputs = [random_values[name] for name in INPUT_NAMES]
    # dep_gen is force-disabled here: --validate-fwd runs two on-device programs in
    # one process (decode_fwd, then the host-ref loop's decode_fwd_layers), and the
    # dep_gen collector cannot register host buffers for the second program
    # (`halHostRegister for dep_gen SHM failed: 8` -> `init_dep_gen failed: 8`). On
    # this 40-layer graph dep_gen also overflows ("records dropped", no deps.json),
    # so it provides nothing of value here regardless.
    run_cfg = RunConfig(
        platform=args.platform,
        device_id=args.device,
        backend_type=_backend_type(args.platform),
        enable_l2_swimlane=args.enable_l2_swimlane,
        enable_dep_gen=False,
        dump_passes=False,
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
        # seq_lens / block_table / slot_mapping / rope tables are shared across layers
        # (NOT per-layer stacked); the PAGED KV pool kc/vc IS stacked N times (one
        # paged pool per layer, indexed by layer_cache_base).
        stacked = [
            hs, stack0(irw, N), stack0(wq_, N), stack0(wk_, N), stack0(wv_, N),
            stack0(qn, N), stack0(kn, N), sl, bt, sm, rc, rs, stack0(kc, N), stack0(vc, N),
            stack0(wo_, N), stack0(wg, N), stack0(wu, N), stack0(wd, N), stack0(prw, N),
            final_norm_w, lm_head_w,
        ]
        logits = torch.zeros(BATCH, VOCAB, dtype=torch.float32)
        embed_weight = torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16)
        sampled_ids_in = torch.zeros(BATCH, SAMPLED_IDS_PAD, dtype=torch.int32)
        for b in range(BATCH):
            sampled_ids_in[b, 0] = b
            embed_weight[b] = hs[b]
        sampled_ids_out = torch.zeros(BATCH, SAMPLED_IDS_PAD, dtype=torch.int32)
        next_hidden = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
        decode_fwd(
            *stacked,
            logits,
            embed_weight,
            sampled_ids_in,
            sampled_ids_out,
            next_hidden,
            config=run_cfg,
        )
        # Perf-only mode: the L2 swimlane collector cannot register host buffers for a
        # second on-device program in the same process (the host-ref call below would
        # `init_l2_swimlane failed: 8`). decode_fwd already emitted the swimlane table,
        # so skip the reference comparison and exit cleanly.
        if args.enable_l2_swimlane:
            print(f"[stacked-fwd {N}L+LMhead] swimlane perf run complete "
                  f"(host-ref argmax check skipped under --enable-l2-swimlane)")
            raise SystemExit(0)
        # host ref: run one N-layer decode_fwd_layers chunk -> final RMSNorm -> lm_head.
        # _CHUNK_NLAYERS = N keeps the inter-layer residual FP32 (chunk casts BF16 only at
        # its boundaries), matching decode_fwd's FP32-carry-until-LM-head path. A per-layer
        # chain (N single-layer dispatches) would re-enter each layer from BF16, diverging
        # from decode_fwd and making the argmax check pass/fail for the wrong reason.
        _CHUNK_NLAYERS = N
        ref_out = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
        decode_fwd_layers(*stacked[:len(INPUT_NAMES)], ref_out, config=run_cfg)
        hn = ref_out.float()
        inv = torch.rsqrt(hn.pow(2).mean(-1, keepdim=True) + EPS)
        ref_normed = (hn * inv) * final_norm_w.float()
        ref_logits = ref_normed @ lm_head_w.float().t()  # [BATCH, VOCAB]
        a = logits.cpu(); e = ref_logits.cpu()
        # compare argmax (the actual generation signal) + value closeness
        amax_k = a.argmax(-1); amax_r = e.argmax(-1)
        sample_k = sampled_ids_out[:, 0].cpu()
        argmax_match = int((amax_k == amax_r).sum())
        sample_match = int((sample_k == amax_k).sum())
        close = torch.isclose(a, e, rtol=5e-2, atol=5e-2)
        print(f"[stacked-fwd {N}L+LMhead] argmax match {argmax_match}/{BATCH} | "
              f"sample match {sample_match}/{BATCH} | "
              f"logits {int(close.sum())/a.numel():.4%} within 5e-2 | "
              f"max_abs_err={(a-e).abs().max():.4f} | kernel_argmax={amax_k.tolist()} "
              f"sampled={sample_k.tolist()} ref_argmax={amax_r.tolist()}")
        raise SystemExit(0 if argmax_match == BATCH and sample_match == BATCH else 1)
