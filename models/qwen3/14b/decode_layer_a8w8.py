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

import os

import pypto.language as pl

from config import VOCAB  # vocab size for the fused decode_fwd LM head / logits
from rms_lm_head import rms_lm_head  # LM head for the fused multi-layer decode_fwd

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
_DECODE_DEBUG_STAGE_NAME = os.environ.get("DECODE_A8W8_DEBUG_STAGE", "")
DECODE_DEBUG_STAGE_ID = (
    1 if _DECODE_DEBUG_STAGE_NAME == "q_proj" else
    2 if _DECODE_DEBUG_STAGE_NAME == "k_proj" else
    3 if _DECODE_DEBUG_STAGE_NAME == "v_proj" else
    4 if _DECODE_DEBUG_STAGE_NAME == "attn_out" else
    5 if _DECODE_DEBUG_STAGE_NAME == "attn_proj" else
    6 if _DECODE_DEBUG_STAGE_NAME == "post_resid" else
    7 if _DECODE_DEBUG_STAGE_NAME == "mlp_norm_in" else
    8 if _DECODE_DEBUG_STAGE_NAME == "gate_acc" else
    9 if _DECODE_DEBUG_STAGE_NAME == "up_acc" else
    10 if _DECODE_DEBUG_STAGE_NAME == "mlp_tile" else
    11 if _DECODE_DEBUG_STAGE_NAME == "down_acc" else
    12 if _DECODE_DEBUG_STAGE_NAME == "post_inv_rms" else
    13 if _DECODE_DEBUG_STAGE_NAME == "mlp_tile_5k" else
    14 if _DECODE_DEBUG_STAGE_NAME == "mlp_tile_10k" else
    15 if _DECODE_DEBUG_STAGE_NAME == "mlp_tile_12k" else
    16 if _DECODE_DEBUG_STAGE_NAME == "down_acc_tile2" else
    17 if _DECODE_DEBUG_STAGE_NAME == "out_partial_tile2" else
    18 if _DECODE_DEBUG_STAGE_NAME == "x_gamma" else
    19 if _DECODE_DEBUG_STAGE_NAME == "act_i8" else
    20 if _DECODE_DEBUG_STAGE_NAME == "act_scale" else
    21 if _DECODE_DEBUG_STAGE_NAME == "q_rope" else
    22 if _DECODE_DEBUG_STAGE_NAME == "k_cur_cache" else
    23 if _DECODE_DEBUG_STAGE_NAME == "v_cur_cache" else
    24 if _DECODE_DEBUG_STAGE_NAME == "q_pre_rope" else
    25 if _DECODE_DEBUG_STAGE_NAME == "k_pre_rope" else
    26 if _DECODE_DEBUG_STAGE_NAME == "k_rope_raw" else
    0
)


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
ACTIVE_BATCH = int(os.environ.get("QWEN_A8W8_ACTIVE_BATCH", str(BATCH)))

# ── Derived shapes — recomputed from the above, don't edit ──
HIDDEN = NUM_HEADS * HEAD_DIM  # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 1024
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS  # 5 (GQA ratio)
HALF_DIM = HEAD_DIM // 2  # 64 (RoPE rotates lo/hi halves)
ATTN_SCALE_MULT = float(os.environ.get("QWEN_A8W8_ATTN_SCALE_MULT", "1.0"))
ATTN_SCALE = ATTN_SCALE_MULT / (HEAD_DIM**0.5)
OUT_PROJ_BF16 = os.environ.get("QWEN_A8W8_OUT_PROJ_BF16", "0") == "1"
SKIP_QK_INV_RMS_CONTROL = os.environ.get("QWEN_A8W8_SKIP_QK_INV_RMS_CONTROL", "0") == "1"
FUSE_MLP_DOWN_MATERIALIZE = os.environ.get("QWEN_A8W8_FUSE_MLP_DOWN_MATERIALIZE", "1") == "1"
FUSE_GATE_UP_PROJ = os.environ.get("QWEN_A8W8_FUSE_GATE_UP_PROJ", "0") == "1"
MLP_SPLITK_ATOMIC = os.environ.get("QWEN_A8W8_MLP_SPLITK_ATOMIC", "0") == "1"
DOWN_SPLITK_ATOMIC = os.environ.get("QWEN_A8W8_DOWN_SPLITK_ATOMIC", "0") == "1"
FUSED_QKV_DEQUANT = os.environ.get("QWEN_A8W8_FUSED_QKV_DEQUANT", "0") == "1"
FUSED_QK_NORM = os.environ.get("QWEN_A8W8_FUSED_QK_NORM", "0") == "1"
Q_ROPE_BATCH_EXPLICIT = os.environ.get("QWEN_A8W8_Q_ROPE_BATCH_EXPLICIT", "0") == "1"
ACC_DEQUANT_OP = os.environ.get("QWEN_A8W8_ACC_DEQUANT_OP", "0") == "1"
SKIP_QK_ACT_SCALE = os.environ.get("QWEN_A8W8_SKIP_QK_ACT_SCALE", "0") == "1"
QKV_SPLITK_ATOMIC = os.environ.get("QWEN_A8W8_QKV_SPLITK_ATOMIC", "0") == "1"
GROUP_KV_PROJ = os.environ.get("QWEN_A8W8_GROUP_KV_PROJ", "0") == "1"
BF16_KV_CACHE = os.environ.get("QWEN_A8W8_BF16_KV", "0") == "1"
FUSE_OUT_PROJ_NPAIR = os.environ.get("QWEN_A8W8_FUSE_OUT_PROJ_NPAIR", "0") == "1"
FUSE_SILU_DOWN_PROJ = os.environ.get("QWEN_A8W8_FUSE_SILU_DOWN_PROJ", "0") == "1"
QUANT_ROWMAX_TRANSPOSE = os.environ.get("QWEN_A8W8_QUANT_ROWMAX_TRANSPOSE", "0") == "1"
ACT_ROWMAX_TRANSPOSE = os.environ.get("QWEN_A8W8_ACT_ROWMAX_TRANSPOSE", "1" if QUANT_ROWMAX_TRANSPOSE else "0") == "1"
OUT_ROWMAX_TRANSPOSE = os.environ.get("QWEN_A8W8_OUT_ROWMAX_TRANSPOSE", "1" if QUANT_ROWMAX_TRANSPOSE else "0") == "1"
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM  # per-head QK-norm RMSNorm denominator

# Attention head grouping. Q_HEAD_BATCH = Q_PER_KV (one attn lane per KV head).
Q_HEAD_BATCH = Q_PER_KV  # 5: Q heads bundled per attention task
Q_HEAD_PAD = ((Q_HEAD_BATCH + 15) // 16) * 16  # 16: rounded up to matmul L0 inner-dim multiple of 16
# Real-vs-padded handling for the fused-attn scratch: tiles stay PHYSICALLY
# Q_HEAD_PAD rows while set_validshape marks only the Q_HEAD_BATCH (5) real rows,
# so each tile.load / tile.store transfers 5 rows from GM (rest auto-padded).
# Never shrink the tile (that trips alloc_tile alignment).


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
TM = BATCH  # M tile = BATCH (OM = 1; M is not split)
TN = 256  # inner N sub-tile
TK = 512  # inner K chunk; INT8 matmul K offsets must stay 512-aligned on a2a3
QKV_N_TILE = 512  # outer N-tile width (one ON unit) = N_SUB inner TN subtiles
N_SUB = QKV_N_TILE // TN  # 2 inner N-subtiles per outer N-tile
Q_ON = HIDDEN // QKV_N_TILE  # 10 outer N-tiles (Q)
KV_ON = KV_HIDDEN // QKV_N_TILE  # 2 outer N-tiles (K, V)
QKV_OK = int(os.environ.get("QWEN_A8W8_QKV_OK", "5" if QKV_SPLITK_ATOMIC else "1"))
QKV_K_SLICE = HIDDEN // QKV_OK  # 1280 K per split
QKV_K_CHUNKS = QKV_K_SLICE // TK  # 5 inner TK chunks per split

# ── Scope 2 · grouped-query attention (fused, PAGED) ──
# BLOCK_SIZE is the serving paged-cache page size. ATTN_TILE is the per-partial
# attention tile used inside decode; splitting a 128-row page into two 64-row
# partials keeps the A8W8 INT8-dequant + softmax Vec footprint under 910B limits
# without changing the cache/page contract shared with prefill.
BLOCK_SIZE = 128
ATTN_TILE = int(os.environ.get("QWEN_A8W8_ATTN_TILE", "64"))
PAGE_ATTN_PARTS = BLOCK_SIZE // ATTN_TILE
SEQ_TILE = BLOCK_SIZE
MAX_CTX_BLOCKS = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE  # logical cache pages per seq
MAX_ATTN_PARTS = MAX_CTX_BLOCKS * PAGE_ATTN_PARTS
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
TOKENS_PER_SPLIT = ATTN_TILE
KV_SPLITS = MAX_ATTN_PARTS

# GP_SIZE = number of KV heads bundled per fa_fused work item (the `gp` loop).
# All bundled heads process the SAME block index, so an item stays equal-cost
# (GP_SIZE head-blocks) and the per-item scalar setup — notably
# pl.read(seq_lens, ...) — is amortized across GP_SIZE heads. Larger GP_SIZE =
# fewer work items / fewer seq_lens reads, at a slightly coarser balance
# granularity. Must divide NUM_KV_HEADS. GP_SIZE = NUM_KV_HEADS (8) bundles all
# heads → one head-group per (batch, block).
GP_SIZE = 8
HEAD_GROUPS = NUM_KV_HEADS // GP_SIZE  # head-groups per (batch, block) (8//8 = 1)
ROPE_CORES = 32
ROPE_ITEMS_PER_CORE = (NUM_KV_HEADS * BATCH) // ROPE_CORES

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
FA_TABLE_CAP = BATCH * MAX_ATTN_PARTS  # upper bound on real attention partials
OS_WORK = BATCH * NUM_KV_HEADS  # online_softmax work items (128)

# ── Scope 3a · out_proj (split-K × split-N, atomic-add into attn_proj_fp32) ──
K_SPLITS_OUT = 5
N_SPLITS_OUT = 40
OUT_INNER_TK = int(os.environ.get("QWEN_A8W8_OUT_INNER_TK", "512"))
OUT_TN = HIDDEN // N_SPLITS_OUT  # 128 output N per task; 256-wide tiles exceed Vec buffer on a2a3.
OUT_TK = HIDDEN // K_SPLITS_OUT  # 1024 K per task
OUT_N_SUB_K = OUT_TK // OUT_INNER_TK  # inner K iters per task

# ── Scope 3b · residual + BF16 cast + RMS reduce ──
K_CHUNK = int(os.environ.get("QWEN_A8W8_K_CHUNK", "512"))  # inner pipe width for residual_rms_cast and post_rms_reduce

# ── Scope 3b · MLP gate / up (split-K, atomic-add into per-batch FP32) ──
MLP_TN = 256  # output N-tile per task (= silu task N-width = DOWN_TN)
K_SPLITS_MLP = 5
MLP_INNER_TK = int(os.environ.get("QWEN_A8W8_MLP_INNER_TK", "512"))
MLP_K_SLICE = HIDDEN // K_SPLITS_MLP  # 1024 K per task
MLP_N_SUB_K = MLP_K_SLICE // MLP_INNER_TK  # inner K iters per task
MLP_ON = INTERMEDIATE // MLP_TN  # 17 output N-blocks (= silu task count)

# ── Scope 3b · silu (MLP_TN-wide tasks, inner pipe over MLP_OUT_CHUNK sub-tiles) ──
MLP_OUT_CHUNK = 256  # silu inner-pipe sub-tile width
SILU_INNER_CHUNKS = MLP_TN // MLP_OUT_CHUNK  # 4 sub-tiles per silu task

# ── Scope 3b · down (split-K, atomic-add into down_acc_all) ──
DOWN_TN = 256  # output N-tile per task (must equal MLP_TN, see assert)
DOWN_K_SLICE = int(os.environ.get("QWEN_A8W8_DOWN_K_SLICE", "512"))
DOWN_TK = int(os.environ.get("QWEN_A8W8_DOWN_TK", "512"))  # inner K iter
DOWN_ON = HIDDEN // DOWN_TN  # 5 output N-blocks
K_SPLITS = INTERMEDIATE // DOWN_K_SLICE  # K-slices per N-block
N_SUB_K = DOWN_K_SLICE // DOWN_TK  # inner K iters per task

# ── Cross-stage wiring constraints ──
N_PER_CAST_K = MLP_K_SLICE // OUT_TN  # 2

# Geometry assertions — keep at the bottom so all constants are defined first.
assert QKV_N_TILE % TN == 0, "TN must divide the outer N-tile"
assert HIDDEN % QKV_N_TILE == 0 and KV_HIDDEN % QKV_N_TILE == 0, "QKV_N_TILE must divide Q and KV widths"
assert HIDDEN % QKV_OK == 0, "OK must divide HIDDEN (K dim)"
assert QKV_K_SLICE % TK == 0, "TK must divide the split-K slice"
assert Q_ON == HIDDEN // QKV_N_TILE and KV_ON == KV_HIDDEN // QKV_N_TILE
assert TM == BATCH and N_SUB in {1, 2}
assert QKV_OK * QKV_K_CHUNKS == HIDDEN // TK
assert 1 <= ACTIVE_BATCH <= BATCH
assert BLOCK_SIZE % ATTN_TILE == 0, "ATTN_TILE must divide the paged cache page size"
assert PAGE_ATTN_PARTS * ATTN_TILE == BLOCK_SIZE
assert KV_SPLITS == MAX_ATTN_PARTS
assert NUM_KV_HEADS % GP_SIZE == 0, "GP_SIZE must divide NUM_KV_HEADS"
assert HEAD_GROUPS * GP_SIZE == NUM_KV_HEADS
assert (NUM_KV_HEADS * BATCH) % ROPE_CORES == 0
assert GP_SIZE == NUM_KV_HEADS, "block-level work table encodes (b, block); needs HEAD_GROUPS == 1"
assert FA_TABLE_CAP == BATCH * MAX_ATTN_PARTS
assert DOWN_TN % MLP_OUT_CHUNK == 0, "DOWN_TN must be a multiple of MLP_OUT_CHUNK"
assert DOWN_ON * DOWN_TN == HIDDEN
assert K_SPLITS * DOWN_K_SLICE == INTERMEDIATE
assert N_SUB_K * DOWN_TK == DOWN_K_SLICE
assert MLP_ON * MLP_TN == INTERMEDIATE
assert K_SPLITS_MLP * MLP_K_SLICE == HIDDEN
assert MLP_TN % MLP_OUT_CHUNK == 0
assert MLP_TN == DOWN_TN, "silu/down K-slice alignment requires MLP_TN == DOWN_TN"
assert N_SPLITS_OUT * OUT_TN == HIDDEN
assert K_SPLITS_OUT * OUT_TK == HIDDEN
assert OUT_N_SUB_K * OUT_INNER_TK == OUT_TK
assert N_PER_CAST_K * OUT_TN == MLP_K_SLICE
assert N_SPLITS_OUT % 2 == 0

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
    wq_scale: pl.Tensor,
    wk_scale: pl.Tensor,
    wv_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    k_cache_scale: pl.Tensor,
    v_cache_scale: pl.Tensor,
    wo: pl.Tensor,
    wo_scale: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    # Per-layer offsets into the STACKED weights / PAGED KV cache. decode_fwd
    # passes the running loop index so each iteration addresses its own layer.
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
    inv_rms_states = pl.create_tensor([BATCH], dtype=pl.FP32)  # deferred 1/rms denominator
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    # ── A8W8 activation quant (per-row) — computed ONCE and reused by Q/K/V. ──
    normed_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
    act_scales = pl.create_tensor([BATCH, 1], dtype=pl.FP32)

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
            xg_k0 = kb * RMSNORM_K_CHUNK
            xg_chunk = pl.cast(hidden_states[:, xg_k0 : xg_k0 + RMSNORM_K_CHUNK], target_type=pl.FP32)
            xg_gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [layer_idx, xg_k0])
            xg_scaled = pl.col_expand_mul(xg_chunk, xg_gamma)
            normed_states = pl.assemble(normed_states, pl.cast(xg_scaled, target_type=pl.BF16), [0, xg_k0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip"):
        partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):
            rms_k0 = kb * RMSNORM_K_CHUNK
            rms_chunk = pl.cast(hidden_states[:, rms_k0 : rms_k0 + RMSNORM_K_CHUNK], target_type=pl.FP32)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(rms_chunk, rms_chunk)), [1, BATCH]),
            )
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_b in pl.unroll(BATCH):
            pl.tensor.write(inv_rms_states, [rms_b], pl.tensor.read(inv_rms, [rms_b, 0]))

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="act_quant"):
        if ACT_ROWMAX_TRANSPOSE:
            act_quant_amax_t = pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for act_quant_amax_kb_t in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                act_quant_amax_k0_t = act_quant_amax_kb_t * RMSNORM_K_CHUNK
                act_quant_amax_bf16_t = normed_states[:, act_quant_amax_k0_t : act_quant_amax_k0_t + RMSNORM_K_CHUNK]
                act_quant_amax_f_t = pl.cast(act_quant_amax_bf16_t, target_type=pl.FP32)
                act_quant_amax_abs_t = pl.maximum(act_quant_amax_f_t, pl.neg(act_quant_amax_f_t))
                act_quant_amax_row_t = pl.reshape(pl.row_max(act_quant_amax_abs_t), [1, BATCH])
                act_quant_amax_t = pl.maximum(act_quant_amax_t, act_quant_amax_row_t)
            for act_quant_row_t in pl.range(BATCH):
                act_quant_row_amax_t = pl.tensor.read(act_quant_amax_t, [0, act_quant_row_t])
                act_quant_scale_q_t = INT8_SCALE_MAX / act_quant_row_amax_t
                pl.tensor.write(act_scales, [act_quant_row_t, 0], act_quant_row_amax_t / INT8_SCALE_MAX)
                for act_quant_write_kb_t in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                    act_quant_write_k0_t = act_quant_write_kb_t * RMSNORM_K_CHUNK
                    act_quant_write_bf16_t = pl.slice(
                        normed_states, [1, RMSNORM_K_CHUNK], [act_quant_row_t, act_quant_write_k0_t]
                    )
                    act_quant_write_f_t = pl.cast(act_quant_write_bf16_t, target_type=pl.FP32)
                    act_quant_scaled_t = pl.mul(act_quant_write_f_t, act_quant_scale_q_t)
                    act_quant_i32_t = pl.cast(act_quant_scaled_t, target_type=pl.INT32, mode="rint")
                    act_quant_i32_t = pl.minimum(
                        pl.maximum(act_quant_i32_t, pl.full([1, RMSNORM_K_CHUNK], dtype=pl.INT32, value=-127)),
                        pl.full([1, RMSNORM_K_CHUNK], dtype=pl.INT32, value=127),
                    )
                    act_quant_half_t = pl.cast(act_quant_i32_t, target_type=pl.FP16, mode="round")
                    act_quant_i8_t = pl.cast(act_quant_half_t, target_type=pl.INT8, mode="trunc")
                    normed_i8 = pl.assemble(normed_i8, act_quant_i8_t, [act_quant_row_t, act_quant_write_k0_t])
        else:
            # Keep the decode A8W8 activation quantization row-scalar. The vector
            # row_max/reshape path can produce a wrong row-0 scale on current PyPTO.
            for act_quant_row in pl.range(BATCH):
                act_quant_amax = INT8_AMAX_EPS
                for act_quant_amax_kb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                    act_quant_amax_k0 = act_quant_amax_kb * RMSNORM_K_CHUNK
                    act_quant_amax_bf16 = pl.slice(
                        normed_states, [1, RMSNORM_K_CHUNK], [act_quant_row, act_quant_amax_k0]
                    )
                    act_quant_amax_f = pl.cast(act_quant_amax_bf16, target_type=pl.FP32)
                    act_quant_amax_abs = pl.maximum(act_quant_amax_f, pl.neg(act_quant_amax_f))
                    for act_quant_hd in pl.range(RMSNORM_K_CHUNK):
                        act_quant_amax = pl.max(
                            act_quant_amax,
                            pl.tensor.read(act_quant_amax_abs, [0, act_quant_hd]),
                        )
                act_quant_scale_q = INT8_SCALE_MAX / act_quant_amax
                pl.tensor.write(act_scales, [act_quant_row, 0], act_quant_amax / INT8_SCALE_MAX)
                for act_quant_write_kb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                    act_quant_write_k0 = act_quant_write_kb * RMSNORM_K_CHUNK
                    act_quant_write_bf16 = pl.slice(
                        normed_states, [1, RMSNORM_K_CHUNK], [act_quant_row, act_quant_write_k0]
                    )
                    act_quant_write_f = pl.cast(act_quant_write_bf16, target_type=pl.FP32)
                    act_quant_scaled = pl.mul(act_quant_write_f, act_quant_scale_q)
                    act_quant_i32 = pl.cast(act_quant_scaled, target_type=pl.INT32, mode="rint")
                    act_quant_i32 = pl.minimum(
                        pl.maximum(act_quant_i32, pl.full([1, RMSNORM_K_CHUNK], dtype=pl.INT32, value=-127)),
                        pl.full([1, RMSNORM_K_CHUNK], dtype=pl.INT32, value=127),
                    )
                    act_quant_half = pl.cast(act_quant_i32, target_type=pl.FP16, mode="round")
                    act_quant_i8 = pl.cast(act_quant_half, target_type=pl.INT8, mode="trunc")
                    normed_i8 = pl.assemble(normed_i8, act_quant_i8, [act_quant_row, act_quant_write_k0])

    # ── Scope 1: Q/K/V projection — A8W8 act quant + INT8 matmul + dequant. ──
    if QKV_SPLITK_ATOMIC:
        q_proj_i32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed") as q_seed_tid:
            for q_seed_on in pl.pipeline(Q_ON, stage=2):
                q_seed_n0 = q_seed_on * QKV_N_TILE
                q_zero = pl.full([BATCH, QKV_N_TILE], dtype=pl.INT32, value=0)
                q_proj_i32 = pl.assemble(q_proj_i32, q_zero, [0, q_seed_n0])
        for q_on in pl.parallel(Q_ON):
            q_n_region = q_on * QKV_N_TILE
            for q_ks in pl.range(QKV_OK):
                q_k_base = q_ks * QKV_K_SLICE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj_splitk", deps=[q_seed_tid]):
                    for n_sub in pl.range(N_SUB):
                        q_n0 = q_n_region + n_sub * TN
                        q_acc = pl.matmul(
                            pl.tensor.set_validshape(normed_i8[:, q_k_base : q_k_base + TK], ACTIVE_BATCH, TK),
                            wq[layer_hidden_base + q_k_base : layer_hidden_base + q_k_base + TK, q_n0 : q_n0 + TN],
                            out_dtype=pl.INT32,
                        )
                        for kc in pl.range(1, QKV_K_CHUNKS):
                            q_kk = q_k_base + kc * TK
                            q_acc = pl.matmul_acc(
                                q_acc,
                                pl.tensor.set_validshape(normed_i8[:, q_kk : q_kk + TK], ACTIVE_BATCH, TK),
                                wq[layer_hidden_base + q_kk : layer_hidden_base + q_kk + TK, q_n0 : q_n0 + TN],
                            )
                        q_proj_i32 = pl.assemble(q_proj_i32, q_acc, [0, q_n0], atomic=pl.AtomicType.Add)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_dequant"):
            for on_idx in pl.pipeline(Q_ON, stage=2):
                q_deq_n0 = on_idx * QKV_N_TILE
                q_deq_w_scale = pl.reshape(pl.slice(wq_scale, [1, QKV_N_TILE], [layer_idx, q_deq_n0]), [1, QKV_N_TILE])
                q_deq_acc = q_proj_i32[:, q_deq_n0 : q_deq_n0 + QKV_N_TILE]
                q_deq_acc_f = pl.cast(q_deq_acc, target_type=pl.FP32)
                q_deq_weighted = pl.col_expand_mul(q_deq_acc_f, q_deq_w_scale)
                q_deq = pl.mul(q_deq_weighted, act_scales)
                q_proj = pl.assemble(q_proj, q_deq, [0, q_deq_n0])

        k_proj_i32 = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_seed") as k_seed_tid:
            for k_seed_on in pl.pipeline(KV_ON, stage=2):
                k_seed_n0 = k_seed_on * QKV_N_TILE
                k_zero = pl.full([BATCH, QKV_N_TILE], dtype=pl.INT32, value=0)
                k_proj_i32 = pl.assemble(k_proj_i32, k_zero, [0, k_seed_n0])
        for k_on in pl.parallel(KV_ON):
            k_n_region = k_on * QKV_N_TILE
            for k_ks in pl.range(QKV_OK):
                k_k_base = k_ks * QKV_K_SLICE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj_splitk", deps=[k_seed_tid]):
                    for n_sub in pl.range(N_SUB):
                        k_n0 = k_n_region + n_sub * TN
                        k_acc = pl.matmul(
                            pl.tensor.set_validshape(normed_i8[:, k_k_base : k_k_base + TK], ACTIVE_BATCH, TK),
                            wk[layer_hidden_base + k_k_base : layer_hidden_base + k_k_base + TK, k_n0 : k_n0 + TN],
                            out_dtype=pl.INT32,
                        )
                        for kc in pl.range(1, QKV_K_CHUNKS):
                            k_kk = k_k_base + kc * TK
                            k_acc = pl.matmul_acc(
                                k_acc,
                                pl.tensor.set_validshape(normed_i8[:, k_kk : k_kk + TK], ACTIVE_BATCH, TK),
                                wk[layer_hidden_base + k_kk : layer_hidden_base + k_kk + TK, k_n0 : k_n0 + TN],
                            )
                        k_proj_i32 = pl.assemble(k_proj_i32, k_acc, [0, k_n0], atomic=pl.AtomicType.Add)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_dequant"):
            for on_idx in pl.pipeline(KV_ON, stage=2):
                k_deq_n0 = on_idx * QKV_N_TILE
                k_deq_w_scale = pl.reshape(pl.slice(wk_scale, [1, QKV_N_TILE], [layer_idx, k_deq_n0]), [1, QKV_N_TILE])
                k_deq_acc = k_proj_i32[:, k_deq_n0 : k_deq_n0 + QKV_N_TILE]
                k_deq_acc_f = pl.cast(k_deq_acc, target_type=pl.FP32)
                k_deq_weighted = pl.col_expand_mul(k_deq_acc_f, k_deq_w_scale)
                k_deq = pl.mul(k_deq_weighted, act_scales)
                k_proj = pl.assemble(k_proj, k_deq, [0, k_deq_n0])

        v_proj_i32 = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_seed") as v_seed_tid:
            for v_seed_on in pl.pipeline(KV_ON, stage=2):
                v_seed_n0 = v_seed_on * QKV_N_TILE
                v_zero = pl.full([BATCH, QKV_N_TILE], dtype=pl.INT32, value=0)
                v_proj_i32 = pl.assemble(v_proj_i32, v_zero, [0, v_seed_n0])
        for v_on in pl.parallel(KV_ON):
            v_n_region = v_on * QKV_N_TILE
            for v_ks in pl.range(QKV_OK):
                v_k_base = v_ks * QKV_K_SLICE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj_splitk", deps=[v_seed_tid]):
                    for n_sub in pl.range(N_SUB):
                        v_n0 = v_n_region + n_sub * TN
                        v_acc = pl.matmul(
                            pl.tensor.set_validshape(normed_i8[:, v_k_base : v_k_base + TK], ACTIVE_BATCH, TK),
                            wv[layer_hidden_base + v_k_base : layer_hidden_base + v_k_base + TK, v_n0 : v_n0 + TN],
                            out_dtype=pl.INT32,
                        )
                        for kc in pl.range(1, QKV_K_CHUNKS):
                            v_kk = v_k_base + kc * TK
                            v_acc = pl.matmul_acc(
                                v_acc,
                                pl.tensor.set_validshape(normed_i8[:, v_kk : v_kk + TK], ACTIVE_BATCH, TK),
                                wv[layer_hidden_base + v_kk : layer_hidden_base + v_kk + TK, v_n0 : v_n0 + TN],
                            )
                        v_proj_i32 = pl.assemble(v_proj_i32, v_acc, [0, v_n0], atomic=pl.AtomicType.Add)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_dequant"):
            for on_idx in pl.pipeline(KV_ON, stage=2):
                v_deq_n0 = on_idx * QKV_N_TILE
                v_deq_w_scale = pl.reshape(pl.slice(wv_scale, [1, QKV_N_TILE], [layer_idx, v_deq_n0]), [1, QKV_N_TILE])
                v_deq_acc = v_proj_i32[:, v_deq_n0 : v_deq_n0 + QKV_N_TILE]
                v_deq_acc_f = pl.cast(v_deq_acc, target_type=pl.FP32)
                v_deq_weighted = pl.col_expand_mul(v_deq_acc_f, v_deq_w_scale)
                v_deq = pl.mul(v_deq_weighted, act_scales)
                v_proj = pl.assemble(v_proj, v_deq, [0, v_deq_n0])
    elif FUSED_QKV_DEQUANT:
        for q_on in pl.parallel(Q_ON):
            q_n_region = q_on * QKV_N_TILE
            for n_sub in pl.range(N_SUB):
                q_n0 = q_n_region + n_sub * TN
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj_fused_dequant"):
                    q_acc = pl.matmul(
                        pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
                        wq[layer_hidden_base + 0 : layer_hidden_base + TK, q_n0 : q_n0 + TN],
                        out_dtype=pl.INT32,
                    )
                    if ACC_DEQUANT_OP:
                        for kc in pl.range(1, QKV_K_CHUNKS - 1):
                            q_kk = kc * TK
                            q_acc = pl.matmul_acc(
                                q_acc,
                                pl.tensor.set_validshape(normed_i8[:, q_kk : q_kk + TK], ACTIVE_BATCH, TK),
                                wq[layer_hidden_base + q_kk : layer_hidden_base + q_kk + TK, q_n0 : q_n0 + TN],
                            )
                        q_w_scale = pl.reshape(pl.slice(wq_scale, [1, TN], [layer_idx, q_n0]), [1, TN])
                        q_last_kk = (QKV_K_CHUNKS - 1) * TK
                        q_deq_fused = pl.a8w8_matmul_dequant_acc(
                            q_acc,
                            pl.tensor.set_validshape(normed_i8[:, q_last_kk : q_last_kk + TK], ACTIVE_BATCH, TK),
                            wq[layer_hidden_base + q_last_kk : layer_hidden_base + q_last_kk + TK, q_n0 : q_n0 + TN],
                            act_scales,
                            q_w_scale,
                            out_dtype=pl.FP32,
                        )
                    else:
                        for kc in pl.range(1, QKV_K_CHUNKS):
                            q_kk = kc * TK
                            q_acc = pl.matmul_acc(
                                q_acc,
                                pl.tensor.set_validshape(normed_i8[:, q_kk : q_kk + TK], ACTIVE_BATCH, TK),
                                wq[layer_hidden_base + q_kk : layer_hidden_base + q_kk + TK, q_n0 : q_n0 + TN],
                        )
                        q_w_scale = pl.reshape(pl.slice(wq_scale, [1, TN], [layer_idx, q_n0]), [1, TN])
                        q_deq_weighted_tile = pl.col_expand_mul(pl.cast(q_acc, target_type=pl.FP32), q_w_scale)
                        if SKIP_QK_ACT_SCALE:
                            q_deq_fused = q_deq_weighted_tile
                        else:
                            q_deq_fused = pl.mul(q_deq_weighted_tile, act_scales)
                    q_proj = pl.assemble(q_proj, q_deq_fused, [0, q_n0])

        if GROUP_KV_PROJ:
            for kv_on in pl.parallel(KV_ON):
                kv_n_region = kv_on * QKV_N_TILE
                for n_sub in pl.range(N_SUB):
                    kv_n0 = kv_n_region + n_sub * TN
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_fused_dequant"):
                        k_acc = pl.matmul(
                            pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
                            wk[layer_hidden_base + 0 : layer_hidden_base + TK, kv_n0 : kv_n0 + TN],
                            out_dtype=pl.INT32,
                        )
                        if ACC_DEQUANT_OP:
                            for kc in pl.range(1, QKV_K_CHUNKS - 1):
                                k_kk = kc * TK
                                k_acc = pl.matmul_acc(
                                    k_acc,
                                    pl.tensor.set_validshape(normed_i8[:, k_kk : k_kk + TK], ACTIVE_BATCH, TK),
                                    wk[layer_hidden_base + k_kk : layer_hidden_base + k_kk + TK, kv_n0 : kv_n0 + TN],
                                )
                            k_w_scale = pl.reshape(pl.slice(wk_scale, [1, TN], [layer_idx, kv_n0]), [1, TN])
                            k_last_kk = (QKV_K_CHUNKS - 1) * TK
                            k_deq_fused = pl.a8w8_matmul_dequant_acc(
                                k_acc,
                                pl.tensor.set_validshape(normed_i8[:, k_last_kk : k_last_kk + TK], ACTIVE_BATCH, TK),
                                wk[
                                    layer_hidden_base + k_last_kk : layer_hidden_base + k_last_kk + TK,
                                    kv_n0 : kv_n0 + TN,
                                ],
                                act_scales,
                                k_w_scale,
                                out_dtype=pl.FP32,
                            )
                        else:
                            for kc in pl.range(1, QKV_K_CHUNKS):
                                k_kk = kc * TK
                                k_acc = pl.matmul_acc(
                                    k_acc,
                                    pl.tensor.set_validshape(normed_i8[:, k_kk : k_kk + TK], ACTIVE_BATCH, TK),
                                    wk[layer_hidden_base + k_kk : layer_hidden_base + k_kk + TK, kv_n0 : kv_n0 + TN],
                                )
                            k_w_scale = pl.reshape(pl.slice(wk_scale, [1, TN], [layer_idx, kv_n0]), [1, TN])
                            k_deq_weighted_tile = pl.col_expand_mul(pl.cast(k_acc, target_type=pl.FP32), k_w_scale)
                            if SKIP_QK_ACT_SCALE:
                                k_deq_fused = k_deq_weighted_tile
                            else:
                                k_deq_fused = pl.mul(k_deq_weighted_tile, act_scales)
                        k_proj = pl.assemble(k_proj, k_deq_fused, [0, kv_n0])

                        v_acc = pl.matmul(
                            pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
                            wv[layer_hidden_base + 0 : layer_hidden_base + TK, kv_n0 : kv_n0 + TN],
                            out_dtype=pl.INT32,
                        )
                        if ACC_DEQUANT_OP:
                            for kc in pl.range(1, QKV_K_CHUNKS - 1):
                                v_kk = kc * TK
                                v_acc = pl.matmul_acc(
                                    v_acc,
                                    pl.tensor.set_validshape(normed_i8[:, v_kk : v_kk + TK], ACTIVE_BATCH, TK),
                                    wv[layer_hidden_base + v_kk : layer_hidden_base + v_kk + TK, kv_n0 : kv_n0 + TN],
                                )
                            v_w_scale = pl.reshape(pl.slice(wv_scale, [1, TN], [layer_idx, kv_n0]), [1, TN])
                            v_last_kk = (QKV_K_CHUNKS - 1) * TK
                            v_deq_fused = pl.a8w8_matmul_dequant_acc(
                                v_acc,
                                pl.tensor.set_validshape(normed_i8[:, v_last_kk : v_last_kk + TK], ACTIVE_BATCH, TK),
                                wv[
                                    layer_hidden_base + v_last_kk : layer_hidden_base + v_last_kk + TK,
                                    kv_n0 : kv_n0 + TN,
                                ],
                                act_scales,
                                v_w_scale,
                                out_dtype=pl.FP32,
                            )
                        else:
                            for kc in pl.range(1, QKV_K_CHUNKS):
                                v_kk = kc * TK
                                v_acc = pl.matmul_acc(
                                    v_acc,
                                    pl.tensor.set_validshape(normed_i8[:, v_kk : v_kk + TK], ACTIVE_BATCH, TK),
                                    wv[layer_hidden_base + v_kk : layer_hidden_base + v_kk + TK, kv_n0 : kv_n0 + TN],
                                )
                            v_w_scale = pl.reshape(pl.slice(wv_scale, [1, TN], [layer_idx, kv_n0]), [1, TN])
                            v_deq_fused = pl.mul(
                                pl.col_expand_mul(pl.cast(v_acc, target_type=pl.FP32), v_w_scale),
                                act_scales,
                            )
                        v_proj = pl.assemble(v_proj, v_deq_fused, [0, kv_n0])
        else:
            for k_on in pl.parallel(KV_ON):
                k_n_region = k_on * QKV_N_TILE
                for n_sub in pl.range(N_SUB):
                    k_n0 = k_n_region + n_sub * TN
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj_fused_dequant"):
                        k_acc = pl.matmul(
                            pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
                            wk[layer_hidden_base + 0 : layer_hidden_base + TK, k_n0 : k_n0 + TN],
                            out_dtype=pl.INT32,
                        )
                        if ACC_DEQUANT_OP:
                            for kc in pl.range(1, QKV_K_CHUNKS - 1):
                                k_kk = kc * TK
                                k_acc = pl.matmul_acc(
                                    k_acc,
                                    pl.tensor.set_validshape(normed_i8[:, k_kk : k_kk + TK], ACTIVE_BATCH, TK),
                                    wk[layer_hidden_base + k_kk : layer_hidden_base + k_kk + TK, k_n0 : k_n0 + TN],
                                )
                            k_w_scale = pl.reshape(pl.slice(wk_scale, [1, TN], [layer_idx, k_n0]), [1, TN])
                            k_last_kk = (QKV_K_CHUNKS - 1) * TK
                            k_deq_fused = pl.a8w8_matmul_dequant_acc(
                                k_acc,
                                pl.tensor.set_validshape(normed_i8[:, k_last_kk : k_last_kk + TK], ACTIVE_BATCH, TK),
                                wk[
                                    layer_hidden_base + k_last_kk : layer_hidden_base + k_last_kk + TK,
                                    k_n0 : k_n0 + TN,
                                ],
                                act_scales,
                                k_w_scale,
                                out_dtype=pl.FP32,
                            )
                        else:
                            for kc in pl.range(1, QKV_K_CHUNKS):
                                k_kk = kc * TK
                                k_acc = pl.matmul_acc(
                                    k_acc,
                                    pl.tensor.set_validshape(normed_i8[:, k_kk : k_kk + TK], ACTIVE_BATCH, TK),
                                    wk[layer_hidden_base + k_kk : layer_hidden_base + k_kk + TK, k_n0 : k_n0 + TN],
                            )
                            k_w_scale = pl.reshape(pl.slice(wk_scale, [1, TN], [layer_idx, k_n0]), [1, TN])
                            k_deq_weighted_tile = pl.col_expand_mul(pl.cast(k_acc, target_type=pl.FP32), k_w_scale)
                            if SKIP_QK_ACT_SCALE:
                                k_deq_fused = k_deq_weighted_tile
                            else:
                                k_deq_fused = pl.mul(k_deq_weighted_tile, act_scales)
                        k_proj = pl.assemble(k_proj, k_deq_fused, [0, k_n0])

            for v_on in pl.parallel(KV_ON):
                v_n_region = v_on * QKV_N_TILE
                for n_sub in pl.range(N_SUB):
                    v_n0 = v_n_region + n_sub * TN
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj_fused_dequant"):
                        v_acc = pl.matmul(
                            pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
                            wv[layer_hidden_base + 0 : layer_hidden_base + TK, v_n0 : v_n0 + TN],
                            out_dtype=pl.INT32,
                        )
                        if ACC_DEQUANT_OP:
                            for kc in pl.range(1, QKV_K_CHUNKS - 1):
                                v_kk = kc * TK
                                v_acc = pl.matmul_acc(
                                    v_acc,
                                    pl.tensor.set_validshape(normed_i8[:, v_kk : v_kk + TK], ACTIVE_BATCH, TK),
                                    wv[layer_hidden_base + v_kk : layer_hidden_base + v_kk + TK, v_n0 : v_n0 + TN],
                                )
                            v_w_scale = pl.reshape(pl.slice(wv_scale, [1, TN], [layer_idx, v_n0]), [1, TN])
                            v_last_kk = (QKV_K_CHUNKS - 1) * TK
                            v_deq_fused = pl.a8w8_matmul_dequant_acc(
                                v_acc,
                                pl.tensor.set_validshape(normed_i8[:, v_last_kk : v_last_kk + TK], ACTIVE_BATCH, TK),
                                wv[
                                    layer_hidden_base + v_last_kk : layer_hidden_base + v_last_kk + TK,
                                    v_n0 : v_n0 + TN,
                                ],
                                act_scales,
                                v_w_scale,
                                out_dtype=pl.FP32,
                            )
                        else:
                            for kc in pl.range(1, QKV_K_CHUNKS):
                                v_kk = kc * TK
                                v_acc = pl.matmul_acc(
                                    v_acc,
                                    pl.tensor.set_validshape(normed_i8[:, v_kk : v_kk + TK], ACTIVE_BATCH, TK),
                                    wv[layer_hidden_base + v_kk : layer_hidden_base + v_kk + TK, v_n0 : v_n0 + TN],
                                )
                            v_w_scale = pl.reshape(pl.slice(wv_scale, [1, TN], [layer_idx, v_n0]), [1, TN])
                            v_deq_fused = pl.mul(
                                pl.col_expand_mul(pl.cast(v_acc, target_type=pl.FP32), v_w_scale),
                                act_scales,
                            )
                        v_proj = pl.assemble(v_proj, v_deq_fused, [0, v_n0])
    else:
        q_proj_i32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT32)
        q_ready = pl.create_tensor([Q_ON], dtype=pl.INT32)
        for q_on in pl.parallel(Q_ON):
            q_n_region = q_on * QKV_N_TILE
            for n_sub in pl.range(N_SUB):
                q_n0 = q_n_region + n_sub * TN
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj_accum"):
                    q_acc = pl.matmul(
                        pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
                        wq[layer_hidden_base + 0 : layer_hidden_base + TK, q_n0 : q_n0 + TN],
                        out_dtype=pl.INT32,
                    )
                    for kc in pl.range(1, QKV_K_CHUNKS):
                        q_kk = kc * TK
                        q_acc = pl.matmul_acc(
                            q_acc,
                            pl.tensor.set_validshape(normed_i8[:, q_kk : q_kk + TK], ACTIVE_BATCH, TK),
                            wq[layer_hidden_base + q_kk : layer_hidden_base + q_kk + TK, q_n0 : q_n0 + TN],
                        )
                    q_proj_i32 = pl.assemble(q_proj_i32, q_acc, [0, q_n0])
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_ready"):
                pl.tensor.write(q_ready, [q_on], pl.cast(q_on * 0 + 1, target_type=pl.INT32))

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_dequant"):
            for on_idx in pl.pipeline(Q_ON, stage=2):
                q_deq_n0 = on_idx * QKV_N_TILE
                q_deq_w_scale = pl.reshape(pl.slice(wq_scale, [1, QKV_N_TILE], [layer_idx, q_deq_n0]), [1, QKV_N_TILE])
                q_deq_acc = q_proj_i32[:, q_deq_n0 : q_deq_n0 + QKV_N_TILE]
                q_deq_acc_f = pl.cast(q_deq_acc, target_type=pl.FP32)
                q_deq_weighted = pl.col_expand_mul(q_deq_acc_f, q_deq_w_scale)
                q_deq = pl.mul(q_deq_weighted, act_scales)
                q_proj = pl.assemble(q_proj, q_deq, [0, q_deq_n0])

        k_proj_i32 = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.INT32)
        for k_on in pl.parallel(KV_ON):
            k_n_region = k_on * QKV_N_TILE
            for n_sub in pl.range(N_SUB):
                k_n0 = k_n_region + n_sub * TN
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj_accum"):
                    k_acc = pl.matmul(
                        pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
                        wk[layer_hidden_base + 0 : layer_hidden_base + TK, k_n0 : k_n0 + TN],
                        out_dtype=pl.INT32,
                    )
                    for kc in pl.range(1, QKV_K_CHUNKS):
                        k_kk = kc * TK
                        k_acc = pl.matmul_acc(
                            k_acc,
                            pl.tensor.set_validshape(normed_i8[:, k_kk : k_kk + TK], ACTIVE_BATCH, TK),
                            wk[layer_hidden_base + k_kk : layer_hidden_base + k_kk + TK, k_n0 : k_n0 + TN],
                        )
                    k_proj_i32 = pl.assemble(k_proj_i32, k_acc, [0, k_n0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_dequant"):
            for on_idx in pl.pipeline(KV_ON, stage=2):
                k_deq_n0 = on_idx * QKV_N_TILE
                k_deq_w_scale = pl.reshape(pl.slice(wk_scale, [1, QKV_N_TILE], [layer_idx, k_deq_n0]), [1, QKV_N_TILE])
                k_deq_acc = k_proj_i32[:, k_deq_n0 : k_deq_n0 + QKV_N_TILE]
                k_deq_acc_f = pl.cast(k_deq_acc, target_type=pl.FP32)
                k_deq_weighted = pl.col_expand_mul(k_deq_acc_f, k_deq_w_scale)
                k_deq = pl.mul(k_deq_weighted, act_scales)
                k_proj = pl.assemble(k_proj, k_deq, [0, k_deq_n0])

        v_proj_i32 = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.INT32)
        for v_on in pl.parallel(KV_ON):
            v_n_region = v_on * QKV_N_TILE
            for n_sub in pl.range(N_SUB):
                v_n0 = v_n_region + n_sub * TN
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj_accum"):
                    v_acc = pl.matmul(
                        pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
                        wv[layer_hidden_base + 0 : layer_hidden_base + TK, v_n0 : v_n0 + TN],
                        out_dtype=pl.INT32,
                    )
                    for kc in pl.range(1, QKV_K_CHUNKS):
                        v_kk = kc * TK
                        v_acc = pl.matmul_acc(
                            v_acc,
                            pl.tensor.set_validshape(normed_i8[:, v_kk : v_kk + TK], ACTIVE_BATCH, TK),
                            wv[layer_hidden_base + v_kk : layer_hidden_base + v_kk + TK, v_n0 : v_n0 + TN],
                        )
                    v_proj_i32 = pl.assemble(v_proj_i32, v_acc, [0, v_n0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_dequant"):
            for on_idx in pl.pipeline(KV_ON, stage=2):
                v_deq_n0 = on_idx * QKV_N_TILE
                v_deq_w_scale = pl.reshape(pl.slice(wv_scale, [1, QKV_N_TILE], [layer_idx, v_deq_n0]), [1, QKV_N_TILE])
                v_deq_acc = v_proj_i32[:, v_deq_n0 : v_deq_n0 + QKV_N_TILE]
                v_deq_acc_f = pl.cast(v_deq_acc, target_type=pl.FP32)
                v_deq_weighted = pl.col_expand_mul(v_deq_acc_f, v_deq_w_scale)
                v_deq = pl.mul(v_deq_weighted, act_scales)
                v_proj = pl.assemble(v_proj, v_deq, [0, v_deq_n0])

    # ── Scope 2: RoPE + KV cache update + fused flash-attention. ──
    all_q_padded = pl.create_tensor([BATCH * NUM_KV_HEADS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    all_oi_tmp = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_ATTN_PARTS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32
    )
    all_k_deq_tmp = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_ATTN_PARTS * ATTN_TILE, HEAD_DIM], dtype=pl.BF16
    )
    all_cur_mi = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_ATTN_PARTS * Q_HEAD_PAD, 1], dtype=pl.FP32
    )
    all_cur_li = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_ATTN_PARTS * Q_HEAD_PAD, 1], dtype=pl.FP32
    )
    kv_ready = pl.create_tensor([BATCH], dtype=pl.INT32)
    qkd_done = pl.create_tensor([FA_TABLE_CAP], dtype=pl.INT32)
    fa_done = pl.create_tensor([FA_TABLE_CAP], dtype=pl.INT32)

    # Block-level (dense) work list. `fa_work_table[w]` encodes the w-th REAL
    # seq-block as `b * MAX_CTX_BLOCKS + p`; `fa_total[0]` holds the number of
    # real blocks. Built once by the `fa_work_build` AIV task below, consumed by
    # fa_fused's grid-stride. Sized to the worst case (every sequence full).
    fa_work_table = pl.create_tensor([FA_TABLE_CAP], dtype=pl.INT32)
    fa_total = pl.create_tensor([1], dtype=pl.INT32)

    # ── Scope 2 prep: build the dense block-level work list on an AIV task. ──
    # Walk batches in order; for each, append its real blocks [0, ctx_blocks[b])
    # to the table at a running cursor (prefix-sum), so the table is gap-free and
    # `core = w % NUM_CORES` over [0, fa_total) load-balances equal-cost blocks
    # across cores regardless of per-batch length skew.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_work_build"):
        cursor = pl.read(seq_lens, [0]) * 0  # scalar 0 (INDEX)
        for wb in pl.unroll(BATCH):
            wb_ctx = (pl.read(seq_lens, [wb]) + (ATTN_TILE - 1)) // ATTN_TILE
            for wp in pl.range(wb_ctx):
                # fa_work_table is INT32 (fixed-width for the host↔ptoas ABI; see
                # KNOWN_ISSUES "pl.INDEX GM tensor width mismatch") — cast the
                # index-typed encoding to match the tensor dtype.
                pl.tensor.write(
                    fa_work_table, [cursor + wp], pl.cast(wb * MAX_ATTN_PARTS + wp, target_type=pl.INT32)
                )
            cursor = cursor + wb_ctx
        pl.tensor.write(fa_total, [0], pl.cast(cursor, target_type=pl.INT32))

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
    v_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_inv_states = pl.create_tensor([NUM_KV_HEADS * BATCH * Q_PER_KV], dtype=pl.FP32)
    k_inv_states = pl.create_tensor([NUM_KV_HEADS * BATCH], dtype=pl.FP32)

    # inv_rms[b] as a [BATCH, 1] column — applied row-wise to q_proj / k_proj
    # BEFORE QK-norm in both sub-steps below (the control-experiment scale).
    inv_rms_col = pl.reshape(inv_rms_states, [BATCH, 1])

    if FUSED_QK_NORM:
        # Fold gamma and qk reciprocal into one per-KV-head task. This reads each
        # q/k tile once and makes rope consume fully-normalized q_proj_norm/k_proj_norm.
        for h in pl.parallel(NUM_KV_HEADS):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm"):
                q0 = h * Q_PER_KV * HEAD_DIM
                q_chunk = pl.slice(q_proj, [BATCH, Q_PER_KV * HEAD_DIM], [0, q0])
                if not SKIP_QK_INV_RMS_CONTROL:
                    q_chunk = pl.row_expand_mul(q_chunk, inv_rms_col)
                q_flat = pl.reshape(q_chunk, [BATCH * Q_PER_KV, HEAD_DIM])
                q_g = pl.col_expand_mul(q_flat, q_norm_w)
                q_ss = pl.row_sum(pl.mul(q_flat, q_flat))
                q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
                q_g = pl.row_expand_mul(q_g, q_inv)
                q_proj_norm = pl.assemble(
                    q_proj_norm,
                    pl.reshape(q_g, [BATCH, Q_PER_KV * HEAD_DIM]),
                    [0, q0],
                )

                k0 = h * HEAD_DIM
                k_chunk = pl.slice(k_proj, [BATCH, HEAD_DIM], [0, k0])
                if not SKIP_QK_INV_RMS_CONTROL:
                    k_chunk = pl.row_expand_mul(k_chunk, inv_rms_col)
                k_g = pl.col_expand_mul(k_chunk, k_norm_w)
                k_ss = pl.row_sum(pl.mul(k_chunk, k_chunk))
                k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS)))
                k_g = pl.row_expand_mul(k_g, k_inv)
                k_proj_norm = pl.assemble(k_proj_norm, k_g, [0, k0])
    else:
        # Step (a): input inv_rms (control scale) + `* gamma` — elementwise along
        # HEAD_DIM, no reduction dep.
        for h in pl.parallel(NUM_KV_HEADS):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_gamma"):
                qk_gamma_q0 = h * Q_PER_KV * HEAD_DIM
                qk_gamma_q_chunk = pl.slice(q_proj, [BATCH, Q_PER_KV * HEAD_DIM], [0, qk_gamma_q0])
                if not SKIP_QK_INV_RMS_CONTROL:
                    qk_gamma_q_chunk = pl.row_expand_mul(qk_gamma_q_chunk, inv_rms_col)
                q_proj_norm = pl.assemble(
                    q_proj_norm,
                    pl.reshape(
                        pl.col_expand_mul(
                            pl.reshape(qk_gamma_q_chunk, [BATCH * Q_PER_KV, HEAD_DIM]),
                            q_norm_w,
                        ),
                        [BATCH, Q_PER_KV * HEAD_DIM],
                    ),
                    [0, qk_gamma_q0],
                )
                qk_gamma_k0 = h * HEAD_DIM
                qk_gamma_k_chunk = pl.slice(k_proj, [BATCH, HEAD_DIM], [0, qk_gamma_k0])
                if not SKIP_QK_INV_RMS_CONTROL:
                    qk_gamma_k_chunk = pl.row_expand_mul(qk_gamma_k_chunk, inv_rms_col)
                qk_gamma_k_g = pl.col_expand_mul(qk_gamma_k_chunk, k_norm_w)
                k_proj_norm = pl.assemble(k_proj_norm, qk_gamma_k_g, [0, qk_gamma_k0])

        # Step (b): per-head 1/rms reciprocal — DEFERRED, folded into rope_qkv below.
        # Same inv_rms[b] control scale on the input; its 1/inv_rms in the denominator
        # cancels the inv_rms in step (a) (see CONTROL EXPERIMENT note above).
        for h in pl.parallel(NUM_KV_HEADS):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_recip"):
                qk_recip_q0 = h * Q_PER_KV * HEAD_DIM
                qk_recip_q_chunk = pl.slice(q_proj, [BATCH, Q_PER_KV * HEAD_DIM], [0, qk_recip_q0])
                if not SKIP_QK_INV_RMS_CONTROL:
                    qk_recip_q_chunk = pl.row_expand_mul(qk_recip_q_chunk, inv_rms_col)
                qk_recip_q_flat = pl.reshape(qk_recip_q_chunk, [BATCH * Q_PER_KV, HEAD_DIM])
                q_ss = pl.row_sum(pl.mul(qk_recip_q_flat, qk_recip_q_flat))
                q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
                q_inv_base_h = h * BATCH * Q_PER_KV
                for q_inv_i in pl.unroll(BATCH * Q_PER_KV):
                    pl.tensor.write(q_inv_states, [q_inv_base_h + q_inv_i], pl.tensor.read(q_inv, [q_inv_i, 0]))
                qk_recip_k0 = h * HEAD_DIM
                qk_recip_k_chunk = pl.slice(k_proj, [BATCH, HEAD_DIM], [0, qk_recip_k0])
                if not SKIP_QK_INV_RMS_CONTROL:
                    qk_recip_k_chunk = pl.row_expand_mul(qk_recip_k_chunk, inv_rms_col)
                qk_recip_k_ss = pl.row_sum(pl.mul(qk_recip_k_chunk, qk_recip_k_chunk))
                k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(qk_recip_k_ss, HEAD_DIM_INV), EPS)))
                k_inv_base_h = h * BATCH
                for k_inv_i in pl.unroll(BATCH):
                    pl.tensor.write(k_inv_states, [k_inv_base_h + k_inv_i], pl.tensor.read(k_inv, [k_inv_i, 0]))

    for h in pl.parallel(NUM_KV_HEADS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_norm"):
            v_norm_k0 = h * HEAD_DIM
            v_norm_chunk = pl.row_expand_mul(
                pl.slice(v_proj, [BATCH, HEAD_DIM], [0, v_norm_k0]), inv_rms_col
            )
            v_proj_norm = pl.assemble(v_proj_norm, v_norm_chunk, [0, v_norm_k0])

    # Flatten rope/cache work over (KV head, batch), matching the BF16 fast path:
    # one wider SPMD launch replaces per-batch serial KV-head loops.
    for rope_core in pl.spmd(ROPE_CORES, name_hint="rope_qkv", allow_early_resolve=True):
        for rope_it in pl.pipeline(ROPE_ITEMS_PER_CORE, stage=2):
            g_idx = rope_core * ROPE_ITEMS_PER_CORE + rope_it
            ki = g_idx // BATCH
            b = g_idx % BATCH
            ctx_len = pl.read(seq_lens, [b])
            # Deferred RMSNorm denominator: scale this row's q/k/v by inv_rms[b] (a
            # per-row scalar) here, instead of normalizing before the projection.
            # Read on-device so it auto-deps on the rms_recip producer.
            inv_rms_b = pl.read(inv_rms_states, [b])
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
            kv_col = ki * HEAD_DIM
            # K carries the qk_norm gamma (applied batch-wide in qk_gamma above);
            # fold in the DEFERRED per-head qk_inv scalar here (one scalar mul before
            # the linear RoPE — read on-device so it auto-deps on qk_recip). The
            # input-RMSNorm inv_rms cancels inside qk_norm, so no extra inv_rms factor
            # is needed on K. Then split lo/hi for RoPE.
            if FUSED_QK_NORM:
                k_full = k_proj_norm[b : b + 1, kv_col : kv_col + HEAD_DIM]
            else:
                k_inv_b = pl.read(k_inv_states, [ki * BATCH + b])
                k_full = pl.mul(k_proj_norm[b : b + 1, kv_col : kv_col + HEAD_DIM], k_inv_b)
            k_lo = k_full[:, 0:HALF_DIM]
            k_hi = k_full[:, HALF_DIM:HEAD_DIM]
            rot_lo = pl.sub(pl.mul(k_lo, cos_lo), pl.mul(k_hi, sin_lo))
            rot_hi = pl.add(pl.mul(k_hi, cos_hi), pl.mul(k_lo, sin_hi))
            cache_row = layer_cache_base + (wr_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + wr_slot_offset
            v_row_fp32 = v_proj_norm[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM]
            if BF16_KV_CACHE:
                k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_row, 0])
                k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_row, HALF_DIM])
                v_cache = pl.assemble(v_cache, pl.cast(v_row_fp32, target_type=pl.BF16), [cache_row, 0])
            else:
                # ── KV cache INT8 quant: write as INT8 payload + per-row FP32 scale ──
                k_rot_fp32_lo = rot_lo
                k_rot_fp32_hi = rot_hi
                # Per-row amax for K (lo+hi combined max per cache row).
                k_rot_abs_lo = pl.maximum(k_rot_fp32_lo, pl.neg(k_rot_fp32_lo))
                k_rot_abs_hi = pl.maximum(k_rot_fp32_hi, pl.neg(k_rot_fp32_hi))
                k_cache_amax = pl.max(pl.tensor.read(k_rot_abs_lo, [0, 0]), INT8_AMAX_EPS)
                for hd in pl.range(1, HALF_DIM):
                    k_cache_amax = pl.max(k_cache_amax, pl.tensor.read(k_rot_abs_lo, [0, hd]))
                for hd in pl.range(HALF_DIM):
                    k_cache_amax = pl.max(k_cache_amax, pl.tensor.read(k_rot_abs_hi, [0, hd]))
                k_cache_scale_q = INT8_SCALE_MAX / k_cache_amax
                k_cache_scale_fp32 = k_cache_amax / INT8_SCALE_MAX
                # Quantize rot_lo and rot_hi to INT8 per cache row.
                kq_lo_scaled = pl.mul(k_rot_fp32_lo, k_cache_scale_q)
                kq_lo_i32 = pl.cast(kq_lo_scaled, target_type=pl.INT32, mode="rint")
                kq_lo_i32 = pl.minimum(
                    pl.maximum(kq_lo_i32, pl.full([1, HALF_DIM], dtype=pl.INT32, value=-127)),
                    pl.full([1, HALF_DIM], dtype=pl.INT32, value=127),
                )
                kq_lo_i8 = pl.cast(pl.cast(kq_lo_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
                kq_hi_scaled = pl.mul(k_rot_fp32_hi, k_cache_scale_q)
                kq_hi_i32 = pl.cast(kq_hi_scaled, target_type=pl.INT32, mode="rint")
                kq_hi_i32 = pl.minimum(
                    pl.maximum(kq_hi_i32, pl.full([1, HALF_DIM], dtype=pl.INT32, value=-127)),
                    pl.full([1, HALF_DIM], dtype=pl.INT32, value=127),
                )
                kq_hi_i8 = pl.cast(pl.cast(kq_hi_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
                k_cache = pl.assemble(k_cache, kq_lo_i8, [cache_row, 0])
                k_cache = pl.assemble(k_cache, kq_hi_i8, [cache_row, HALF_DIM])
                pl.tensor.write(k_cache_scale, [cache_row, 0], k_cache_scale_fp32)
                # V cache quant: per-row scale over the full HEAD_DIM row.
                v_abs = pl.maximum(v_row_fp32, pl.neg(v_row_fp32))
                v_cache_amax = pl.max(pl.tensor.read(v_abs, [0, 0]), INT8_AMAX_EPS)
                for hd in pl.range(1, HEAD_DIM):
                    v_cache_amax = pl.max(v_cache_amax, pl.tensor.read(v_abs, [0, hd]))
                v_cache_scale_q = INT8_SCALE_MAX / v_cache_amax
                v_cache_scale_fp32 = v_cache_amax / INT8_SCALE_MAX
                vq_scaled = pl.mul(v_row_fp32, v_cache_scale_q)
                vq_i32 = pl.cast(vq_scaled, target_type=pl.INT32, mode="rint")
                vq_i32 = pl.minimum(
                    pl.maximum(vq_i32, pl.full([1, HEAD_DIM], dtype=pl.INT32, value=-127)),
                    pl.full([1, HEAD_DIM], dtype=pl.INT32, value=127),
                )
                vq_i8 = pl.cast(pl.cast(vq_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
                v_cache = pl.assemble(v_cache, vq_i8, [cache_row, 0])
                pl.tensor.write(v_cache_scale, [cache_row, 0], v_cache_scale_fp32)

            q_base = ki * Q_PER_KV
            q_pad_row0 = b * NUM_KV_HEADS * Q_HEAD_PAD + ki * Q_HEAD_PAD
            # Q carries the qk_norm gamma (applied batch-wide in qk_gamma above);
            # fold in the DEFERRED per-head qk_inv scalar here. The Q_PER_KV Q heads
            # bundled under this KV head each have their OWN qk_inv scalar, so we
            # scale + rotate one head at a time (a bundled [Q_PER_KV, 1] column is
            # too small for the 32B col-major tile-alloc constraint; per-head
            # pl.read scalars sidestep it). inv_rms cancels inside qk_norm, so no
            # inv_rms factor on Q either.
            if FUSED_QK_NORM and Q_ROPE_BATCH_EXPLICIT:
                q_heads = pl.reshape(
                    q_proj_norm[
                        b : b + 1, q_base * HEAD_DIM : (q_base + Q_PER_KV) * HEAD_DIM
                    ],
                    [Q_PER_KV, HEAD_DIM],
                )
                q_lo = q_heads[:, 0:HALF_DIM]
                q_hi = q_heads[:, HALF_DIM:HEAD_DIM]

                cos_lo_full = pl.full([Q_PER_KV, HALF_DIM], dtype=pl.FP32, value=0.0)
                cos_hi_full = pl.full([Q_PER_KV, HALF_DIM], dtype=pl.FP32, value=0.0)
                sin_lo_full = pl.full([Q_PER_KV, HALF_DIM], dtype=pl.FP32, value=0.0)
                sin_hi_full = pl.full([Q_PER_KV, HALF_DIM], dtype=pl.FP32, value=0.0)
                for qj in pl.unroll(Q_PER_KV):
                    cos_lo_full = pl.assemble(cos_lo_full, cos_lo, [qj, 0])
                    cos_hi_full = pl.assemble(cos_hi_full, cos_hi, [qj, 0])
                    sin_lo_full = pl.assemble(sin_lo_full, sin_lo, [qj, 0])
                    sin_hi_full = pl.assemble(sin_hi_full, sin_hi, [qj, 0])

                q_rot_lo = pl.sub(pl.mul(q_lo, cos_lo_full), pl.mul(q_hi, sin_lo_full))
                q_rot_hi = pl.add(pl.mul(q_hi, cos_hi_full), pl.mul(q_lo, sin_hi_full))
                q_rot = pl.concat(q_rot_lo, q_rot_hi)
                all_q_padded = pl.assemble(
                    all_q_padded, pl.cast(q_rot, target_type=pl.BF16), [q_pad_row0, 0]
                )
            elif FUSED_QK_NORM:
                for qj in pl.range(Q_PER_KV):
                    q_head = q_proj_norm[
                        b : b + 1, (q_base + qj) * HEAD_DIM : (q_base + qj + 1) * HEAD_DIM
                    ]
                    q_one_lo = q_head[:, 0:HALF_DIM]
                    q_one_hi = q_head[:, HALF_DIM:HEAD_DIM]
                    q_one_rot_lo = pl.sub(pl.mul(q_one_lo, cos_lo), pl.mul(q_one_hi, sin_lo))
                    q_one_rot_hi = pl.add(pl.mul(q_one_hi, cos_hi), pl.mul(q_one_lo, sin_hi))
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(q_one_rot_lo, target_type=pl.BF16), [q_pad_row0 + qj, 0]
                    )
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(q_one_rot_hi, target_type=pl.BF16), [q_pad_row0 + qj, HALF_DIM]
                    )
            else:
                q_inv_base = ki * BATCH * Q_PER_KV + b * Q_PER_KV
                for qj in pl.range(Q_PER_KV):
                    q_inv_bj = pl.read(q_inv_states, [q_inv_base + qj])
                    q_head = pl.mul(
                        q_proj_norm[
                            b : b + 1, (q_base + qj) * HEAD_DIM : (q_base + qj + 1) * HEAD_DIM
                        ],
                        q_inv_bj,
                    )
                    q_one_lo = q_head[:, 0:HALF_DIM]
                    q_one_hi = q_head[:, HALF_DIM:HEAD_DIM]
                    q_one_rot_lo = pl.sub(pl.mul(q_one_lo, cos_lo), pl.mul(q_one_hi, sin_lo))
                    q_one_rot_hi = pl.add(pl.mul(q_one_hi, cos_hi), pl.mul(q_one_lo, sin_hi))
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(q_one_rot_lo, target_type=pl.BF16), [q_pad_row0 + qj, 0]
                    )
                    all_q_padded = pl.assemble(
                        all_q_padded, pl.cast(q_one_rot_hi, target_type=pl.BF16), [q_pad_row0 + qj, HALF_DIM]
                    )
            q_pad_zero = pl.cast(
                pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),
                target_type=pl.BF16,
            )
            all_q_padded = pl.assemble(all_q_padded, q_pad_zero, [q_pad_row0 + Q_HEAD_BATCH, 0])
            pl.tensor.write(kv_ready, [b], pl.cast(ctx_len * 0 + 1, target_type=pl.INT32))

    # ── Scope 3b accumulator allocations. ──
    # These tensors are fully overwritten by their projection tasks below. Do not
    # launch zero-fill seed tasks: with the current single-split accumulation
    # pattern they are independent WAW writers and can race with the real writes.
    down_acc_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    gate_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)
    up_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)

    # Dequantize the INT8 K cache blocks needed by fa_fused into a BF16 tensor
    # scratch. Keeping this as its own SPMD task makes the scratch a normal
    # inter-task tensor input for the following QK matmul.
    for qkd_core in pl.spmd(NUM_CORES, name_hint="qk_dequant"):
        qkd_total_blocks = pl.cast(pl.read(fa_total, [0]), target_type=pl.INDEX)
        for qkd_w in pl.range(qkd_core, qkd_total_blocks, NUM_CORES):
            qkd_enc = pl.cast(pl.read(fa_work_table, [qkd_w]), target_type=pl.INDEX)
            qkd_b = qkd_enc // MAX_ATTN_PARTS
            qkd_ready_raw = pl.read(kv_ready, [qkd_b])
            qkd_ready = pl.cast(qkd_ready_raw, target_type=pl.INDEX)
            qkd_part = qkd_enc % MAX_ATTN_PARTS
            qkd_p = qkd_part // PAGE_ATTN_PARTS
            qkd_page_part = qkd_part - qkd_p * PAGE_ATTN_PARTS
            qkd_pbid = pl.cast(
                pl.tensor.read(block_table, [qkd_b * max_blocks_per_seq + qkd_p]), pl.INDEX
            ) + (qkd_ready - qkd_ready)
            for qkd_gi in pl.range(NUM_KV_HEADS):
                qkd_cache_row = (
                    layer_cache_base + (qkd_pbid * NUM_KV_HEADS + qkd_gi) * BLOCK_SIZE + qkd_page_part * ATTN_TILE
                )
                qkd_tmp_base = (
                    (qkd_b * NUM_KV_HEADS + qkd_gi) * MAX_ATTN_PARTS * ATTN_TILE + qkd_part * ATTN_TILE
                )
                if BF16_KV_CACHE:
                    qkd_tile = pl.slice(k_cache, [ATTN_TILE, HEAD_DIM], [qkd_cache_row, 0])
                    all_k_deq_tmp = pl.assemble(all_k_deq_tmp, qkd_tile, [qkd_tmp_base, 0])
                else:
                    for qkd_ti in pl.range(ATTN_TILE):
                        qkd_scale = pl.tensor.read(k_cache_scale, [qkd_cache_row + qkd_ti, 0])
                        qkd_row_i8 = pl.slice(k_cache, [1, HEAD_DIM], [qkd_cache_row + qkd_ti, 0])
                        qkd_row_fp32 = pl.cast(pl.cast(qkd_row_i8, target_type=pl.FP16), target_type=pl.FP32)
                        all_k_deq_tmp = pl.assemble(
                            all_k_deq_tmp,
                            pl.cast(pl.mul(qkd_row_fp32, qkd_scale), target_type=pl.BF16),
                            [qkd_tmp_base + qkd_ti, 0],
                        )
            pl.tensor.write(qkd_done, [qkd_w], pl.cast(1, target_type=pl.INT32))

    # fa_fused: ONE mixed cube+vec root (QK -> softmax -> SV), BLOCK-LEVEL dense
    # static dispatch. Each grid-stride step processes exactly ONE real seq-block
    # (×GP_SIZE heads), decoded from the dense work table. Because the table holds
    # only real blocks, core = w % NUM_CORES distributes total_real_blocks evenly
    # across cores (≈1.25x of ideal), independent of per-batch length skew.
    for fa_core in pl.spmd(NUM_CORES, name_hint="fa_fused"):
        # Read the device-computed block count ON-DEVICE (inside the kernel) so
        # `fa_total` enters fa_fused as an INPUT and the normal task auto-dep
        # orders it after the `fa_work_build` producer. Reading it at
        # orchestration scope instead lowers to a host get_tensor_data that does
        # NOT wait for the producer (stale ~0 read → empty dispatch; see
        # KNOWN_ISSUES). This mirrors the existing on-device fa_work_table read.
        fa_total_blocks = pl.cast(pl.read(fa_total, [0]), target_type=pl.INDEX)
        # Grid-stride over the dense real-block list: core fa_core owns table
        # entries fa_core, fa_core+NUM_CORES, … < fa_total_blocks.
        for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):
            fa_enc = pl.cast(pl.read(fa_work_table, [fa_w]), target_type=pl.INDEX)
            fa_qkd_ready = pl.cast(pl.read(qkd_done, [fa_w]), target_type=pl.FP32)
            fa_b = fa_enc // MAX_ATTN_PARTS
            fa_ready_raw = pl.read(kv_ready, [fa_b])
            fa_ready = pl.cast(fa_ready_raw, target_type=pl.INDEX)
            fa_part = fa_enc % MAX_ATTN_PARTS
            fa_p = fa_part // PAGE_ATTN_PARTS
            fa_page_part = fa_part - fa_p * PAGE_ATTN_PARTS
            fa_hg = 0  # HEAD_GROUPS == 1 (GP_SIZE == NUM_KV_HEADS)
            fa_ctx_len = pl.read(seq_lens, [fa_b])
            # Table holds only real blocks → exactly one block per entry, at fa_p.
            sb = fa_p  # logical KV block index (no inner loop — old p_blocks was 1)
            s0 = fa_part * ATTN_TILE
            valid_len = pl.min(ATTN_TILE, fa_ctx_len - s0)
            # Paged read: map logical block sb -> physical page via this request's
            # block_table row. SEQ_TILE == page_size, so one page is exactly one
            # contiguous SEQ_TILE-row slice of the pool (shared with the kvh below).
            fa_pbid = pl.cast(
                pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX
            ) + (fa_ready - fa_ready)

            # Keep one KV head live at a time. The stage=2 pipeline form overlaps
            # adjacent heads but doubles the INT8-dequant/softmax live Vec tiles on
            # 910B, exceeding the AIV buffer in the A8W8 cache path.
            for gp in pl.range(GP_SIZE):
                gi = fa_hg * GP_SIZE + gp
                kvh = gi  # Q_GROUPS=1
                q_pad_row_g = fa_b * NUM_KV_HEADS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]
                g_base = (fa_b * NUM_KV_HEADS + gi) * MAX_ATTN_PARTS * Q_HEAD_PAD
                cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE + fa_page_part * ATTN_TILE

                # Decode keeps the paged KV cache in INT8, but dequantizes K to
                # BF16 before QK. Write K through a tensor scratch before matmul,
                # matching the prefill A8W8 pattern and avoiding unsupported
                # Vec->Right tmov from a freshly computed RHS tile.
                k_tmp_base = (fa_b * NUM_KV_HEADS + gi) * MAX_ATTN_PARTS * ATTN_TILE + fa_part * ATTN_TILE
                k_tile = all_k_deq_tmp[k_tmp_base : k_tmp_base + ATTN_TILE, :]
                raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                # Keep the QKD -> FA dependency visible without letting the
                # readiness marker participate as a numeric scale in softmax.
                scores_scaled = pl.add(pl.mul(raw_scores, ATTN_SCALE), pl.mul(fa_qkd_ready, 0.0))
                scores_valid = pl.tensor.set_validshape(scores_scaled, Q_HEAD_PAD, valid_len)
                scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                cur_mi = pl.row_max(scores)
                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                cur_li = pl.row_sum(exp_scores_fp32)

                if BF16_KV_CACHE:
                    v_tile_bf16_cache = pl.slice(v_cache, [ATTN_TILE, HEAD_DIM], [cache_row, 0])
                    oi_raw_bf16_cache = pl.matmul(exp_scores_bf16, v_tile_bf16_cache, out_dtype=pl.FP32)
                    oi_valid_bf16_cache = pl.tensor.set_validshape(oi_raw_bf16_cache, Q_HEAD_BATCH, HEAD_DIM)
                    all_oi_tmp = pl.assemble(
                        all_oi_tmp,
                        oi_valid_bf16_cache,
                        [g_base + fa_part * Q_HEAD_PAD, 0],
                    )
                else:
                    v_tile_deq = pl.create_tensor([ATTN_TILE, HEAD_DIM], dtype=pl.BF16)
                    for v_scale_ti in pl.range(ATTN_TILE):
                        v_scale = pl.tensor.read(v_cache_scale, [cache_row + v_scale_ti, 0])
                        v_row_i8 = pl.slice(v_cache, [1, HEAD_DIM], [cache_row + v_scale_ti, 0])
                        v_row_fp32 = pl.cast(pl.cast(v_row_i8, target_type=pl.FP16), target_type=pl.FP32)
                        v_tile_deq = pl.assemble(
                            v_tile_deq,
                            pl.cast(pl.mul(v_row_fp32, v_scale), target_type=pl.BF16),
                            [v_scale_ti, 0],
                        )
                    oi_raw_deq = pl.matmul(exp_scores_bf16, v_tile_deq, out_dtype=pl.FP32)
                    oi_valid_deq = pl.tensor.set_validshape(oi_raw_deq, Q_HEAD_BATCH, HEAD_DIM)
                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_valid_deq, [g_base + fa_part * Q_HEAD_PAD, 0])
                all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + fa_part * Q_HEAD_PAD, 0])
                all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + fa_part * Q_HEAD_PAD, 0])
            pl.tensor.write(fa_done, [fa_w], pl.cast(1, target_type=pl.INT32))

    # online_softmax: flat top-level spmd, writes attn_out directly. Reduces the
    # per-block partials across ALL blocks of a lane (hence across the KV
    # partitions that produced them); ordinary tensor auto-deps are enough in the
    # current API as the downstream out_proj reads attn_out directly.
    for os_core in pl.spmd(NUM_CORES * 2, name_hint="online_softmax"):
        # Grid-stride over online_softmax work items (one per (b, kvh) lane).
        for os_spmd_idx in pl.range(os_core, OS_WORK, NUM_CORES * 2):
            os_b = os_spmd_idx // NUM_KV_HEADS
            os_gi = os_spmd_idx % NUM_KV_HEADS
            os_ctx_len = pl.read(seq_lens, [os_b])
            os_ctx_blocks = (os_ctx_len + ATTN_TILE - 1) // ATTN_TILE
            os_kvh = os_gi  # Q_GROUPS=1
            os_q_base = os_kvh * Q_PER_KV
            os_g_base = (os_b * NUM_KV_HEADS + os_gi) * MAX_ATTN_PARTS * Q_HEAD_PAD
            os_w_base = pl.cast(os_ctx_len * 0, target_type=pl.INDEX)
            for os_prev_b in pl.range(os_b):
                os_w_base = os_w_base + (pl.read(seq_lens, [os_prev_b]) + (ATTN_TILE - 1)) // ATTN_TILE

            # Loop-carried accumulators: full Q_HEAD_PAD (16) rows. They can't be
            # valid-shaped because the recurrence (add / maximum) drops valid_shape,
            # which would break loop-carry type consistency. Only the per-block GM
            # reads below are trimmed to the 5 real rows via pl.slice valid_shape.
            os_done0 = pl.cast(pl.read(fa_done, [os_w_base]), target_type=pl.FP32)
            oi = all_oi_tmp[os_g_base : os_g_base + Q_HEAD_PAD, :]
            mi = pl.add(all_cur_mi[os_g_base : os_g_base + Q_HEAD_PAD, :], pl.mul(os_done0, 0.0))
            li = all_cur_li[os_g_base : os_g_base + Q_HEAD_PAD, :]
            for sb in pl.range(1, os_ctx_blocks):
                rec = os_g_base + sb * Q_HEAD_PAD
                os_done_sb = pl.cast(pl.read(fa_done, [os_w_base + sb]), target_type=pl.FP32)
                # Load a full padded tile here. The final write still slices the
                # first Q_HEAD_BATCH rows, and all recurrence math is row-wise, so
                # padded rows cannot affect valid rows. Avoiding valid_shape on
                # this loop-carried path also sidesteps multi-partial tile-load
                # instability seen when ctx_len first exceeds one ATTN_TILE.
                oi_tmp_valid = all_oi_tmp[rec : rec + Q_HEAD_PAD, :]
                online_cur_mi = pl.add(
                    all_cur_mi[rec : rec + Q_HEAD_PAD, :],
                    pl.mul(os_done_sb, 0.0),
                )
                online_cur_li = all_cur_li[rec : rec + Q_HEAD_PAD, :]
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
    mlp_down_tile = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)

    # ── A8W8: quantize attn_out per-row once, reused by all out_proj tasks. ──
    attn_out_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
    attn_out_scales = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_act_quant"):
        if OUT_ROWMAX_TRANSPOSE:
            out_quant_amax_t = pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for out_quant_amax_kb_t in pl.range(HIDDEN // K_CHUNK):
                out_quant_amax_k0_t = out_quant_amax_kb_t * K_CHUNK
                out_quant_amax_bf16_t = attn_out[:, out_quant_amax_k0_t : out_quant_amax_k0_t + K_CHUNK]
                out_quant_amax_f_t = pl.cast(out_quant_amax_bf16_t, target_type=pl.FP32)
                out_quant_amax_abs_t = pl.maximum(out_quant_amax_f_t, pl.neg(out_quant_amax_f_t))
                out_quant_amax_row_t = pl.reshape(pl.row_max(out_quant_amax_abs_t), [1, BATCH])
                out_quant_amax_t = pl.maximum(out_quant_amax_t, out_quant_amax_row_t)
            for out_quant_row_t in pl.range(BATCH):
                out_quant_row_amax_t = pl.tensor.read(out_quant_amax_t, [0, out_quant_row_t])
                out_quant_scale_q_t = INT8_SCALE_MAX / out_quant_row_amax_t
                pl.tensor.write(attn_out_scales, [out_quant_row_t, 0], out_quant_row_amax_t / INT8_SCALE_MAX)
                for out_quant_write_kb_t in pl.range(HIDDEN // K_CHUNK):
                    out_quant_write_k0_t = out_quant_write_kb_t * K_CHUNK
                    out_quant_write_bf16_t = pl.slice(
                        attn_out, [1, K_CHUNK], [out_quant_row_t, out_quant_write_k0_t]
                    )
                    out_quant_write_f_t = pl.cast(out_quant_write_bf16_t, target_type=pl.FP32)
                    out_quant_scaled_t = pl.mul(out_quant_write_f_t, out_quant_scale_q_t)
                    out_quant_i32_t = pl.cast(out_quant_scaled_t, target_type=pl.INT32, mode="rint")
                    out_quant_i32_t = pl.minimum(
                        pl.maximum(out_quant_i32_t, pl.full([1, K_CHUNK], dtype=pl.INT32, value=-127)),
                        pl.full([1, K_CHUNK], dtype=pl.INT32, value=127),
                    )
                    out_quant_half_t = pl.cast(out_quant_i32_t, target_type=pl.FP16, mode="round")
                    out_quant_i8_t = pl.cast(out_quant_half_t, target_type=pl.INT8, mode="trunc")
                    attn_out_i8 = pl.assemble(attn_out_i8, out_quant_i8_t, [out_quant_row_t, out_quant_write_k0_t])
        else:
            for out_quant_row in pl.range(BATCH):
                out_quant_amax = INT8_AMAX_EPS
                for out_quant_amax_kb in pl.range(HIDDEN // K_CHUNK):
                    out_quant_amax_k0 = out_quant_amax_kb * K_CHUNK
                    out_quant_amax_bf16 = pl.slice(attn_out, [1, K_CHUNK], [out_quant_row, out_quant_amax_k0])
                    out_quant_amax_f = pl.cast(out_quant_amax_bf16, target_type=pl.FP32)
                    out_quant_amax_abs = pl.maximum(out_quant_amax_f, pl.neg(out_quant_amax_f))
                    for out_quant_hd in pl.range(K_CHUNK):
                        out_quant_amax = pl.max(
                            out_quant_amax,
                            pl.tensor.read(out_quant_amax_abs, [0, out_quant_hd]),
                        )
                out_quant_scale_q = INT8_SCALE_MAX / out_quant_amax
                pl.tensor.write(attn_out_scales, [out_quant_row, 0], out_quant_amax / INT8_SCALE_MAX)
                for out_quant_write_kb in pl.range(HIDDEN // K_CHUNK):
                    out_quant_write_k0 = out_quant_write_kb * K_CHUNK
                    out_quant_write_bf16 = pl.slice(attn_out, [1, K_CHUNK], [out_quant_row, out_quant_write_k0])
                    out_quant_write_f = pl.cast(out_quant_write_bf16, target_type=pl.FP32)
                    out_quant_scaled = pl.mul(out_quant_write_f, out_quant_scale_q)
                    out_quant_i32 = pl.cast(out_quant_scaled, target_type=pl.INT32, mode="rint")
                    out_quant_i32 = pl.minimum(
                        pl.maximum(out_quant_i32, pl.full([1, K_CHUNK], dtype=pl.INT32, value=-127)),
                        pl.full([1, K_CHUNK], dtype=pl.INT32, value=127),
                    )
                    out_quant_half = pl.cast(out_quant_i32, target_type=pl.FP16, mode="round")
                    out_quant_i8 = pl.cast(out_quant_half, target_type=pl.INT8, mode="trunc")
                    attn_out_i8 = pl.assemble(attn_out_i8, out_quant_i8, [out_quant_row, out_quant_write_k0])

    # ── Scope 3b: auto-dep scope for out_proj + post-RMS + MLP. ──
    # Split-K split-N out_proj: 40 N-tiles x 5 K-slices (A8W8).
    if FUSE_OUT_PROJ_NPAIR:
        for n_pair in pl.parallel(N_SPLITS_OUT // 2):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_npair"):
                for n_sub_pair in pl.range(2):
                    n_op = (n_pair * 2 + n_sub_pair) * OUT_TN
                    out_c_acc = pl.full([BATCH, OUT_TN], dtype=pl.INT32, value=0)
                    for k_split_out in pl.range(K_SPLITS_OUT):
                        k_op = k_split_out * OUT_TK
                        out_acc_k = pl.matmul(
                            pl.tensor.set_validshape(attn_out_i8[:, k_op : k_op + OUT_INNER_TK], ACTIVE_BATCH, OUT_INNER_TK),
                            wo[layer_hidden_base + k_op : layer_hidden_base + OUT_INNER_TK + k_op, n_op : n_op + OUT_TN],
                            out_dtype=pl.INT32,
                        )
                        for out_lk in pl.range(1, OUT_N_SUB_K):
                            out_ks_off = out_lk * OUT_INNER_TK
                            out_a_k = pl.tensor.set_validshape(
                                attn_out_i8[:, k_op + out_ks_off : k_op + out_ks_off + OUT_INNER_TK],
                                ACTIVE_BATCH,
                                OUT_INNER_TK,
                            )
                            out_w_k = wo[
                                layer_hidden_base
                                + k_op
                                + out_ks_off : layer_hidden_base
                                + k_op
                                + out_ks_off
                                + OUT_INNER_TK,
                                n_op : n_op + OUT_TN,
                            ]
                            out_acc_k = pl.matmul_acc(out_acc_k, out_a_k, out_w_k)
                        out_c_acc = pl.add(out_c_acc, out_acc_k)
                    w_scale_col = pl.reshape(pl.slice(wo_scale, [1, OUT_TN], [layer_idx, n_op]), [1, OUT_TN])
                    out_fp32 = pl.mul(
                        pl.col_expand_mul(pl.cast(out_c_acc, target_type=pl.FP32), w_scale_col),
                        attn_out_scales,
                    )
                    attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_fp32, [0, n_op])
    else:
        for n_out_proj in pl.parallel(N_SPLITS_OUT):
            n_op = n_out_proj * OUT_TN
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj"):
                out_c_acc = pl.full([BATCH, OUT_TN], dtype=pl.INT32, value=0)
                for k_split_out in pl.range(K_SPLITS_OUT):
                    k_op = k_split_out * OUT_TK
                    out_acc_k = pl.matmul(
                        pl.tensor.set_validshape(attn_out_i8[:, k_op : k_op + OUT_INNER_TK], ACTIVE_BATCH, OUT_INNER_TK),
                        wo[layer_hidden_base + k_op : layer_hidden_base + OUT_INNER_TK + k_op, n_op : n_op + OUT_TN],
                        out_dtype=pl.INT32,
                    )
                    for out_lk in pl.range(1, OUT_N_SUB_K):
                        out_ks_off = out_lk * OUT_INNER_TK
                        out_a_k = pl.tensor.set_validshape(
                            attn_out_i8[:, k_op + out_ks_off : k_op + out_ks_off + OUT_INNER_TK],
                            ACTIVE_BATCH,
                            OUT_INNER_TK,
                        )
                        out_w_k = wo[
                            layer_hidden_base
                            + k_op
                            + out_ks_off : layer_hidden_base
                            + k_op
                            + out_ks_off
                            + OUT_INNER_TK,
                            n_op : n_op + OUT_TN,
                        ]
                        out_acc_k = pl.matmul_acc(out_acc_k, out_a_k, out_w_k)
                    out_c_acc = pl.add(out_c_acc, out_acc_k)
                w_scale_col = pl.reshape(pl.slice(wo_scale, [1, OUT_TN], [layer_idx, n_op]), [1, OUT_TN])
                out_fp32 = pl.mul(
                    pl.col_expand_mul(pl.cast(out_c_acc, target_type=pl.FP32), w_scale_col),
                    attn_out_scales,
                )
                attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_fp32, [0, n_op])

    # Tiled residual + BF16 cast.
    for k_slice in pl.unroll(K_SPLITS_MLP):
        k_base = k_slice * MLP_K_SLICE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_rms_cast"):
            for kb in pl.pipeline(MLP_K_SLICE // K_CHUNK, stage=2):
                resid_k0 = k_base + kb * K_CHUNK
                resid_attn_chunk = attn_proj_fp32[:, resid_k0 : resid_k0 + K_CHUNK]
                resid_hidden_chunk = hidden_states[:, resid_k0 : resid_k0 + K_CHUNK]
                resid_fp32 = pl.add(resid_attn_chunk, pl.cast(resid_hidden_chunk, target_type=pl.FP32))
                # Raw residual h1 — added back after down_proj (must NOT be gamma-scaled).
                post_norm_partial = pl.assemble(
                    post_norm_partial, pl.cast(resid_fp32, target_type=pl.BF16), [0, resid_k0]
                )
                # Explicit post-RMS gamma: gate/up input = h1 * post_gamma. gamma is
                # per-K (the matmul contraction dim) so it canNOT defer past the matmul
                # like inv_rms does — it scales the input here (with raw w_gate/w_up).
                post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, resid_k0])
                mlp_norm_in = pl.assemble(
                    mlp_norm_in,
                    pl.cast(pl.col_expand_mul(resid_fp32, post_gamma), target_type=pl.BF16),
                    [0, resid_k0],
                )

    # RMS reduction reads all of attn_proj_fp32.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rms_reduce"):
        sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=2):
            post_rms_k0 = kb * K_CHUNK
            post_rms_attn_chunk = attn_proj_fp32[:, post_rms_k0 : post_rms_k0 + K_CHUNK]
            post_rms_hidden_chunk = hidden_states[:, post_rms_k0 : post_rms_k0 + K_CHUNK]
            resid_chunk = pl.add(post_rms_attn_chunk, pl.cast(post_rms_hidden_chunk, target_type=pl.FP32))
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH]),
            )
        post_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
        post_inv_rms_col = pl.reshape(post_inv_rms, [BATCH, 1])
        inv_rms_tile = pl.assemble(inv_rms_tile, post_inv_rms_col, [0, 0])

    # Split-K gate + up.  The atomic path mirrors the BF16 decode layout:
    # parallelize K-slices and accumulate partial FP32 projections in GM.
    if MLP_SPLITK_ATOMIC:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_up_seed") as gate_up_seed_tid:
            for n_out in pl.pipeline(MLP_ON, stage=2):
                gate_n0 = n_out * MLP_TN
                zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                gate_acc_all = pl.assemble(gate_acc_all, zero, [0, gate_n0])
                up_acc_all = pl.assemble(up_acc_all, zero, [0, gate_n0])

        for n_out in pl.parallel(MLP_ON):
            gate_n0 = n_out * MLP_TN
            for k_split in pl.range(K_SPLITS_MLP):
                gate_k0 = k_split * MLP_K_SLICE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj", deps=[gate_up_seed_tid]):
                    gate_c_acc = pl.matmul(
                        pl.tensor.set_validshape(mlp_norm_in[:, gate_k0 : gate_k0 + MLP_INNER_TK], ACTIVE_BATCH, MLP_INNER_TK),
                        w_gate[
                            layer_hidden_base + gate_k0 : layer_hidden_base + gate_k0 + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ],
                        out_dtype=pl.FP32,
                    )
                    for lk in pl.range(1, MLP_N_SUB_K):
                        gate_ks_off = lk * MLP_INNER_TK
                        gate_a_k = pl.tensor.set_validshape(
                            mlp_norm_in[:, gate_k0 + gate_ks_off : gate_k0 + gate_ks_off + MLP_INNER_TK],
                            ACTIVE_BATCH,
                            MLP_INNER_TK,
                        )
                        gate_w_k = w_gate[
                            layer_hidden_base
                            + gate_k0
                            + gate_ks_off : layer_hidden_base
                            + gate_k0
                            + gate_ks_off
                            + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ]
                        gate_c_acc = pl.matmul_acc(gate_c_acc, gate_a_k, gate_w_k)
                    gate_acc_all = pl.assemble(gate_acc_all, gate_c_acc, [0, gate_n0], atomic=pl.AtomicType.Add)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj", deps=[gate_up_seed_tid]):
                    up_c_acc = pl.matmul(
                        pl.tensor.set_validshape(mlp_norm_in[:, gate_k0 : gate_k0 + MLP_INNER_TK], ACTIVE_BATCH, MLP_INNER_TK),
                        w_up[
                            layer_hidden_base + gate_k0 : layer_hidden_base + gate_k0 + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ],
                        out_dtype=pl.FP32,
                    )
                    for lk in pl.range(1, MLP_N_SUB_K):
                        up_ks_off = lk * MLP_INNER_TK
                        up_a_k = pl.tensor.set_validshape(
                            mlp_norm_in[:, gate_k0 + up_ks_off : gate_k0 + up_ks_off + MLP_INNER_TK],
                            ACTIVE_BATCH,
                            MLP_INNER_TK,
                        )
                        up_w_k = w_up[
                            layer_hidden_base
                            + gate_k0
                            + up_ks_off : layer_hidden_base
                            + gate_k0
                            + up_ks_off
                            + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ]
                        up_c_acc = pl.matmul_acc(up_c_acc, up_a_k, up_w_k)
                    up_acc_all = pl.assemble(up_acc_all, up_c_acc, [0, gate_n0], atomic=pl.AtomicType.Add)
    elif FUSE_GATE_UP_PROJ:
        for n_out in pl.parallel(MLP_ON):
            gate_n0 = n_out * MLP_TN
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_up_proj"):
                gate_acc = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                for k_split in pl.range(K_SPLITS_MLP):
                    gate_k0 = k_split * MLP_K_SLICE
                    gate_c_acc = pl.matmul(
                        pl.tensor.set_validshape(mlp_norm_in[:, gate_k0 : gate_k0 + MLP_INNER_TK], ACTIVE_BATCH, MLP_INNER_TK),
                        w_gate[
                            layer_hidden_base + gate_k0 : layer_hidden_base + gate_k0 + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ],
                        out_dtype=pl.FP32,
                    )
                    for lk in pl.range(1, MLP_N_SUB_K):
                        gate_ks_off = lk * MLP_INNER_TK
                        gate_a_k = pl.tensor.set_validshape(
                            mlp_norm_in[:, gate_k0 + gate_ks_off : gate_k0 + gate_ks_off + MLP_INNER_TK],
                            ACTIVE_BATCH,
                            MLP_INNER_TK,
                        )
                        gate_w_k = w_gate[
                            layer_hidden_base
                            + gate_k0
                            + gate_ks_off : layer_hidden_base
                            + gate_k0
                            + gate_ks_off
                            + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ]
                        gate_c_acc = pl.matmul_acc(gate_c_acc, gate_a_k, gate_w_k)
                    gate_acc = pl.add(gate_acc, gate_c_acc)
                gate_acc_all = pl.assemble(gate_acc_all, gate_acc, [0, gate_n0])

                up_acc = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                for k_split in pl.range(K_SPLITS_MLP):
                    up_k0 = k_split * MLP_K_SLICE
                    up_c_acc = pl.matmul(
                        pl.tensor.set_validshape(mlp_norm_in[:, up_k0 : up_k0 + MLP_INNER_TK], ACTIVE_BATCH, MLP_INNER_TK),
                        w_up[
                            layer_hidden_base + up_k0 : layer_hidden_base + up_k0 + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ],
                        out_dtype=pl.FP32,
                    )
                    for lk in pl.range(1, MLP_N_SUB_K):
                        up_ks_off = lk * MLP_INNER_TK
                        up_a_k = pl.tensor.set_validshape(
                            mlp_norm_in[:, up_k0 + up_ks_off : up_k0 + up_ks_off + MLP_INNER_TK],
                            ACTIVE_BATCH,
                            MLP_INNER_TK,
                        )
                        up_w_k = w_up[
                            layer_hidden_base
                            + up_k0
                            + up_ks_off : layer_hidden_base
                            + up_k0
                            + up_ks_off
                            + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ]
                        up_c_acc = pl.matmul_acc(up_c_acc, up_a_k, up_w_k)
                    up_acc = pl.add(up_acc, up_c_acc)
                up_acc_all = pl.assemble(up_acc_all, up_acc, [0, gate_n0])
    else:
        for n_out in pl.parallel(MLP_ON):
            gate_n0 = n_out * MLP_TN
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
                gate_acc = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                for k_split in pl.range(K_SPLITS_MLP):
                    gate_k0 = k_split * MLP_K_SLICE
                    gate_c_acc = pl.matmul(
                        pl.tensor.set_validshape(mlp_norm_in[:, gate_k0 : gate_k0 + MLP_INNER_TK], ACTIVE_BATCH, MLP_INNER_TK),
                        w_gate[
                            layer_hidden_base + gate_k0 : layer_hidden_base + gate_k0 + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ],
                        out_dtype=pl.FP32,
                    )
                    for lk in pl.range(1, MLP_N_SUB_K):
                        gate_ks_off = lk * MLP_INNER_TK
                        gate_a_k = pl.tensor.set_validshape(
                            mlp_norm_in[:, gate_k0 + gate_ks_off : gate_k0 + gate_ks_off + MLP_INNER_TK],
                            ACTIVE_BATCH,
                            MLP_INNER_TK,
                        )
                        gate_w_k = w_gate[
                            layer_hidden_base
                            + gate_k0
                            + gate_ks_off : layer_hidden_base
                            + gate_k0
                            + gate_ks_off
                            + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ]
                        gate_c_acc = pl.matmul_acc(gate_c_acc, gate_a_k, gate_w_k)
                    gate_acc = pl.add(gate_acc, gate_c_acc)
                gate_acc_all = pl.assemble(gate_acc_all, gate_acc, [0, gate_n0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
                up_acc = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                for k_split in pl.range(K_SPLITS_MLP):
                    up_k0 = k_split * MLP_K_SLICE
                    up_c_acc = pl.matmul(
                        pl.tensor.set_validshape(mlp_norm_in[:, up_k0 : up_k0 + MLP_INNER_TK], ACTIVE_BATCH, MLP_INNER_TK),
                        w_up[
                            layer_hidden_base + up_k0 : layer_hidden_base + up_k0 + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ],
                        out_dtype=pl.FP32,
                    )
                    for lk in pl.range(1, MLP_N_SUB_K):
                        up_ks_off = lk * MLP_INNER_TK
                        up_a_k = pl.tensor.set_validshape(
                            mlp_norm_in[:, up_k0 + up_ks_off : up_k0 + up_ks_off + MLP_INNER_TK],
                            ACTIVE_BATCH,
                            MLP_INNER_TK,
                        )
                        up_w_k = w_up[
                            layer_hidden_base
                            + up_k0
                            + up_ks_off : layer_hidden_base
                            + up_k0
                            + up_ks_off
                            + MLP_INNER_TK,
                            gate_n0 : gate_n0 + MLP_TN,
                        ]
                        up_c_acc = pl.matmul_acc(up_c_acc, up_a_k, up_w_k)
                    up_acc = pl.add(up_acc, up_c_acc)
                up_acc_all = pl.assemble(up_acc_all, up_acc, [0, gate_n0])

    # silu.
    if not FUSE_SILU_DOWN_PROJ:
        for n_out in pl.range(MLP_ON):
            silu_n0 = n_out * MLP_TN
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
                silu_inv_rms_chunk = inv_rms_tile[:, 0:1]
                for sub in pl.range(SILU_INNER_CHUNKS):
                    silu_off = silu_n0 + sub * MLP_OUT_CHUNK
                    gate_chunk = gate_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]
                    up_chunk = up_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]
                    scaled_gate = pl.row_expand_mul(gate_chunk, silu_inv_rms_chunk)
                    scaled_up = pl.row_expand_mul(up_chunk, silu_inv_rms_chunk)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(scaled_gate)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(scaled_gate, sigmoid), scaled_up)
                    mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                    mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, silu_off])
                    if FUSE_MLP_DOWN_MATERIALIZE:
                        mlp_down_tile = pl.assemble(mlp_down_tile, mlp_chunk_bf16, [0, silu_off])

    if (not FUSE_SILU_DOWN_PROJ) and (not FUSE_MLP_DOWN_MATERIALIZE):
        # Keep the SiLU producer/output tensor separate from the down-projection input.
        # The direct parallel-yield value can miss MLIR mapping in fused decode.
        for n_out in pl.range(MLP_ON):
            mlp_down_n0 = n_out * MLP_TN
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mlp_down_materialize"):
                for sub in pl.range(SILU_INNER_CHUNKS):
                    mlp_down_off = mlp_down_n0 + sub * MLP_OUT_CHUNK
                    mlp_down_chunk = mlp_tile[:, mlp_down_off : mlp_down_off + MLP_OUT_CHUNK]
                    mlp_down_tile = pl.assemble(mlp_down_tile, mlp_down_chunk, [0, mlp_down_off])

    if DOWN_SPLITK_ATOMIC:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_seed") as down_seed_tid:
            for n_out in pl.pipeline(DOWN_ON, stage=2):
                down_n0 = n_out * DOWN_TN
                down_zero = pl.full([BATCH, DOWN_TN], dtype=pl.FP32, value=0.0)
                down_acc_all = pl.assemble(down_acc_all, down_zero, [0, down_n0])

        for n_out in pl.parallel(DOWN_ON):
            down_n0 = n_out * DOWN_TN
            for k_split in pl.range(K_SPLITS):
                down_k0 = k_split * DOWN_K_SLICE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj", deps=[down_seed_tid]):
                    down_c_acc = pl.matmul(
                        pl.tensor.set_validshape(mlp_down_tile[:, down_k0 : down_k0 + DOWN_TK], ACTIVE_BATCH, DOWN_TK),
                        w_down[
                            layer_inter_base + down_k0 : layer_inter_base + down_k0 + DOWN_TK,
                            down_n0 : down_n0 + DOWN_TN,
                        ],
                        out_dtype=pl.FP32,
                    )
                    for lk in pl.range(1, N_SUB_K):
                        down_ks_off = lk * DOWN_TK
                        down_a_k = pl.tensor.set_validshape(
                            mlp_down_tile[:, down_k0 + down_ks_off : down_k0 + down_ks_off + DOWN_TK],
                            ACTIVE_BATCH,
                            DOWN_TK,
                        )
                        down_w_k = w_down[
                            layer_inter_base
                            + down_k0
                            + down_ks_off : layer_inter_base
                            + down_k0
                            + down_ks_off
                            + DOWN_TK,
                            down_n0 : down_n0 + DOWN_TN,
                        ]
                        down_c_acc = pl.matmul_acc(down_c_acc, down_a_k, down_w_k)
                    down_acc_all = pl.assemble(down_acc_all, down_c_acc, [0, down_n0], atomic=pl.AtomicType.Add)
    else:
        for n_out in pl.parallel(DOWN_ON):
            down_n0 = n_out * DOWN_TN
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj"):
                down_acc = pl.full([BATCH, DOWN_TN], dtype=pl.FP32, value=0.0)
                for k_split in pl.range(K_SPLITS):
                    down_k0 = k_split * DOWN_K_SLICE
                    if FUSE_SILU_DOWN_PROJ:
                        sd_inv_rms_chunk = inv_rms_tile[:, 0:1]
                        sd_gate_chunk = gate_acc_all[:, down_k0 : down_k0 + DOWN_TK]
                        sd_up_chunk = up_acc_all[:, down_k0 : down_k0 + DOWN_TK]
                        sd_scaled_gate = pl.row_expand_mul(sd_gate_chunk, sd_inv_rms_chunk)
                        sd_scaled_up = pl.row_expand_mul(sd_up_chunk, sd_inv_rms_chunk)
                        sd_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(sd_scaled_gate)), 1.0))
                        sd_mlp_down_chunk = pl.cast(
                            pl.mul(pl.mul(sd_scaled_gate, sd_sigmoid), sd_scaled_up),
                            target_type=pl.BF16,
                        )
                    else:
                        sd_mlp_down_chunk = mlp_down_tile[:, down_k0 : down_k0 + DOWN_TK]
                    down_c_acc = pl.matmul(
                        pl.tensor.set_validshape(sd_mlp_down_chunk, ACTIVE_BATCH, DOWN_TK),
                        w_down[layer_inter_base + down_k0 : layer_inter_base + down_k0 + DOWN_TK, down_n0 : down_n0 + DOWN_TN],
                        out_dtype=pl.FP32,
                    )
                    for lk in pl.range(1, N_SUB_K):
                        down_ks_off = lk * DOWN_TK
                        if FUSE_SILU_DOWN_PROJ:
                            sd_inv_rms_next = inv_rms_tile[:, 0:1]
                            gate_next = gate_acc_all[:, down_k0 + down_ks_off : down_k0 + down_ks_off + DOWN_TK]
                            up_next = up_acc_all[:, down_k0 + down_ks_off : down_k0 + down_ks_off + DOWN_TK]
                            scaled_gate_next = pl.row_expand_mul(gate_next, sd_inv_rms_next)
                            scaled_up_next = pl.row_expand_mul(up_next, sd_inv_rms_next)
                            sigmoid_next = pl.recip(pl.add(pl.exp(pl.neg(scaled_gate_next)), 1.0))
                            down_a_src = pl.cast(
                                pl.mul(pl.mul(scaled_gate_next, sigmoid_next), scaled_up_next),
                                target_type=pl.BF16,
                            )
                        else:
                            down_a_src = mlp_down_tile[:, down_k0 + down_ks_off : down_k0 + down_ks_off + DOWN_TK]
                        down_a_k = pl.tensor.set_validshape(
                            down_a_src,
                            ACTIVE_BATCH,
                            DOWN_TK,
                        )
                        down_w_k = w_down[
                            layer_inter_base + down_k0 + down_ks_off : layer_inter_base + down_k0 + down_ks_off + DOWN_TK,
                            down_n0 : down_n0 + DOWN_TN,
                        ]
                        down_c_acc = pl.matmul_acc(down_c_acc, down_a_k, down_w_k)
                    down_acc = pl.add(down_acc, down_c_acc)
                down_acc_all = pl.assemble(down_acc_all, down_acc, [0, down_n0])

    with pl.spmd(DOWN_ON, name_hint="down_cast_residual") as dcr_tid:
        n_out = pl.tile.get_block_idx()
        down_cast_n0 = n_out * DOWN_TN
        resid_block_bf16 = post_norm_partial[:, down_cast_n0 : down_cast_n0 + DOWN_TN]
        resid_block = pl.cast(resid_block_bf16, target_type=pl.FP32)
        acc_chunk_bf16 = pl.cast(
            down_acc_all[:, down_cast_n0 : down_cast_n0 + DOWN_TN],
            target_type=pl.BF16,
        )
        acc_chunk = pl.cast(acc_chunk_bf16, target_type=pl.FP32)
        out_chunk = pl.add(acc_chunk, resid_block)
        out_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
        if DECODE_DEBUG_STAGE_ID == 17:
            out_partial = pl.assemble(out_partial, out_bf16, [0, down_cast_n0])
        else:
            out = pl.assemble(out, out_bf16, [0, down_cast_n0])

    if DECODE_DEBUG_STAGE_ID == 1:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_q_proj_out"):
                out = pl.assemble(
                    out,
                    pl.cast(q_proj[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK], target_type=pl.BF16),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 2:
        for debug_kb in pl.parallel(KV_HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_k_proj_out"):
                out = pl.assemble(
                    out,
                    pl.cast(k_proj[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK], target_type=pl.BF16),
                    [0, debug_k0],
                )
        for debug_kb in pl.parallel(KV_HIDDEN // RMSNORM_K_CHUNK, HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_k_proj_pad"):
                out = pl.assemble(
                    out,
                    pl.full([BATCH, RMSNORM_K_CHUNK], dtype=pl.BF16, value=0.0),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 3:
        for debug_kb in pl.parallel(KV_HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_v_proj_out"):
                out = pl.assemble(
                    out,
                    pl.cast(v_proj[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK], target_type=pl.BF16),
                    [0, debug_k0],
                )
        for debug_kb in pl.parallel(KV_HIDDEN // RMSNORM_K_CHUNK, HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_v_proj_pad"):
                out = pl.assemble(
                    out,
                    pl.full([BATCH, RMSNORM_K_CHUNK], dtype=pl.BF16, value=0.0),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 4:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_attn_out"):
                out = pl.assemble(
                    out,
                    attn_out[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK],
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 5:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_attn_proj"):
                out = pl.assemble(
                    out,
                    pl.cast(attn_proj_fp32[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK], target_type=pl.BF16),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 6:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_post_resid"):
                out = pl.assemble(
                    out,
                    post_norm_partial[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK],
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 7:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_mlp_norm_in"):
                out = pl.assemble(
                    out,
                    mlp_norm_in[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK],
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 8:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_gate_acc"):
                out = pl.assemble(
                    out,
                    pl.cast(gate_acc_all[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK], target_type=pl.BF16),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 9:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_up_acc"):
                out = pl.assemble(
                    out,
                    pl.cast(up_acc_all[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK], target_type=pl.BF16),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 10:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_mlp_tile"):
                out = pl.assemble(
                    out,
                    mlp_tile[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK],
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 11:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_down_acc"):
                out = pl.assemble(
                    out,
                    pl.cast(down_acc_all[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK], target_type=pl.BF16),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 12:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_post_inv_rms"):
                inv_rms_debug = pl.row_expand_mul(
                    pl.full([BATCH, RMSNORM_K_CHUNK], dtype=pl.FP32, value=1.0),
                    inv_rms_tile[:, 0:1],
                )
                out = pl.assemble(
                    out,
                    pl.cast(inv_rms_debug, target_type=pl.BF16),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 13:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_mlp_tile_5k"):
                out = pl.assemble(
                    out,
                    mlp_tile[:, 5120 + debug_k0 : 5120 + debug_k0 + RMSNORM_K_CHUNK],
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 14:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_mlp_tile_10k"):
                out = pl.assemble(
                    out,
                    mlp_tile[:, 10240 + debug_k0 : 10240 + debug_k0 + RMSNORM_K_CHUNK],
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 15:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_mlp_tile_12k"):
                out = pl.assemble(
                    out,
                    mlp_tile[:, 12288 + debug_k0 : 12288 + debug_k0 + RMSNORM_K_CHUNK],
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 16:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_down_acc_tile2"):
            out = pl.assemble(
                out,
                pl.cast(down_acc_all[:, 512:768], target_type=pl.BF16),
                [0, 512],
            )
    elif DECODE_DEBUG_STAGE_ID == 17:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_out_partial_tile2"):
            out = pl.assemble(
                out,
                out_partial[:, 512:768],
                [0, 512],
            )
    elif DECODE_DEBUG_STAGE_ID == 18:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_x_gamma"):
                out = pl.assemble(
                    out,
                    normed_states[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK],
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 19:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_act_i8"):
                out = pl.assemble(
                    out,
                    pl.cast(
                        pl.cast(
                            pl.cast(
                                normed_i8[:, debug_k0 : debug_k0 + RMSNORM_K_CHUNK],
                                target_type=pl.INT32,
                            ),
                            target_type=pl.FP32,
                        ),
                        target_type=pl.BF16,
                    ),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 20:
        for debug_kb in pl.parallel(HIDDEN // RMSNORM_K_CHUNK):
            debug_k0 = debug_kb * RMSNORM_K_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_act_scale"):
                act_scale_debug = pl.row_expand_mul(
                    pl.full([BATCH, RMSNORM_K_CHUNK], dtype=pl.FP32, value=1.0),
                    act_scales,
                )
                out = pl.assemble(
                    out,
                    pl.cast(act_scale_debug, target_type=pl.BF16),
                    [0, debug_k0],
                )
    elif DECODE_DEBUG_STAGE_ID == 21:
        for debug_b in pl.parallel(BATCH):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_q_rope"):
                for debug_ki in pl.range(NUM_KV_HEADS):
                    debug_q_base = debug_ki * Q_PER_KV
                    debug_q_pad_base = debug_b * NUM_KV_HEADS * Q_HEAD_PAD + debug_ki * Q_HEAD_PAD
                    for debug_qj in pl.range(Q_PER_KV):
                        debug_q_row = debug_q_pad_base + debug_qj
                        debug_q_col = (debug_q_base + debug_qj) * HEAD_DIM
                        out = pl.assemble(
                            out,
                            all_q_padded[debug_q_row : debug_q_row + 1, :],
                            [debug_b, debug_q_col],
                        )
    elif DECODE_DEBUG_STAGE_ID == 22:
        for debug_b in pl.parallel(BATCH):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_k_cur_cache"):
                debug_wr_slot = pl.cast(pl.tensor.read(slot_mapping, [debug_b]), pl.INDEX)
                debug_wr_block = debug_wr_slot // BLOCK_SIZE
                debug_wr_offset = debug_wr_slot - debug_wr_block * BLOCK_SIZE
                for debug_ki in pl.range(NUM_KV_HEADS):
                    debug_cache_row = (
                        layer_cache_base
                        + (debug_wr_block * NUM_KV_HEADS + debug_ki) * BLOCK_SIZE
                        + debug_wr_offset
                    )
                    debug_k_scale = pl.tensor.read(k_cache_scale, [debug_cache_row, 0])
                    debug_k_i8 = pl.slice(k_cache, [1, HEAD_DIM], [debug_cache_row, 0])
                    debug_k_f = pl.mul(
                        pl.cast(pl.cast(debug_k_i8, target_type=pl.FP16), target_type=pl.FP32),
                        debug_k_scale,
                    )
                    out = pl.assemble(
                        out,
                        pl.cast(debug_k_f, target_type=pl.BF16),
                        [debug_b, debug_ki * HEAD_DIM],
                    )
                for debug_pad_kb in pl.range(KV_HIDDEN // RMSNORM_K_CHUNK, HIDDEN // RMSNORM_K_CHUNK):
                    debug_pad_k0 = debug_pad_kb * RMSNORM_K_CHUNK
                    out = pl.assemble(
                        out,
                        pl.full([1, RMSNORM_K_CHUNK], dtype=pl.BF16, value=0.0),
                        [debug_b, debug_pad_k0],
                    )
    elif DECODE_DEBUG_STAGE_ID == 23:
        for debug_b in pl.parallel(BATCH):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_v_cur_cache"):
                debug_wr_slot = pl.cast(pl.tensor.read(slot_mapping, [debug_b]), pl.INDEX)
                debug_wr_block = debug_wr_slot // BLOCK_SIZE
                debug_wr_offset = debug_wr_slot - debug_wr_block * BLOCK_SIZE
                for debug_ki in pl.range(NUM_KV_HEADS):
                    debug_cache_row = (
                        layer_cache_base
                        + (debug_wr_block * NUM_KV_HEADS + debug_ki) * BLOCK_SIZE
                        + debug_wr_offset
                    )
                    debug_v_scale = pl.tensor.read(v_cache_scale, [debug_cache_row, 0])
                    debug_v_i8 = pl.slice(v_cache, [1, HEAD_DIM], [debug_cache_row, 0])
                    debug_v_f = pl.mul(
                        pl.cast(pl.cast(debug_v_i8, target_type=pl.FP16), target_type=pl.FP32),
                        debug_v_scale,
                    )
                    out = pl.assemble(
                        out,
                        pl.cast(debug_v_f, target_type=pl.BF16),
                        [debug_b, debug_ki * HEAD_DIM],
                    )
                for debug_pad_kb in pl.range(KV_HIDDEN // RMSNORM_K_CHUNK, HIDDEN // RMSNORM_K_CHUNK):
                    debug_pad_k0 = debug_pad_kb * RMSNORM_K_CHUNK
                    out = pl.assemble(
                        out,
                        pl.full([1, RMSNORM_K_CHUNK], dtype=pl.BF16, value=0.0),
                        [debug_b, debug_pad_k0],
                    )
    elif DECODE_DEBUG_STAGE_ID == 24:
        for debug_b in pl.parallel(BATCH):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_q_pre_rope"):
                for debug_ki in pl.range(NUM_KV_HEADS):
                    debug_q_base = debug_ki * Q_PER_KV
                    for debug_qj in pl.range(Q_PER_KV):
                        debug_q_col = (debug_q_base + debug_qj) * HEAD_DIM
                        debug_q_head = q_proj_norm[debug_b : debug_b + 1, debug_q_col : debug_q_col + HEAD_DIM]
                        if not FUSED_QK_NORM:
                            debug_q_inv_base = debug_ki * BATCH * Q_PER_KV + debug_b * Q_PER_KV
                            debug_q_inv = pl.read(q_inv_states, [debug_q_inv_base + debug_qj])
                            debug_q_head = pl.mul(debug_q_head, debug_q_inv)
                        out = pl.assemble(
                            out,
                            pl.cast(debug_q_head, target_type=pl.BF16),
                            [debug_b, debug_q_col],
                        )
    elif DECODE_DEBUG_STAGE_ID == 25:
        for debug_b in pl.parallel(BATCH):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_k_pre_rope"):
                for debug_ki in pl.range(NUM_KV_HEADS):
                    debug_k_col = debug_ki * HEAD_DIM
                    debug_k_head = k_proj_norm[debug_b : debug_b + 1, debug_k_col : debug_k_col + HEAD_DIM]
                    if not FUSED_QK_NORM:
                        debug_k_inv = pl.read(k_inv_states, [debug_ki * BATCH + debug_b])
                        debug_k_head = pl.mul(debug_k_head, debug_k_inv)
                    out = pl.assemble(
                        out,
                        pl.cast(debug_k_head, target_type=pl.BF16),
                        [debug_b, debug_k_col],
                    )
                for debug_pad_kb in pl.range(KV_HIDDEN // RMSNORM_K_CHUNK, HIDDEN // RMSNORM_K_CHUNK):
                    debug_pad_k0 = debug_pad_kb * RMSNORM_K_CHUNK
                    out = pl.assemble(
                        out,
                        pl.full([1, RMSNORM_K_CHUNK], dtype=pl.BF16, value=0.0),
                        [debug_b, debug_pad_k0],
                    )
    elif DECODE_DEBUG_STAGE_ID == 26:
        for debug_b in pl.parallel(BATCH):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="debug_k_rope_raw"):
                debug_ctx_len = pl.read(seq_lens, [debug_b])
                debug_pos = debug_ctx_len - 1
                debug_cos_lo = rope_cos[debug_pos : debug_pos + 1, 0:HALF_DIM]
                debug_cos_hi = rope_cos[debug_pos : debug_pos + 1, HALF_DIM:HEAD_DIM]
                debug_sin_lo = rope_sin[debug_pos : debug_pos + 1, 0:HALF_DIM]
                debug_sin_hi = rope_sin[debug_pos : debug_pos + 1, HALF_DIM:HEAD_DIM]
                for debug_ki in pl.range(NUM_KV_HEADS):
                    debug_k_col = debug_ki * HEAD_DIM
                    debug_k_head = k_proj_norm[debug_b : debug_b + 1, debug_k_col : debug_k_col + HEAD_DIM]
                    if not FUSED_QK_NORM:
                        debug_k_inv = pl.read(k_inv_states, [debug_ki * BATCH + debug_b])
                        debug_k_head = pl.mul(debug_k_head, debug_k_inv)
                    debug_k_lo = debug_k_head[:, 0:HALF_DIM]
                    debug_k_hi = debug_k_head[:, HALF_DIM:HEAD_DIM]
                    debug_rot_lo = pl.sub(pl.mul(debug_k_lo, debug_cos_lo), pl.mul(debug_k_hi, debug_sin_lo))
                    debug_rot_hi = pl.add(pl.mul(debug_k_hi, debug_cos_hi), pl.mul(debug_k_lo, debug_sin_hi))
                    out = pl.assemble(out, pl.cast(debug_rot_lo, target_type=pl.BF16), [debug_b, debug_k_col])
                    out = pl.assemble(
                        out,
                        pl.cast(debug_rot_hi, target_type=pl.BF16),
                        [debug_b, debug_k_col + HALF_DIM],
                    )
                for debug_pad_kb in pl.range(KV_HIDDEN // RMSNORM_K_CHUNK, HIDDEN // RMSNORM_K_CHUNK):
                    debug_pad_k0 = debug_pad_kb * RMSNORM_K_CHUNK
                    out = pl.assemble(
                        out,
                        pl.full([1, RMSNORM_K_CHUNK], dtype=pl.BF16, value=0.0),
                        [debug_b, debug_pad_k0],
                    )
    return out


NUM_LAYERS = 40  # full Qwen3-14B depth, for the fused decode_fwd loop


@pl.jit
def decode_fwd(  # noqa: PLR0913 — device-side fused NUM_LAYERS decode + LM head
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    wq_scale: pl.Tensor,
    wk_scale: pl.Tensor,
    wv_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    k_cache_scale: pl.Tensor,
    v_cache_scale: pl.Tensor,
    wo: pl.Tensor,
    wo_scale: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    final_norm_weight: pl.Tensor,
    lm_head_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for cb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                cur = pl.assemble(
                    cur, pl.slice(hidden_states, [BATCH, RMSNORM_K_CHUNK], [cb0, ck0]), [cb0, ck0]
                )
    for layer_idx in pl.range(NUM_LAYERS):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        cur = _decode_layer(
            cur, input_rms_weight,
            wq, wk, wv, wq_scale, wk_scale, wv_scale,
            q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin,
            k_cache, v_cache, k_cache_scale, v_cache_scale,
            wo, wo_scale,
            w_gate, w_up, w_down, post_rms_weight, next_hidden, layer_idx,
        )
    out = rms_lm_head(cur, final_norm_weight, lm_head_weight, seq_lens, out)
    return out
