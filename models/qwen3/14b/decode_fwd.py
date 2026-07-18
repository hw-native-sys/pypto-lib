# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim    # CI marker: external CCEC attention is A2/A3 onboard only
# ci: no-dep-gen  # The external attention kernel requires an uninstrumented full mixed-core cohort.
"""Qwen3-14B decode with FP32 inter-layer carry and direct CANN attention.

The projection, QK-norm, RoPE, output projection, MLP, and dependency topology
follow the main implementation. The attention stage calls CANN
FusedInferAttentionScore through ``pl.jit.extern``. Its public ABI matches vLLM:
Q/O are active TND and the flat paged K/V buffers contain BSND bytes ordered as
``[page, token, kv_head, dim]``.

``decode_fwd`` accepts public batch 1 or 16 while keeping the model pipeline
internally padded to 16 rows. The inter-layer residual remains FP32; BF16
conversion occurs only at the external chunk boundaries and model-defined
compute boundaries.
"""

import argparse
import os
from pathlib import Path

import pypto.language as pl
import torch
from pypto.backend import BackendType, set_backend_type
from pypto.runtime import RunConfig

from config import (
    QWEN3_14B_DIMS as D,
    QWEN3_14B_TILING as T,
    QWEN3_14B as M,
)  # vocab size for the fused decode_fwd LM head / logits
from paged_attention_cce import (
    DEFAULT_BLOCK_DIM as PA_DEFAULT_BLOCK_DIM,
    METADATA_BYTES as PA_METADATA_BYTES,
    SUPPORTED_PLATFORMS as PA_SUPPORTED_PLATFORMS,
    WORKSPACE_BYTES as PA_WORKSPACE_BYTES,
    build_paged_attention_metadata,
    paged_attention_cce,
)
from rms_lm_head import rms_lm_head_fp32  # LM head for the fused multi-layer decode_fwd

KV_CACHE_ROWS_DYN = D.kv_cache_rows

BATCH = M.batch
NUM_HEADS = M.num_heads
NUM_KV_HEADS = M.num_kv_heads
HEAD_DIM = M.head_dim
HIDDEN = M.hidden
INTERMEDIATE = M.intermediate
KV_HIDDEN = M.kv_hidden
VOCAB = M.vocab
REAL_VOCAB = M.real_vocab
NUM_LAYERS = M.num_layers
SAMPLED_IDS_PAD = M.sampled_ids_pad
EPS = M.eps
HIDDEN_INV = M.hidden_inv
HEAD_DIM_INV = M.head_dim_inv
ATTN_SCALE = M.attn_scale
HALF_DIM = M.half_dim
Q_PER_KV = M.q_per_kv
Q_HEAD_BATCH = M.q_head_batch
Q_HEAD_PAD = M.q_head_pad

# ══════════════════════════════════════════════════════════════════════════════
# Functional config — model architecture + workload.
# ══════════════════════════════════════════════════════════════════════════════

# MAX_SEQ is env-overridable for the e2e generate harness: it sizes the standalone
# paged KV pool (CACHE_ROWS) and the RoPE tables, so a 512-token run can use a much
# smaller pool than the 4096 micro-benchmark default (less KV memory).
MAX_SEQ = int(os.environ.get("PTO2_MANUAL_MAX_SEQ", str(M.max_seq)))

# ── Derived shapes — recomputed from the above, don't edit ──

EMBED_HIDDEN_CHUNK = 256
SAMPLE_VOCAB_CHUNK = T.vocab_chunk
SAMPLE_CHUNK_PAD = T.vocab_chunk
SAMPLE_TOPK = 16
SAMPLE_NUM_VOCAB_CHUNKS = VOCAB // SAMPLE_VOCAB_CHUNK
SAMPLE_REAL_NUM_FULL_VOCAB_CHUNKS = REAL_VOCAB // SAMPLE_VOCAB_CHUNK
SAMPLE_REAL_VOCAB_TAIL = REAL_VOCAB % SAMPLE_VOCAB_CHUNK
SAMPLE_REAL_NUM_VOCAB_CHUNKS = SAMPLE_REAL_NUM_FULL_VOCAB_CHUNKS + (1 if SAMPLE_REAL_VOCAB_TAIL != 0 else 0)

assert HIDDEN % EMBED_HIDDEN_CHUNK == 0
assert VOCAB % SAMPLE_VOCAB_CHUNK == 0
assert SAMPLE_NUM_VOCAB_CHUNKS <= SAMPLE_CHUNK_PAD
assert REAL_VOCAB <= VOCAB

# Q_HEAD_PAD keeps the QK-norm reduction column 32-byte aligned. Only the
# Q_HEAD_BATCH real rows are stored into the compact TND query buffer.


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

# ── Scope 2 · direct CANN paged attention ──
SEQ_TILE = T.seq_tile
BLOCK_SIZE = T.block_size
assert SEQ_TILE == BLOCK_SIZE
DECODE_MAX_BLOCKS_PER_SEQ = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
NUM_PAGES = BATCH * DECODE_MAX_BLOCKS_PER_SEQ
CACHE_ROWS = NUM_PAGES * NUM_KV_HEADS * BLOCK_SIZE

ROPE_CORES = 32
ROPE_ITEMS_PER_CORE = (NUM_KV_HEADS * BATCH) // ROPE_CORES
assert (NUM_KV_HEADS * BATCH) % ROPE_CORES == 0

# Q/K/V are SPMD dispatches, so each has one TASK_ID.  Rope depends on
# those scalar ids directly; per-tile copies would create duplicate edges.

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
N_OUT_DIRECT_BLOCKS = 24

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
    seq_lens: pl.Tensor[[BATCH], pl.INT32],
    block_table: pl.Tensor[[D.block_table_flat], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH], pl.INT32],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    pa_metadata: pl.Tensor[[PA_METADATA_BYTES], pl.UINT8],
    pa_workspace: pl.Tensor[[PA_WORKSPACE_BYTES], pl.UINT8],
    pa_tiling_tid: pl.Scalar[pl.TASK_ID],
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
    # x_gamma0 / dcr_xgamma are SPMD producers, so each carry has exactly one
    # task id.  Keep that id in a length-one Array across inline-loop boundaries:
    # Array mutation is preserved by the inline lowering, unlike a re-bound Scalar.
    prev_out_tid: pl.Array[1, pl.TASK_ID],
    prev_normed_tid: pl.Array[1, pl.TASK_ID],
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
    user_batch = BATCH
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
    # suppressed; every cross-task edge below is explicit.  normed_in is
    # produced by the previous dcr_xgamma (or x_gamma0 for layer 0).

    # ── Scope 2 allocations (hoisted before the manual scope). ──
    q_tnd_flat = pl.create_tensor([BATCH * NUM_HEADS, HEAD_DIM], dtype=pl.BF16)
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

    with pl.manual_scope():
        # Unflagged barrier from c61f710: prevent RMS/KV setup from taking
        # the Q projection's early-dispatch window.
        seed_dummy = pl.system.task_dummy(deps=[])
        prev_normed_seed_deps = pl.array.create(2, pl.TASK_ID)
        prev_normed_seed_deps[0] = prev_normed_tid[0]
        prev_normed_seed_deps[1] = seed_dummy

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="attn_out_seed",
            allow_early_resolve=True,
        ) as attn_out_seed_tid:
            for b in pl.range(user_batch, BATCH):
                attn_out = pl.assemble(
                    attn_out,
                    pl.full([1, HIDDEN], dtype=pl.BF16, value=0.0),
                    [b, 0],
                )

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="rms_recip",
            allow_early_resolve=True,
            deps=[prev_normed_seed_deps[i] for i in range(2)],
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

        # ── Scope 1: Q projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed", allow_early_resolve=True) as q_seed_tid:  # no explicit dep: runtime q_proj WAR hazard orders it after prev rope_qkv (now the q_proj reader)
            for snb in pl.pipeline(Q_ON, stage=2):
                q_proj = pl.assemble(
                    q_proj, pl.full([BATCH, QKV_N_TILE], dtype=pl.FP32, value=0.0), [0, snb * QKV_N_TILE]
                )
        # Carry task ids are length-one Arrays; build the explicit dependency
        # array before passing it through deps= so inline lowering retains the
        # prev_normed edge.
        prev_normed_q_deps = pl.array.create(2, pl.TASK_ID)
        prev_normed_q_deps[0] = prev_normed_tid[0]
        prev_normed_q_deps[1] = q_seed_tid
        with pl.spmd(
            Q_ON * QKV_OK,
            name_hint="q_proj",
            allow_early_resolve=True,
            deps=[prev_normed_q_deps[i] for i in range(2)],
        ) as q_proj_tid:
            q_blk = pl.get_block_idx()
            q_nt = q_blk // QKV_OK
            q_ks = q_blk % QKV_OK
            q_n_region = q_nt * QKV_N_TILE
            q_k_base = q_ks * QKV_K_SLICE
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
                        q_acc,
                        normed_in[:, kk : kk + TK],
                        wq[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN],
                    )
                q_proj = pl.assemble(q_proj, q_acc, [0, n0], atomic=pl.AtomicType.Add)

        # ── Scope 1: K projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="kv_seed",
            deps=[prev_normed_seed_deps[i] for i in range(2)],
        ) as kv_seed_tid:
            k_proj = pl.assemble(k_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])
            v_proj = pl.assemble(v_proj, pl.full([BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])

        # Create the MLP/output accumulators and their single seed immediately
        # after kv_seed, matching c61f710's task-generation order.  The
        # unflagged seed_dummy keeps this setup out of Q's early window.
        down_acc_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        gate_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)
        up_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32)
        attn_proj_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="mlp_out_seed",
            allow_early_resolve=True,
            deps=[prev_normed_seed_deps[i] for i in range(2)],
        ) as mlp_out_seed_tid:
            for nb in pl.pipeline(DOWN_ON, stage=2):
                n0 = nb * DOWN_TN
                zero = pl.full([BATCH, DOWN_TN], dtype=pl.FP32, value=0.0)
                down_acc_all = pl.assemble(down_acc_all, zero, [0, n0])
            for nb in pl.pipeline(MLP_ON, stage=2):
                n0 = nb * MLP_TN
                zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                gate_acc_all = pl.assemble(gate_acc_all, zero, [0, n0])
            for nb in pl.pipeline(MLP_ON, stage=2):
                n0 = nb * MLP_TN
                zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
                up_acc_all = pl.assemble(up_acc_all, zero, [0, n0])
            for nb in pl.pipeline(N_SPLITS_OUT, stage=2):
                out_seed_n0 = nb * OUT_TN
                out_zero = pl.full([BATCH, OUT_TN], dtype=pl.FP32, value=0.0)
                attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_zero, [0, out_seed_n0])

        with pl.spmd(
            KV_ON * QKV_OK,
            name_hint="k_proj",
            allow_early_resolve=True,
            deps=[kv_seed_tid],
        ) as k_proj_tid:
            k_blk = pl.get_block_idx()
            k_nt = k_blk // QKV_OK
            k_ks = k_blk % QKV_OK
            k_n_region = k_nt * QKV_N_TILE
            k_k_base = k_ks * QKV_K_SLICE
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
                        k_acc,
                        normed_in[:, kk : kk + TK],
                        wk[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN],
                    )
                k_proj = pl.assemble(k_proj, k_acc, [0, n0], atomic=pl.AtomicType.Add)

        with pl.spmd(
            KV_ON * QKV_OK,
            name_hint="v_proj",
            allow_early_resolve=True,
            deps=[kv_seed_tid],
        ) as v_proj_tid:
            v_blk = pl.get_block_idx()
            v_nt = v_blk // QKV_OK
            v_ks = v_blk % QKV_OK
            v_n_region = v_nt * QKV_N_TILE
            v_k_base = v_ks * QKV_K_SLICE
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
                        v_acc,
                        normed_in[:, kk : kk + TK],
                        wv[layer_hidden_base + kk : layer_hidden_base + kk + TK, n0 : n0 + TN],
                    )
                v_proj = pl.assemble(v_proj, v_acc, [0, n0], atomic=pl.AtomicType.Add)

        # QK-norm and RoPE retain the main implementation's fused arithmetic,
        # but run as a standalone producer for the external CANN attention.
        with pl.spmd(
            ROPE_CORES,
            name_hint="rope_qkv",
            deps=[q_proj_tid, k_proj_tid, v_proj_tid, rms_tid],
        ) as rope_tid:
            rope_core = pl.get_block_idx()
            q_red_pad = pl.full(
                [1, (Q_HEAD_PAD - Q_PER_KV) * HEAD_DIM],
                dtype=pl.FP32,
                value=0.0,
            )
            k_red_pad = pl.full(
                [1, (K_RED_ROWS - 1) * HEAD_DIM],
                dtype=pl.FP32,
                value=0.0,
            )
            for it in pl.pipeline(ROPE_ITEMS_PER_CORE, stage=2):
                g_idx = rope_core + it * ROPE_CORES
                if g_idx < NUM_KV_HEADS * user_batch:
                    ki = g_idx // user_batch
                    b = g_idx - ki * user_batch
                    ctx_len = pl.read(seq_lens, [b])
                    inv_rms_b = pl.read(inv_rms_states, [b, 0])
                    pos = ctx_len - 1
                    wr_slot = pl.cast(pl.tensor.read(slot_mapping, [b]), pl.INDEX)
                    wr_slot_block = wr_slot // BLOCK_SIZE
                    wr_slot_offset = wr_slot - wr_slot_block * BLOCK_SIZE
                    cos_lo = rope_cos[pos : pos + 1, 0:HALF_DIM]
                    cos_hi = rope_cos[pos : pos + 1, HALF_DIM:HEAD_DIM]
                    sin_lo = rope_sin[pos : pos + 1, 0:HALF_DIM]
                    sin_hi = rope_sin[pos : pos + 1, HALF_DIM:HEAD_DIM]

                    kv_col = ki * HEAD_DIM
                    k_raw = pl.mul(
                        pl.reshape(
                            pl.concat(
                                k_proj[b : b + 1, kv_col : kv_col + HEAD_DIM],
                                k_red_pad,
                            ),
                            [K_RED_ROWS, HEAD_DIM],
                        ),
                        inv_rms_b,
                    )
                    k_ss = pl.row_sum(pl.mul(k_raw, k_raw))
                    k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS)))
                    k_normed = pl.row_expand_mul(
                        pl.col_expand_mul(k_raw, k_norm_w),
                        k_inv,
                    )
                    k_full = k_normed[0:1, :]
                    k_lo = k_full[:, 0:HALF_DIM]
                    k_hi = k_full[:, HALF_DIM:HEAD_DIM]
                    rot_lo = pl.sub(
                        pl.col_expand_mul(k_lo, cos_lo),
                        pl.col_expand_mul(k_hi, sin_lo),
                    )
                    rot_hi = pl.add(
                        pl.col_expand_mul(k_hi, cos_hi),
                        pl.col_expand_mul(k_lo, sin_hi),
                    )
                    cache_row = (
                        layer_cache_base
                        + (wr_slot_block * BLOCK_SIZE + wr_slot_offset) * NUM_KV_HEADS
                        + ki
                    )
                    k_cache = pl.assemble(
                        k_cache,
                        pl.cast(pl.concat(rot_lo, rot_hi), target_type=pl.BF16),
                        [cache_row, 0],
                    )
                    v_row_bf16 = pl.cast(
                        pl.mul(
                            v_proj[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM],
                            inv_rms_b,
                        ),
                        target_type=pl.BF16,
                    )
                    v_cache = pl.assemble(v_cache, v_row_bf16, [cache_row, 0])

                    q_base = ki * Q_PER_KV
                    q_raw = pl.mul(
                        pl.reshape(
                            pl.concat(
                                q_proj[
                                    b : b + 1,
                                    q_base * HEAD_DIM : (q_base + Q_PER_KV) * HEAD_DIM,
                                ],
                                q_red_pad,
                            ),
                            [Q_HEAD_PAD, HEAD_DIM],
                        ),
                        inv_rms_b,
                    )
                    q_ss = pl.row_sum(pl.mul(q_raw, q_raw))
                    q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
                    q_heads = pl.row_expand_mul(
                        pl.col_expand_mul(q_raw, q_norm_w),
                        q_inv,
                    )
                    q_lo = q_heads[:, 0:HALF_DIM]
                    q_hi = q_heads[:, HALF_DIM:HEAD_DIM]
                    q_rot_lo = pl.sub(
                        pl.col_expand_mul(q_lo, cos_lo),
                        pl.col_expand_mul(q_hi, sin_lo),
                    )
                    q_rot_hi = pl.add(
                        pl.col_expand_mul(q_hi, cos_hi),
                        pl.col_expand_mul(q_lo, sin_hi),
                    )
                    q_row = b * NUM_HEADS + q_base
                    q_tnd_flat = pl.assemble(
                        q_tnd_flat,
                        pl.cast(
                            pl.concat(q_rot_lo, q_rot_hi)[0:Q_PER_KV, :],
                            target_type=pl.BF16,
                        ),
                        [q_row, 0],
                    )

        q_tnd = pl.reshape(q_tnd_flat, [BATCH, NUM_HEADS, HEAD_DIM])
        attn_out_tnd = pl.reshape(attn_out, [BATCH, NUM_HEADS, HEAD_DIM])
        attention_core_num = PA_DEFAULT_BLOCK_DIM
        with pl.spmd(
            attention_core_num,
            name_hint="fa_fused",
            allow_early_resolve=True,
            sync_start=True,
            deps=[
                rope_tid,
                pa_tiling_tid,
                attn_out_seed_tid,
                mlp_out_seed_tid,
            ],
        ) as attn_done_tid:
            attn_out_tnd = paged_attention_cce(
                q_tnd,
                k_cache,
                v_cache,
                block_table,
                attn_out_tnd,
                pa_workspace,
                pa_metadata,
                layer_cache_base,
            )
        attn_out = pl.reshape(attn_out_tnd, [BATCH, HIDDEN])
        # Scope-3 allocations. (down_acc_all / gate_acc_all / up_acc_all / attn_proj_fp32
        # are created earlier, alongside their hoisted seed tasks between rope and attn.)
        post_norm_partial = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)  # raw residual h1 (add-back); FP32 (was BF16)
        mlp_norm_in = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # h1 * post_gamma (gate/up input)
        inv_rms_tile = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
        mlp_tile = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)

        # ── Scope 3b: manual_scope MLP block. ──
        silu_tids = pl.array.create(MLP_ON, pl.TASK_ID)
        # down_tids is hoisted to orchestration scope (declared before this
        # manual_scope) so the post-scope consolidated writer can gate on it;
        # it is FILLED here in the down_proj loop below.
        gate_tids = pl.array.create(MLP_ON * K_SPLITS_MLP, pl.TASK_ID)
        up_tids = pl.array.create(MLP_ON * K_SPLITS_MLP, pl.TASK_ID)
        cast_tids = pl.array.create(K_SPLITS_MLP, pl.TASK_ID)
        out_tids = pl.array.create(N_SPLITS_OUT * K_SPLITS_OUT, pl.TASK_ID)

        # 14e2635 critical-wave split: defer the first 26 tiles through an
        # unflagged dummy, while the final 24 tiles are one SPMD dispatch
        # directly gated by FAI.  The direct dispatch intentionally has no
        # allow_early_resolve flag.
        out_proj_dummy = pl.system.task_dummy(deps=[attn_done_tid])
        N_OUT_DIRECT = N_SPLITS_OUT * K_SPLITS_OUT - N_OUT_DIRECT_BLOCKS
        for out_idx in pl.parallel(0, N_OUT_DIRECT):
            n_out_proj = out_idx // K_SPLITS_OUT
            k_split_out = out_idx % K_SPLITS_OUT
            n_op = n_out_proj * OUT_TN
            k_op = k_split_out * OUT_TK
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="out_proj",
                deps=[out_proj_dummy],
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
                attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_c_acc, [0, n_op], atomic=pl.AtomicType.Add)
            out_tids[out_idx] = out_tid

        with pl.spmd(
            N_OUT_DIRECT_BLOCKS,
            name_hint="out_proj",
            deps=[attn_done_tid],
        ) as out_proj_direct_tid:
            out_idx = N_OUT_DIRECT + pl.get_block_idx()
            n_out_proj = out_idx // K_SPLITS_OUT
            k_split_out = out_idx % K_SPLITS_OUT
            n_op = n_out_proj * OUT_TN
            k_op = k_split_out * OUT_TK
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
            attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_c_acc, [0, n_op], atomic=pl.AtomicType.Add)
        for _block in pl.unroll(N_OUT_DIRECT_BLOCKS):
            out_tids[N_OUT_DIRECT + _block] = out_proj_direct_tid

        # Tiled residual + BF16 cast.
        for k_slice in pl.unroll(K_SPLITS_MLP):
            k_base = k_slice * MLP_K_SLICE
            n_split_base = k_slice * N_PER_CAST_K
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="residual_rms_cast",
                allow_early_resolve=True,
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
                    deps=[cast_tids[k_split]],  # gate_seed funneled via fa_fused (cast_tid -> out_proj -> fa_fused)
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
                    deps=[cast_tids[k_split]],  # up_seed funneled via fa_fused (cast_tid -> out_proj -> fa_fused)
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
                    allow_early_resolve=True,
                    deps=[silu_tids[k_split]],  # down_seed funneled via fa_fused (silu -> ... -> out_proj -> fa_fused)
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
        allow_early_resolve=True,
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
    # Mutate the length-one carry arrays in place.  dcr_xgamma produces both
    # `out` and `normed_out`, so its one SPMD dispatch id is the carry for both.
    prev_out_tid[0] = dcr_tid
    prev_normed_tid[0] = dcr_tid
    return out


@pl.jit.inline
def _token_embed_inline(
    sampled_ids: pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    user_batch = BATCH
    for b in pl.parallel(user_batch):
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
    user_batch = BATCH
    for b in pl.parallel(user_batch):
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
            winning_logits = pl.slice(
                logits,
                [1, SAMPLE_VOCAB_CHUNK],
                [pl.cast(b, pl.INDEX), chunk_base_idx],
            )
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
_FWD_NLAYERS = NUM_LAYERS  # decode_fwd loop bound; overridable for layer-count tests


@pl.jit
def decode_fwd(  # noqa: PLR0913 — device-side fused NUM_LAYERS decode + LM head
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor[[BATCH], pl.INT32],
    block_table: pl.Tensor[[D.block_table_flat], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH], pl.INT32],
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
    out: pl.Out[pl.Tensor[[BATCH, VOCAB], pl.FP32]],
    embed_weight: pl.Tensor,
    sampled_ids_in: pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32],
    sampled_ids_out: pl.Out[pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32]],
    next_hidden: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
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
    block_table.bind_dynamic(0, D.block_table_flat)
    k_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    user_batch = BATCH

    pa_metadata = pl.create_tensor([PA_METADATA_BYTES], dtype=pl.UINT8)
    pa_workspace = pl.create_tensor([PA_WORKSPACE_BYTES], dtype=pl.UINT8)
    pa_num_layers = pl.tensor.dim(input_rms_weight, 0)
    pa_num_pages = pl.tensor.dim(k_cache, 0) // (pa_num_layers * BLOCK_SIZE * NUM_KV_HEADS)
    pa_max_blocks = pl.tensor.dim(block_table, 0) // user_batch
    pa_num_pages_i32 = pl.cast(pa_num_pages, pl.INT32)
    pa_max_blocks_i32 = pl.cast(pa_max_blocks, pl.INT32)
    pa_tiling_tid = build_paged_attention_metadata(
        seq_lens,
        pa_max_blocks_i32,
        pa_num_pages_i32,
        pa_metadata,
    )
    next_hidden = _token_embed_inline(sampled_ids_in, embed_weight, next_hidden)

    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)  # FP32 inter-layer carry (was BF16)
    prev_out_tid = pl.array.create(1, pl.TASK_ID)
    prev_out_tid[0] = pl.system.task_dummy(deps=[])
    for cb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden", allow_early_resolve=True) as ch_tid:
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                # FIRST-layer boundary: cast the external BF16 embed input -> FP32 once,
                # so every layer consumes FP32 hidden with no per-boundary round-trip.
                cur = pl.assemble(
                    cur,
                    pl.cast(
                        pl.fillpad(
                            pl.slice(
                                next_hidden,
                                [BATCH, RMSNORM_K_CHUNK],
                                [cb0, ck0],
                                valid_shape=[user_batch, RMSNORM_K_CHUNK],
                            ),
                            pad_value=pl.PadValue.zero,
                        ),
                        target_type=pl.FP32,
                    ),
                    [cb0, ck0],
                )
        prev_out_tid[0] = ch_tid

    # ── Pre-loop x_gamma_0: layer 0's normed = cur_0 * gamma_0 (BF16). For layers 1+,
    # the per-layer normed is produced by the PREVIOUS layer's fused dcr_xgamma; only
    # layer 0 (whose cur comes from copy_hidden, not a dcr) needs this standalone task.
    normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    prev_normed_tid = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(
            XG_BLOCKS,
            name_hint="x_gamma0",
            allow_early_resolve=True,
            deps=[prev_out_tid[0]],
        ) as xgamma_tid:
            xg_k0 = pl.tile.get_block_idx() * (HIDDEN // XG_BLOCKS)
            for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK // XG_BLOCKS, stage=2):
                k0 = xg_k0 + kb * RMSNORM_K_CHUNK
                x_chunk = cur[:, k0 : k0 + RMSNORM_K_CHUNK]
                gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [0, k0])
                xg = pl.col_expand_mul(x_chunk, gamma)
                normed = pl.assemble(normed, pl.cast(xg, target_type=pl.BF16), [0, k0])
        prev_normed_tid[0] = xgamma_tid

    for layer_idx in pl.range(_FWD_NLAYERS):
        layer_next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)  # FP32 layer output
        next_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # next layer's x*gamma
        next_gamma_idx = pl.min(layer_idx + 1, _FWD_NLAYERS - 1)  # clamp: last layer's normed unused
        cur = _decode_layer(
            cur, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache,
            pa_metadata, pa_workspace, pa_tiling_tid, wo, w_gate, w_up, w_down,
            post_rms_weight, layer_next_hidden, normed, next_normed, layer_idx, next_gamma_idx,
            prev_out_tid, prev_normed_tid,
        )
        normed = next_normed
    out = rms_lm_head_fp32(cur, final_norm_weight, lm_head_weight, seq_lens, out)
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
    # Fused B16 decode of _CHUNK_NLAYERS consecutive layers, output = hidden
    # (NO LM head). Dynamic public batching is provided by decode_fwd.
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
    user_batch = BATCH
    pa_metadata = pl.create_tensor([PA_METADATA_BYTES], dtype=pl.UINT8)
    pa_workspace = pl.create_tensor([PA_WORKSPACE_BYTES], dtype=pl.UINT8)
    pa_num_layers = pl.tensor.dim(input_rms_weight, 0)
    pa_num_pages = pl.tensor.dim(k_cache, 0) // (pa_num_layers * BLOCK_SIZE * NUM_KV_HEADS)
    pa_max_blocks = pl.tensor.dim(block_table, 0) // user_batch
    pa_num_pages_i32 = pl.cast(pa_num_pages, pl.INT32)
    pa_max_blocks_i32 = pl.cast(pa_max_blocks, pl.INT32)
    pa_tiling_tid = build_paged_attention_metadata(
        seq_lens,
        pa_max_blocks_i32,
        pa_num_pages_i32,
        pa_metadata,
    )

    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    prev_out_tid = pl.array.create(1, pl.TASK_ID)
    prev_out_tid[0] = pl.system.task_dummy(deps=[])
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
        prev_out_tid[0] = ch_tid

    # Pre-loop x_gamma_0: chunk layer 0's normed = cur * gamma_0 (mirrors decode_fwd).
    normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    prev_normed_tid = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(
            XG_BLOCKS,
            name_hint="x_gamma0",
            allow_early_resolve=True,
            deps=[prev_out_tid[0]],
        ) as xgamma_tid:
            xg_k0 = pl.tile.get_block_idx() * (HIDDEN // XG_BLOCKS)
            for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK // XG_BLOCKS, stage=2):
                k0 = xg_k0 + kb * RMSNORM_K_CHUNK
                x_chunk = cur[:, k0 : k0 + RMSNORM_K_CHUNK]
                gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [0, k0])
                xg = pl.col_expand_mul(x_chunk, gamma)
                normed = pl.assemble(normed, pl.cast(xg, target_type=pl.BF16), [0, k0])
        prev_normed_tid[0] = xgamma_tid

    for i in pl.range(_CHUNK_NLAYERS):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
        next_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        next_gamma_idx = pl.min(i + 1, _CHUNK_NLAYERS - 1)
        cur = _decode_layer(
            cur, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache,
            pa_metadata, pa_workspace, pa_tiling_tid, wo, w_gate, w_up, w_down,
            post_rms_weight, next_hidden, normed, next_normed, i, next_gamma_idx,
            prev_out_tid, prev_normed_tid,
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
    physical pages [b*DECODE_MAX_BLOCKS_PER_SEQ .. ]; slot_mapping[b] is the current
    token's (pos=seq_len-1) physical row. Mirrors the serving runner's layout."""
    block_table = torch.arange(BATCH * DECODE_MAX_BLOCKS_PER_SEQ, dtype=torch.int32)
    slot_mapping = torch.empty(BATCH, dtype=torch.int32)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        logical_block = pos // BLOCK_SIZE
        phys_page = b * DECODE_MAX_BLOCKS_PER_SEQ + logical_block
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
    if platform not in PA_SUPPORTED_PLATFORMS:
        raise ValueError(f"direct CANN decode attention does not support platform {platform!r}")
    return BackendType.Ascend910B


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

    # The flat cache buffers represent BSND bytes: [page, token, kv_head, dim].
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
    k_cache = values["k_cache"].view(NUM_PAGES * BLOCK_SIZE, KV_HIDDEN).float()
    v_cache = values["v_cache"].view(NUM_PAGES * BLOCK_SIZE, KV_HIDDEN).float()

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
            # Each physical page is [BLOCK_SIZE, KV_HIDDEN] in BSND order.
            k_lane = torch.empty(slen, HEAD_DIM)
            v_lane = torch.empty(slen, HEAD_DIM)
            for sb in range(n_blocks):
                pbid = int(block_table[b * DECODE_MAX_BLOCKS_PER_SEQ + sb].item())
                row = pbid * BLOCK_SIZE
                col = kvh * HEAD_DIM
                lo = sb * BLOCK_SIZE
                blk = min(BLOCK_SIZE, slen - lo)
                k_lane[lo : lo + blk] = k_cache[row : row + blk, col : col + HEAD_DIM]
                v_lane[lo : lo + blk] = v_cache[row : row + blk, col : col + HEAD_DIM]
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
        choices=PA_SUPPORTED_PLATFORMS,
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
    parser.add_argument("--no-dep-gen", action="store_true", default=False,
                        help="deprecated no-op: dep_gen is already forced off for the direct "
                             "CCE attention cohort; kept for CLI / CI back-compat.")
    parser.add_argument("--validate-fwd", action="store_true", default=False,
                        help="validate the fused decode_fwd (N stacked layers + on-device LM head "
                             "-> logits) against a host chain reference, instead of the default "
                             "single-layer golden test.")
    parser.add_argument("--fwd-layers", type=int, default=4, help="layer count N for --validate-fwd")
    parser.add_argument("--decode-steps", type=int, default=1,
                        help="under --validate-fwd, run this many decode steps for a device-cost "
                             "timing sweep at growing context: each step feeds the previous step's "
                             "sampled token back as input and grows the context by one (seq_lens += 1, "
                             "slot_mapping advanced to the new token's KV slot). The context starts at "
                             "MAX_SEQ (PTO2_MANUAL_MAX_SEQ) and grows to MAX_SEQ + decode_steps - 1; the "
                             "paged KV pool (block_table / slot_mapping / kc / vc) is enlarged by "
                             "decode_steps tokens up front so the growing seq_lens never overflow it. "
                             "Read the device `Effective` span (kc/vc are re-uploaded per step, so it "
                             "measures decode cost at growing context, not a correct generation). The "
                             "host-ref argmax check is skipped for N > 1.")
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
    # `python decode_fwd.py -p a2a3sim` sweep — codegen regressions are still
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
        print(f"[decode_fwd] single-layer golden unit test | platform={args.platform} "
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
                # The direct CCE attention launch requires an uninstrumented full
                # mixed-core cohort. --no-dep-gen remains an accepted CLI no-op.
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
        _n_steps = args.decode_steps
        if _n_steps < 1:
            parser.error(f"--decode-steps must be >= 1, got {_n_steps}")
        # Multi-step decode-timing sweep: grow the context from MAX_SEQ to
        # MAX_SEQ + decode_steps - 1, one token per step. The paged KV pool
        # (block_table / slot_mapping and the kc/vc pools, all runtime-dynamic in
        # the kernel) is enlarged by decode_steps tokens up front so the growing
        # seq_lens never overflow it, regardless of MAX_SEQ.
        #
        # This measures the device decode *cost* at growing context — read the
        # device `Effective` span, not host wall-clock. kc/vc are plain host
        # inputs (not InOut), so each dispatch re-uploads them and the on-device
        # KV writes don't persist across steps; that is fine here because the
        # per-step read cost tracks seq_lens exactly, and the argmax correctness
        # check is skipped for N > 1. It is NOT a correct autoregressive
        # generation (the KV content per step is the re-uploaded prefill KV).
        if _n_steps > 1:
            _grow_blocks = (MAX_SEQ + _n_steps + BLOCK_SIZE - 1) // BLOCK_SIZE
            _grow_pages = BATCH * _grow_blocks
            _grow_rows = _grow_pages * NUM_KV_HEADS * BLOCK_SIZE
            bt = torch.arange(_grow_pages, dtype=torch.int32)
            _pad_rows = _grow_rows - kc.shape[0]
            if _pad_rows > 0:
                kc = torch.cat(
                    [kc, torch.empty([_pad_rows, HEAD_DIM], dtype=kc.dtype).normal_() * 0.01],
                    dim=0,
                ).contiguous()
                vc = torch.cat(
                    [vc, torch.empty([_pad_rows, HEAD_DIM], dtype=vc.dtype).normal_() * 0.02],
                    dim=0,
                ).contiguous()

            def _grow_slot_mapping(seq_lens: torch.Tensor) -> torch.Tensor:
                """slot_mapping over the enlarged pool (_grow_blocks pages per seq)."""
                sm_ = torch.empty(BATCH, dtype=torch.int32)
                for _b in range(BATCH):
                    pos = int(seq_lens[_b].item()) - 1
                    sm_[_b] = (_b * _grow_blocks + pos // BLOCK_SIZE) * BLOCK_SIZE + (pos % BLOCK_SIZE)
                return sm_

            sl = torch.full([BATCH], MAX_SEQ, dtype=torch.int32)
            sm = _grow_slot_mapping(sl)
            print(f"[stacked-fwd {N}L+LMhead] autoregressive decode: seq_lens {MAX_SEQ} -> "
                  f"{MAX_SEQ + _n_steps - 1} over {_n_steps} steps "
                  f"(KV pool enlarged to {_grow_blocks} pages/seq for {MAX_SEQ + _n_steps} tokens)")
        stacked = [
            stack0(irw, N), stack0(wq_, N), stack0(wk_, N), stack0(wv_, N),
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
        for _step in range(_n_steps):
            decode_fwd(
                *stacked,
                logits,
                embed_weight,
                sampled_ids_in,
                sampled_ids_out,
                next_hidden,
                config=run_cfg,
            )
            if _step + 1 < _n_steps:
                # Feed the sampled token back and grow the context by one token:
                # seq_lens += 1 and re-point slot_mapping to the new token's KV slot
                # in the enlarged pool.
                sampled_ids_in.copy_(sampled_ids_out)
                sl.add_(1)
                sm.copy_(_grow_slot_mapping(sl))
        if _n_steps > 1:
            print(f"[stacked-fwd {N}L+LMhead] {_n_steps}-step autoregressive decode complete "
                  f"(host-ref argmax check skipped for --decode-steps > 1)")
            raise SystemExit(0)
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
        decode_fwd_layers(hs, *stacked[:len(INPUT_NAMES) - 1], ref_out, config=run_cfg)
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
