# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim    # Runtime simulation is disabled; a2a3sim compile-only smoke is supported.
"""Qwen3-14B decode with FP32 inter-layer carry and native PyPTO paged attention.

The projection, output projection, MLP, and dependency topology follow the main
implementation. The attention stage uses native PyPTO Phase 0 plus paged
attention. Its public ABI matches vLLM: Q/O are active TND and the flat paged
K/V buffers contain BSND bytes ordered as
``[page, token, kv_head, dim]``.

``decode_fwd`` accepts ANY public batch >= 1 while keeping the model pipeline
internally padded to 16 rows: a batch above that width runs as
ceil(batch / BATCH_PAD) consecutive row windows, each executing the full layer
stack and the LM head for its rows, with the token embedding and the sampling
shared across the whole batch. Windows are serialized while reusing the native
paged-attention scratch buffers, and weights are re-read per window, so throughput is flat
past 16 rows -- correctness scales, cost does not amortize. The inter-layer
residual remains FP32; BF16 conversion occurs only at the external chunk
boundaries and model-defined compute boundaries.
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
from paged_attention_pypto import (
    FFTS_WORKSPACE_ELEMENTS as PA_FFTS_WORKSPACE_ELEMENTS,
    STACK_TOKENS as PA_STACK_TOKENS,
    TRANSFER_ROWS as PA_TRANSFER_ROWS,
    paged_attention_pypto_swpipe,
)
from rms_lm_head import rms_lm_head_fp32  # LM head for the fused multi-layer decode_fwd

PA_SUPPORTED_PLATFORMS = ("a2a3", "a2a3sim")

KV_CACHE_ROWS_DYN = D.kv_cache_rows

BATCH_PAD = M.batch_pad  # padded pipeline width (M of every matmul)
BATCH_DYN = D.batch  # public batch; any batch >= 1 is processed in BATCH_PAD windows
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
SAMPLE_REDUCE_ROWS = 8  # Align FP32 column reductions to the A2/A3 32-byte requirement.
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
TM = 16  # M tile = BATCH_PAD (OM = 1; M is not split)
TN = 256  # inner N sub-tile
TK = 256  # inner K chunk
QKV_N_TILE = 512  # outer N-tile width (one ON unit) = N_SUB inner TN subtiles
N_SUB = QKV_N_TILE // TN  # 2 inner N-subtiles per outer N-tile
Q_ON = HIDDEN // QKV_N_TILE  # 10 outer N-tiles (Q)
KV_ON = KV_HIDDEN // QKV_N_TILE  # 2 outer N-tiles (K, V)
QKV_OK = 5  # split-K slices (atomic-add)  # 5 -> QKV_K_SLICE=1024 = normed slab (1:1 partial-Q)
QKV_K_SLICE = HIDDEN // QKV_OK  # 1280 K per split
QKV_K_CHUNKS = QKV_K_SLICE // TK  # 5 inner TK chunks per split

# ── Scope 2 · native PyPTO paged attention ──
SEQ_TILE = T.seq_tile
BLOCK_SIZE = T.block_size
assert SEQ_TILE == BLOCK_SIZE
DECODE_MAX_BLOCKS_PER_SEQ = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
NUM_PAGES = BATCH_PAD * DECODE_MAX_BLOCKS_PER_SEQ
CACHE_ROWS = NUM_PAGES * NUM_KV_HEADS * BLOCK_SIZE

ROPE_CORES = 32
ROPE_ITEMS_PER_CORE = (NUM_KV_HEADS * BATCH_PAD) // ROPE_CORES
assert (NUM_KV_HEADS * BATCH_PAD) % ROPE_CORES == 0

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
GATE_UP_SPMD_N = 6  # critical-wave: n_out 0..5 → pl.spmd; n≥6 deferred via dummy

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
assert TM == BATCH_PAD and N_SUB == 2 and QKV_K_CHUNKS == 4
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
assert GATE_UP_SPMD_N < MLP_ON


@pl.jit.inline(auto_scope=False)
def _run_paged_attention(  # noqa: PLR0913 -- paged-attention adapter ABI
    q_tnd_flat: pl.Tensor,
    attn_out: pl.Tensor,
    key_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    value_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor,
    seq_lens: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    q_proj: pl.Tensor,
    k_proj: pl.Tensor,
    v_proj: pl.Tensor,
    q_norm_w: pl.Tensor,
    k_norm_w: pl.Tensor,
    inv_rms_states: pl.Tensor,
    layer_cache_base_token_rows: pl.Scalar[pl.INDEX],
    score_transfer: pl.Tensor[[PA_TRANSFER_ROWS, PA_STACK_TOKENS], pl.FP32],
    probability_transfer: pl.Tensor[[PA_TRANSFER_ROWS, PA_STACK_TOKENS], pl.BF16],
    pv_transfer: pl.Tensor[[PA_TRANSFER_ROWS, HEAD_DIM], pl.FP32],
    ffts_workspace: pl.Tensor[[PA_FFTS_WORKSPACE_ELEMENTS], pl.INT64],
    q_proj_tid: pl.Scalar[pl.TASK_ID],
    k_proj_tid: pl.Scalar[pl.TASK_ID],
    v_proj_tid: pl.Scalar[pl.TASK_ID],
    rms_tid: pl.Scalar[pl.TASK_ID],
    attn_out_seed_tid: pl.Scalar[pl.TASK_ID],
    mlp_out_seed_tid: pl.Scalar[pl.TASK_ID],
    scratch_ready_tid: pl.Scalar[pl.TASK_ID],
):
    """Run fused native Q/K norm, RoPE, cache append, and paged attention."""
    attn_out_tnd = pl.reshape(attn_out, [BATCH_PAD, NUM_HEADS, HEAD_DIM])
    attn_done_tid = paged_attention_pypto_swpipe(
        q_tnd_flat,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        inv_rms_states,
        slot_mapping,
        rope_cos,
        rope_sin,
        q_proj,
        k_proj,
        v_proj,
        q_norm_w,
        k_norm_w,
        layer_cache_base_token_rows,
        attn_out_tnd,
        score_transfer,
        probability_transfer,
        pv_transfer,
        ffts_workspace,
        q_proj_tid,
        k_proj_tid,
        v_proj_tid,
        rms_tid,
        attn_out_seed_tid,
        mlp_out_seed_tid,
        scratch_ready_tid,
    )
    return attn_out, attn_done_tid


# ──────────────────────────────────────────────────────────────────────────────
# Monolithic JIT entry.
# ──────────────────────────────────────────────────────────────────────────────


@pl.jit.inline
def _decode_layer(  # noqa: PLR0913 — model signature is intrinsic
    hidden_states: pl.Tensor[[BATCH_PAD, HIDDEN], pl.FP32],  # FP32: inter-layer carry (was BF16)
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[D.block_table_flat], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    score_transfer: pl.Tensor[[PA_TRANSFER_ROWS, PA_STACK_TOKENS], pl.FP32],
    probability_transfer: pl.Tensor[[PA_TRANSFER_ROWS, PA_STACK_TOKENS], pl.BF16],
    pv_transfer: pl.Tensor[[PA_TRANSFER_ROWS, HEAD_DIM], pl.FP32],
    ffts_workspace: pl.Tensor[[PA_FFTS_WORKSPACE_ELEMENTS], pl.INT64],
    scratch_ready: pl.Array[1, pl.TASK_ID],
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Tensor[[BATCH_PAD, HIDDEN], pl.FP32],  # FP32: inter-layer carry (was BF16)
    # normed_in: THIS layer's x*gamma (BF16), produced by the previous layer's fused
    # dcr_xgamma (x_gamma_0 for layer 0). Consumed by QKV — replaces the local x_gamma.
    normed_in: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    # normed_out: NEXT layer's x*gamma, written by THIS layer's dcr_xgamma. Escapes via
    # the inline alias (verified: inline out-param tensor writes are visible to caller).
    normed_out: pl.Tensor[[BATCH_PAD, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
    next_gamma_idx: pl.Scalar[pl.INT32],  # clamped min(layer_idx+1, N-1) for dcr_xgamma's gamma
    # x_gamma0 / dcr_xgamma are SPMD producers, so each carry has exactly one
    # task id.  Keep that id in a length-one Array across inline-loop boundaries:
    # Array mutation is preserved by the inline lowering, unlike a re-bound Scalar.
    prev_out_tid: pl.Array[1, pl.TASK_ID],
    prev_normed_tid: pl.Array[1, pl.TASK_ID],
    # Public batch WINDOW this call serves. The row-indexed tensors above
    # (hidden_states, normed_in/out, out) already hold the window's rows;
    # seq_lens / slot_mapping / block_table are whole-batch; native PA receives
    # runtime-shaped views of the rows starting at batch_offset.
    batch_offset: pl.Scalar[pl.INDEX],
    batch_count: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH_PAD, HIDDEN], pl.FP32]:
    # Per-layer offsets into the STACKED weights / PAGED KV cache. decode_fwd passes
    # the running loop index 0.._FWD_NLAYERS-1; decode_fwd_layers passes
    # 0.._CHUNK_NLAYERS-1 (per-chunk weight slices).
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    # Paged KV: the public ABI is flat [head_rows, HEAD_DIM]. Native PyPTO packs
    # all KV heads into one BSND token row through this zero-copy view.
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    cache_rows = pl.tensor.dim(k_cache, 0) // NUM_KV_HEADS
    layer_cache_rows = cache_rows // num_layers_actual
    layer_cache_base_token_rows = layer_idx * layer_cache_rows
    batch = batch_count  # live rows in THIS window; pipeline rows above it are masked
    batch_rows = pl.cast(batch_count, pl.INDEX)

    # Native PyPTO PA derives its active batch and block-table row stride from
    # the descriptors it receives. Give it a dynamic view of this public-batch
    # window while preserving the absolute physical page / slot ids stored in
    # the underlying tensors.
    public_batch = pl.tensor.dim(seq_lens, 0)
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // public_batch
    window_seq_lens = pl.slice(
        seq_lens,
        [batch_rows],
        [batch_offset],
    )
    window_slot_mapping = pl.slice(
        slot_mapping,
        [batch_rows],
        [batch_offset],
    )
    window_block_table = pl.slice(
        block_table,
        [batch_rows * max_blocks_per_seq],
        [batch_offset * max_blocks_per_seq],
    )
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
    inv_rms_states = pl.create_tensor([BATCH_PAD, 1], dtype=pl.FP32)  # deferred 1/rms denominator
    q_proj = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH_PAD, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH_PAD, KV_HIDDEN], dtype=pl.FP32)

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
    q_tnd_flat = pl.create_tensor([BATCH_PAD * NUM_HEADS, HEAD_DIM], dtype=pl.BF16)
    attn_out = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)

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
            # Static trip count + guard, not pl.range(batch, BATCH_PAD): a
            # dynamic LOWER bound here would also make this a dynamic-offset GM
            # store inside an allow_early_resolve pl.at. Mirrors prefill_fwd.
            for b in pl.range(BATCH_PAD):
                if b >= batch:
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
            partial_sq = pl.full([1, BATCH_PAD], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):
                k0 = kb * RMSNORM_K_CHUNK
                x_chunk = hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK]  # FP32 already (was cast from BF16)
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_PAD]),
                )
            variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH_PAD, 1])
            inv_rms = pl.recip(pl.sqrt(variance))
            inv_rms_states = pl.assemble(inv_rms_states, inv_rms, [0, 0])

        # ── Scope 1: Q projection — SPLIT-K + inner N/K tiling, SPMD (seed + atomic). ──
        with (
            pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed", allow_early_resolve=True) as q_seed_tid
        ):  # no explicit dep: runtime q_proj WAR hazard orders it after the previous fused PA reader
            for snb in pl.pipeline(Q_ON, stage=2):
                q_proj = pl.assemble(
                    q_proj, pl.full([BATCH_PAD, QKV_N_TILE], dtype=pl.FP32, value=0.0), [0, snb * QKV_N_TILE]
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
            k_proj = pl.assemble(k_proj, pl.full([BATCH_PAD, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])
            v_proj = pl.assemble(v_proj, pl.full([BATCH_PAD, KV_HIDDEN], dtype=pl.FP32, value=0.0), [0, 0])

        # Create the MLP/output accumulators and their single seed immediately
        # after kv_seed, matching c61f710's task-generation order.  The
        # unflagged seed_dummy keeps this setup out of Q's early window.
        down_acc_all = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)
        gate_acc_all = pl.create_tensor([BATCH_PAD, INTERMEDIATE], dtype=pl.FP32)
        up_acc_all = pl.create_tensor([BATCH_PAD, INTERMEDIATE], dtype=pl.FP32)
        attn_proj_fp32 = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="mlp_out_seed",
            allow_early_resolve=True,
            deps=[prev_normed_seed_deps[i] for i in range(2)],
        ) as mlp_out_seed_tid:
            for nb in pl.pipeline(DOWN_ON, stage=2):
                n0 = nb * DOWN_TN
                zero = pl.full([BATCH_PAD, DOWN_TN], dtype=pl.FP32, value=0.0)
                down_acc_all = pl.assemble(down_acc_all, zero, [0, n0])
            for nb in pl.pipeline(MLP_ON, stage=2):
                n0 = nb * MLP_TN
                zero = pl.full([BATCH_PAD, MLP_TN], dtype=pl.FP32, value=0.0)
                gate_acc_all = pl.assemble(gate_acc_all, zero, [0, n0])
            for nb in pl.pipeline(MLP_ON, stage=2):
                n0 = nb * MLP_TN
                zero = pl.full([BATCH_PAD, MLP_TN], dtype=pl.FP32, value=0.0)
                up_acc_all = pl.assemble(up_acc_all, zero, [0, n0])
            for nb in pl.pipeline(N_SPLITS_OUT, stage=2):
                out_seed_n0 = nb * OUT_TN
                out_zero = pl.full([BATCH_PAD, OUT_TN], dtype=pl.FP32, value=0.0)
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

        # PyPTO emits one fused Q/K/V/RMS -> Phase-0 -> syncall -> PA task over
        # the shared BSND cache root. The row-window views keep its local batch
        # indexing 0-based while their page and slot values remain globally addressed.
        # Materialize the loop-carried array element as a named scalar so the
        # nested JIT dependency binder does not leave a free subscript value.
        scratch_ready_tid = scratch_ready[0]
        attn_out, attn_done_tid = _run_paged_attention(
            q_tnd_flat,
            attn_out,
            k_cache,
            v_cache,
            window_block_table,
            window_seq_lens,
            window_slot_mapping,
            rope_cos,
            rope_sin,
            q_proj,
            k_proj,
            v_proj,
            q_norm_w,
            k_norm_w,
            inv_rms_states,
            layer_cache_base_token_rows,
            score_transfer,
            probability_transfer,
            pv_transfer,
            ffts_workspace,
            q_proj_tid,
            k_proj_tid,
            v_proj_tid,
            rms_tid,
            attn_out_seed_tid,
            mlp_out_seed_tid,
            scratch_ready_tid,
        )
        # Explicit caller-owned carry serializes reuse of the single scratch set.
        scratch_ready[0] = attn_done_tid
        # Scope-3 allocations. (down_acc_all / gate_acc_all / up_acc_all / attn_proj_fp32
        # are created earlier, alongside their hoisted seed tasks between rope and attn.)
        post_norm_partial = pl.create_tensor(
            [BATCH_PAD, HIDDEN], dtype=pl.FP32
        )  # raw residual h1 (add-back); FP32 (was BF16)
        mlp_norm_in = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)  # h1 * post_gamma (gate/up input)
        inv_rms_tile = pl.create_tensor([BATCH_PAD, 1], dtype=pl.FP32)
        mlp_tile = pl.create_tensor([BATCH_PAD, INTERMEDIATE], dtype=pl.BF16)

        # ── Scope 3b: manual_scope MLP block. ──
        silu_tids = pl.array.create(MLP_ON, pl.TASK_ID)
        # down_tids is hoisted to orchestration scope (declared before this
        # manual_scope) so the post-scope consolidated writer can gate on it;
        # it is FILLED here in the down_proj loop below.
        gate_tids = pl.array.create(MLP_ON * K_SPLITS_MLP, pl.TASK_ID)
        up_tids = pl.array.create(MLP_ON * K_SPLITS_MLP, pl.TASK_ID)
        cast_tids = pl.array.create(K_SPLITS_MLP, pl.TASK_ID)
        # Per-k task_dummy funnels for the deferred (non-critical) gate/up tiles.
        # Filled in the k_split-keyed critical-wave loop, consumed by the
        # [n_out outer, k_split inner] deferred-tile loop (symbolic pl.range
        # indexing requires a pl.array, not a Python list).
        gate_late_tids = pl.array.create(K_SPLITS_MLP, pl.TASK_ID)
        up_late_tids = pl.array.create(K_SPLITS_MLP, pl.TASK_ID)
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
                out_w0 = wo[
                    layer_hidden_base + k_op : layer_hidden_base + k_op + OUT_INNER_TK, n_op : n_op + OUT_TN
                ]
                out_c_acc = pl.matmul(out_a0, out_w0, out_dtype=pl.FP32)
                for out_lk in pl.pipeline(1, OUT_N_SUB_K, stage=2):
                    out_ks_off = out_lk * OUT_INNER_TK
                    out_a_k = attn_out[:, k_op + out_ks_off : k_op + out_ks_off + OUT_INNER_TK]
                    out_w_k = wo[
                        layer_hidden_base + k_op + out_ks_off : layer_hidden_base
                        + k_op
                        + out_ks_off
                        + OUT_INNER_TK,
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
            out_w0 = wo[
                layer_hidden_base + k_op : layer_hidden_base + k_op + OUT_INNER_TK, n_op : n_op + OUT_TN
            ]
            out_c_acc = pl.matmul(out_a0, out_w0, out_dtype=pl.FP32)
            for out_lk in pl.pipeline(1, OUT_N_SUB_K, stage=2):
                out_ks_off = out_lk * OUT_INNER_TK
                out_a_k = attn_out[:, k_op + out_ks_off : k_op + out_ks_off + OUT_INNER_TK]
                out_w_k = wo[
                    layer_hidden_base + k_op + out_ks_off : layer_hidden_base
                    + k_op
                    + out_ks_off
                    + OUT_INNER_TK,
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
                    post_norm_partial = pl.assemble(
                        post_norm_partial, resid_fp32, [0, k0]
                    )  # FP32 (no BF16 cast)
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
            sq_sum = pl.full([1, BATCH_PAD], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=2):
                k0 = kb * K_CHUNK
                attn_chunk = attn_proj_fp32[:, k0 : k0 + K_CHUNK]
                hidden_chunk = hidden_states[:, k0 : k0 + K_CHUNK]  # FP32 already
                resid_chunk = pl.add(attn_chunk, hidden_chunk)
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH_PAD]),
                )
            post_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
            post_inv_rms_col = pl.reshape(post_inv_rms, [BATCH_PAD, 1])
            inv_rms_tile = pl.assemble(inv_rms_tile, post_inv_rms_col, [0, 0])

        # Split-K gate + up critical-wave (per cast / k_split), like out_proj:
        #   * n_out ∈ [0, GATE_UP_SPMD_N) → pl.spmd(GATE_UP_SPMD_N, deps=[cast])
        #   * n_out ∈ [GATE_UP_SPMD_N, MLP_ON) → task_dummy(cast) → pl.at
        #
        # EXPERIMENT: this is split into two loops so the deferred (non-critical)
        # tiles run in the baseline [n_out outer, k_split inner] order:
        #   Loop 1 (k_split-keyed, pl.unroll): the critical-wave SPMD dispatches
        #     for the leading GATE_UP_SPMD_N tiles + the per-k task_dummy funnels
        #     (gate_late_tids/up_late_tids).
        #   Loop 2 (n_out outer / pl.parallel, k_split inner / pl.range): the
        #     deferred tiles' gate/up matmuls, each depending on the funnel dummy
        #     for its k_split.
        # The two n-tile ranges write disjoint n0 columns of gate/up_acc_all
        # (atomic-add over k), so the split is value-equivalent to the fused form.
        #
        # NOTE: Loop 1 still runs at the top level of a `pl.unroll` body, so it
        # keeps the dedicated `gu_k0` (a bare `k0` would leak as a constant into
        # the later down_proj `pl.parallel(DOWN_ON)` loop — orchestration codegen
        # rejects the literal-init iter_arg). Loop 2 reassigns `k0` inside a real
        # `pl.range` scope, so that leak does not apply there.
        for k_split in pl.unroll(K_SPLITS_MLP):
            gu_k0 = k_split * MLP_K_SLICE
            gate_late_tids[k_split] = pl.system.task_dummy(deps=[cast_tids[k_split]])
            up_late_tids[k_split] = pl.system.task_dummy(deps=[cast_tids[k_split]])

            with pl.spmd(
                GATE_UP_SPMD_N,
                name_hint="gate_proj",
                deps=[cast_tids[k_split]],
            ) as gate_spmd_tid:
                spmd_gate_n_out = pl.get_block_idx()
                spmd_gate_n0 = spmd_gate_n_out * MLP_TN
                spmd_gate_a0 = mlp_norm_in[:, gu_k0 : gu_k0 + MLP_INNER_TK]
                spmd_gate_w0 = w_gate[
                    layer_hidden_base + gu_k0 : layer_hidden_base + gu_k0 + MLP_INNER_TK,
                    spmd_gate_n0 : spmd_gate_n0 + MLP_TN,
                ]
                spmd_gate_c_acc = pl.matmul(spmd_gate_a0, spmd_gate_w0, out_dtype=pl.FP32)
                for spmd_gate_lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):
                    spmd_gate_ks_off = spmd_gate_lk * MLP_INNER_TK
                    spmd_gate_a_k = mlp_norm_in[
                        :, gu_k0 + spmd_gate_ks_off : gu_k0 + spmd_gate_ks_off + MLP_INNER_TK
                    ]
                    spmd_gate_w_k = w_gate[
                        layer_hidden_base + gu_k0 + spmd_gate_ks_off : layer_hidden_base
                        + gu_k0
                        + spmd_gate_ks_off
                        + MLP_INNER_TK,
                        spmd_gate_n0 : spmd_gate_n0 + MLP_TN,
                    ]
                    spmd_gate_c_acc = pl.matmul_acc(spmd_gate_c_acc, spmd_gate_a_k, spmd_gate_w_k)
                gate_acc_all = pl.assemble(
                    gate_acc_all, spmd_gate_c_acc, [0, spmd_gate_n0], atomic=pl.AtomicType.Add
                )
            for spmd_n_out in pl.unroll(GATE_UP_SPMD_N):
                gate_tids[spmd_n_out * K_SPLITS_MLP + k_split] = gate_spmd_tid

            with pl.spmd(
                GATE_UP_SPMD_N,
                name_hint="up_proj",
                deps=[cast_tids[k_split]],
            ) as up_spmd_tid:
                spmd_up_n_out = pl.get_block_idx()
                spmd_up_n0 = spmd_up_n_out * MLP_TN
                spmd_up_a0 = mlp_norm_in[:, gu_k0 : gu_k0 + MLP_INNER_TK]
                spmd_up_w0 = w_up[
                    layer_hidden_base + gu_k0 : layer_hidden_base + gu_k0 + MLP_INNER_TK,
                    spmd_up_n0 : spmd_up_n0 + MLP_TN,
                ]
                spmd_up_c_acc = pl.matmul(spmd_up_a0, spmd_up_w0, out_dtype=pl.FP32)
                for spmd_up_lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):
                    spmd_up_ks_off = spmd_up_lk * MLP_INNER_TK
                    spmd_up_a_k = mlp_norm_in[
                        :, gu_k0 + spmd_up_ks_off : gu_k0 + spmd_up_ks_off + MLP_INNER_TK
                    ]
                    spmd_up_w_k = w_up[
                        layer_hidden_base + gu_k0 + spmd_up_ks_off : layer_hidden_base
                        + gu_k0
                        + spmd_up_ks_off
                        + MLP_INNER_TK,
                        spmd_up_n0 : spmd_up_n0 + MLP_TN,
                    ]
                    spmd_up_c_acc = pl.matmul_acc(spmd_up_c_acc, spmd_up_a_k, spmd_up_w_k)
                up_acc_all = pl.assemble(up_acc_all, spmd_up_c_acc, [0, spmd_up_n0], atomic=pl.AtomicType.Add)
            for spmd_n_out in pl.unroll(GATE_UP_SPMD_N):
                up_tids[spmd_n_out * K_SPLITS_MLP + k_split] = up_spmd_tid

        # Deferred (non-critical) gate/up tiles n_out ∈ [GATE_UP_SPMD_N, MLP_ON),
        # restored to the baseline [n_out outer, k_split inner] loop order. Each
        # tile funnels its cast dependency through the per-k dummy so it carries a
        # single edge to cast_tids[k_split] instead of one edge per tile.
        for n_out in pl.parallel(GATE_UP_SPMD_N, MLP_ON):
            n0 = n_out * MLP_TN
            for k_split in pl.range(K_SPLITS_MLP):
                k0 = k_split * MLP_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="gate_proj",
                    deps=[gate_late_tids[k_split]],
                ) as gate_tid:
                    a0 = mlp_norm_in[:, k0 : k0 + MLP_INNER_TK]
                    w0 = w_gate[
                        layer_hidden_base + k0 : layer_hidden_base + k0 + MLP_INNER_TK, n0 : n0 + MLP_TN
                    ]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):
                        ks_off = lk * MLP_INNER_TK
                        a_k = mlp_norm_in[:, k0 + ks_off : k0 + ks_off + MLP_INNER_TK]
                        w_k = w_gate[
                            layer_hidden_base + k0 + ks_off : layer_hidden_base + k0 + ks_off + MLP_INNER_TK,
                            n0 : n0 + MLP_TN,
                        ]
                        c_acc = pl.matmul_acc(c_acc, a_k, w_k)
                    gate_acc_all = pl.assemble(gate_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)
                gate_tids[n_out * K_SPLITS_MLP + k_split] = gate_tid

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="up_proj",
                    deps=[up_late_tids[k_split]],
                ) as up_tid:
                    a0 = mlp_norm_in[:, k0 : k0 + MLP_INNER_TK]
                    w0 = w_up[
                        layer_hidden_base + k0 : layer_hidden_base + k0 + MLP_INNER_TK, n0 : n0 + MLP_TN
                    ]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, MLP_N_SUB_K, stage=2):
                        ks_off = lk * MLP_INNER_TK
                        a_k = mlp_norm_in[:, k0 + ks_off : k0 + ks_off + MLP_INNER_TK]
                        w_k = w_up[
                            layer_hidden_base + k0 + ks_off : layer_hidden_base + k0 + ks_off + MLP_INNER_TK,
                            n0 : n0 + MLP_TN,
                        ]
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
                    deps=[
                        silu_tids[k_split]
                    ],  # down_seed flows through MLP, output projection, and attention
                ) as down_tid:
                    a0 = mlp_tile[:, k0 : k0 + DOWN_TK]
                    w0 = w_down[layer_inter_base + k0 : layer_inter_base + k0 + DOWN_TK, n0 : n0 + DOWN_TN]
                    c_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
                    for lk in pl.pipeline(1, N_SUB_K, stage=2):
                        ks_off = lk * DOWN_TK
                        a_k = mlp_tile[:, k0 + ks_off : k0 + ks_off + DOWN_TK]
                        w_k = w_down[
                            layer_inter_base + k0 + ks_off : layer_inter_base + k0 + ks_off + DOWN_TK,
                            n0 : n0 + DOWN_TN,
                        ]
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
        out_chunk = pl.add(down_acc_all[:, n0 : n0 + DOWN_TN], post_norm_partial[:, n0 : n0 + DOWN_TN])
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
    sampled_ids: pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH_DYN, HIDDEN], pl.BF16],
) -> pl.Tensor[[BATCH_DYN, HIDDEN], pl.BF16]:
    batch = pl.tensor.dim(sampled_ids, 0)
    for b in pl.parallel(batch):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_embed"):
            token_id = pl.read(sampled_ids, [b, 0])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            for k0 in pl.range(0, HIDDEN, EMBED_HIDDEN_CHUNK):
                hidden_chunk = pl.slice(embed_weight, [1, EMBED_HIDDEN_CHUNK], [token_row, k0])
                next_hidden = pl.assemble(next_hidden, hidden_chunk, [b, k0])
    return next_hidden


@pl.jit.inline
def _greedy_sample_inline(
    logits: pl.Tensor[[BATCH_DYN, VOCAB], pl.FP32],
    sampled_ids: pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32],
) -> pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32]:
    batch = pl.tensor.dim(logits, 0)
    for b in pl.parallel(batch):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="greedy_sample"):
            # Page Attention uses explicit scratch MemRefs, which intentionally
            # skips module-wide PlanMemory.  Use the tile reduction form with
            # explicit scratch so greedy top-1 remains planner-independent.
            reduce_tmp = pl.create_tile(
                [SAMPLE_REDUCE_ROWS, SAMPLE_VOCAB_CHUNK],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            chunk_vals = pl.tile.full(
                [SAMPLE_REDUCE_ROWS, SAMPLE_CHUNK_PAD],
                dtype=pl.FP32,
                value=-3.402823e38,
            )
            for c in pl.range(SAMPLE_REAL_NUM_VOCAB_CHUNKS):
                c0 = c * SAMPLE_VOCAB_CHUNK
                local_scores = pl.load(
                    logits,
                    [b, c0],
                    [1, SAMPLE_VOCAB_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                local_valid_cols = pl.cast(SAMPLE_VOCAB_CHUNK, pl.INDEX)
                if c == SAMPLE_REAL_NUM_FULL_VOCAB_CHUNKS:
                    local_valid_cols = pl.cast(SAMPLE_REAL_VOCAB_TAIL, pl.INDEX)
                local_scores_valid = pl.tile.set_validshape(local_scores, 1, local_valid_cols)
                local_scores_rows = pl.tile.full(
                    [SAMPLE_REDUCE_ROWS, SAMPLE_VOCAB_CHUNK],
                    dtype=pl.FP32,
                    value=-3.402823e38,
                )
                local_scores_rows = pl.tile.assemble(local_scores_rows, local_scores_valid, [0, 0])
                local_max = pl.tile.row_max(local_scores_rows, reduce_tmp)
                best_val = pl.tile.read(local_max, [0, 0])
                pl.tile.write(chunk_vals, [0, c], best_val)

            chunk_max = pl.tile.row_max(chunk_vals, reduce_tmp)
            best_val = pl.tile.read(chunk_max, [0, 0])
            chunk_i32 = pl.cast(0, pl.INT32)
            for c in pl.range(SAMPLE_REAL_NUM_VOCAB_CHUNKS):
                scan_c = (SAMPLE_REAL_NUM_VOCAB_CHUNKS - 1) - c
                val = pl.tile.read(chunk_vals, [0, scan_c])
                if val == best_val:
                    chunk_i32 = pl.cast(scan_c, pl.INT32)

            local_token = pl.cast(0, pl.INT32)
            chunk_base = chunk_i32 * pl.cast(SAMPLE_VOCAB_CHUNK, target_type=pl.INT32)
            chunk_base_idx = pl.cast(chunk_base, target_type=pl.INDEX)
            winning_logits = pl.load(
                logits,
                [pl.cast(b, pl.INDEX), chunk_base_idx],
                [1, SAMPLE_VOCAB_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            if SAMPLE_REAL_VOCAB_TAIL != 0:
                if chunk_i32 == pl.cast(SAMPLE_REAL_NUM_FULL_VOCAB_CHUNKS, target_type=pl.INT32):
                    winning_logits_valid = pl.tile.set_validshape(winning_logits, 1, SAMPLE_REAL_VOCAB_TAIL)
                    winning_logits_padded = pl.tile.fillpad(winning_logits_valid, pad_value=pl.PadValue.min)
                    for t in pl.range(SAMPLE_VOCAB_CHUNK):
                        scan_t = (SAMPLE_VOCAB_CHUNK - 1) - t
                        val = pl.tile.read(winning_logits_padded, [0, pl.cast(scan_t, pl.INDEX)])
                        if val == best_val:
                            local_token = pl.cast(scan_t, pl.INT32)
                else:
                    for t in pl.range(SAMPLE_VOCAB_CHUNK):
                        scan_t = (SAMPLE_VOCAB_CHUNK - 1) - t
                        val = pl.tile.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)])
                        if val == best_val:
                            local_token = pl.cast(scan_t, pl.INT32)
            else:
                for t in pl.range(SAMPLE_VOCAB_CHUNK):
                    scan_t = (SAMPLE_VOCAB_CHUNK - 1) - t
                    val = pl.tile.read(winning_logits, [0, pl.cast(scan_t, pl.INDEX)])
                    if val == best_val:
                        local_token = pl.cast(scan_t, pl.INT32)
            token_id = chunk_base + local_token
            if token_id >= pl.cast(REAL_VOCAB, target_type=pl.INT32):
                token_id = pl.cast(0, pl.INT32)
            token_out = pl.tile.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
            pl.tile.write(token_out, [0, 0], token_id)
            pl.store(token_out, [b, 0], sampled_ids)
    return sampled_ids


_FWD_NLAYERS = NUM_LAYERS  # decode_fwd loop bound; overridable for layer-count tests


@pl.jit.inline(auto_scope=False)
def _decode_fwd_body(  # noqa: PLR0913 — PyPTO-state fused decode body
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[D.block_table_flat], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH_DYN], pl.INT32],
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
    out: pl.Tensor[[BATCH_DYN, VOCAB], pl.FP32],
    embed_weight: pl.Tensor,
    sampled_ids_in: pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32],
    sampled_ids_out: pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32],
    next_hidden: pl.Tensor[[BATCH_DYN, HIDDEN], pl.BF16],
    score_transfer: pl.Tensor[[PA_TRANSFER_ROWS, PA_STACK_TOKENS], pl.FP32],
    probability_transfer: pl.Tensor[[PA_TRANSFER_ROWS, PA_STACK_TOKENS], pl.BF16],
    pv_transfer: pl.Tensor[[PA_TRANSFER_ROWS, HEAD_DIM], pl.FP32],
    ffts_workspace: pl.Tensor[[PA_FFTS_WORKSPACE_ELEMENTS], pl.INT64],
    scratch_ready: pl.Array[1, pl.TASK_ID],
):
    # Device-side fused decode: embed the previous sampled token id, loop the inline
    # body over all _FWD_NLAYERS layers, run the LM head, then sample the next token
    # id. Weights are STACKED [_FWD_NLAYERS*HIDDEN, ...] /
    # [_FWD_NLAYERS*INTERMEDIATE, ...]; k_cache / v_cache cover the public-batch
    # paged pool, and out holds public-batch logits [BATCH_DYN, VOCAB].
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
    seq_lens.bind_dynamic(0, BATCH_DYN)
    slot_mapping.bind_dynamic(0, BATCH_DYN)
    out.bind_dynamic(0, BATCH_DYN)
    sampled_ids_in.bind_dynamic(0, BATCH_DYN)
    sampled_ids_out.bind_dynamic(0, BATCH_DYN)
    next_hidden.bind_dynamic(0, BATCH_DYN)
    batch = pl.tensor.dim(seq_lens, 0)

    next_hidden = _token_embed_inline(sampled_ids_in, embed_weight, next_hidden)

    # ── Batch CHUNKING: the pipeline is padded to BATCH_PAD rows and the attention
    # kernel handles at most BATCH_PAD sequences per call, so a larger public batch
    # runs as consecutive row windows. Embed and sampling stay outside because both
    # already walk the whole public batch.
    #
    # The caller-owned scratch_ready TaskId serializes each native PA invocation,
    # including reuse of the transfer and FFTS workspaces across row windows.
    num_chunks = (batch + BATCH_PAD - 1) // BATCH_PAD
    for chunk_idx in pl.range(num_chunks):
        chunk_row0 = pl.cast(chunk_idx * BATCH_PAD, pl.INDEX)
        chunk_rows = pl.min(BATCH_PAD, batch - chunk_row0)  # < BATCH_PAD on the tail window
        chunk_rows_i32 = pl.cast(chunk_rows, pl.INT32)

        cur = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)  # FP32 inter-layer carry (was BF16)
        prev_out_tid = pl.array.create(1, pl.TASK_ID)
        prev_out_tid[0] = pl.system.task_dummy(deps=[])
        for cb0 in pl.parallel(0, BATCH_PAD, BATCH_PAD):
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="copy_hidden", allow_early_resolve=True
            ) as ch_tid:
                for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                    ck0 = ckb * RMSNORM_K_CHUNK
                    # FIRST-layer boundary: cast the external BF16 embed input -> FP32 once,
                    # so every layer consumes FP32 hidden with no per-boundary round-trip.
                    # The read starts at this window's first public row; the pipeline row
                    # stays 0-based, and chunk_rows zero-fills the tail window's pad rows.
                    cur = pl.assemble(
                        cur,
                        pl.cast(
                            pl.fillpad(
                                pl.slice(
                                    next_hidden,
                                    [BATCH_PAD, RMSNORM_K_CHUNK],
                                    [chunk_row0 + cb0, ck0],
                                    valid_shape=[chunk_rows, RMSNORM_K_CHUNK],
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
        normed = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)
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
            layer_next_hidden = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)  # FP32 layer output
            next_normed = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)  # next layer's x*gamma
            next_gamma_idx = pl.min(layer_idx + 1, _FWD_NLAYERS - 1)  # clamp: last layer's normed unused
            cur = _decode_layer(
                cur,
                input_rms_weight,
                wq,
                wk,
                wv,
                q_norm_weight,
                k_norm_weight,
                seq_lens,
                block_table,
                slot_mapping,
                rope_cos,
                rope_sin,
                k_cache,
                v_cache,
                score_transfer,
                probability_transfer,
                pv_transfer,
                ffts_workspace,
                scratch_ready,
                wo,
                w_gate,
                w_up,
                w_down,
                post_rms_weight,
                layer_next_hidden,
                normed,
                next_normed,
                layer_idx,
                next_gamma_idx,
                prev_out_tid,
                prev_normed_tid,
                chunk_row0,
                chunk_rows_i32,
            )
            normed = next_normed
        out = rms_lm_head_fp32(cur, final_norm_weight, lm_head_weight, out, chunk_row0, chunk_rows)
        # Gate the next window's first PA on this window's consolidated final
        # layer writer, matching the upstream window-serialization contract.
        scratch_ready[0] = prev_out_tid[0]
    sampled_ids_out = _greedy_sample_inline(out, sampled_ids_out)
    return out, sampled_ids_out, next_hidden


@pl.jit
def decode_fwd(  # noqa: PLR0913 -- public model ABI
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[D.block_table_flat], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH_DYN], pl.INT32],
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
    out: pl.Out[pl.Tensor[[BATCH_DYN, VOCAB], pl.FP32]],
    embed_weight: pl.Tensor,
    sampled_ids_in: pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32],
    sampled_ids_out: pl.Out[pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32]],
    next_hidden: pl.Out[pl.Tensor[[BATCH_DYN, HIDDEN], pl.BF16]],
):
    score_transfer = pl.create_tensor([PA_TRANSFER_ROWS, PA_STACK_TOKENS], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([PA_TRANSFER_ROWS, PA_STACK_TOKENS], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([PA_TRANSFER_ROWS, HEAD_DIM], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([PA_FFTS_WORKSPACE_ELEMENTS], dtype=pl.INT64)
    # TASK_ID arrays start invalid.  Leave the initial scratch carry invalid so
    # the first PA has no artificial, unflagged dummy predecessor; every later
    # carry is a real allow_early_resolve producer.
    scratch_ready = pl.array.create(1, pl.TASK_ID)
    out, sampled_ids_out, next_hidden = _decode_fwd_body(
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        w_gate,
        w_up,
        w_down,
        post_rms_weight,
        final_norm_weight,
        lm_head_weight,
        out,
        embed_weight,
        sampled_ids_in,
        sampled_ids_out,
        next_hidden,
        score_transfer,
        probability_transfer,
        pv_transfer,
        ffts_workspace,
        scratch_ready,
    )
    return out, sampled_ids_out, next_hidden


_CHUNK_NLAYERS = 8  # layers per decode_fwd_layers dispatch (chunked fused decode)


@pl.jit.inline(auto_scope=False)
def _decode_fwd_layers_body(  # noqa: PLR0913 — PyPTO-state chunk body
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
    out: pl.Tensor,
    score_transfer: pl.Tensor[[PA_TRANSFER_ROWS, PA_STACK_TOKENS], pl.FP32],
    probability_transfer: pl.Tensor[[PA_TRANSFER_ROWS, PA_STACK_TOKENS], pl.BF16],
    pv_transfer: pl.Tensor[[PA_TRANSFER_ROWS, HEAD_DIM], pl.FP32],
    ffts_workspace: pl.Tensor[[PA_FFTS_WORKSPACE_ELEMENTS], pl.INT64],
    scratch_ready: pl.Array[1, pl.TASK_ID],
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
    batch = pl.tensor.dim(seq_lens, 0)
    # Layer-chunk entry: STATIC BATCH_PAD-row hidden in/out, so it serves one
    # window (rows 0..batch). Public batches above BATCH_PAD are decode_fwd's job.
    batch_i32 = pl.cast(batch, pl.INT32)
    cur = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)
    prev_out_tid = pl.array.create(1, pl.TASK_ID)
    prev_out_tid[0] = pl.system.task_dummy(deps=[])
    for cb0 in pl.parallel(0, BATCH_PAD, BATCH_PAD):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden") as ch_tid:
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                cur = pl.assemble(
                    cur,
                    pl.cast(
                        pl.slice(hidden_states, [BATCH_PAD, RMSNORM_K_CHUNK], [cb0, ck0]),
                        target_type=pl.FP32,
                    ),
                    [cb0, ck0],
                )
        prev_out_tid[0] = ch_tid

    # Pre-loop x_gamma_0: chunk layer 0's normed = cur * gamma_0 (mirrors decode_fwd).
    normed = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)
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
        next_hidden = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.FP32)
        next_normed = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)
        next_gamma_idx = pl.min(i + 1, _CHUNK_NLAYERS - 1)
        cur = _decode_layer(
            cur,
            input_rms_weight,
            wq,
            wk,
            wv,
            q_norm_weight,
            k_norm_weight,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos,
            rope_sin,
            k_cache,
            v_cache,
            score_transfer,
            probability_transfer,
            pv_transfer,
            ffts_workspace,
            scratch_ready,
            wo,
            w_gate,
            w_up,
            w_down,
            post_rms_weight,
            next_hidden,
            normed,
            next_normed,
            i,
            next_gamma_idx,
            prev_out_tid,
            prev_normed_tid,
            0,
            batch_i32,
        )
        normed = next_normed
    for ob0 in pl.parallel(0, BATCH_PAD, BATCH_PAD):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_out"):
            for okb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ok0 = okb * RMSNORM_K_CHUNK
                out = pl.assemble(
                    out,
                    pl.cast(pl.slice(cur, [BATCH_PAD, RMSNORM_K_CHUNK], [ob0, ok0]), target_type=pl.BF16),
                    [ob0, ok0],
                )
    return out


@pl.jit
def decode_fwd_layers(  # noqa: PLR0913 -- public model ABI
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
    score_transfer = pl.create_tensor([PA_TRANSFER_ROWS, PA_STACK_TOKENS], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([PA_TRANSFER_ROWS, PA_STACK_TOKENS], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([PA_TRANSFER_ROWS, HEAD_DIM], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([PA_FFTS_WORKSPACE_ELEMENTS], dtype=pl.INT64)
    # The first PA does not reuse scratch; its invalid carry is omitted from the
    # generated dependency list.  Subsequent carries come from flagged PA tasks.
    scratch_ready = pl.array.create(1, pl.TASK_ID)
    return _decode_fwd_layers_body(
        hidden_states,
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        w_gate,
        w_up,
        w_down,
        post_rms_weight,
        out,
        score_transfer,
        probability_transfer,
        pv_transfer,
        ffts_workspace,
        scratch_ready,
    )


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
    rows = seq_lens.shape[0]
    block_table = torch.arange(rows * DECODE_MAX_BLOCKS_PER_SEQ, dtype=torch.int32)
    slot_mapping = torch.empty(rows, dtype=torch.int32)
    for b in range(rows):
        pos = int(seq_lens[b].item()) - 1
        logical_block = pos // BLOCK_SIZE
        phys_page = b * DECODE_MAX_BLOCKS_PER_SEQ + logical_block
        slot_mapping[b] = phys_page * BLOCK_SIZE + (pos % BLOCK_SIZE)
    return block_table, slot_mapping


def _decode_smoke_inputs(batch: int = BATCH_PAD + 1) -> list[torch.Tensor]:
    """One-layer production-decode inputs with a non-empty tail window.

    Only shapes and dtypes are consumed by compile-only smoke.  ``batch=17`` is
    deliberate: it traces ``decode_fwd``'s dynamic public-batch path as one full
    16-row window plus a one-row tail, matching the entry delegated to by the
    serving HOST wrapper.
    """
    layers = 1
    cache_rows = layers * batch * DECODE_MAX_BLOCKS_PER_SEQ * NUM_KV_HEADS * BLOCK_SIZE
    return [
        torch.empty([layers, HIDDEN], dtype=torch.float32),
        torch.empty([layers * HIDDEN, HIDDEN], dtype=torch.bfloat16),
        torch.empty([layers * HIDDEN, KV_HIDDEN], dtype=torch.bfloat16),
        torch.empty([layers * HIDDEN, KV_HIDDEN], dtype=torch.bfloat16),
        torch.empty([layers, HEAD_DIM], dtype=torch.float32),
        torch.empty([layers, HEAD_DIM], dtype=torch.float32),
        torch.empty([batch], dtype=torch.int32),
        torch.empty([batch * DECODE_MAX_BLOCKS_PER_SEQ], dtype=torch.int32),
        torch.empty([batch], dtype=torch.int32),
        torch.empty([MAX_SEQ, HEAD_DIM], dtype=torch.float32),
        torch.empty([MAX_SEQ, HEAD_DIM], dtype=torch.float32),
        torch.empty([cache_rows, HEAD_DIM], dtype=torch.bfloat16),
        torch.empty([cache_rows, HEAD_DIM], dtype=torch.bfloat16),
        torch.empty([layers * HIDDEN, HIDDEN], dtype=torch.bfloat16),
        torch.empty([layers * HIDDEN, INTERMEDIATE], dtype=torch.bfloat16),
        torch.empty([layers * HIDDEN, INTERMEDIATE], dtype=torch.bfloat16),
        torch.empty([layers * INTERMEDIATE, HIDDEN], dtype=torch.bfloat16),
        torch.empty([layers, HIDDEN], dtype=torch.float32),
        torch.empty([1, HIDDEN], dtype=torch.float32),
        torch.empty([VOCAB, HIDDEN], dtype=torch.bfloat16),
        torch.empty([batch, VOCAB], dtype=torch.float32),
        torch.empty([VOCAB, HIDDEN], dtype=torch.bfloat16),
        torch.empty([batch, SAMPLED_IDS_PAD], dtype=torch.int32),
        torch.empty([batch, SAMPLED_IDS_PAD], dtype=torch.int32),
        torch.empty([batch, HIDDEN], dtype=torch.bfloat16),
    ]


def _backend_type(platform: str) -> BackendType:
    if platform not in PA_SUPPORTED_PLATFORMS:
        raise ValueError(f"Qwen decode attention does not support platform {platform!r}")
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


def random_inputs(
    full_seq: bool = False,
    seed: int = 1234,
    batch: int = BATCH_PAD,
    seq_lens_values: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Deterministic random fixture (name -> tensor) for decode_fwd_layers (N==1).

    full_seq: set every sequence length to MAX_SEQ (full KV cache) for a stable,
    maximum-load performance run; otherwise sample varied lengths in [1, MAX_SEQ].
    batch: public batch — sizes the per-row tensors and the paged pool. Above
    BATCH_PAD it is only meaningful for decode_fwd, which chunks internally.
    """
    g = torch.Generator().manual_seed(seed)

    def rn(shape, std=1.0, bias=0.0):
        return torch.empty(shape).normal_(0.0, std, generator=g) + bias

    if full_seq and seq_lens_values is not None:
        raise ValueError("full_seq and seq_lens_values are mutually exclusive")
    if seq_lens_values is not None:
        if len(seq_lens_values) != batch or any(not 1 <= value <= MAX_SEQ for value in seq_lens_values):
            raise ValueError(f"seq_lens_values must contain {batch} integers in [1, {MAX_SEQ}]")
        seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32)
    elif full_seq:
        seq_lens = torch.full([batch], MAX_SEQ, dtype=torch.int32)
    else:
        seq_lens = torch.randint(1, MAX_SEQ + 1, (batch,), generator=g, dtype=torch.int32)
    cache_rows = batch * DECODE_MAX_BLOCKS_PER_SEQ * NUM_KV_HEADS * BLOCK_SIZE

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
        "hidden_states": rn([batch, HIDDEN], 1.0).to(torch.bfloat16),
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
        "k_cache": rn([cache_rows, HEAD_DIM], 0.01).to(torch.bfloat16),
        "v_cache": rn([cache_rows, HEAD_DIM], 0.02, 0.3).to(torch.bfloat16),
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
    x = values["hidden_states"].float()  # [B,H], residual source
    gamma_in = values["input_rms_weight"].float()[0]  # [H]
    inv_rms = _rmsnorm_inv(x)  # [B,1] (deferred)

    normed = _bf16(x * gamma_in)  # bf16 normed (no inv_rms yet)
    q_proj = normed @ values["wq"].float()  # [B,H]
    k_proj = normed @ values["wk"].float()  # [B,KVH]
    v_proj = normed @ values["wv"].float()  # [B,KVH]

    qn = values["q_norm_weight"].float()[0]
    kn = values["k_norm_weight"].float()[0]
    qh = (q_proj * inv_rms).reshape(BATCH_PAD, NUM_HEADS, HEAD_DIM)
    qh = qh * _rmsnorm_inv(qh) * qn  # per-head QK-norm
    kh = (k_proj * inv_rms).reshape(BATCH_PAD, NUM_KV_HEADS, HEAD_DIM)
    kh = kh * _rmsnorm_inv(kh) * kn
    v_heads = (v_proj * inv_rms).reshape(BATCH_PAD, NUM_KV_HEADS, HEAD_DIM)

    seq_lens = values["seq_lens"]
    block_table = values["block_table"]
    rope_cos = values["rope_cos"].float()
    rope_sin = values["rope_sin"].float()
    k_cache = values["k_cache"].view(NUM_PAGES * BLOCK_SIZE, KV_HIDDEN).float()
    v_cache = values["v_cache"].view(NUM_PAGES * BLOCK_SIZE, KV_HIDDEN).float()

    attn_out = torch.zeros(BATCH_PAD, HIDDEN)
    for b in range(BATCH_PAD):
        slen = int(seq_lens[b].item())
        p = slen - 1
        cos_p, sin_p = rope_cos[p], rope_sin[p]
        q_b = _bf16(_rope_half(qh[b], cos_p, sin_p))  # [40,128] current Q (bf16)
        k_cur = _bf16(_rope_half(kh[b], cos_p, sin_p))  # [8,128] current K (bf16)
        v_cur = _bf16(v_heads[b])  # [8,128]
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
            k_lane[p] = k_cur[kvh]  # current token (kernel writes it first)
            v_lane[p] = v_cur[kvh]
            for j in range(Q_PER_KV):
                hq = kvh * Q_PER_KV + j
                scores = (q_b[hq].unsqueeze(0) * k_lane).sum(-1) * _REF_ATTN_SCALE
                w = torch.softmax(scores, dim=-1)
                attn_out[b, hq * HEAD_DIM : (hq + 1) * HEAD_DIM] = (w.unsqueeze(-1) * v_lane).sum(0)
    attn_out = _bf16(attn_out)

    attn_proj = attn_out @ values["wo"].float()  # out_proj (FP32)
    h1 = x + attn_proj  # raw residual (FP32, NOT bf16-rounded)
    post_gamma = values["post_rms_weight"].float()[0]
    post_inv = _rmsnorm_inv(h1)  # deferred into silu (FP32 h1)
    mlp_in = _bf16(h1 * post_gamma)  # gamma before inv_rms
    gate = mlp_in @ values["w_gate"].float()
    up = mlp_in @ values["w_up"].float()
    sg = gate * post_inv
    su = up * post_inv
    mlp = _bf16(sg * torch.sigmoid(sg) * su)  # SwiGLU
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
    specs.append(TensorSpec("out", [BATCH_PAD, HIDDEN], torch.bfloat16, is_output=True))
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
        "--enable-chip-swimlane",
        nargs="?",
        const=4,
        default=0,
        type=int,
        metavar="PERF_LEVEL",
        help="Enable chip swimlane perf capture at the given granularity level. Bare flag "
        "= level 4 (full). Levels: 1=AICore timing, 2=+dispatch/fanout, 3=+sched "
        "phases, 4=+orch phases; 0 (default) disables.",
    )
    parser.add_argument(
        "--max-seq",
        action="store_true",
        default=False,
        help="set EVERY sequence length to MAX_SEQ (full KV cache) for a stable, "
        "maximum-load performance run; default samples varied random lengths.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for the random fixture (reproducible inputs + golden).",
    )
    parser.add_argument(
        "--seq-lens",
        help=f"exact comma-separated sequence lengths for the single-layer fixture; must contain "
        f"{BATCH_PAD} values in [1, MAX_SEQ] and is mutually exclusive with --max-seq",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "build_output" / "data",
        help="only used by --validate-fwd (pre-generated stacked-fwd inputs).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="compile-only (no device); also the implicit behavior on *sim platforms.",
    )
    parser.add_argument(
        "--enable-dep-gen",
        action="store_true",
        default=False,
        help="capture the task dependency graph to dfx_outputs/deps.json "
        "(render with simpler_setup.tools.deps_viewer). Opt-in AICPU-side "
        "DFX, off by default. Keep --fwd-layers small when capturing: the "
        "per-run SHM record buffer can overflow ('records dropped') on the "
        "full 40-layer graph.",
    )
    parser.add_argument(
        "--dep-output-dir",
        type=Path,
        help="save the selected invocation's kernels and dependency capture under this directory; "
        "requires --enable-dep-gen",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        default=False,
        help="under --validate-fwd, stop after the selected production decode invocation so its "
        "dependency capture cannot be overwritten by the reference invocation",
    )
    parser.add_argument(
        "--validate-fwd",
        action="store_true",
        default=False,
        help="validate the fused decode_fwd (N stacked layers + on-device LM head "
        "-> logits) against a host chain reference, instead of the default "
        "single-layer golden test.",
    )
    parser.add_argument("--fwd-layers", type=int, default=4, help="layer count N for --validate-fwd")
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=BATCH_PAD,
        help="PUBLIC batch for --validate-fwd (>= 1, no upper bound). decode_fwd's\n"
        "batch axes are pl.dynamic, so one compiled program serves any value; the\n"
        "pipeline stays internally padded to BATCH_PAD rows and a larger batch runs as\n"
        "ceil(batch / BATCH_PAD) row windows. Ignored by the default single-layer test,\n"
        "which drives the static decode_fwd_layers.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=1,
        help="under --validate-fwd, run this many decode steps for a device-cost "
        "timing sweep at growing context: each step feeds the previous step's "
        "sampled token back as input and grows the context by one (seq_lens += 1, "
        "slot_mapping advanced to the new token's KV slot). The context starts at "
        "MAX_SEQ (PTO2_MANUAL_MAX_SEQ) and grows to MAX_SEQ + decode_steps - 1; the "
        "paged KV pool (block_table / slot_mapping / kc / vc) is enlarged by "
        "decode_steps tokens up front so the growing seq_lens never overflow it. "
        "Read the device `Effective` span (kc/vc are re-uploaded per step, so it "
        "measures decode cost at growing context, not a correct generation). The "
        "host-ref argmax check is skipped for N > 1.",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        default=False,
        help="persist inputs + golden for replay (off: large fixtures)",
    )
    args = parser.parse_args()

    if args.dep_output_dir is not None and not args.enable_dep_gen:
        parser.error("--dep-output-dir requires --enable-dep-gen")
    if args.skip_reference and not args.validate_fwd:
        parser.error("--skip-reference requires --validate-fwd")
    if args.validate_fwd and args.dep_output_dir is not None and not args.skip_reference:
        parser.error("--validate-fwd with --dep-output-dir also requires --skip-reference")
    requested_seq_lens: list[int] | None = None
    if args.seq_lens is not None:
        if args.max_seq:
            parser.error("--seq-lens and --max-seq are mutually exclusive")
        try:
            requested_seq_lens = [int(value) for value in args.seq_lens.split(",")]
        except ValueError:
            parser.error("--seq-lens must be a comma-separated list of integers")
        expected_seq_lens = args.batch if args.validate_fwd else BATCH_PAD
        if len(requested_seq_lens) != expected_seq_lens or any(
            not 1 <= value <= MAX_SEQ for value in requested_seq_lens
        ):
            parser.error(f"--seq-lens must contain {expected_seq_lens} values in [1, {MAX_SEQ}]")

    set_backend_type(_backend_type(args.platform))

    # The default golden drives a one-layer decode_fwd_layers chunk.  The smoke
    # instead traces the production dynamic decode_fwd entry for one layer.
    # Both loop bounds are Python globals read at trace time, so bind them before
    # either entry can compile.
    _CHUNK_NLAYERS = 1
    _FWD_NLAYERS = 1

    # Full-codegen smoke: explicit --smoke, or any *sim platform.  Compile the
    # production dynamic entry at B17, not the static B16 layer harness: the
    # second one-row window exercises the same sliced block-table shapes used by
    # serving.  `.lower()` alone does not exercise kernel-wrapper extraction.
    if args.smoke or args.platform.endswith("sim"):
        compiled = decode_fwd.compile(
            *_decode_smoke_inputs(),
            config=RunConfig(platform=args.platform),
        )
        print(f"Compiled dynamic B{BATCH_PAD + 1} decode smoke: {compiled.output_dir}")
        raise SystemExit(0)

    # ── Default single-layer unit test: RANDOM inputs, on-the-fly torch golden,
    # on-device run + compare, all through the golden/ harness. ──
    if not args.validate_fwd:
        from golden import ratio_allclose, run_jit

        inputs = random_inputs(
            full_seq=args.max_seq,
            seed=args.seed,
            seq_lens_values=requested_seq_lens,
        )
        specs = _build_specs(inputs)
        print(
            f"[decode_fwd] single-layer golden unit test | platform={args.platform} "
            f"device={args.device} seq={'MAX' if args.max_seq else 'varied'} seed={args.seed} "
            f"seq_lens={inputs['seq_lens'].tolist()}"
        )
        # Ratio tolerance: bf16 outputs cannot satisfy a strict 100% allclose at
        # rtol/atol=3e-3 (one bf16 ULP at value 1 is 2**-8 ≈ 0.0039 > 3e-3), so allow
        # up to 2% outliers — the codebase's ratio_allclose convention. Remaining
        # mismatches are 1-2 ULP bf16 quantization, not errors.
        result = run_jit(
            fn=decode_fwd_layers,
            specs=specs,
            golden_fn=golden_decode_layer,
            compile_cfg=dict(
                dump_passes=False,
                save_kernels=args.dep_output_dir is not None,
                save_kernels_dir=(
                    str(args.dep_output_dir.resolve()) if args.dep_output_dir is not None else None
                ),
            ),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_chip_swimlane=args.enable_chip_swimlane,
                enable_dep_gen=args.enable_dep_gen,
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
        # Generate the fixture at the requested public batch: above BATCH_PAD the
        # per-row tensors and the paged pool must cover every row decode_fwd will
        # chunk over, not just one pipeline width.
        random_values = random_inputs(
            full_seq=args.max_seq,
            seed=args.seed,
            batch=args.batch,
            seq_lens_values=requested_seq_lens,
        )
        inputs = [random_values[name] for name in INPUT_NAMES]
    # dep_gen (--enable-dep-gen) is an AICPU-side DFX collector for the producer->
    # consumer task graph; it does not reduce the AICore cohort, so it coexists with
    # the full-occupancy attention op. --validate-fwd runs two on-device programs
    # (decode_fwd, then the host-ref decode_fwd_layers), and each writes its own
    # dfx_outputs/deps.json. The per-run SHM record buffer can overflow ("records
    # dropped") on the full 40-layer --max-seq graph, so capture with a small
    # --fwd-layers.
    run_cfg = RunConfig(
        platform=args.platform,
        device_id=args.device,
        backend_type=_backend_type(args.platform),
        enable_chip_swimlane=args.enable_chip_swimlane,
        enable_dep_gen=args.enable_dep_gen,
        save_kernels=args.dep_output_dir is not None,
        save_kernels_dir=(str(args.dep_output_dir.resolve()) if args.dep_output_dir is not None else None),
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
        UB = args.batch
        if UB < 1:
            parser.error(f"--batch must be >= 1, got {UB}")
        if UB != hs.shape[0]:
            # Public batch axes are dynamic, so the HOST tensors track UB rows.
            # block_table in particular MUST be UB * DECODE_MAX_BLOCKS_PER_SEQ: the
            # kernel derives the paged row stride as len(block_table) // batch, so a
            # table sized for a different row count would hand the FAI tiler a wrong
            # stride and misaddress every b > 0.
            sl = sl[:UB].contiguous()
            bt = torch.arange(UB * DECODE_MAX_BLOCKS_PER_SEQ, dtype=torch.int32)
            sm = torch.empty(UB, dtype=torch.int32)
            for _b in range(UB):
                _pos = int(sl[_b].item()) - 1
                sm[_b] = (_b * DECODE_MAX_BLOCKS_PER_SEQ + _pos // BLOCK_SIZE) * BLOCK_SIZE + (
                    _pos % BLOCK_SIZE
                )
        torch.manual_seed(1234)
        final_norm_w = torch.empty([1, HIDDEN], dtype=torch.float32).normal_() * 0.1 + 1.0
        lm_head_w = torch.empty([VOCAB, HIDDEN], dtype=torch.bfloat16).normal_() * 0.02
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
            _grow_pages = UB * _grow_blocks
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
                sm_ = torch.empty(UB, dtype=torch.int32)
                for _b in range(UB):
                    pos = int(seq_lens[_b].item()) - 1
                    sm_[_b] = (_b * _grow_blocks + pos // BLOCK_SIZE) * BLOCK_SIZE + (pos % BLOCK_SIZE)
                return sm_

            sl = torch.full([UB], MAX_SEQ, dtype=torch.int32)
            sm = _grow_slot_mapping(sl)
            print(
                f"[stacked-fwd {N}L+LMhead] autoregressive decode: seq_lens {MAX_SEQ} -> "
                f"{MAX_SEQ + _n_steps - 1} over {_n_steps} steps "
                f"(KV pool enlarged to {_grow_blocks} pages/seq for {MAX_SEQ + _n_steps} tokens)"
            )
        stacked = [
            stack0(irw, N),
            stack0(wq_, N),
            stack0(wk_, N),
            stack0(wv_, N),
            stack0(qn, N),
            stack0(kn, N),
            sl,
            bt,
            sm,
            rc,
            rs,
            stack0(kc, N),
            stack0(vc, N),
            stack0(wo_, N),
            stack0(wg, N),
            stack0(wu, N),
            stack0(wd, N),
            stack0(prw, N),
            final_norm_w,
            lm_head_w,
        ]
        logits = torch.zeros(UB, VOCAB, dtype=torch.float32)
        embed_weight = torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16)
        sampled_ids_in = torch.zeros(UB, SAMPLED_IDS_PAD, dtype=torch.int32)
        for b in range(UB):
            embed_weight[b] = hs[b]
        for b in range(UB):
            sampled_ids_in[b, 0] = b
        sampled_ids_out = torch.zeros(UB, SAMPLED_IDS_PAD, dtype=torch.int32)
        next_hidden = torch.zeros(UB, HIDDEN, dtype=torch.bfloat16)
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
            print(
                f"[stacked-fwd {N}L+LMhead] {_n_steps}-step autoregressive decode complete "
                f"(host-ref argmax check skipped for --decode-steps > 1)"
            )
            raise SystemExit(0)
        # Perf-only mode: the chip swimlane collector cannot register host buffers for a
        # second on-device program in the same process (the host-ref call below would
        # `init_chip_swimlane failed: 8`). decode_fwd already emitted the swimlane table,
        # so skip the reference comparison and exit cleanly.
        if args.enable_chip_swimlane:
            print(
                f"[stacked-fwd {N}L+LMhead] swimlane perf run complete "
                f"(host-ref argmax check skipped under --enable-chip-swimlane)"
            )
            raise SystemExit(0)
        if args.skip_reference:
            print(
                f"[stacked-fwd {N}L+LMhead] selected production invocation complete "
                "(host reference skipped by --skip-reference)"
            )
            raise SystemExit(0)
        # host ref: run one N-layer decode_fwd_layers chunk -> final RMSNorm -> lm_head.
        # _CHUNK_NLAYERS = N keeps the inter-layer residual FP32 (chunk casts BF16 only at
        # its boundaries), matching decode_fwd's FP32-carry-until-LM-head path. A per-layer
        # chain (N single-layer dispatches) would re-enter each layer from BF16, diverging
        # from decode_fwd and making the argmax check pass/fail for the wrong reason.
        # decode_fwd_layers is the BATCH_PAD-wide layer stack, so a public batch above
        # it is referenced one window at a time -- the same rows decode_fwd chunks
        # over, but as independent dispatches. The KV pool and the weights are passed
        # whole; only the per-row inputs (hidden, seq_lens, block_table, slot_mapping)
        # are windowed, and the block-table slice keeps its ABSOLUTE page ids so each
        # window addresses the same pages the fused run did.
        _CHUNK_NLAYERS = N
        _BPS = DECODE_MAX_BLOCKS_PER_SEQ

        # `stacked` drops hidden_states, so a windowed input sits one slot before
        # its INPUT_NAMES position. Derive the slots instead of writing them out:
        # reordering INPUT_NAMES would otherwise silently window the wrong tensor.
        def _slot(name: str) -> int:
            return INPUT_NAMES.index(name) - 1

        ref_rows = []
        for _w0 in range(0, UB, BATCH_PAD):
            _n = min(BATCH_PAD, UB - _w0)
            _hs_w = torch.zeros(BATCH_PAD, HIDDEN, dtype=hs.dtype)
            _hs_w[:_n] = hs[_w0 : _w0 + _n]
            _win = list(stacked[: len(INPUT_NAMES) - 1])
            _win[_slot("seq_lens")] = sl[_w0 : _w0 + _n].contiguous()
            _win[_slot("block_table")] = bt[_w0 * _BPS : (_w0 + _n) * _BPS].contiguous()
            _win[_slot("slot_mapping")] = sm[_w0 : _w0 + _n].contiguous()
            _ref_w = torch.zeros(BATCH_PAD, HIDDEN, dtype=torch.bfloat16)
            decode_fwd_layers(_hs_w, *_win, _ref_w, config=run_cfg)
            ref_rows.append(_ref_w[:_n])
        ref_out = torch.cat(ref_rows, dim=0)
        hn = ref_out.float()
        inv = torch.rsqrt(hn.pow(2).mean(-1, keepdim=True) + EPS)
        ref_normed = (hn * inv) * final_norm_w.float()
        ref_logits = ref_normed @ lm_head_w.float().t()  # [BATCH_PAD, VOCAB]
        # Only rows [0, UB) carry real sequences. The reference shares the UB-row
        # seq_lens, so decode_fwd_layers computes the same UB rows; rows past UB
        # are padding on both sides and are deliberately not compared.
        a = logits.cpu()[:UB]
        e = ref_logits.cpu()[:UB]
        # compare argmax (the actual generation signal) + value closeness
        amax_k = a.argmax(-1)
        amax_r = e.argmax(-1)
        sample_k = sampled_ids_out[:UB, 0].cpu()
        argmax_match = int((amax_k == amax_r).sum())
        sample_match = int((sample_k == amax_k).sum())
        close = torch.isclose(a, e, rtol=5e-2, atol=5e-2)
        _windows = (UB + BATCH_PAD - 1) // BATCH_PAD
        print(
            f"[stacked-fwd {N}L+LMhead] batch={UB} "
            f"({_windows}x{BATCH_PAD}-row window(s)) | "
            f"argmax match {argmax_match}/{UB} | "
            f"sample match {sample_match}/{UB} | "
            f"logits {int(close.sum()) / a.numel():.4%} within 5e-2 | "
            f"max_abs_err={(a - e).abs().max():.4f} | kernel_argmax={amax_k.tolist()} "
            f"sampled={sample_k.tolist()} ref_argmax={amax_r.tolist()}"
        )
        raise SystemExit(0 if argmax_match == UB and sample_match == UB else 1)


@pl.jit.host
def qwen3_decode_host(  # noqa: PLR0913 — HOST entry over decode_fwd for serving signature-mode compile
    input_rms_weight: pl.Tensor[[NUM_LAYERS, HIDDEN], pl.FP32],
    wq: pl.Tensor[[NUM_LAYERS * HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[NUM_LAYERS * HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[NUM_LAYERS * HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[NUM_LAYERS, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[D.block_table_flat], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[D.rope_seq, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[D.rope_seq, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[NUM_LAYERS * HIDDEN, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[NUM_LAYERS * HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[NUM_LAYERS * HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[NUM_LAYERS * INTERMEDIATE, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[NUM_LAYERS, HIDDEN], pl.FP32],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH_DYN, VOCAB], pl.FP32]],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    sampled_ids_in: pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[BATCH_DYN, SAMPLED_IDS_PAD], pl.INT32]],
    next_hidden: pl.Out[pl.Tensor[[BATCH_DYN, HIDDEN], pl.BF16]],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    logits, sampled_ids, next_hidden = decode_fwd(
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        w_gate,
        w_up,
        w_down,
        post_rms_weight,
        final_norm_weight,
        lm_head_weight,
        out,
        embed_weight,
        sampled_ids_in,
        sampled_ids,
        next_hidden,
    )
    return logits, sampled_ids, next_hidden
