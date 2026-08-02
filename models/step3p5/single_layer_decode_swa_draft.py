# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 single-layer decode draft — SLIDING-WINDOW attention (layer 1).

Phase 2b of the step3p5 migration plan (see ``MIGRATION_PLAN.md``). Produces
the standalone decode draft for one SWA layer of step3p5 (e.g. layer 1).
The companion file ``single_layer_decode_full_draft.py`` covers the
full-attention layer (e.g. layer 0); the two converge into a parameterised
``decode_layer.py`` in Phase 3.

Layer-1 contract (from the step3p5 checkpoint config and vllm reference):

    | param                  | value          |
    |------------------------|----------------|
    | num_heads              | 96             |
    | num_kv_heads           | 8              |
    | q_per_kv               | 12             |
    | head_dim               | 128            |
    | hidden                 | 4096           |
    | intermediate (dense)   | 11264          |
    | rope_theta             | 1e4            |
    | partial_rotary_factor  | 1.0 (full)     |
    | yarn scaling           | NO             |
    | sliding_window         | 512            |
    | zero-centered RMSNorm  | YES (gamma_eff = stored_gamma + 1.0) |
    | per-head q_norm/k_norm | YES (shape [128]) |
    | head-wise attn gate    | YES (g_proj [96, 4096]) |
    | mlp activation         | plain SiLU * up (SwigluStep limit = 0) |

Structural blueprint (mirrors the dense-GQA reference ``decode_layer.py`` fa_fused with
the step3p5 SWA increments highlighted):

Scope 1
  1. Zero-centered RMSNorm of input hidden states.
  2. Q / K / V projection matmuls (single-layer-flat weights here; Phase 3
     converts these to the per-layer LAYER_HIDDEN_ROWS_DYN form).
  3. Per-head q_norm / k_norm (zero-centered, weights shape [HEAD_DIM]).
  4. Head-wise attention gate matmul: gate_logits = x @ g_proj.

Scope 2 (sliding-window flash decode)
  1. Full RoPE on the leading HEAD_DIM lanes of Q / K (partial_rotary=1.0).
  2. K/V paged cache write at the current decode slot.
  3. fa_fused: pl.spmd(BATCH * (TOTAL_Q_GROUPS // 2)) with a pl.pipeline(2)
     inner over the paired Q-group. The K/V iteration count is clamped to
     ``eff_ctx_blocks = ceil(min(seq_len, SLIDING_WINDOW) / BLOCK_SIZE)`` so
     the softmax denominator only sums over the most recent ``SLIDING_WINDOW``
     positions. The last block uses tail-masking on
     ``valid_len = min(BLOCK_SIZE, eff_ctx_len - sb * BLOCK_SIZE)`` to clear
     the lanes past the trimmed window.

Scope 2.5 (head-wise gate)
  Multiply each head slice of attn_out by sigmoid(gate_logits[batch, head]).

Scope 3
  1. Output projection: attn_out @ wo + residual.
  2. Zero-centered RMSNorm of the post-attn hidden.
  3. Dense MLP: gate_proj / up_proj / SiLU * up / down_proj.
  4. Residual add to produce next_hidden.

Tiling parameters come from ``models/step3p5/config.py``. SWA-specific
choices versus the dense-GQA reference's fa_fused tuning:
  - Q_HEAD_BATCH = Q_PER_KV_SWA = 12 (vs. the dense-GQA reference = 5).
  - Q_HEAD_PAD   = 24 — even, multiple of 4, and Q_HEAD_PAD//2 = 12 >=
    Q_HEAD_BATCH, satisfying the same ptoas constraint the dense-GQA reference documents.
  - TOTAL_Q_GROUPS = NUM_KV_HEADS_SWA * (Q_PER_KV_SWA // Q_HEAD_BATCH) =
    8 * 1 = 8 (even, fa_fused pairs Q-groups).

Phase 3 dedup boundaries are marked with ``TODO(phase-3 dedup)`` comments —
the inline helpers (zero-centered RMSNorm, per-head qk-norm broadcast,
partial-rope step, head-wise gate) intentionally share function names and
docstrings with ``single_layer_decode_full_draft.py`` so the consolidator
can lift them into a shared ``_ops.py`` without renaming.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl

from models.step3p5.config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_SWA,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE,
    K_CHUNK,
    KV_HIDDEN,
    KV_OUT_CHUNK,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ_DEFAULT,
    MLP_OUT_CHUNK,
    NUM_HEADS_SWA,
    NUM_KV_HEADS,
    OUT_PROJ_K_CHUNK,
    OUT_PROJ_N_CHUNK,
    Q_HEAD_BATCH_SWA,
    Q_HEAD_PAD_SWA,
    Q_OUT_CHUNK,
    Q_PER_KV_SWA,
    ROTARY_HALF_SWA,
    SLIDING_WINDOW,
)

# -----------------------------------------------------------------------------
# Local single-layer-draft aliases. Phase 3 will replace these with the
# LAYER_DYN-based per-layer slice arithmetic that ``decode_layer.py`` uses.
# -----------------------------------------------------------------------------
NUM_HEADS = NUM_HEADS_SWA            # 96
HIDDEN_Q = HIDDEN_Q_SWA              # 12288
Q_PER_KV = Q_PER_KV_SWA              # 12
Q_HEAD_BATCH = Q_HEAD_BATCH_SWA      # 12
Q_HEAD_PAD = Q_HEAD_PAD_SWA          # 24
# Rotation slice covers the full HEAD_DIM since partial_rotary_factor = 1.0.
ROTARY_HALF = ROTARY_HALF_SWA        # 64 == HEAD_DIM // 2

# SWA decode: each query only attends to the most recent SLIDING_WINDOW K/V
# rows. With BLOCK_SIZE = 128 and SLIDING_WINDOW = 512, the window spans
# exactly 4 paged blocks per KV head. The single-layer draft assumes the
# host stages the visible window into the first ``WIN_BLOCKS`` slots of each
# batch's block table (the per-layer kv-cache layout in Phase 3 will mirror
# this convention; the rotating-slot scheme the TP-aware SWA reference
# uses is left for that integration step).
WIN_BLOCKS = (SLIDING_WINDOW + BLOCK_SIZE - 1) // BLOCK_SIZE  # 4

# fa_fused Q-group geometry (same shape as the dense-GQA reference fa_fused, with the SWA
# Q-row counts substituted in):
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH                      # 1
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS                 # 8
assert TOTAL_Q_GROUPS % 2 == 0, (
    f"TOTAL_Q_GROUPS ({TOTAL_Q_GROUPS}) must be even (fa_fused pairs Q groups)"
)
assert Q_HEAD_PAD % 4 == 0 and Q_HEAD_PAD // 2 >= Q_HEAD_BATCH, (
    f"Q_HEAD_PAD ({Q_HEAD_PAD}) must be a multiple of 4 with "
    f"Q_HEAD_PAD // 2 ({Q_HEAD_PAD // 2}) >= Q_HEAD_BATCH ({Q_HEAD_BATCH})"
)
# Sliding-window must align with BLOCK_SIZE so we can express
# ``eff_ctx_blocks = ceil(min(ctx_len, SLIDING_WINDOW) / BLOCK_SIZE)`` as a
# simple integer; the draft does not need to handle a non-multiple window.
assert SLIDING_WINDOW % BLOCK_SIZE == 0, (
    f"SLIDING_WINDOW ({SLIDING_WINDOW}) must be a multiple of BLOCK_SIZE ({BLOCK_SIZE})"
)


# ============================================================================
# Phase-3 dedup boundary START — shared helper signatures.
#
# The four ``@pl.jit.inline`` helpers below ARE INTENTIONALLY DUPLICATED in
# ``single_layer_decode_full_draft.py``. They will be hoisted into a single
# ``models/step3p5/_ops.py`` in Phase 3. Do not rename them in isolation —
# pick a new name in BOTH drafts at once.
# ============================================================================


# TODO(phase-3 dedup): hoist `rmsnorm_zero_centered_row` to shared _ops.py.
# Mirror copy lives in single_layer_decode_full_draft.py.
@pl.jit.inline
def rmsnorm_zero_centered_row(
    x_chunk_fp32,  # pl.Tensor[[rows, k_chunk], pl.FP32]
    inv_rms,       # pl.Tensor[[rows, 1], pl.FP32]
    gamma_stored,  # pl.Tensor[[1, k_chunk], pl.FP32]  (stored, not gamma + 1)
):
    """Zero-centered RMSNorm body for one [rows, k_chunk] tile.

    Step3p5 stores RMSNorm gammas zero-centered: the effective gamma is
    ``stored_gamma + 1.0`` (vllm's OptimusRMSNorm with ``zero_centered=True``
    -- ``norm_weight_bias=1.0`` in ``fused_qknorm_rope_forward_impl``).

    This helper performs the per-tile arithmetic that comes AFTER the
    row-wise variance reduction is already known: it scales the chunk by
    ``inv_rms`` and multiplies by ``gamma_stored + 1.0`` lane-wise. Pull the
    sq_sum / inv_rms outside so we can reuse it across hidden-dim chunks.
    """
    gamma_eff = pl.adds(gamma_stored, 1.0)
    normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk_fp32, inv_rms), gamma_eff)
    return normed


# TODO(phase-3 dedup): hoist `per_head_qk_norm_block` to shared _ops.py.
# Mirror copy lives in single_layer_decode_full_draft.py.
@pl.jit.inline
def per_head_qk_norm_block(
    q_chunk,        # pl.Tensor[[rows * q_per_kv, HEAD_DIM], pl.FP32]
    k_chunk,        # pl.Tensor[[rows, HEAD_DIM], pl.FP32]
    q_norm_gamma,   # pl.Tensor[[1, HEAD_DIM], pl.FP32]  (stored, zero-centered)
    k_norm_gamma,   # pl.Tensor[[1, HEAD_DIM], pl.FP32]  (stored, zero-centered)
):
    """Per-head zero-centered RMSNorm on a (Q_PER_KV Q-rows + 1 K-row) bundle.

    Step3p5 applies a per-head RMSNorm to Q and K BEFORE RoPE (vllm
    `fused_qknorm_rope_forward_impl`). The norm dim is the per-head
    ``HEAD_DIM = 128`` axis; the weight is shared across all heads (one
    [HEAD_DIM] vector per Q / K). Zero-centered means we use
    ``gamma_eff = stored_gamma + 1.0``.

    Returns the (q_chunk_normed, k_chunk_normed) pair, both FP32.
    """
    head_dim_inv = HEAD_DIM_INV
    q_sq = pl.row_sum(pl.mul(q_chunk, q_chunk))
    q_inv = pl.rsqrt(pl.add(pl.mul(q_sq, head_dim_inv), EPS))
    q_gamma_eff = pl.adds(q_norm_gamma, 1.0)
    q_normed = pl.col_expand_mul(pl.row_expand_mul(q_chunk, q_inv), q_gamma_eff)

    k_sq = pl.row_sum(pl.mul(k_chunk, k_chunk))
    k_inv = pl.rsqrt(pl.add(pl.mul(k_sq, head_dim_inv), EPS))
    k_gamma_eff = pl.adds(k_norm_gamma, 1.0)
    k_normed = pl.col_expand_mul(pl.row_expand_mul(k_chunk, k_inv), k_gamma_eff)
    return q_normed, k_normed


# TODO(phase-3 dedup): hoist `rope_partial_rotate` to shared _ops.py.
# Mirror copy lives in single_layer_decode_full_draft.py. For the SWA layer
# `partial=1.0` means we rotate the leading HEAD_DIM lanes (rotary_half =
# HEAD_DIM // 2 = 64), so the "partial" rotation degenerates to a full
# rotation; the function still accepts the per-layer half-dim to stay
# generic for Phase 3.
@pl.jit.inline
def rope_partial_rotate(
    lo,        # pl.Tensor[[rows, ROTARY_HALF], pl.FP32]
    hi,        # pl.Tensor[[rows, ROTARY_HALF], pl.FP32]
    cos_lo,    # pl.Tensor[[1, ROTARY_HALF], pl.FP32]
    cos_hi,    # pl.Tensor[[1, ROTARY_HALF], pl.FP32]
    sin_lo,    # pl.Tensor[[1, ROTARY_HALF], pl.FP32]
    sin_hi,    # pl.Tensor[[1, ROTARY_HALF], pl.FP32]
):
    """Rotate one (lo, hi) half-pair of an RoPE slice (interleaved layout).

    Step3p5 uses the llama-style RoPE where the rotation pair is
    ``(x[..., :half], x[..., half:rotary_dim])``. For SWA layers
    ``rotary_dim == head_dim`` (partial=1.0); for full-attention layers it
    is ``head_dim // 2`` (partial=0.5) -- the helper does not care, the
    caller supplies the right half-dim slices.
    """
    rot_lo = pl.sub(pl.col_expand_mul(lo, cos_lo), pl.col_expand_mul(hi, sin_lo))
    rot_hi = pl.add(pl.col_expand_mul(hi, cos_hi), pl.col_expand_mul(lo, sin_hi))
    return rot_lo, rot_hi


# TODO(phase-3 dedup): hoist `head_wise_gate_apply` to shared _ops.py.
# Mirror copy lives in single_layer_decode_full_draft.py.
@pl.jit.inline
def head_wise_gate_apply(
    attn_head_slice,    # pl.Tensor[[rows, HEAD_DIM], pl.BF16]
    gate_logit_col,     # pl.Tensor[[rows, 1], pl.FP32]
):
    """Multiply one attn-out head slab by its per-head sigmoid gate.

    ``gate = sigmoid(x @ g_proj)`` is precomputed at scope-1 time and lives
    as a single column [rows, 1] for each head ``h``. We broadcast it across
    HEAD_DIM lanes and multiply into ``attn_head_slice``.
    """
    gate = pl.recip(pl.add(pl.exp(pl.neg(gate_logit_col)), 1.0))
    gated_fp32 = pl.col_expand_mul(pl.cast(attn_head_slice, target_type=pl.FP32), gate)
    return pl.cast(gated_fp32, target_type=pl.BF16)


# ============================================================================
# Phase-3 dedup boundary END.
# ============================================================================


# Single-layer-draft entry. Shapes mirror the dense-GQA reference's `test_decode_layer`
# fixture: every layer-stacked weight is collapsed to one layer (no
# LAYER_HIDDEN_ROWS_DYN slicing) so the SWA draft can be built and tested
# in isolation. Phase 3 re-introduces the per-layer LAYER_DYN base offsets.
@pl.jit
def decode_swa_layer(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN_Q], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    g_proj: pl.Tensor[[HIDDEN, NUM_HEADS], pl.BF16],
    seq_lens: pl.Tensor[[BATCH], pl.INT32],
    block_table: pl.Tensor[[BATCH * MAX_BLOCKS_PER_SEQ], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH], pl.INT32],
    rope_cos: pl.Tensor[[MAX_SEQ_DEFAULT, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ_DEFAULT, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[BATCH * MAX_BLOCKS_PER_SEQ * NUM_KV_HEADS * BLOCK_SIZE, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[BATCH * MAX_BLOCKS_PER_SEQ * NUM_KV_HEADS * BLOCK_SIZE, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[HIDDEN_Q, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    hidden_blocks = HIDDEN // K_CHUNK
    decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
    decode_attn_scale = ATTN_SCALE
    bt_stride = MAX_BLOCKS_PER_SEQ

    # Single-layer draft: no layer offset arithmetic (Phase 3 reintroduces it).
    layer_idx = 0

    # Bridges between scope 1 sub-regions and scope 2.
    q_proj = pl.create_tensor([BATCH, HIDDEN_Q], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN_Q], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    # Head-wise gate logits — produced in scope 1, consumed in scope 2.5.
    gate_logits = pl.create_tensor([BATCH, NUM_HEADS], dtype=pl.FP32)

    # ---- Scope 1.a — zero-centered RMSNorm of the input hidden states. ----
    for rms_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="swa_rmsnorm"):
        rms_b0 = rms_spmd_idx * BATCH_TILE
        partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(decode_scope1_hidden_blocks):
            sq_k0 = kb * INPUT_PROJ_K_CHUNK
            sq_chunk = pl.cast(
                pl.slice(hidden_states, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [rms_b0, sq_k0]),
                target_type=pl.FP32,
            )
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(sq_chunk, sq_chunk)), [1, BATCH_TILE]),
            )
        variance = pl.reshape(
            pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
            [BATCH_TILE, 1],
        )
        inv_rms = pl.recip(pl.sqrt(variance))
        for kb in pl.range(decode_scope1_hidden_blocks):
            norm_k0 = kb * INPUT_PROJ_K_CHUNK
            norm_chunk = pl.cast(
                pl.slice(hidden_states, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [rms_b0, norm_k0]),
                target_type=pl.FP32,
            )
            gamma_stored = pl.slice(
                input_rms_weight, [1, INPUT_PROJ_K_CHUNK], [layer_idx, norm_k0],
            )
            normed = rmsnorm_zero_centered_row(norm_chunk, inv_rms, gamma_stored)
            normed_all = pl.assemble(
                normed_all,
                pl.cast(normed, target_type=pl.BF16),
                [rms_b0, norm_k0],
            )

    # ---- Scope 1.b — Q projection (12288 cols / 256 = 48 spmd lanes per batch tile). ----
    for q_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (HIDDEN_Q // Q_OUT_CHUNK),
        name_hint="swa_q_proj",
    ):
        q_b_idx = q_spmd_idx // (HIDDEN_Q // Q_OUT_CHUNK)
        q_ob = q_spmd_idx % (HIDDEN_Q // Q_OUT_CHUNK)
        q_b0 = q_b_idx * BATCH_TILE
        q_o0 = q_ob * Q_OUT_CHUNK

        q_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, 0])
        q_tile_b_0 = pl.slice(wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [0, q_o0])
        q_acc = pl.matmul(q_tile_a_0, q_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            q_k0 = kb * INPUT_PROJ_K_CHUNK
            q_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, q_k0])
            q_tile_b = pl.slice(wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [q_k0, q_o0])
            q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
        q_proj = pl.assemble(q_proj, q_acc, [q_b0, q_o0])

    # ---- Scope 1.c — K projection. ----
    for k_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (KV_HIDDEN // KV_OUT_CHUNK),
        name_hint="swa_k_proj",
    ):
        k_b_idx = k_spmd_idx // (KV_HIDDEN // KV_OUT_CHUNK)
        k_ob = k_spmd_idx % (KV_HIDDEN // KV_OUT_CHUNK)
        k_b0 = k_b_idx * BATCH_TILE
        k_o0 = k_ob * KV_OUT_CHUNK

        k_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [k_b0, 0])
        k_tile_b_0 = pl.slice(wk, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [0, k_o0])
        k_acc = pl.matmul(k_tile_a_0, k_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            k_k0 = kb * INPUT_PROJ_K_CHUNK
            k_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [k_b0, k_k0])
            k_tile_b = pl.slice(wk, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [k_k0, k_o0])
            k_acc = pl.matmul_acc(k_acc, k_tile_a, k_tile_b)
        k_proj = pl.assemble(k_proj, k_acc, [k_b0, k_o0])

    # ---- Scope 1.d — V projection. ----
    for v_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (KV_HIDDEN // KV_OUT_CHUNK),
        name_hint="swa_v_proj",
    ):
        v_b_idx = v_spmd_idx // (KV_HIDDEN // KV_OUT_CHUNK)
        v_ob = v_spmd_idx % (KV_HIDDEN // KV_OUT_CHUNK)
        v_b0 = v_b_idx * BATCH_TILE
        v_o0 = v_ob * KV_OUT_CHUNK

        v_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [v_b0, 0])
        v_tile_b_0 = pl.slice(wv, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [0, v_o0])
        v_acc = pl.matmul(v_tile_a_0, v_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            v_k0 = kb * INPUT_PROJ_K_CHUNK
            v_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [v_b0, v_k0])
            v_tile_b = pl.slice(wv, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [v_k0, v_o0])
            v_acc = pl.matmul_acc(v_acc, v_tile_a, v_tile_b)
        v_proj = pl.assemble(v_proj, v_acc, [v_b0, v_o0])

    # ---- Scope 1.e — per-head zero-centered q_norm / k_norm. ----
    # One KV head per spmd lane; the Q_PER_KV = 12 Q heads tied to that KV
    # head are normed in the same tile. The norm weight is broadcast across
    # all heads (per-head [HEAD_DIM] vector).
    for qkn_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * NUM_KV_HEADS,
        name_hint="swa_qk_norm",
    ):
        qkn_b_idx = qkn_spmd_idx // NUM_KV_HEADS
        qkn_h = qkn_spmd_idx % NUM_KV_HEADS
        qkn_b0 = qkn_b_idx * BATCH_TILE

        qkn_q0 = qkn_h * Q_PER_KV * HEAD_DIM
        q_chunk = pl.reshape(
            pl.slice(q_proj, [BATCH_TILE, Q_HEAD_BATCH * HEAD_DIM], [qkn_b0, qkn_q0]),
            [BATCH_TILE * Q_HEAD_BATCH, HEAD_DIM],
        )
        qkn_k0 = qkn_h * HEAD_DIM
        k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [qkn_b0, qkn_k0])

        q_gamma = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
        k_gamma = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
        q_normed, k_normed = per_head_qk_norm_block(q_chunk, k_chunk, q_gamma, k_gamma)

        q_normed_flat = pl.reshape(q_normed, [BATCH_TILE, Q_HEAD_BATCH * HEAD_DIM])
        q_proj_norm = pl.assemble(q_proj_norm, q_normed_flat, [qkn_b0, qkn_q0])
        k_proj_norm = pl.assemble(k_proj_norm, k_normed, [qkn_b0, qkn_k0])

    # ---- Scope 1.f — head-wise attention gate matmul. ----
    # gate_logits = hidden_states @ g_proj  with g_proj shape [HIDDEN, NUM_HEADS].
    # NUM_HEADS = 96 is too narrow to chunk on the N axis; we keep one tile.
    for gp_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="swa_gate_proj"):
        gp_b0 = gp_spmd_idx * BATCH_TILE
        a0 = pl.slice(hidden_states, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [gp_b0, 0])
        b0t = pl.slice(g_proj, [INPUT_PROJ_K_CHUNK, NUM_HEADS], [0, 0])
        gp_acc = pl.matmul(a0, b0t, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            k0 = kb * INPUT_PROJ_K_CHUNK
            a = pl.slice(hidden_states, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [gp_b0, k0])
            b = pl.slice(g_proj, [INPUT_PROJ_K_CHUNK, NUM_HEADS], [k0, 0])
            gp_acc = pl.matmul_acc(gp_acc, a, b)
        gate_logits = pl.assemble(gate_logits, gp_acc, [gp_b0, 0])

    # ---- Scope 2 — RoPE + paged KV cache write + fa_fused (sliding). ----
    attn_out = pl.create_tensor([BATCH, HIDDEN_Q], dtype=pl.BF16)
    all_q_padded = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16,
    )

    # Per-batch RoPE + K/V scatter. Same shape as the dense-GQA reference's rope_kv_cache
    # except that the rotation slice covers the full HEAD_DIM lanes (no
    # pass-through tail) since partial_rotary_factor = 1.0 on this layer.
    for b in pl.parallel(BATCH):
        ctx_len = pl.tensor.read(seq_lens, [b])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE
        cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, ROTARY_HALF], [0, 0])
        cos_hi = pl.slice(cos_row, [1, ROTARY_HALF], [0, ROTARY_HALF])
        sin_lo = pl.slice(sin_row, [1, ROTARY_HALF], [0, 0])
        sin_hi = pl.slice(sin_row, [1, ROTARY_HALF], [0, ROTARY_HALF])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_kv_cache"):
            for ki in pl.range(NUM_KV_HEADS):
                kv_col = ki * HEAD_DIM
                cache_row = (slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + slot_offset
                k_lo = pl.slice(k_proj_norm, [1, ROTARY_HALF], [b, kv_col])
                k_hi = pl.slice(k_proj_norm, [1, ROTARY_HALF], [b, kv_col + ROTARY_HALF])
                rot_k_lo, rot_k_hi = rope_partial_rotate(
                    k_lo, k_hi, cos_lo, cos_hi, sin_lo, sin_hi,
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_lo, target_type=pl.BF16), [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_hi, target_type=pl.BF16), [cache_row, ROTARY_HALF],
                )
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(
                        pl.slice(v_proj, [1, HEAD_DIM], [b, kv_col]),
                        target_type=pl.BF16,
                    ),
                    [cache_row, 0],
                )

                q_base = ki * Q_PER_KV
                q_block = pl.reshape(
                    pl.slice(q_proj_norm, [1, Q_HEAD_BATCH * HEAD_DIM], [b, q_base * HEAD_DIM]),
                    [Q_HEAD_BATCH, HEAD_DIM],
                )
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH, ROTARY_HALF], [0, 0])
                q_hi = pl.slice(q_block, [Q_HEAD_BATCH, ROTARY_HALF], [0, ROTARY_HALF])
                rot_q_lo, rot_q_hi = rope_partial_rotate(
                    q_lo, q_hi, cos_lo, cos_hi, sin_lo, sin_hi,
                )
                rot_q_lo_bf16 = pl.cast(rot_q_lo, target_type=pl.BF16)
                rot_q_hi_bf16 = pl.cast(rot_q_hi, target_type=pl.BF16)
                all_q_padded = pl.assemble(
                    all_q_padded,
                    rot_q_lo_bf16,
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD, 0],
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    rot_q_hi_bf16,
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD, ROTARY_HALF],
                )
                # Zero the padded tail rows (Q_HEAD_BATCH..Q_HEAD_PAD).
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(
                        pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),
                        target_type=pl.BF16,
                    ),
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                )

    # ---- fa_fused (SWA): pl.spmd(BATCH * (TOTAL_Q_GROUPS // 2)) lanes. ----
    # SWA-specific change versus the dense-GQA reference fa_fused: clamp the K/V iteration count
    # to ``eff_ctx_blocks = ceil(eff_ctx_len / BLOCK_SIZE)`` where
    # ``eff_ctx_len = min(ctx_len, SLIDING_WINDOW)``. Everything else is the
    # same online-softmax recurrence + dual-AIV no-op replay machinery.
    # TODO(phase-3 SWA cache layout): this draft uses the LINEAR cache
    # layout (blocks 0..eff_ctx_blocks-1 of each batch's block table hold
    # the visible window, same as the dense-GQA reference). vLLM's reference SWA path
    # and the in-tree TP-aware SWA reference use a ROTATING-SLOT scheme
    # ((start_pos + s) % WIN). Per team-lead 2026-06-03: stay with the
    # linear form here; Phase 3 consolidation revisits once Phase 5
    # decode_fwd settles on which layout it wants.
    for fa_spmd_idx in pl.spmd(
        BATCH * (TOTAL_Q_GROUPS // 2),
        name_hint="swa_fa_fused",
    ):
        fa_b = fa_spmd_idx // (TOTAL_Q_GROUPS // 2)
        fa_g2 = fa_spmd_idx % (TOTAL_Q_GROUPS // 2)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b])
        # Window-clamp: each query only attends to the most recent
        # SLIDING_WINDOW K/V rows. In a single-query decode this collapses
        # to capping the iteration count and softmax denominator.
        fa_eff_ctx_len = pl.min(fa_ctx_len, SLIDING_WINDOW)
        fa_ctx_blocks = (fa_eff_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        fa_block_table_base = fa_b * bt_stride

        for gp in pl.pipeline(2, stage=2):
            gi = fa_g2 * 2 + gp
            kvh = gi // Q_GROUPS
            qg = gi - kvh * Q_GROUPS
            q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH
            q_padded_row = fa_b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
            q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, HEAD_DIM], [q_padded_row, 0])

            # Online-softmax sentinels (mirrors the dense-GQA reference fa_fused).
            mi_flat = pl.full([1, Q_HEAD_PAD], dtype=pl.FP32, value=-3.0e38)
            mi = pl.reshape(mi_flat, [Q_HEAD_PAD, 1])
            li_flat = pl.full([1, Q_HEAD_PAD], dtype=pl.FP32, value=0.0)
            li = pl.reshape(li_flat, [Q_HEAD_PAD, 1])
            oi = pl.full([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)

            for sb in pl.range(fa_ctx_blocks):
                s0 = sb * BLOCK_SIZE
                valid_len = pl.min(BLOCK_SIZE, fa_eff_ctx_len - s0)
                fa_block_table_idx = fa_block_table_base + sb
                fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_block_table_idx]), pl.INDEX)
                fa_cache_row = (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE

                k_tile = k_cache[fa_cache_row : fa_cache_row + BLOCK_SIZE, :]
                raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                scores_scaled = pl.mul(raw_scores, decode_attn_scale)
                # TODO(npu-tuning): Q_HEAD_PAD // 2 == 12 here (SWA uses
                # Q_HEAD_PAD_SWA=24 vs the dense-GQA reference's 16). The ptoas verifier's
                # dual-AIV no-op replay path has only been exercised at
                # half-pad == 8 in the dense-GQA reference; the value-12 slow-path is
                # untested on hardware. Expect a tuning bounce on the
                # first NPU run.
                scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_PAD // 2, valid_len)
                scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                cur_mi = pl.row_max(scores)
                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                cur_li = pl.row_sum(exp_scores_fp32)

                v_tile = v_cache[fa_cache_row : fa_cache_row + BLOCK_SIZE, :]
                oi_tmp = pl.matmul(exp_scores_bf16, v_tile, out_dtype=pl.FP32)

                mi_new = pl.maximum(mi, cur_mi)
                alpha = pl.exp(pl.sub(mi, mi_new))
                beta = pl.exp(pl.sub(cur_mi, mi_new))
                li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp, beta))
                mi = mi_new

            ctx = pl.row_expand_div(oi, li)
            ctx_valid = ctx[0:Q_HEAD_BATCH, :]
            ctx_flat_bf16 = pl.cast(
                pl.reshape(ctx_valid, [1, Q_HEAD_BATCH * HEAD_DIM]),
                target_type=pl.BF16,
            )
            attn_out = pl.assemble(attn_out, ctx_flat_bf16, [fa_b, q_base * HEAD_DIM])

    # ---- Scope 2.5 — head-wise sigmoid gate on attn_out (per-head). ----
    # Multiply each head slab of attn_out by sigmoid(gate_logits[b, h]).
    gated_attn_out = pl.create_tensor([BATCH, HIDDEN_Q], dtype=pl.BF16)
    for gate_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * NUM_HEADS,
        name_hint="swa_head_gate",
    ):
        gate_b_idx = gate_spmd_idx // NUM_HEADS
        gate_h = gate_spmd_idx % NUM_HEADS
        gate_b0 = gate_b_idx * BATCH_TILE
        gate_h0 = gate_h * HEAD_DIM
        head_slice = pl.slice(attn_out, [BATCH_TILE, HEAD_DIM], [gate_b0, gate_h0])
        gate_col = pl.slice(gate_logits, [BATCH_TILE, 1], [gate_b0, gate_h])
        gated = head_wise_gate_apply(head_slice, gate_col)
        gated_attn_out = pl.assemble(gated_attn_out, gated, [gate_b0, gate_h0])

    # ---- Scope 3 — o_proj + residual + post_rmsnorm + dense MLP + residual. ----
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.FP32)

        # out_proj: gated_attn_out @ wo + residual.
        # K reduction iterates HIDDEN_Q / OUT_PROJ_K_CHUNK = 12288 / 256 = 48 blocks.
        out_proj_k_blocks = HIDDEN_Q // OUT_PROJ_K_CHUNK
        for ob in pl.spmd(
            HIDDEN // OUT_PROJ_N_CHUNK,
            name_hint="swa_out_proj",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            o0 = ob * OUT_PROJ_N_CHUNK
            a_chunk_0 = pl.slice(gated_attn_out, [BATCH_TILE, OUT_PROJ_K_CHUNK], [b0, 0])
            w_chunk_0 = pl.slice(wo, [OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK], [0, o0])
            o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
            for kb in pl.range(1, out_proj_k_blocks):
                k0 = kb * OUT_PROJ_K_CHUNK
                a_chunk = pl.slice(gated_attn_out, [BATCH_TILE, OUT_PROJ_K_CHUNK], [b0, k0])
                w_chunk = pl.slice(wo, [OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK], [k0, o0])
                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)
            resid = pl.cast(
                pl.slice(hidden_states, [BATCH_TILE, OUT_PROJ_N_CHUNK], [b0, o0]),
                target_type=pl.FP32,
            )
            resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

        # Post-attention zero-centered RMSNorm.
        post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_post_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden_blocks):
                post_sq_k0 = kb * K_CHUNK
                post_sq_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_sq_k0])
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(post_sq_chunk, post_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
            inv_rms_col = pl.reshape(inv_rms_s3, [BATCH_TILE, 1])
            for kb in pl.range(hidden_blocks):
                post_norm_k0 = kb * K_CHUNK
                post_norm_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_norm_k0])
                post_gamma_stored = pl.slice(
                    post_rms_weight, [1, K_CHUNK], [layer_idx, post_norm_k0],
                )
                post_normed = rmsnorm_zero_centered_row(
                    post_norm_chunk, inv_rms_col, post_gamma_stored,
                )
                normed_bf16 = pl.cast(post_normed, target_type=pl.BF16)
                post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, post_norm_k0])

        # Dense MLP: gate_proj / up_proj / SiLU * up (plain — SwigluStep
        # limit = 0 on layer 1) / down_proj.
        mlp_tile = pl.create_tensor([BATCH_TILE, INTERMEDIATE], dtype=pl.BF16)
        for ob in pl.spmd(
            INTERMEDIATE // MLP_OUT_CHUNK,
            name_hint="swa_gate_up_silu",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            mlp_o0 = ob * MLP_OUT_CHUNK
            post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
            wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, mlp_o0])
            wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, mlp_o0])
            gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
            up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
            for kb in pl.range(1, hidden_blocks):
                k0 = kb * K_CHUNK
                post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, mlp_o0])
                wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, mlp_o0])
                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
            mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, mlp_o0])

        # down_proj + residual (no GM round-trip; UP_DOWN-split mixed region).
        # INTERMEDIATE = 11264 / 256 = 44 K-blocks; HIDDEN = 4096 / 256 = 16
        # output blocks.
        mlp_k_blocks = INTERMEDIATE // MLP_OUT_CHUNK
        for dob in pl.spmd(
            HIDDEN // K_CHUNK,
            name_hint="swa_down_proj",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            d0 = dob * K_CHUNK
            mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
            w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
            for ob in pl.range(1, mlp_k_blocks):
                down_o0 = ob * MLP_OUT_CHUNK
                down_mlp_chunk_bf16 = pl.slice(
                    mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, down_o0],
                )
                w_down_chunk = pl.slice(
                    w_down, [MLP_OUT_CHUNK, K_CHUNK], [down_o0, d0],
                )
                down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)
            resid_chunk_fp32 = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])
            out_chunk = pl.add(down_acc, resid_chunk_fp32)
            out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [b0, d0])

    return out


# ============================================================================
# Torch reference for bit-pass-rate compare against the JIT kernel above.
#
# The reference mirrors every step of the kernel in plain torch — same tile
# arithmetic so BF16 round-off lines up — but expressed as straightforward
# matmuls / softmax. The KV-cache convention assumed by the kernel is:
#
#   - ``block_table[b, j]`` lists the physical block ids for batch ``b``.
#     Slots 0..eff_ctx_blocks-1 hold the sliding-window K/V; slot
#     ``slot_block`` (== (eff_ctx_len - 1) // BLOCK_SIZE) holds the current
#     decode row at offset ``slot - slot_block * BLOCK_SIZE``.
#   - ``slot_mapping[b]`` is the absolute slot index where this decode's
#     fresh K/V row is written (per-batch).
#   - ``seq_lens[b]`` may be larger than SLIDING_WINDOW; the kernel reads
#     ``eff_ctx_len = min(seq_lens[b], SLIDING_WINDOW)`` blocks.
# ============================================================================


def _torch_golden_swa(tensors):
    """PyTorch reference for ``decode_swa_layer`` (Phase 2b draft)."""
    import math

    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    g_proj = tensors["g_proj"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = hidden_states.shape[0]
    head_dim = HEAD_DIM
    num_kv_heads = NUM_KV_HEADS
    num_heads = NUM_HEADS
    q_per_kv = Q_PER_KV
    q_groups = Q_GROUPS
    half = ROTARY_HALF
    scale = 1.0 / math.sqrt(head_dim)
    eps = EPS

    # ---- zero-centered RMSNorm of the input. ----
    def _rmsnorm_zero_centered_torch(x, gamma_stored):
        gamma_eff = gamma_stored.float() + 1.0
        var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        return x.float() * torch.rsqrt(var + eps) * gamma_eff

    normed_bf16 = _rmsnorm_zero_centered_torch(hidden_states, input_rms_weight[0:1, :]).bfloat16()

    # ---- Q / K / V projections. ----
    q_proj = normed_bf16.float() @ wq.float()
    k_proj = normed_bf16.float() @ wk.float()
    v_proj = normed_bf16.float() @ wv.float()

    # ---- per-head zero-centered q_norm / k_norm. ----
    def _per_head_norm_torch(x_flat, num_h, gamma_stored):
        gamma_eff = gamma_stored.float() + 1.0
        x_h = x_flat.view(batch, num_h, head_dim).float()
        var = (x_h ** 2).mean(dim=-1, keepdim=True)
        return (x_h * torch.rsqrt(var + eps) * gamma_eff).view(batch, num_h * head_dim)

    q_proj_norm = _per_head_norm_torch(q_proj, num_heads, q_norm_weight[0:1, :])
    k_proj_norm = _per_head_norm_torch(k_proj, num_kv_heads, k_norm_weight[0:1, :])

    # ---- head-wise gate matmul (logits only; sigmoid applied per-head later). ----
    gate_logits = hidden_states.float() @ g_proj.float()  # [batch, num_heads]

    attn_out = torch.zeros(batch, num_heads * head_dim, dtype=torch.bfloat16)
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ

    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        eff_ctx_len = min(ctx_len, SLIDING_WINDOW)
        eff_ctx_blocks = (eff_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pos = ctx_len - 1

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        # ---- K RoPE + cache write at slot_mapping[b]. ----
        slot = int(slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE
        k_heads = k_proj_norm[b].view(num_kv_heads, head_dim).float()
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
            dim=-1,
        )
        for ki in range(num_kv_heads):
            cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
            k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_row, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        # ---- Q RoPE (per head). ----
        q_heads = q_proj_norm[b].view(num_heads, head_dim).float()
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
            dim=-1,
        )

        # ---- fa_fused: per Q-group online softmax over the SWA window. ----
        attn_row = torch.zeros(1, num_heads * head_dim, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(eff_ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = min(BLOCK_SIZE, eff_ctx_len - s0)
                    pbid = int(block_table[b * max_ctx_blocks + sb].item())
                    cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                    k_tile = k_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]
                    v_tile = v_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < BLOCK_SIZE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                ctx_flat_bf16 = ctx.reshape(1, -1).to(torch.bfloat16)
                attn_row[:, q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim] = ctx_flat_bf16

        attn_out[b : b + 1, :] = attn_row

    # ---- head-wise sigmoid gate on attn_out (per-head). ----
    gate = torch.sigmoid(gate_logits).unsqueeze(-1)  # [batch, num_heads, 1]
    gated = (
        attn_out.view(batch, num_heads, head_dim).float() * gate
    ).view(batch, num_heads * head_dim).to(torch.bfloat16)

    # ---- o_proj + residual. ----
    o_proj_out = gated.float() @ wo.float()
    resid1 = o_proj_out + hidden_states.float()

    # ---- post-attn zero-centered RMSNorm. ----
    var = (resid1 ** 2).mean(dim=-1, keepdim=True)
    post_gamma_eff = post_rms_weight[0:1, :].float() + 1.0
    post_normed_bf16 = (resid1 * torch.rsqrt(var + eps) * post_gamma_eff).bfloat16()

    # ---- dense MLP: SiLU(gate) * up, then down. ----
    gate_mat = post_normed_bf16.float() @ w_gate.float()
    up_mat = post_normed_bf16.float() @ w_up.float()
    mlp_bf16 = (gate_mat * torch.sigmoid(gate_mat) * up_mat).bfloat16()
    down = mlp_bf16.float() @ w_down.float()

    tensors["out"][:] = (down + resid1).bfloat16()


# ----------------------------------------------------------------------------
# Tensor specs / build harness for ``run_jit``.
# ----------------------------------------------------------------------------


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ_DEFAULT,
):
    """Tensor specs for ``decode_swa_layer``.

    Mirrors the in-tree dense-GQA reference's single-layer fixture: every
    layer-stacked weight is collapsed to one layer, the KV cache holds
    ``batch * MAX_BLOCKS_PER_SEQ * NUM_KV_HEADS * BLOCK_SIZE`` rows of
    HEAD_DIM columns. Seq lengths span both the in-window and over-window
    regimes so the sliding clamp is exercised.
    """
    import torch
    from golden import TensorSpec

    hidden_size = HIDDEN
    kv_hidden = KV_HIDDEN
    hidden_q = HIDDEN_Q
    inter = INTERMEDIATE
    head_dim = HEAD_DIM
    num_heads = NUM_HEADS
    num_kv_heads = NUM_KV_HEADS
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    cache_rows = num_blocks * num_kv_heads * BLOCK_SIZE
    synthetic_proj_scale = 0.5

    # Seq-len pattern covers (a) entirely inside the window, (b) exactly at
    # the window edge, and (c) past the window (so the clamp activates).
    # Pad the pattern up to ``batch`` and clamp to [1, max_seq].
    seq_len_pattern = torch.tensor(
        [9, 31, 62, SLIDING_WINDOW - 1, SLIDING_WINDOW, SLIDING_WINDOW + 1, max_seq, max_seq // 2],
        dtype=torch.int32,
    )
    repeat = (batch + seq_len_pattern.numel() - 1) // seq_len_pattern.numel()
    seq_lens_seed = seq_len_pattern.repeat(repeat)[:batch].clone()
    seq_lens_seed = torch.clamp(seq_lens_seed, min=1, max=max_seq)

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_input_rms_weight():
        return (torch.rand(1, hidden_size) - 0.5) * 0.1  # small zero-centered gamma

    def init_wq():
        return torch.rand(hidden_size, hidden_q) / hidden_size ** 0.5

    def init_wk():
        return torch.rand(hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_wv():
        return synthetic_proj_scale * torch.rand(hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_q_norm_weight():
        return (torch.rand(1, head_dim) - 0.5) * 0.1

    def init_k_norm_weight():
        return (torch.rand(1, head_dim) - 0.5) * 0.1

    def init_g_proj():
        return synthetic_proj_scale * (torch.rand(hidden_size, num_heads) - 0.5) / hidden_size ** 0.5

    def init_seq_lens():
        return seq_lens_seed.clone()

    def init_block_table():
        return torch.arange(num_blocks, dtype=torch.int32)

    def init_slot_mapping():
        # Place the decode K/V row inside the current window block. With the
        # SWA convention used by the draft, the "window" lives in blocks
        # 0..eff_ctx_blocks - 1 of each batch's block table; the current
        # decode token sits at the last visible slot.
        slots = torch.empty(batch, dtype=torch.int32)
        for b in range(batch):
            ctx_len = int(seq_lens_seed[b].item())
            eff_ctx_len = min(ctx_len, SLIDING_WINDOW)
            slot_pos = eff_ctx_len - 1
            logical_block = slot_pos // BLOCK_SIZE
            page_offset = slot_pos % BLOCK_SIZE
            phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
            slots[b] = phys_block * BLOCK_SIZE + page_offset
        return slots

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return synthetic_proj_scale * (torch.rand(cache_rows, head_dim) - 0.5)

    def init_wo():
        return synthetic_proj_scale * (torch.rand(hidden_q, hidden_size) - 0.5) / hidden_q ** 0.5

    def init_post_rms_weight():
        return (torch.rand(1, hidden_size) - 0.5) * 0.1

    def init_w_gate():
        return synthetic_proj_scale * (torch.rand(hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return synthetic_proj_scale * (torch.rand(hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return synthetic_proj_scale * (torch.rand(inter, hidden_size) - 0.5) / inter ** 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32,
                   init_value=init_input_rms_weight),
        TensorSpec("wq", [hidden_size, hidden_q], torch.bfloat16,
                   init_value=init_wq),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wk),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wv),
        TensorSpec("q_norm_weight", [1, head_dim], torch.float32,
                   init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [1, head_dim], torch.float32,
                   init_value=init_k_norm_weight),
        TensorSpec("g_proj", [hidden_size, num_heads], torch.bfloat16,
                   init_value=init_g_proj),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("block_table", [batch * MAX_BLOCKS_PER_SEQ], torch.int32,
                   init_value=init_block_table),
        TensorSpec("slot_mapping", [batch], torch.int32, init_value=init_slot_mapping),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32,
                   init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32,
                   init_value=init_rope_sin),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_v_cache),
        TensorSpec("wo", [hidden_q, hidden_size], torch.bfloat16,
                   init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32,
                   init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, inter], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, inter], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("w_down", [inter, hidden_size], torch.bfloat16,
                   init_value=init_w_down),
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2b draft — Step3p5 sliding-window single-layer decode "
            "(layer 1 contract). Compare against the torch golden at "
            "pass_rate >= 0.98."
        ),
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=BATCH)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ_DEFAULT)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=decode_swa_layer,
        specs=build_tensor_specs(batch=args.batch, max_seq=args.max_seq),
        golden_fn=_torch_golden_swa,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            # BF16 long-tail. Phase-2 acceptance: pass_rate >= 0.98.
            "out": ratio_reldiff(diff_thd=1e-2, pct_thd=2e-2, max_diff_hd=10),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
