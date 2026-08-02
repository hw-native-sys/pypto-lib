# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 Phase-2a single-layer decode draft — FULL-attention layer 0.

This file mirrors the the dense-GQA reference ``decode_layer.py`` fa_fused decode skeleton
and adds the four step3p5 increments required for layer 0:

1. **Zero-centred RMSNorm** (vllm ``GemmaRMSNorm``). The stored gamma is
   centred at 0; the effective per-channel scale is ``stored_gamma + 1.0``.
   Applied to the input RMSNorm, the post-attention RMSNorm, AND the
   per-head q_norm / k_norm. Centralized in the ``_zero_centered_rmsnorm_apply``
   helper (see TODO note: Phase 3 should hoist this into a shared
   ``models/step3p5/_ops.py`` so the SWA twin can reuse it bit-for-bit).
2. **Partial RoPE with rotary_dim = 64** (= ``HEAD_DIM * partial_rotary_factor``
   with the layer-0 factor of 0.5). The leading 64 lanes of each head split
   into 32 low + 32 high pairs and rotate; the trailing 64 lanes pass through
   unchanged. ``ROTARY_HALF_FULL = 32`` (see ``config.py``) is the half-dim of
   the rotation; ``ROTARY_DIM_FULL = 64`` is the full rotary slice.
3. **Per-head q_norm / k_norm with zero-centred [HEAD_DIM=128] gamma** applied
   AFTER QKV projection and BEFORE RoPE. The [HEAD_DIM]-length gamma is
   broadcast across every Q head (Q_PER_KV_FULL = 8 heads per KV head) and
   across the single K head per KV group.
4. **Head-wise attention gate** ``g_proj`` of shape ``[NUM_HEADS_FULL=64,
   HIDDEN]``. After fa_fused emits per-head attention output, the kernel
   computes ``gate_logits = current_hidden @ g_proj.T`` of shape
   ``[BATCH, 64]``, applies sigmoid, then multiplies each Q head's
   ``HEAD_DIM`` lanes by its scalar gate value before the o_proj matmul.

The MLP at layer 0 is a **dense SwiGLU** (no MoE; layers 3..44 are MoE).
``SWIGLU_LIMITS[0] = SWIGLU_LIMITS_SHARED[0] = 0.0`` in ``config.py`` confirms
the activation is plain ``silu(gate) * up`` — i.e. no ``SwigluStep`` clipping
at layer 0. ``INTERMEDIATE = 11264`` is the dense MLP hidden width.

For the rope cos/sin generation in the torch harness AND in the kernel's
input-gen path we apply **llama3 yarn scaling** (factor=2.0, low/high freq
factors 1.0/32.0, original max position 131072). Yarn scaling only applies
to full-attention layers in step3p5 (``yarn_only_types=["full_attention"]``)
so the SWA twin must NOT apply it. The kernel itself just consumes whatever
cos/sin tables it is handed; both kernel and torch golden read identical
tensors so the golden compare stays bit-tight.

Run:
    python single_layer_decode_full_draft.py -p a2a3sim

Target: ``pass_rate >= 0.98`` on BF16 (matches the the dense-GQA reference ``decode_fwd``
threshold).

TODO(phase3): Hoist ``_zero_centered_rmsnorm_apply`` and the head-wise gate
epilogue out of both single-layer drafts (full + SWA) into a shared
``models/step3p5/_ops.py`` module so ``attention_full.py`` / ``attention_swa.py``
can drop in the same building blocks. See ``MIGRATION_PLAN.md`` §五 Phase 3.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl

from config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    DOWN_MLP_CHUNK,
    DOWN_OUT_CHUNK,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN,
    KV_OUT_CHUNK,
    KV_PROJ_K_CHUNK,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ_DEFAULT,
    MLP_OUT_CHUNK,
    NUM_HEADS_FULL,
    NUM_KV_HEADS,
    OUT_PROJ_K_CHUNK,
    OUT_PROJ_N_CHUNK,
    Q_HEAD_BATCH_FULL,
    Q_HEAD_PAD_FULL,
    Q_OUT_CHUNK,
    Q_PER_KV_FULL,
    ROPE_SCALING,
    ROPE_SEQ_DYN,
    ROTARY_HALF_FULL,
    USER_BATCH_DYN,
)

# -----------------------------------------------------------------------------
# Local dynamic dims used only by this single-layer draft. Mirrors the
# the dense-GQA reference ``LAYER_HIDDEN_ROWS_DYN`` pattern but adds two new dims because
# step3p5's full-attention layer has DIFFERENT row counts on wq vs. wo, and
# the head-wise gate ``g_proj`` introduces a third per-layer row block:
#
#   wq rows         = LAYERS * HIDDEN            (LAYER_HIDDEN_ROWS_DYN)
#   wo rows         = LAYERS * HIDDEN_Q_FULL     (LAYER_QHIDDEN_ROWS_DYN)
#   g_proj rows     = LAYERS * NUM_HEADS_FULL    (LAYER_QGATE_ROWS_DYN)
#   w_down rows     = LAYERS * INTERMEDIATE      (LAYER_INTER_ROWS_DYN — shared with the dense-GQA reference convention)
#
# Phase 3 (``attention_full.py``) is expected to fold these into the
# generic per-layer dyn-dim table when both attention variants share a
# single decode_layer entry.
# -----------------------------------------------------------------------------
LAYER_QHIDDEN_ROWS_DYN = pl.dynamic("LAYER_QHIDDEN_ROWS_DYN")
LAYER_QGATE_ROWS_DYN = pl.dynamic("LAYER_QGATE_ROWS_DYN")

# -----------------------------------------------------------------------------
# Local compile-time constants.
# -----------------------------------------------------------------------------
# Half-dim of the rotary slice (config exposes ROTARY_HALF_FULL = 32).
# Full rotary slice width = ROTARY_DIM_FULL = 64 lanes per head; the trailing
# (HEAD_DIM - ROTARY_DIM_FULL) = 64 lanes pass through.
ROTARY_DIM_FULL = ROTARY_HALF_FULL * 2
ROTARY_PASS_FULL = HEAD_DIM - ROTARY_DIM_FULL
# Q-group split: with Q_PER_KV_FULL = 8 and Q_HEAD_BATCH_FULL = 8, each KV
# head feeds exactly one Q group of 8 heads. Mirrors the dense-GQA reference's Q_GROUPS = 1.
Q_GROUPS_FULL = Q_PER_KV_FULL // Q_HEAD_BATCH_FULL
TOTAL_Q_GROUPS_FULL = NUM_KV_HEADS * Q_GROUPS_FULL
# Layer-0 of step3p5 is dense, so this draft compiles a single-layer
# specialization (LAYER_DYN == 1, layer_idx == 0).

assert Q_PER_KV_FULL == Q_HEAD_BATCH_FULL, (
    f"Q_PER_KV_FULL ({Q_PER_KV_FULL}) must equal Q_HEAD_BATCH_FULL "
    f"({Q_HEAD_BATCH_FULL}) — qk_norm / rope_kv_cache assume one Q group "
    f"per KV head; the draft does not yet support Q_GROUPS_FULL > 1."
)
assert Q_HEAD_PAD_FULL % 4 == 0 and Q_HEAD_PAD_FULL // 2 >= Q_HEAD_BATCH_FULL, (
    f"Q_HEAD_PAD_FULL ({Q_HEAD_PAD_FULL}) must be a multiple of 4 with "
    f"Q_HEAD_PAD_FULL // 2 ({Q_HEAD_PAD_FULL // 2}) >= Q_HEAD_BATCH_FULL "
    f"({Q_HEAD_BATCH_FULL})  — see the dense-GQA tiling rationale."
)
assert TOTAL_Q_GROUPS_FULL % 2 == 0, (
    f"TOTAL_Q_GROUPS_FULL ({TOTAL_Q_GROUPS_FULL}) must be even (fa_fused "
    f"pairs Q groups via pl.pipeline(2, stage=2))."
)
assert (INTERMEDIATE // MLP_OUT_CHUNK) > 0
assert HIDDEN_Q_FULL == NUM_HEADS_FULL * HEAD_DIM


# =============================================================================
# RMSNorm note — zero-centred gamma.
# =============================================================================
# Step3p5 stores RMSNorm gammas centred at 0 (vllm ``GemmaRMSNorm``); the
# effective per-channel scale is ``stored_gamma + 1.0``. We inline the
# `(gamma + 1.0)` shift at each call site (per pypto frontend rules — bare
# Python helpers are not callable from @pl.jit / @pl.function bodies); the
# host loader still passes raw checkpoint weights unchanged.


# =============================================================================
# Inline kernel — full-attention decode body for a single step3p5 layer.
# =============================================================================
@pl.jit.inline
def single_layer_decode_full(
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM_FULL], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM_FULL], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    # Compile-time constants shared across the body. The fa_fused / out_proj /
    # gate_up / down_proj top-level pl.spmd dispatches MUST spell their block
    # counts inline at the call site (pl.spmd outlines its body to a top-level
    # function and SSA-verifies the count outside the jit-inlined scope), so
    # we keep these aliases as a reference only.
    decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
    qhidden_blocks = HIDDEN_Q_FULL // K_CHUNK
    head_dim_inv = HEAD_DIM_INV
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    bt_stride = pl.tensor.dim(block_table, 0) // user_batch
    batch_padded = BATCH

    # Per-layer base offsets into the layer-strided weight / kv-cache tensors.
    # For the single-layer draft layer_idx == 0; these still need to be
    # spelled correctly so the Phase-5 unified entry can drop in the same
    # inline body unchanged.
    layer_hidden_base = layer_idx * HIDDEN
    layer_qhidden_base = layer_idx * HIDDEN_Q_FULL
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cache_base = layer_idx * decode_layer_cache_rows

    # Intermediate tensors flowing between top-level pl.spmd dispatches.
    q_proj = pl.create_tensor([BATCH, HIDDEN_Q_FULL], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN_Q_FULL], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    # Bridge buffer between input RMSNorm and Q/K/V projection. Promoted to
    # top level so the downstream q_proj / k_proj / v_proj spmd dispatches
    # can slice it (same pattern as the dense-GQA reference).
    normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

    # =====================================================================
    # Scope 1: input RMSNorm (zero-centred) + Q/K/V projection + per-head
    # zero-centred q_norm / k_norm.
    # =====================================================================

    # Input RMSNorm with zero-centred gamma. Standard RMSNorm row reduction,
    # then the gamma broadcast goes through the zero-centred helper.
    for rms_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="input_rmsnorm_zc"):
        rms_b0 = rms_spmd_idx * BATCH_TILE
        partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(decode_scope1_hidden_blocks):
            sq_k0 = kb * INPUT_PROJ_K_CHUNK
            sq_chunk = pl.cast(
                pl.slice(
                    current_hidden,
                    [BATCH_TILE, INPUT_PROJ_K_CHUNK],
                    [rms_b0, sq_k0],
                ),
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
                pl.slice(
                    current_hidden,
                    [BATCH_TILE, INPUT_PROJ_K_CHUNK],
                    [rms_b0, norm_k0],
                ),
                target_type=pl.FP32,
            )
            gamma = pl.slice(input_rms_weight, [1, INPUT_PROJ_K_CHUNK], [layer_idx, norm_k0])
            scaled = pl.row_expand_mul(norm_chunk, inv_rms)
            # Zero-centred broadcast multiply: scaled * (gamma + 1.0).
            normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
            normed_all = pl.assemble(
                normed_all,
                pl.cast(normed, target_type=pl.BF16),
                [rms_b0, norm_k0],
            )

    # Q projection (HIDDEN_Q_FULL = 8192 wide). Same peeled-first + matmul_acc
    # K-loop pattern as the dense-GQA reference, just with a wider output dim.
    for q_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (HIDDEN_Q_FULL // Q_OUT_CHUNK),
        name_hint="q_proj_full",
    ):
        q_b_idx = q_spmd_idx // (HIDDEN_Q_FULL // Q_OUT_CHUNK)
        q_ob = q_spmd_idx % (HIDDEN_Q_FULL // Q_OUT_CHUNK)
        q_b0 = q_b_idx * BATCH_TILE
        q_o0 = q_ob * Q_OUT_CHUNK

        q_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, 0])
        q_tile_b_0 = pl.slice(wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, q_o0])
        q_acc = pl.matmul(q_tile_a_0, q_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            q_k0 = kb * INPUT_PROJ_K_CHUNK
            q_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, q_k0])
            q_tile_b = pl.slice(wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + q_k0, q_o0])
            q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
        q_proj = pl.assemble(q_proj, q_acc, [q_b0, q_o0])

    # K projection.
    for k_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (KV_HIDDEN // KV_OUT_CHUNK),
        name_hint="k_proj_full",
    ):
        k_b_idx = k_spmd_idx // (KV_HIDDEN // KV_OUT_CHUNK)
        k_ob = k_spmd_idx % (KV_HIDDEN // KV_OUT_CHUNK)
        k_b0 = k_b_idx * BATCH_TILE
        k_o0 = k_ob * KV_OUT_CHUNK

        k_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [k_b0, 0])
        k_tile_b_0 = pl.slice(wk, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base, k_o0])
        k_acc = pl.matmul(k_tile_a_0, k_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            k_k0 = kb * INPUT_PROJ_K_CHUNK
            k_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [k_b0, k_k0])
            k_tile_b = pl.slice(wk, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k_k0, k_o0])
            k_acc = pl.matmul_acc(k_acc, k_tile_a, k_tile_b)
        k_proj = pl.assemble(k_proj, k_acc, [k_b0, k_o0])

    # V projection.
    for v_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (KV_HIDDEN // KV_OUT_CHUNK),
        name_hint="v_proj_full",
    ):
        v_b_idx = v_spmd_idx // (KV_HIDDEN // KV_OUT_CHUNK)
        v_ob = v_spmd_idx % (KV_HIDDEN // KV_OUT_CHUNK)
        v_b0 = v_b_idx * BATCH_TILE
        v_o0 = v_ob * KV_OUT_CHUNK

        v_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [v_b0, 0])
        v_tile_b_0 = pl.slice(wv, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base, v_o0])
        v_acc = pl.matmul(v_tile_a_0, v_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            v_k0 = kb * INPUT_PROJ_K_CHUNK
            v_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [v_b0, v_k0])
            v_tile_b = pl.slice(wv, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + v_k0, v_o0])
            v_acc = pl.matmul_acc(v_acc, v_tile_a, v_tile_b)
        v_proj = pl.assemble(v_proj, v_acc, [v_b0, v_o0])

    # Per-head zero-centred q_norm / k_norm — one KV head per spmd block.
    # Each block normalizes the Q_HEAD_BATCH_FULL = 8 Q heads tied to its KV
    # head plus the single K head. The gamma is broadcast across all heads
    # in the block via the standard col_expand_mul, and the +1 zero-centring
    # is folded in by ``_zero_centered_rmsnorm_apply``.
    for qkn_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * NUM_KV_HEADS,
        name_hint="qk_norm_zc",
    ):
        qkn_b_idx = qkn_spmd_idx // NUM_KV_HEADS
        qkn_h = qkn_spmd_idx % NUM_KV_HEADS
        qkn_b0 = qkn_b_idx * BATCH_TILE

        qkn_q0 = qkn_h * Q_PER_KV_FULL * HEAD_DIM
        q_chunk = pl.reshape(
            pl.slice(
                q_proj,
                [BATCH_TILE, Q_HEAD_BATCH_FULL * HEAD_DIM],
                [qkn_b0, qkn_q0],
            ),
            [BATCH_TILE * Q_HEAD_BATCH_FULL, HEAD_DIM],
        )
        q_sq_sum = pl.row_sum(pl.mul(q_chunk, q_chunk))
        q_inv_rms = pl.rsqrt(pl.add(pl.mul(q_sq_sum, head_dim_inv), EPS))
        q_scaled = pl.row_expand_mul(q_chunk, q_inv_rms)
        q_chunk_norm = pl.col_expand_mul(
            q_scaled,
            pl.add(pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0]), 1.0),
        )
        q_chunk_norm_flat = pl.reshape(
            q_chunk_norm, [BATCH_TILE, Q_HEAD_BATCH_FULL * HEAD_DIM],
        )
        q_proj_norm = pl.assemble(q_proj_norm, q_chunk_norm_flat, [qkn_b0, qkn_q0])

        qkn_k0 = qkn_h * HEAD_DIM
        k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [qkn_b0, qkn_k0])
        k_sq_sum = pl.row_sum(pl.mul(k_chunk, k_chunk))
        k_inv_rms = pl.rsqrt(pl.add(pl.mul(k_sq_sum, head_dim_inv), EPS))
        k_scaled = pl.row_expand_mul(k_chunk, k_inv_rms)
        k_chunk_norm = pl.col_expand_mul(
            k_scaled,
            pl.add(pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0]), 1.0),
        )
        k_proj_norm = pl.assemble(k_proj_norm, k_chunk_norm, [qkn_b0, qkn_k0])

    # =====================================================================
    # Scope 2: partial RoPE + paged KV cache write + flash decode attention.
    # =====================================================================
    # ``attn_out`` holds the per-batch per-head context after fa_fused; we
    # then apply the head-wise gate in a follow-on spmd before o_proj.
    attn_out = pl.create_tensor([BATCH, HIDDEN_Q_FULL], dtype=pl.BF16)
    all_q_padded = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS_FULL * Q_HEAD_PAD_FULL, HEAD_DIM], dtype=pl.BF16,
    )

    # Per-batch rope_kv_cache. Partial-RoPE-aware: only the leading
    # ROTARY_DIM_FULL = 64 lanes of each head rotate; the trailing
    # ROTARY_PASS_FULL = 64 lanes are written through unchanged. V is never
    # rotated (RoPE applies to Q/K only); v_cache writes are identical to
    # the the dense-GQA reference form.
    for b in pl.parallel(user_batch):
        ctx_len = pl.tensor.read(seq_lens, [b])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE
        cos_row = pl.slice(rope_cos, [1, ROTARY_DIM_FULL], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, ROTARY_DIM_FULL], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, ROTARY_HALF_FULL], [0, 0])
        cos_hi = pl.slice(cos_row, [1, ROTARY_HALF_FULL], [0, ROTARY_HALF_FULL])
        sin_lo = pl.slice(sin_row, [1, ROTARY_HALF_FULL], [0, 0])
        sin_hi = pl.slice(sin_row, [1, ROTARY_HALF_FULL], [0, ROTARY_HALF_FULL])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache_full"):
            for ki in pl.range(NUM_KV_HEADS):
                kv_col = ki * HEAD_DIM
                cache_row = (
                    layer_cache_base
                    + (slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE
                    + slot_offset
                )
                # K partial RoPE: rotate lanes [0:32] + [32:64], pass [64:128].
                k_lo = pl.slice(k_proj_norm, [1, ROTARY_HALF_FULL], [b, kv_col])
                k_hi = pl.slice(
                    k_proj_norm, [1, ROTARY_HALF_FULL], [b, kv_col + ROTARY_HALF_FULL],
                )
                k_pass = pl.slice(
                    k_proj_norm, [1, ROTARY_PASS_FULL], [b, kv_col + ROTARY_DIM_FULL],
                )
                k_rot_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                k_rot_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                k_cache = pl.assemble(
                    k_cache,
                    pl.cast(k_rot_lo, target_type=pl.BF16),
                    [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache,
                    pl.cast(k_rot_hi, target_type=pl.BF16),
                    [cache_row, ROTARY_HALF_FULL],
                )
                # Pass-through tail [64:128] — k_proj_norm is FP32, cast to BF16.
                k_cache = pl.assemble(
                    k_cache,
                    pl.cast(k_pass, target_type=pl.BF16),
                    [cache_row, ROTARY_DIM_FULL],
                )
                # V is written as-is (no RoPE).
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(
                        pl.slice(v_proj, [1, HEAD_DIM], [b, kv_col]),
                        target_type=pl.BF16,
                    ),
                    [cache_row, 0],
                )

                # Q partial RoPE for the Q_HEAD_BATCH_FULL = 8 heads tied to
                # this KV head. Same lo/hi/pass-through split as K.
                q_base = ki * Q_PER_KV_FULL
                q_block = pl.reshape(
                    pl.slice(
                        q_proj_norm,
                        [1, Q_HEAD_BATCH_FULL * HEAD_DIM],
                        [b, q_base * HEAD_DIM],
                    ),
                    [Q_HEAD_BATCH_FULL, HEAD_DIM],
                )
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH_FULL, ROTARY_HALF_FULL], [0, 0])
                q_hi = pl.slice(
                    q_block,
                    [Q_HEAD_BATCH_FULL, ROTARY_HALF_FULL],
                    [0, ROTARY_HALF_FULL],
                )
                q_pass = pl.slice(
                    q_block,
                    [Q_HEAD_BATCH_FULL, ROTARY_PASS_FULL],
                    [0, ROTARY_DIM_FULL],
                )
                q_rot_lo_bf16 = pl.cast(
                    pl.sub(
                        pl.col_expand_mul(q_lo, cos_lo),
                        pl.col_expand_mul(q_hi, sin_lo),
                    ),
                    target_type=pl.BF16,
                )
                q_rot_hi_bf16 = pl.cast(
                    pl.add(
                        pl.col_expand_mul(q_hi, cos_hi),
                        pl.col_expand_mul(q_lo, sin_hi),
                    ),
                    target_type=pl.BF16,
                )
                q_pass_bf16 = pl.cast(q_pass, target_type=pl.BF16)

                pad_row_base = (
                    b * TOTAL_Q_GROUPS_FULL * Q_HEAD_PAD_FULL + ki * Q_HEAD_PAD_FULL
                )
                all_q_padded = pl.assemble(
                    all_q_padded, q_rot_lo_bf16, [pad_row_base, 0],
                )
                all_q_padded = pl.assemble(
                    all_q_padded, q_rot_hi_bf16, [pad_row_base, ROTARY_HALF_FULL],
                )
                all_q_padded = pl.assemble(
                    all_q_padded, q_pass_bf16, [pad_row_base, ROTARY_DIM_FULL],
                )
                # Zero-pad the tail rows from Q_HEAD_BATCH_FULL up to Q_HEAD_PAD_FULL.
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(
                        pl.full(
                            [Q_HEAD_PAD_FULL - Q_HEAD_BATCH_FULL, HEAD_DIM],
                            dtype=pl.FP32,
                            value=0.0,
                        ),
                        target_type=pl.BF16,
                    ),
                    [pad_row_base + Q_HEAD_BATCH_FULL, 0],
                )

    # fa_fused — same mixed cube+vec body as the dense-GQA reference, parameterized for
    # full-attention shape: BATCH * (TOTAL_Q_GROUPS_FULL // 2) = 64 lanes
    # with Q_HEAD_PAD_FULL = 16, Q_HEAD_BATCH_FULL = 8, Q_PER_KV_FULL = 8.
    # The online-softmax accumulators (mi/li/oi) stay in UB across sb, seeded
    # so sb=0's recurrence reduces to the seed case.
    for fa_spmd_idx in pl.spmd(
        BATCH * (TOTAL_Q_GROUPS_FULL // 2),
        name_hint="fa_fused_full",
    ):
        fa_b = fa_spmd_idx // (TOTAL_Q_GROUPS_FULL // 2)
        fa_g2 = fa_spmd_idx % (TOTAL_Q_GROUPS_FULL // 2)
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_ctx_blocks = (fa_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        fa_block_table_base = fa_b_safe * bt_stride

        for gp in pl.pipeline(2, stage=2):
            gi = fa_g2 * 2 + gp
            kvh = gi // Q_GROUPS_FULL
            qg = gi - kvh * Q_GROUPS_FULL
            q_base = kvh * Q_PER_KV_FULL + qg * Q_HEAD_BATCH_FULL
            q_padded_row = (
                fa_b * TOTAL_Q_GROUPS_FULL * Q_HEAD_PAD_FULL + gi * Q_HEAD_PAD_FULL
            )
            q_padded = pl.slice(
                all_q_padded, [Q_HEAD_PAD_FULL, HEAD_DIM], [q_padded_row, 0],
            )

            mi_flat = pl.full([1, Q_HEAD_PAD_FULL], dtype=pl.FP32, value=-3.0e38)
            mi = pl.reshape(mi_flat, [Q_HEAD_PAD_FULL, 1])
            li_flat = pl.full([1, Q_HEAD_PAD_FULL], dtype=pl.FP32, value=0.0)
            li = pl.reshape(li_flat, [Q_HEAD_PAD_FULL, 1])
            oi = pl.full([Q_HEAD_PAD_FULL, HEAD_DIM], dtype=pl.FP32, value=0.0)

            for sb in pl.range(fa_ctx_blocks):
                s0 = sb * BLOCK_SIZE
                valid_len = pl.min(BLOCK_SIZE, fa_ctx_len - s0)
                fa_block_table_idx = fa_block_table_base + sb
                fa_pbid = pl.cast(
                    pl.tensor.read(block_table, [fa_block_table_idx]), pl.INDEX,
                )
                fa_cache_row = (
                    layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                )

                k_tile = k_cache[fa_cache_row : fa_cache_row + BLOCK_SIZE, :]
                raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                scores_scaled = pl.mul(raw_scores, decode_attn_scale)
                # Q_HEAD_PAD_FULL // 2 = 8 == Q_HEAD_BATCH_FULL (tight); even
                # row count keeps ptoas's static-even constraint happy.
                scores_valid = pl.set_validshape(
                    scores_scaled, Q_HEAD_PAD_FULL // 2, valid_len,
                )
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
            ctx_valid = ctx[0:Q_HEAD_BATCH_FULL, :]
            ctx_flat_bf16 = pl.cast(
                pl.reshape(ctx_valid, [1, Q_HEAD_BATCH_FULL * HEAD_DIM]),
                target_type=pl.BF16,
            )
            attn_out = pl.assemble(attn_out, ctx_flat_bf16, [fa_b, q_base * HEAD_DIM])

    # =====================================================================
    # Scope 2.5: head-wise attention gate.
    # =====================================================================
    # gate_logits = current_hidden @ w_g — shape [BATCH, NUM_HEADS_FULL].
    # w_g is stored as [HIDDEN, NUM_HEADS_FULL] (host transposes the
    # checkpoint's [NUM_HEADS_FULL, HIDDEN] layout). NUM_HEADS_FULL = 64 is
    # tiny on the N axis so we keep the entire output in a single matmul
    # tile per (batch_tile, kv_head_chunk) — no N tiling required.
    #
    # gate = sigmoid(gate_logits) is applied as the vec epilogue right
    # after the cube matmul reduction, then the gated attn_out is written
    # head-by-head (each head's HEAD_DIM lanes multiplied by the same
    # per-batch scalar gate value).
    #
    # NOTE for Phase 3: This kernel uses current_hidden directly (the layer
    # input, BEFORE the input RMSNorm), matching vllm's
    # ``gate_logits, _ = self.g_proj(hidden_states)`` semantics. Do not
    # substitute normed_all here — that would change the math.
    gate_logits = pl.create_tensor([BATCH, NUM_HEADS_FULL], dtype=pl.FP32)
    for gp_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="g_proj"):
        gp_b0 = gp_spmd_idx * BATCH_TILE

        # current_hidden is BF16. Cast happens implicitly via the cube load.
        gp_a_0 = pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [gp_b0, 0])
        gp_b_0 = pl.slice(
            w_g, [INPUT_PROJ_K_CHUNK, NUM_HEADS_FULL], [layer_hidden_base, 0],
        )
        gp_acc = pl.matmul(gp_a_0, gp_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            gp_k0 = kb * INPUT_PROJ_K_CHUNK
            gp_a = pl.slice(
                current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [gp_b0, gp_k0],
            )
            gp_b = pl.slice(
                w_g,
                [INPUT_PROJ_K_CHUNK, NUM_HEADS_FULL],
                [layer_hidden_base + gp_k0, 0],
            )
            gp_acc = pl.matmul_acc(gp_acc, gp_a, gp_b)

        # sigmoid epilogue: 1 / (1 + exp(-x))
        gate_sig = pl.recip(pl.add(pl.exp(pl.neg(gp_acc)), 1.0))
        gate_logits = pl.assemble(gate_logits, gate_sig, [gp_b0, 0])

    # Apply per-head gate to attn_out. Write into a fresh attn_out_gated
    # so o_proj reads from a clean dependency chain.
    attn_out_gated = pl.create_tensor([BATCH, HIDDEN_Q_FULL], dtype=pl.BF16)
    for hg_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * NUM_HEADS_FULL, name_hint="head_gate_apply",
    ):
        hg_b_idx = hg_spmd_idx // NUM_HEADS_FULL
        hg_h = hg_spmd_idx % NUM_HEADS_FULL
        hg_b0 = hg_b_idx * BATCH_TILE
        hg_col = hg_h * HEAD_DIM

        head_slice_bf16 = pl.slice(attn_out, [BATCH_TILE, HEAD_DIM], [hg_b0, hg_col])
        head_slice_fp32 = pl.cast(head_slice_bf16, target_type=pl.FP32)
        gate_col = pl.slice(gate_logits, [BATCH_TILE, 1], [hg_b0, hg_h])
        gated = pl.row_expand_mul(head_slice_fp32, gate_col)
        attn_out_gated = pl.assemble(
            attn_out_gated, pl.cast(gated, target_type=pl.BF16), [hg_b0, hg_col],
        )

    # =====================================================================
    # Scope 3: o_proj + residual + post RMSNorm (zero-centred) + dense MLP
    # (plain SiLU) + residual.
    # =====================================================================
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.FP32)

        # o_proj: attn_out_gated [BATCH, HIDDEN_Q_FULL] @ wo [HIDDEN_Q_FULL, HIDDEN].
        # K-loop reduction over HIDDEN_Q_FULL // K_CHUNK = 32 blocks.
        for ob in pl.spmd(
            HIDDEN // OUT_PROJ_N_CHUNK,
            name_hint="out_proj_full",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            o0 = ob * OUT_PROJ_N_CHUNK

            a_chunk_0 = pl.slice(attn_out_gated, [BATCH_TILE, K_CHUNK], [b0, 0])
            w_chunk_0 = pl.slice(
                wo, [K_CHUNK, OUT_PROJ_N_CHUNK], [layer_qhidden_base, o0],
            )
            o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
            for kb in pl.range(1, qhidden_blocks):
                k0 = kb * K_CHUNK
                a_chunk = pl.slice(attn_out_gated, [BATCH_TILE, K_CHUNK], [b0, k0])
                w_chunk = pl.slice(
                    wo,
                    [K_CHUNK, OUT_PROJ_N_CHUNK],
                    [layer_qhidden_base + k0, o0],
                )
                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

            resid = pl.cast(
                pl.slice(current_hidden, [BATCH_TILE, OUT_PROJ_N_CHUNK], [b0, o0]),
                target_type=pl.FP32,
            )
            resid_sum = pl.add(o_acc, resid)
            resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

        # Post-attention RMSNorm with zero-centred gamma.
        post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)
        hidden_blocks = HIDDEN // K_CHUNK
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm_zc"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden_blocks):
                post_sq_k0 = kb * K_CHUNK
                post_sq_chunk = pl.slice(
                    resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_sq_k0],
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(
                        pl.row_sum(pl.mul(post_sq_chunk, post_sq_chunk)),
                        [1, BATCH_TILE],
                    ),
                )
            inv_rms_s3 = pl.recip(
                pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
            )

            for kb in pl.range(hidden_blocks):
                post_norm_k0 = kb * K_CHUNK
                post_norm_chunk = pl.slice(
                    resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_norm_k0],
                )
                post_gamma = pl.slice(
                    post_rms_weight, [1, K_CHUNK], [layer_idx, post_norm_k0],
                )
                post_scaled = pl.row_expand_mul(
                    post_norm_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1]),
                )
                post_normed = pl.col_expand_mul(post_scaled, pl.add(post_gamma, 1.0))
                normed_bf16 = pl.cast(post_normed, target_type=pl.BF16)
                post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, post_norm_k0])

        # Dense MLP: fused gate_up + plain SiLU (no SwigluStep at layer 0).
        # ``SWIGLU_LIMITS[0] == 0.0`` confirms the activation here is the
        # standard ``silu(gate) * up``; the SwigluStep clipping limits apply
        # only at routed-MoE layers 43/44 and at the shared-expert path of
        # layer 44 (see config.py).
        mlp_tile = pl.create_tensor([BATCH_TILE, INTERMEDIATE], dtype=pl.BF16)
        for ob in pl.spmd(
            INTERMEDIATE // MLP_OUT_CHUNK,
            name_hint="gate_up_silu_full",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            mlp_o0 = ob * MLP_OUT_CHUNK

            post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
            wg_0 = pl.slice(
                w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0],
            )
            wu_0 = pl.slice(
                w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0],
            )
            gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
            up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
            for kb in pl.range(1, hidden_blocks):
                k0 = kb * K_CHUNK
                post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                wg = pl.slice(
                    w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + k0, mlp_o0],
                )
                wu = pl.slice(
                    w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + k0, mlp_o0],
                )
                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
            mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, mlp_o0])

        # Down projection + final residual.
        for dob in pl.spmd(
            HIDDEN // K_CHUNK,
            name_hint="down_proj_full",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            d0 = dob * K_CHUNK

            mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
            w_down_chunk_0 = pl.slice(
                w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base, d0],
            )
            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
            for ob in pl.range(1, INTERMEDIATE // MLP_OUT_CHUNK):
                down_o0 = ob * MLP_OUT_CHUNK
                down_mlp_chunk_bf16 = pl.slice(
                    mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, down_o0],
                )
                w_down_chunk = pl.slice(
                    w_down,
                    [MLP_OUT_CHUNK, K_CHUNK],
                    [layer_inter_base + down_o0, d0],
                )
                down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)

            resid_chunk_fp32 = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])
            out_chunk = pl.add(down_acc, resid_chunk_fp32)
            out_chunk_cast = pl.cast(out_chunk, target_type=pl.BF16)
            next_hidden = pl.assemble(next_hidden, out_chunk_cast, [b0, d0])

    return next_hidden


# =============================================================================
# JIT entry — compiles the layer-0 specialization (LAYER_DYN == 1, layer_idx == 0).
# The returned tensor is the post-layer hidden state (no LM head; that
# arrives at Phase 5 in ``decode_fwd.py`` / ``rms_lm_head.py``).
# =============================================================================
@pl.jit
def test_single_layer_decode_full(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN_Q_FULL], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM_FULL], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM_FULL], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[HIDDEN_Q_FULL, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[HIDDEN, NUM_HEADS_FULL], pl.BF16],
    post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16]],
) -> pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16]:
    hidden_states.bind_dynamic(0, USER_BATCH_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    slot_mapping.bind_dynamic(0, USER_BATCH_DYN)
    out.bind_dynamic(0, USER_BATCH_DYN)
    block_table.bind_dynamic(0, BLOCK_TABLE_FLAT_DYN)
    rope_cos.bind_dynamic(0, ROPE_SEQ_DYN)
    rope_sin.bind_dynamic(0, ROPE_SEQ_DYN)
    k_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)

    user_batch = pl.tensor.dim(hidden_states, 0)
    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden_full"):
            for kb in pl.range(HIDDEN // K_CHUNK):
                copy_k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                current_hidden = pl.assemble(
                    current_hidden, hidden_chunk, [b0, copy_k0],
                )

    next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    current_hidden = single_layer_decode_full(
        current_hidden,
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
        w_g,
        post_rms_weight,
        w_gate,
        w_up,
        w_down,
        next_hidden,
        0,
    )

    # Trim the padded BATCH rows back to the user-visible batch on the way
    # out. ``current_hidden`` is [BATCH, HIDDEN] BF16; ``out`` is
    # [user_batch, HIDDEN] BF16. The valid_shape on the slice makes the
    # write skip rows past user_batch.
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_out_full"):
            for kb in pl.range(HIDDEN // K_CHUNK):
                out_k0 = kb * K_CHUNK
                out_chunk = pl.slice(
                    current_hidden,
                    [BATCH_TILE, K_CHUNK],
                    [b0, out_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                out = pl.assemble(out, out_chunk, [b0, out_k0])
    return out


# =============================================================================
# Torch helpers — llama3 yarn-scaled rope cos/sin tables.
# =============================================================================
def _llama3_yarn_inv_freq(rotary_dim: int, base: float, scaling: dict):
    """Compute llama3-yarn-scaled inv_freq for the full-attention layer.

    Mirrors vllm's ``_compute_llama3_parameters``. The freq spectrum is
    split into three regions by ``low_freq_wavelen`` and
    ``high_freq_wavelen``:
      - wavelen < high_freq_wavelen: no scaling (preserve high freqs).
      - wavelen > low_freq_wavelen:  scale down by ``factor`` (compress
        the low-freq tail).
      - in-between:                  smooth interpolation.

    Args:
        rotary_dim: width of the rope rotation slice (= ``HEAD_DIM * partial``).
        base: layer rope theta (5_000_000.0 for step3p5 full attention).
        scaling: ``ROPE_SCALING`` dict from ``config.py``.

    Returns:
        FP32 tensor of length ``rotary_dim // 2``.
    """
    import math

    import torch

    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_context_length = scaling["original_max_position_embeddings"]

    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )

    low_freq_wavelen = old_context_length / low_freq_factor
    high_freq_wavelen = old_context_length / high_freq_factor
    wavelen = 2.0 * math.pi / inv_freq

    smooth_factor = (
        (old_context_length / wavelen - low_freq_factor)
        / (high_freq_factor - low_freq_factor)
    )
    smoothed_inv_freq = (
        (1.0 - smooth_factor) * inv_freq / factor + smooth_factor * inv_freq
    )

    inv_freq_scaled = torch.where(
        wavelen < high_freq_wavelen,
        inv_freq,
        torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / factor,
            smoothed_inv_freq,
        ),
    )
    return inv_freq_scaled


def _build_rope_tables_full(max_seq: int, rotary_dim: int):
    """Build cos/sin tables for the step3p5 layer-0 full-attention path.

    Shape: ``[max_seq, rotary_dim]`` FP32. The leading ``rotary_dim // 2``
    columns are the low half (paired with the trailing ``rotary_dim // 2``
    columns by the partial RoPE rotation in the kernel). vllm packs the
    same per-position frequency value into both halves (``cat([cos, cos])``,
    ``cat([sin, sin])``).
    """
    import torch

    base = 5_000_000.0  # layer-0 rope_theta (full-attention layers)
    inv_freq = _llama3_yarn_inv_freq(rotary_dim, base, ROPE_SCALING)
    positions = torch.arange(max_seq, dtype=torch.float32).unsqueeze(-1)
    angles = positions * inv_freq.unsqueeze(0)  # [max_seq, rotary_dim // 2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos_full = torch.cat([cos, cos], dim=-1)  # [max_seq, rotary_dim]
    sin_full = torch.cat([sin, sin], dim=-1)
    return cos_full, sin_full


# =============================================================================
# Tensor specs + torch golden for the bit-pass-rate compare.
# =============================================================================
def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ_DEFAULT,
    use_max_seq: bool = False,
):
    import sys
    from pathlib import Path

    import torch

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from golden import TensorSpec

    hidden = HIDDEN
    hidden_q = HIDDEN_Q_FULL
    kv_hidden = KV_HIDDEN
    inter = INTERMEDIATE
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    cache_rows = num_blocks * NUM_KV_HEADS * BLOCK_SIZE
    synthetic_proj_scale = 0.5
    head_dim = HEAD_DIM
    num_heads = NUM_HEADS_FULL
    rotary_dim = ROTARY_DIM_FULL

    if use_max_seq:
        seq_lens_seed = torch.full((batch,), max_seq, dtype=torch.int32)
    else:
        seq_lens_seed = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    rope_cos_table, rope_sin_table = _build_rope_tables_full(max_seq, rotary_dim)

    def init_hidden_states():
        return torch.rand(batch, hidden) - 0.5

    def init_rms_weight():
        # Zero-centred: the stored gamma is small (centred at 0). Sampling
        # from rand-0.5 gives [-0.5, 0.5], which after the +1 shift becomes
        # the effective [0.5, 1.5] gamma range used by the kernel — close
        # enough to a real checkpoint distribution to stress all paths.
        return torch.rand(1, hidden) - 0.5

    def init_wq():
        return torch.rand(hidden, hidden_q) / hidden ** 0.5

    def init_wk():
        return torch.rand(hidden, kv_hidden) / hidden ** 0.5

    def init_wv():
        return synthetic_proj_scale * torch.rand(hidden, kv_hidden) / hidden ** 0.5

    def init_q_norm_weight():
        return torch.rand(1, head_dim) - 0.5

    def init_k_norm_weight():
        return torch.rand(1, head_dim) - 0.5

    def init_seq_lens():
        return seq_lens_seed.clone()

    def init_block_table():
        return torch.arange(num_blocks, dtype=torch.int32)

    def init_slot_mapping():
        slots = torch.empty(batch, dtype=torch.int32)
        for b in range(batch):
            pos = int(seq_lens_seed[b].item()) - 1
            logical_block = pos // BLOCK_SIZE
            page_offset = pos % BLOCK_SIZE
            phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
            slots[b] = phys_block * BLOCK_SIZE + page_offset
        return slots

    def init_rope_cos():
        return rope_cos_table.clone()

    def init_rope_sin():
        return rope_sin_table.clone()

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return synthetic_proj_scale * (torch.rand(cache_rows, head_dim) - 0.5)

    def init_wo():
        return (
            synthetic_proj_scale
            * (torch.rand(hidden_q, hidden) - 0.5)
            / hidden_q ** 0.5
        )

    def init_w_g():
        # g_proj produces gate logits whose sigmoid feeds the head-wise
        # gate. A small synthetic scale keeps the logits in a sane sigmoid
        # range so the multiplicative gate is well-conditioned.
        return synthetic_proj_scale * (torch.rand(hidden, num_heads) - 0.5) / hidden ** 0.5

    def init_post_rms_weight():
        return torch.rand(1, hidden) - 0.5

    def init_w_gate():
        return (
            synthetic_proj_scale * (torch.rand(hidden, inter) - 0.5) / hidden ** 0.5
        )

    def init_w_up():
        return (
            synthetic_proj_scale * (torch.rand(hidden, inter) - 0.5) / hidden ** 0.5
        )

    def init_w_down():
        return (
            synthetic_proj_scale * (torch.rand(inter, hidden) - 0.5) / inter ** 0.5
        )

    return [
        TensorSpec("hidden_states", [batch, hidden], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden], torch.float32,
                   init_value=init_rms_weight),
        TensorSpec("wq", [hidden, hidden_q], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [hidden, kv_hidden], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [hidden, kv_hidden], torch.bfloat16, init_value=init_wv),
        TensorSpec("q_norm_weight", [1, head_dim], torch.float32,
                   init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [1, head_dim], torch.float32,
                   init_value=init_k_norm_weight),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("block_table", [batch * MAX_BLOCKS_PER_SEQ], torch.int32,
                   init_value=init_block_table),
        TensorSpec("slot_mapping", [batch], torch.int32, init_value=init_slot_mapping),
        TensorSpec("rope_cos", [max_seq, rotary_dim], torch.float32,
                   init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, rotary_dim], torch.float32,
                   init_value=init_rope_sin),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_v_cache),
        TensorSpec("wo", [hidden_q, hidden], torch.bfloat16, init_value=init_wo),
        TensorSpec("w_g", [hidden, num_heads], torch.bfloat16, init_value=init_w_g),
        TensorSpec("post_rms_weight", [1, hidden], torch.float32,
                   init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden, inter], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden, inter], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [inter, hidden], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [batch, hidden], torch.bfloat16, is_output=True),
    ]


def golden_single_layer_decode_full(tensors):
    """PyTorch reference for ``test_single_layer_decode_full``.

    Mirrors the step3p5 layer-0 math end-to-end:
      1. input RMSNorm with zero-centred gamma (gamma + 1.0)
      2. Q/K/V projection
      3. per-head q_norm / k_norm with zero-centred gamma
      4. partial RoPE (rotary_dim = 64, leading 64 lanes rotate, trailing
         64 lanes pass through)
      5. KV cache write
      6. paged GQA online-softmax flash attention (same recurrence as the dense-GQA reference)
      7. head-wise attention gate (sigmoid of current_hidden @ w_g)
      8. o_proj + residual
      9. post-attention RMSNorm (zero-centred)
     10. dense SiLU MLP + residual
    """
    import math

    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    w_g = tensors["w_g"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    hidden_q = wq.shape[1]
    head_dim = HEAD_DIM
    num_heads = NUM_HEADS_FULL
    num_kv_heads = NUM_KV_HEADS
    q_per_kv = Q_PER_KV_FULL
    rotary_dim = ROTARY_DIM_FULL
    rotary_half = ROTARY_HALF_FULL
    rotary_pass = ROTARY_PASS_FULL
    q_head_batch = Q_HEAD_BATCH_FULL
    q_groups = Q_GROUPS_FULL
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-6
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ

    def zc_apply(x, gamma):
        """Zero-centred RMSNorm broadcast multiply."""
        return x * (gamma + 1.0)

    # ---- Scope 1: input RMSNorm (zero-centred) + Q/K/V projection ----
    x_fp32 = hidden_states.float()
    sq_sum = torch.zeros(batch, 1, dtype=torch.float32)
    for k0 in range(0, hidden_size, INPUT_PROJ_K_CHUNK):
        x_chunk = x_fp32[:, k0 : k0 + INPUT_PROJ_K_CHUNK]
        sq_sum = sq_sum + (x_chunk * x_chunk).sum(dim=-1, keepdim=True)
    variance = sq_sum / hidden_size + eps
    inv_rms = torch.rsqrt(variance)
    normed_fp32 = zc_apply(x_fp32 * inv_rms, input_rms_weight.float())
    normed_bf16 = normed_fp32.bfloat16()

    q_proj = (normed_bf16.float() @ wq.float()).float()
    k_proj = (normed_bf16.float() @ wk.float()).float()
    v_proj = (normed_bf16.float() @ wv.float()).float()

    # ---- Per-head q_norm / k_norm (zero-centred) BEFORE RoPE ----
    q_heads_all = q_proj.view(batch, num_heads, head_dim)
    q_variance = q_heads_all.pow(2).mean(dim=-1, keepdim=True)
    q_heads_all = zc_apply(
        q_heads_all * torch.rsqrt(q_variance + eps),
        q_norm_weight.float(),
    )

    k_heads_all = k_proj.view(batch, num_kv_heads, head_dim)
    k_variance = k_heads_all.pow(2).mean(dim=-1, keepdim=True)
    k_heads_all = zc_apply(
        k_heads_all * torch.rsqrt(k_variance + eps),
        k_norm_weight.float(),
    )

    # ---- Scope 2: partial RoPE + KV cache write + flash decode ----
    attn_out = torch.zeros(batch, hidden_q, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        cos_row = rope_cos[pos : pos + 1, :]  # [1, rotary_dim]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo = cos_row[:, :rotary_half]
        cos_hi = cos_row[:, rotary_half:rotary_dim]
        sin_lo = sin_row[:, :rotary_half]
        sin_hi = sin_row[:, rotary_half:rotary_dim]

        # K partial RoPE.
        k_heads = k_heads_all[b]  # [num_kv_heads, head_dim]
        k_lo = k_heads[:, :rotary_half]
        k_hi = k_heads[:, rotary_half:rotary_dim]
        k_pass = k_heads[:, rotary_dim : rotary_dim + rotary_pass]
        k_rot = torch.cat(
            [
                k_lo * cos_lo - k_hi * sin_lo,
                k_hi * cos_hi + k_lo * sin_hi,
                k_pass,
            ],
            dim=-1,
        )

        slot = int(slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE
        for ki in range(num_kv_heads):
            cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
            k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_row, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(
                torch.bfloat16
            )

        # Q partial RoPE.
        q_heads = q_heads_all[b]  # [num_heads, head_dim]
        q_lo = q_heads[:, :rotary_half]
        q_hi = q_heads[:, rotary_half:rotary_dim]
        q_pass = q_heads[:, rotary_dim : rotary_dim + rotary_pass]
        q_rot = torch.cat(
            [
                q_lo * cos_lo - q_hi * sin_lo,
                q_hi * cos_hi + q_lo * sin_hi,
                q_pass,
            ],
            dim=-1,
        )

        # GQA online-softmax flash attention over the paged KV cache.
        attn_row = torch.zeros(1, hidden_q, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * q_head_batch
                q_grp_bf16 = q_rot[q_base : q_base + q_head_batch, :].to(torch.bfloat16)

                oi = torch.zeros(q_head_batch, head_dim, dtype=torch.float32)
                li = torch.zeros(q_head_batch, 1, dtype=torch.float32)
                mi = torch.zeros(q_head_batch, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = min(BLOCK_SIZE, ctx_len - s0)
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
                attn_row[
                    :,
                    q_base * head_dim : (q_base + q_head_batch) * head_dim,
                ] = ctx.reshape(1, -1).to(torch.bfloat16)

        attn_out[b : b + 1, :] = attn_row

    # ---- Scope 2.5: head-wise attention gate ----
    # gate_logits = current_hidden @ w_g  (BF16 inputs, FP32 accumulation)
    # gate = sigmoid(gate_logits) -> [batch, num_heads]
    # attn_out_gated[b, h*D:(h+1)*D] = attn_out[b, h*D:(h+1)*D] * gate[b, h]
    gate_logits = hidden_states.float() @ w_g.float()
    gate = torch.sigmoid(gate_logits)
    attn_view = attn_out.view(batch, num_heads, head_dim).float()
    attn_gated = (attn_view * gate.unsqueeze(-1)).to(torch.bfloat16)
    attn_gated_flat = attn_gated.view(batch, hidden_q)

    # ---- Scope 3: o_proj + residual + post RMSNorm (zero-centred) + MLP ----
    o_proj = attn_gated_flat.float() @ wo.float()
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    post_normed = zc_apply(resid1 * inv_rms, post_rms_weight.float())
    post_normed_bf16 = post_normed.bfloat16()

    gate_proj = post_normed_bf16.float() @ w_gate.float()
    up_proj = post_normed_bf16.float() @ w_up.float()
    # Layer-0 activation is plain SiLU (SWIGLU_LIMITS[0] == 0.0). No
    # SwigluStep clipping at this layer.
    mlp_bf16 = (gate_proj * torch.sigmoid(gate_proj) * up_proj).bfloat16()
    down = mlp_bf16.float() @ w_down.float()

    final_hidden = (down + resid1).bfloat16()
    tensors["out"][:] = final_hidden


def make_pass_rate_compare(threshold: float):
    """Pass-rate compare for BF16 ULP long-tail.

    Returns a compare_fn for ``run_jit`` that passes when at least
    ``threshold`` of elements fall within the run's ``atol`` / ``rtol``
    tolerance band. Same shape as the the dense-GQA reference ``decode_fwd`` helper but
    factored locally so this draft is self-contained.
    """

    def cmp(actual, expected, *, rtol, atol, **_):
        import torch

        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        rate = close.float().mean().item()
        n_fail = int((~close).sum().item())
        ok = rate >= threshold
        msg = (
            f"    pass_rate={rate:.6f} (threshold {threshold:.6f}), "
            f"{n_fail}/{actual.numel()} mismatched  rtol={rtol} atol={atol}"
        )
        if not ok:
            flat_a = actual.flatten()
            flat_e = expected.flatten()
            idx = torch.where(~close.flatten())[0][:5]
            lines = [
                f"    [{i.item()}] actual={flat_a[i].item()}, expected={flat_e[i].item()}"
                for i in idx
            ]
            msg += "\n    first {} mismatches:\n".format(idx.numel()) + "\n".join(lines)
        return ok, msg

    cmp.__name__ = f"pass_rate>={threshold:.4f}"
    return cmp


# =============================================================================
# CLI entry — mirrors the in-tree dense-GQA decode_fwd's __main__ block.
# =============================================================================
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "-b", "--batch", type=int, default=BATCH,
        help=(
            "User-visible batch size. Host allocates every batch-dependent "
            "tensor at exactly this size; the kernel internally rounds up "
            f"to BATCH_TILE ({BATCH_TILE}), zero-pads input loads via "
            "valid_shape, and trims the BF16 output by skipping rows past "
            "user_batch. Default: %(default)s"
        ),
    )
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument(
        "--pass-rate", type=float, default=0.98,
        help=(
            "Fraction of `out` elements that must satisfy atol/rtol. "
            "Default 0.98 matches the the dense-GQA reference decode_fwd threshold; this "
            "single-layer step3p5 path is shallower than 40-layer the dense-GQA reference so "
            "0.98 leaves comfortable margin for the BF16 ULP long-tail "
            "(zero-centred RMSNorm + partial RoPE + head-wise gate)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help=(
            "RNG seed for input tensor generation. Fixed by default so the "
            "pass_rate measurement is reproducible."
        ),
    )
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)

    if args.max_seq > MAX_SEQ_DEFAULT:
        raise ValueError(
            f"single_layer_decode_full_draft currently supports max_seq <= "
            f"{MAX_SEQ_DEFAULT}; got {args.max_seq}."
        )

    result = run_jit(
        fn=test_single_layer_decode_full,
        specs=build_tensor_specs(batch=args.batch, max_seq=args.max_seq),
        golden_fn=golden_single_layer_decode_full,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=5e-3,
        atol=5e-3,
        compare_fn={"out": make_pass_rate_compare(args.pass_rate)},
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "single_layer_decode_full",
    "test_single_layer_decode_full",
    "build_tensor_specs",
    "golden_single_layer_decode_full",
    "make_pass_rate_compare",
    "ROTARY_DIM_FULL",
    "ROTARY_PASS_FULL",
    "Q_GROUPS_FULL",
    "TOTAL_Q_GROUPS_FULL",
]
