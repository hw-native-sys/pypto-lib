# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 full-attention kernel — TP=8 in-place refactor (Phase 9 Wave 2).

Each rank holds an attention shard:

  - q_proj output: NUM_HEADS_FULL_LOCAL * HEAD_DIM = 8 * 128 = 1024
  - k_proj/v_proj output: KV_HEADS_LOCAL * HEAD_DIM = 1 * 128 = 128
  - o_proj input: NUM_HEADS_FULL_LOCAL * HEAD_DIM = 1024
  - w_g output:   NUM_HEADS_FULL_LOCAL          = 8
  - q_norm / k_norm gamma [HEAD_DIM=128] — REPLICATED on every rank

Compile-time constants baked in (LOCAL means per-rank-after-TP-slicing):

  - NUM_HEADS  = NUM_HEADS_FULL_LOCAL  (8)
  - HIDDEN_Q   = HIDDEN_Q_FULL_LOCAL   (1024)
  - KV_HIDDEN_DIM = KV_HIDDEN_LOCAL    (128)
  - NUM_KV_HEADS_DIM = KV_HEADS_LOCAL  (1)
  - Q_PER_KV   = Q_PER_KV_FULL         (8 ; invariant under TP)
  - Q_HEAD_BATCH = Q_HEAD_BATCH_FULL   (8)
  - Q_HEAD_PAD = Q_HEAD_PAD_FULL       (16)
  - ROTARY_HALF = ROTARY_HALF_FULL     (32 ; partial_rotary_factor = 0.5)
  - ROTARY_DIM  = 2 * ROTARY_HALF      (64)
  - ROTARY_PASS = HEAD_DIM - ROTARY_DIM (64 ; pass-through lanes)
  - Q_GROUPS  = Q_PER_KV // Q_HEAD_BATCH                  (1)
  - TOTAL_Q_GROUPS = NUM_KV_HEADS_DIM * Q_GROUPS          (1)

TP collective epilogue
----------------------
After the local o_proj (column-sliced) each rank holds a *partial* hidden
``[BATCH, HIDDEN]`` BF16 sum. ``tp_all_reduce`` sums these across the
TP group so every rank ends up with the fully-reduced o_proj output; the
residual add (``+ current_hidden``) happens afterwards (the residual is
replicated across ranks, so adding it post-all-reduce keeps the math
correct).

The caller (Wave-3 ``decode_layer.py`` / ``decode_fwd.py``) must provide
a per-call-site scratch ``tmp_window`` and ``signal_window`` pair, with
documented shapes:

  - ``tmp_window``    : ``pld.DistributedTensor`` view of a
                         ``BATCH * (HIDDEN // TP_WORLD_SIZE) * 2 bytes``
                         ``alloc_window_buffer`` slot (BF16).
  - ``signal_window`` : ``pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32]``,
                        zero-initialised. Each call site allocates a fresh
                        signal-window slot because the ring all-reduce
                        increments the cells across its ``2 * (N - 1)``
                        steps; reusing a slot would corrupt the wait
                        thresholds in subsequent collectives.

Per-layer ``rope_theta`` / ``partial_rotary_factor`` selection and the
host-side rope-table build are unchanged from the single-card draft.
Yarn scaling is always on for full-attention layers (``yarn_only_types =
["full_attention"]``).
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import (
    build_llama3_yarn_rope_tables,
    head_wise_gate_apply,
    partial_rope_rotate,
    per_head_qk_norm,
    zero_centered_rmsnorm_apply,
)
from .config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    INPUT_PROJ_K_CHUNK,
    K_CHUNK,
    KV_PROJ_K_CHUNK,
    KV_PROJ_K_CHUNK_LOCAL,
    KV_CACHE_ROWS_DYN,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_ROPE_THETA,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ_DEFAULT,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_FULL_LOCAL_PAD,
    OUT_PROJ_N_CHUNK,
    Q_HEAD_BATCH_FULL,
    Q_HEAD_PAD_FULL,
    Q_OUT_CHUNK,
    Q_PER_KV_FULL,
    ROPE_SCALING,
    ROPE_SEQ_DYN,
    ROTARY_HALF_FULL,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    is_full_attention,
)

NUM_HEADS = NUM_HEADS_FULL_LOCAL
HIDDEN_Q = HIDDEN_Q_FULL_LOCAL
KV_HIDDEN_DIM = KV_HIDDEN_LOCAL
NUM_KV_HEADS_DIM = KV_HEADS_LOCAL
Q_PER_KV = Q_PER_KV_FULL
Q_HEAD_BATCH = Q_HEAD_BATCH_FULL
Q_HEAD_PAD = Q_HEAD_PAD_FULL
ROTARY_HALF = ROTARY_HALF_FULL
ROTARY_DIM = ROTARY_HALF * 2
ROTARY_PASS = HEAD_DIM - ROTARY_DIM
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH                # 1
TOTAL_Q_GROUPS = NUM_KV_HEADS_DIM * Q_GROUPS       # 1

# Local override for KV projection's output chunk: KV_HIDDEN_LOCAL (128) is
# below the global KV_OUT_CHUNK=256 default, so we pick the whole local KV
# hidden in a single chunk.
KV_OUT_CHUNK_LOCAL = KV_HIDDEN_LOCAL

# Per-layer dyn dim for the o_proj weight (LAYERS * HIDDEN_Q_LOCAL rows).
LAYER_QHIDDEN_ROWS_DYN = pl.dynamic("LAYER_QHIDDEN_ROWS_DYN")

assert Q_HEAD_PAD % 4 == 0 and Q_HEAD_PAD // 2 >= Q_HEAD_BATCH
assert BATCH % 2 == 0, (
    "fa_fused pipelines pairs of batches under TP, so BATCH must be even"
)
assert HIDDEN_Q % K_CHUNK == 0
assert HIDDEN % OUT_PROJ_N_CHUNK == 0
assert HIDDEN % TP_WORLD_SIZE == 0
assert KV_HIDDEN_DIM == KV_OUT_CHUNK_LOCAL


# =============================================================================
# Attention body — local compute through gated attn_out, partial o_proj,
# TP all-reduce, then residual add.
# =============================================================================
@pl.jit.inline
def attention_full(
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_FULL * 2], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_FULL * 2], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16],
    resid1_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[
        [BATCH, HIDDEN // TP_WORLD_SIZE], pl.BF16
    ],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """Step3p5 full-attention layer through TP-reduced o_proj + residual.

    ``my_rank``, ``tmp_window`` and ``signal_window`` are passed in by the
    Wave-3 ``chip_orch`` wrapper (see module docstring for the shape
    contract). Inside this body they only ever flow through the
    :func:`tp_all_reduce` call below; the rest of the kernel is per-rank
    local compute on the rank's head slice.
    """

    decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
    kv_proj_hidden_blocks = HIDDEN // KV_PROJ_K_CHUNK_LOCAL
    qhidden_blocks = HIDDEN_Q_FULL_LOCAL // K_CHUNK
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    bt_stride = pl.tensor.dim(block_table, 0) // user_batch
    batch_padded = BATCH

    layer_hidden_base = layer_idx * HIDDEN
    layer_qhidden_base = layer_idx * HIDDEN_Q_FULL_LOCAL
    layer_cache_base = layer_idx * decode_layer_cache_rows

    q_proj = pl.create_tensor([BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    gate_logits = pl.create_tensor([BATCH, NUM_HEADS_FULL_LOCAL_PAD], dtype=pl.FP32)

    # ----- Scope 1.a — zero-centred input RMSNorm. -----
    # input_rms_weight is replicated across TP ranks (HIDDEN dim is not
    # sliced). Every rank computes the same normed_all tile; this work is
    # duplicated but cheap relative to the projections.
    #
    # Phase A (2026-06-11): mirror qwen3/32b's CORE_GROUP + pl.pipeline form.
    # The previous `pl.spmd(BATCH//BATCH_TILE=1)` with a single worker was
    # the suspected source of the AICore 507018 VEC UB alignment crash at
    # the first MIX SQE task slot (`aicore_kernel_0_mix_aic`). qwen3/32b
    # uses CORE_GROUP + pipeline(stage=4) for the same shape and runs clean.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="full_rmsnorm_zc"):
        partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(decode_scope1_hidden_blocks, stage=4):
            sq_k0 = kb * INPUT_PROJ_K_CHUNK
            sq_chunk = pl.cast(
                pl.slice(current_hidden, [BATCH, INPUT_PROJ_K_CHUNK], [0, sq_k0]),
                target_type=pl.FP32,
            )
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(sq_chunk, sq_chunk)), [1, BATCH]),
            )
        variance = pl.reshape(
            pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1],
        )
        inv_rms = pl.recip(pl.sqrt(variance))
        for kb in pl.pipeline(decode_scope1_hidden_blocks, stage=4):
            norm_k0 = kb * INPUT_PROJ_K_CHUNK
            norm_chunk = pl.cast(
                pl.slice(current_hidden, [BATCH, INPUT_PROJ_K_CHUNK], [0, norm_k0]),
                target_type=pl.FP32,
            )
            gamma = pl.slice(input_rms_weight, [1, INPUT_PROJ_K_CHUNK], [layer_idx, norm_k0])
            scaled = pl.row_expand_mul(norm_chunk, inv_rms)
            normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
            normed_all = pl.assemble(
                normed_all, pl.cast(normed, target_type=pl.BF16), [0, norm_k0],
            )

    # ----- Scope 1.b — Q projection. -----
    # wq is row-sliced (output dim → HIDDEN_Q_FULL_LOCAL per rank), so the
    # SPMD bound shrinks accordingly.
    for q_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (HIDDEN_Q_FULL_LOCAL // Q_OUT_CHUNK), name_hint="full_q_proj",
    ):
        q_b_idx = q_spmd_idx // (HIDDEN_Q_FULL_LOCAL // Q_OUT_CHUNK)
        q_ob = q_spmd_idx % (HIDDEN_Q_FULL_LOCAL // Q_OUT_CHUNK)
        q_b0 = q_b_idx * BATCH_TILE
        q_o0 = q_ob * Q_OUT_CHUNK
        q_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, 0])
        q_tile_b_0 = pl.slice(wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, q_o0])
        q_acc = pl.matmul(q_tile_a_0, q_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            q_k0 = kb * INPUT_PROJ_K_CHUNK
            q_tile_a = pl.slice(normed_all, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [q_b0, q_k0])
            q_tile_b = pl.slice(
                wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + q_k0, q_o0],
            )
            q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
        q_proj = pl.assemble(q_proj, q_acc, [q_b0, q_o0])

    # ----- Scope 1.c — K projection. -----
    # wk is row-sliced (output dim → KV_HEADS_LOCAL * HEAD_DIM = 128 per rank).
    # KV_HIDDEN_LOCAL is 128 (single local KV head), so we drop a single
    # output chunk of width KV_HIDDEN_LOCAL = 128.
    for k_spmd_idx in pl.spmd(
        BATCH // BATCH_TILE, name_hint="full_k_proj",
    ):
        k_b0 = k_spmd_idx * BATCH_TILE
        k_o0 = 0
        k_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [k_b0, 0])
        k_tile_b_0 = pl.slice(
            wk, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base, k_o0],
        )
        k_acc = pl.matmul(k_tile_a_0, k_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, kv_proj_hidden_blocks):
            k_k0 = kb * KV_PROJ_K_CHUNK_LOCAL
            k_tile_a = pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [k_b0, k_k0])
            k_tile_b = pl.slice(
                wk, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL],
                [layer_hidden_base + k_k0, k_o0],
            )
            k_acc = pl.matmul_acc(k_acc, k_tile_a, k_tile_b)
        k_proj = pl.assemble(k_proj, k_acc, [k_b0, k_o0])

    # ----- Scope 1.d — V projection. -----
    for v_spmd_idx in pl.spmd(
        BATCH // BATCH_TILE, name_hint="full_v_proj",
    ):
        v_b0 = v_spmd_idx * BATCH_TILE
        v_o0 = 0
        v_tile_a_0 = pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [v_b0, 0])
        v_tile_b_0 = pl.slice(
            wv, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL], [layer_hidden_base, v_o0],
        )
        v_acc = pl.matmul(v_tile_a_0, v_tile_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, kv_proj_hidden_blocks):
            v_k0 = kb * KV_PROJ_K_CHUNK_LOCAL
            v_tile_a = pl.slice(normed_all, [BATCH_TILE, KV_PROJ_K_CHUNK_LOCAL], [v_b0, v_k0])
            v_tile_b = pl.slice(
                wv, [KV_PROJ_K_CHUNK_LOCAL, KV_HIDDEN_LOCAL],
                [layer_hidden_base + v_k0, v_o0],
            )
            v_acc = pl.matmul_acc(v_acc, v_tile_a, v_tile_b)
        v_proj = pl.assemble(v_proj, v_acc, [v_b0, v_o0])

    # ----- Scope 1.e — per-head zero-centred q_norm / k_norm. -----
    # q_norm / k_norm gamma [HEAD_DIM] are REPLICATED across TP ranks, so
    # this block runs unchanged on each rank — only the per-head loop
    # bounds shrink (KV_HEADS_LOCAL = 1 per rank).
    #
    # Q-heads are processed one-at-a-time inside each spmd block to keep
    # the Vec-memory footprint under the 192 KB platform limit. Packing
    # all Q_HEAD_BATCH_FULL=8 heads together blows past 222 KB (six FP32
    # intermediates × [128 × 128] BF16 → ~217 KB live).
    for qkn_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * KV_HEADS_LOCAL, name_hint="full_qk_norm_zc",
    ):
        qkn_b_idx = qkn_spmd_idx // KV_HEADS_LOCAL
        qkn_h = qkn_spmd_idx % KV_HEADS_LOCAL
        qkn_b0 = qkn_b_idx * BATCH_TILE

        qkn_q0_base = qkn_h * Q_PER_KV_FULL * HEAD_DIM
        q_gamma = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
        for qh in pl.range(Q_HEAD_BATCH_FULL):
            qh_q0 = qkn_q0_base + qh * HEAD_DIM
            q_chunk = pl.slice(q_proj, [BATCH_TILE, HEAD_DIM], [qkn_b0, qh_q0])
            q_sq = pl.row_sum(pl.mul(q_chunk, q_chunk))
            q_inv = pl.rsqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS))
            q_scaled = pl.row_expand_mul(q_chunk, q_inv)
            q_normed = pl.col_expand_mul(q_scaled, pl.add(q_gamma, 1.0))
            q_proj_norm = pl.assemble(q_proj_norm, q_normed, [qkn_b0, qh_q0])

        qkn_k0 = qkn_h * HEAD_DIM
        k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [qkn_b0, qkn_k0])
        k_gamma = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
        k_sq = pl.row_sum(pl.mul(k_chunk, k_chunk))
        k_inv = pl.rsqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS))
        k_scaled = pl.row_expand_mul(k_chunk, k_inv)
        k_normed = pl.col_expand_mul(k_scaled, pl.add(k_gamma, 1.0))
        k_proj_norm = pl.assemble(k_proj_norm, k_normed, [qkn_b0, qkn_k0])

    # ----- Scope 1.f — head-wise gate matmul (current_hidden, NOT normed). -----
    # w_g is row-sliced (output dim → NUM_HEADS_FULL_LOCAL = 8 per rank);
    # gate_logits has the per-rank local heads only.
    for gp_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="full_gate_proj"):
        gp_b0 = gp_spmd_idx * BATCH_TILE
        gp_a_0 = pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [gp_b0, 0])
        gp_b_0 = pl.slice(w_g, [INPUT_PROJ_K_CHUNK, NUM_HEADS_FULL_LOCAL_PAD], [layer_hidden_base, 0])
        gp_acc = pl.matmul(gp_a_0, gp_b_0, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            gp_k0 = kb * INPUT_PROJ_K_CHUNK
            gp_a = pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [gp_b0, gp_k0])
            gp_b = pl.slice(
                w_g, [INPUT_PROJ_K_CHUNK, NUM_HEADS_FULL_LOCAL_PAD], [layer_hidden_base + gp_k0, 0],
            )
            gp_acc = pl.matmul_acc(gp_acc, gp_a, gp_b)
        gate_logits = pl.assemble(gate_logits, gp_acc, [gp_b0, 0])

    # ----- Scope 2 — partial RoPE + paged KV cache write + fa_fused. -----
    # k_cache / v_cache hold KV_HEADS_LOCAL = 1 KV head's history per rank,
    # so the loop over local KV heads collapses to a single iteration (the
    # math below mirrors the single-card draft but the slot/cache strides
    # use KV_HEADS_LOCAL).
    attn_out = pl.create_tensor([BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
    all_q_padded = pl.create_tensor(
        [BATCH * KV_HEADS_LOCAL * (Q_PER_KV_FULL // Q_HEAD_BATCH_FULL) * Q_HEAD_PAD_FULL, HEAD_DIM], dtype=pl.BF16,
    )

    # Phase A reverse-audit (2026-06-12): qwen3/32b uses `for b in pl.parallel(BATCH)`
    # (static constant) here; step3p5 was using `pl.parallel(user_batch)` where
    # user_batch is a DYNAMIC scalar from `pl.tensor.dim(seq_lens, 0)`. The IR
    # for dynamic vs static parallel bounds differs significantly — dynamic
    # generates a host-side C++ loop emitting one rt_submit per iter (each
    # potentially with different UB lifetime), static fully unrolls. The
    # dynamic form is the suspected source of `aicore_kernel_0_mix_aic tslot:6
    # VEC UB not aligned`. Pad batches (b >= user_batch) are already guarded
    # below via `fa_b_safe = pl.min(b, user_batch - 1)` in the 4 attention
    # spmds; here we extend the same guard pattern: clamp the per-batch reads
    # with `b_safe` for pad batches.
    for b in pl.parallel(BATCH):
        b_safe = pl.min(b, user_batch - 1)
        ctx_len = pl.tensor.read(seq_lens, [b_safe])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b_safe])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE
        cos_row = pl.slice(rope_cos, [1, ROTARY_HALF_FULL * 2], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, ROTARY_HALF_FULL * 2], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, ROTARY_HALF_FULL], [0, 0])
        cos_hi = pl.slice(cos_row, [1, ROTARY_HALF_FULL], [0, ROTARY_HALF_FULL])
        sin_lo = pl.slice(sin_row, [1, ROTARY_HALF_FULL], [0, 0])
        sin_hi = pl.slice(sin_row, [1, ROTARY_HALF_FULL], [0, ROTARY_HALF_FULL])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="full_rope_kv_cache"):
            for ki in pl.range(KV_HEADS_LOCAL):
                kv_col = ki * HEAD_DIM
                cache_row = (
                    layer_cache_base
                    + (slot_block * KV_HEADS_LOCAL + ki) * BLOCK_SIZE
                    + slot_offset
                )
                k_lo = pl.slice(k_proj_norm, [1, ROTARY_HALF_FULL], [b, kv_col])
                k_hi = pl.slice(
                    k_proj_norm, [1, ROTARY_HALF_FULL], [b, kv_col + ROTARY_HALF_FULL],
                )
                k_pass = pl.slice(
                    k_proj_norm, [1, HEAD_DIM - ROTARY_HALF_FULL * 2], [b, kv_col + ROTARY_HALF_FULL * 2],
                )
                rot_k_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                rot_k_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                # Phase A (2026-06-11): use qwen3/32b's full-row-cast-then-
                # overwrite idiom instead of the (compile-required, runtime-
                # broken) `pl.add(k_pass, 0.0)` workaround. Cast the entire
                # [1, HEAD_DIM] k_proj_norm row to BF16 once (which is the
                # exact pattern qwen3/32b's v_cache write uses and which
                # AICore lowers cleanly), then overwrite cols 0..2*HALF with
                # the RoPE'd halves. The pass-through tail (cols 2*HALF..)
                # is left as the initial full-row cast.
                k_full_bf16 = pl.cast(
                    pl.slice(k_proj_norm, [1, HEAD_DIM], [b, kv_col]),
                    target_type=pl.BF16,
                )
                k_cache = pl.assemble(k_cache, k_full_bf16, [cache_row, 0])
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_lo, target_type=pl.BF16), [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_hi, target_type=pl.BF16),
                    [cache_row, ROTARY_HALF_FULL],
                )
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(pl.slice(v_proj, [1, HEAD_DIM], [b, kv_col]),
                            target_type=pl.BF16),
                    [cache_row, 0],
                )

                q_base = ki * Q_PER_KV_FULL
                q_block = pl.reshape(
                    pl.slice(
                        q_proj_norm, [1, Q_HEAD_BATCH_FULL * HEAD_DIM],
                        [b, q_base * HEAD_DIM],
                    ),
                    [Q_HEAD_BATCH_FULL, HEAD_DIM],
                )
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH_FULL, ROTARY_HALF_FULL], [0, 0])
                q_hi = pl.slice(q_block, [Q_HEAD_BATCH_FULL, ROTARY_HALF_FULL], [0, ROTARY_HALF_FULL])
                q_pass = pl.slice(q_block, [Q_HEAD_BATCH_FULL, HEAD_DIM - ROTARY_HALF_FULL * 2], [0, ROTARY_HALF_FULL * 2])
                rot_q_lo = pl.sub(
                    pl.col_expand_mul(q_lo, cos_lo),
                    pl.col_expand_mul(q_hi, sin_lo),
                )
                rot_q_hi = pl.add(
                    pl.col_expand_mul(q_hi, cos_hi),
                    pl.col_expand_mul(q_lo, sin_hi),
                )
                rot_q_lo_bf16 = pl.cast(rot_q_lo, target_type=pl.BF16)
                rot_q_hi_bf16 = pl.cast(rot_q_hi, target_type=pl.BF16)
                # Phase A (2026-06-11): full-Q-block cast then overwrite RoPE
                # halves (qwen3/32b idiom). Eliminates the partial-slice ->
                # cast path that previously needed the `pl.add(q_pass, 0.0)`
                # workaround (which compiled but tripped 507018 VEC UB align
                # at runtime).
                q_block_bf16 = pl.cast(q_block, target_type=pl.BF16)

                pad_row_base = b * KV_HEADS_LOCAL * (Q_PER_KV_FULL // Q_HEAD_BATCH_FULL) * Q_HEAD_PAD_FULL + ki * Q_HEAD_PAD_FULL
                all_q_padded = pl.assemble(all_q_padded, q_block_bf16, [pad_row_base, 0])
                all_q_padded = pl.assemble(all_q_padded, rot_q_lo_bf16, [pad_row_base, 0])
                all_q_padded = pl.assemble(
                    all_q_padded, rot_q_hi_bf16, [pad_row_base, ROTARY_HALF_FULL],
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(
                        pl.full([Q_HEAD_PAD_FULL - Q_HEAD_BATCH_FULL, HEAD_DIM],
                                dtype=pl.FP32, value=0.0),
                        target_type=pl.BF16,
                    ),
                    [pad_row_base + Q_HEAD_BATCH_FULL, 0],
                )

    # ----- fa_fused — Phase A (2026-06-11): qwen3/32b-style 4-spmd split. -----
    # The previous fused mixed AIC+AIV single root tripped 507018 / VEC UB
    # not-aligned at this shape (NUM_HEADS_FULL_LOCAL=8, KV_HEADS_LOCAL=1,
    # Q_PER_KV_FULL=Q_HEAD_BATCH_FULL=8, Q_HEAD_PAD_FULL=16, HEAD_DIM=128).
    # We mirror qwen3/32b's split form: four sequential per-batch spmds
    # (qk_matmul / softmax / sv_matmul / online_softmax) with GM scratch
    # carrying raw scores, softmax exp + mi/li, and sv partials between
    # stages. ``pl.slice(..., valid_shape=...)`` replaces set_validshape +
    # fillpad so the VEC lowering goes through a different (proven-safe)
    # path. mi/li stay flat as [Q_HEAD_BATCH_FULL, 1] -- no [16,1] reshape.
    # See docs/step3p5/phases/15-singlerank-npu.md "Phase A route decision".
    MAX_CTX_BLOCKS = MAX_SEQ_DEFAULT // BLOCK_SIZE
    all_raw_scores = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS * Q_HEAD_PAD_FULL, BLOCK_SIZE], dtype=pl.FP32,
    )
    all_exp_padded = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS * Q_HEAD_PAD_FULL, BLOCK_SIZE], dtype=pl.BF16,
    )
    all_cur_mi = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS * Q_HEAD_BATCH_FULL, 1], dtype=pl.FP32,
    )
    all_cur_li = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS * Q_HEAD_BATCH_FULL, 1], dtype=pl.FP32,
    )
    all_oi_tmp = pl.create_tensor(
        [BATCH * MAX_CTX_BLOCKS * Q_HEAD_PAD_FULL, HEAD_DIM], dtype=pl.FP32,
    )

    # Stage 1: QK matmul (cube). One core per batch.
    for fa_b in pl.spmd(BATCH, name_hint="full_qk_matmul"):
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_ctx_blocks = (fa_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        fa_block_table_base = fa_b_safe * bt_stride
        q_padded_row = fa_b * Q_HEAD_PAD_FULL  # KV_HEADS_LOCAL=1, Q_GROUPS=1
        q_padded = pl.slice(
            all_q_padded, [Q_HEAD_PAD_FULL, HEAD_DIM], [q_padded_row, 0],
        )
        for sb in pl.range(fa_ctx_blocks):
            fa_pbid = pl.cast(
                pl.tensor.read(block_table, [fa_block_table_base + sb]), pl.INDEX,
            )
            fa_cache_row = layer_cache_base + fa_pbid * BLOCK_SIZE
            k_tile = pl.slice(
                k_cache, [BLOCK_SIZE, HEAD_DIM], [fa_cache_row, 0],
            )
            raw_scores = pl.matmul(
                q_padded, k_tile, b_trans=True, out_dtype=pl.FP32,
            )
            scratch_row = (fa_b * MAX_CTX_BLOCKS + sb) * Q_HEAD_PAD_FULL
            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [scratch_row, 0])

    # Stage 2: softmax (vec). pl.slice(valid_shape=) marks the real
    # Q_HEAD_BATCH_FULL rows + valid_len columns; fillpad pushes -inf into the
    # masked tail so row_max ignores them. exp/row_sum/cast stay intra-tile
    # (no reshape).
    for fa_b in pl.spmd(BATCH, name_hint="full_softmax"):
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_ctx_blocks = (fa_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for sb in pl.range(fa_ctx_blocks):
            s0 = sb * BLOCK_SIZE
            valid_len = pl.min(BLOCK_SIZE, fa_ctx_len - s0)
            scratch_row = (fa_b * MAX_CTX_BLOCKS + sb) * Q_HEAD_PAD_FULL
            scratch_lm_row = (fa_b * MAX_CTX_BLOCKS + sb) * Q_HEAD_BATCH_FULL
            scores_valid = pl.slice(
                all_raw_scores,
                [Q_HEAD_BATCH_FULL, BLOCK_SIZE],
                [scratch_row, 0],
                valid_shape=[Q_HEAD_BATCH_FULL, valid_len],
            )
            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
            scores = pl.mul(scores_padded, decode_attn_scale)
            cur_mi = pl.row_max(scores)
            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
            exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
            cur_li = pl.row_sum(exp_scores_fp32)
            all_exp_padded = pl.assemble(
                all_exp_padded, exp_scores_bf16, [scratch_row, 0],
            )
            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [scratch_lm_row, 0])
            all_cur_li = pl.assemble(all_cur_li, cur_li, [scratch_lm_row, 0])

    # Stage 3: SV matmul (cube). exp_tile is BF16, v_tile is BF16, oi is FP32.
    for fa_b in pl.spmd(BATCH, name_hint="full_sv_matmul"):
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_ctx_blocks = (fa_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        fa_block_table_base = fa_b_safe * bt_stride
        for sb in pl.range(fa_ctx_blocks):
            fa_pbid = pl.cast(
                pl.tensor.read(block_table, [fa_block_table_base + sb]), pl.INDEX,
            )
            fa_cache_row = layer_cache_base + fa_pbid * BLOCK_SIZE
            v_tile = pl.slice(
                v_cache, [BLOCK_SIZE, HEAD_DIM], [fa_cache_row, 0],
            )
            scratch_row = (fa_b * MAX_CTX_BLOCKS + sb) * Q_HEAD_PAD_FULL
            exp_tile = pl.slice(
                all_exp_padded,
                [Q_HEAD_PAD_FULL, BLOCK_SIZE],
                [scratch_row, 0],
            )
            oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [scratch_row, 0])

    # Stage 4: online softmax accumulation + final normalisation + attn_out
    # write. mi/li/oi carried flat as [Q_HEAD_BATCH_FULL, 1] / [_, HEAD_DIM];
    # no reshape touches the [Q_HEAD_PAD_FULL, 1] tile shape that previously
    # tripped the VEC alignment check in the fused form.
    for fa_b in pl.spmd(BATCH, name_hint="full_online_softmax"):
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_ctx_blocks = (fa_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        oi_row0 = fa_b * MAX_CTX_BLOCKS * Q_HEAD_PAD_FULL
        lm_row0 = fa_b * MAX_CTX_BLOCKS * Q_HEAD_BATCH_FULL
        oi = pl.slice(all_oi_tmp, [Q_HEAD_BATCH_FULL, HEAD_DIM], [oi_row0, 0])
        mi = pl.slice(all_cur_mi, [Q_HEAD_BATCH_FULL, 1], [lm_row0, 0])
        li = pl.slice(all_cur_li, [Q_HEAD_BATCH_FULL, 1], [lm_row0, 0])
        for sb in pl.range(1, fa_ctx_blocks):
            sb_oi_row = oi_row0 + sb * Q_HEAD_PAD_FULL
            sb_lm_row = lm_row0 + sb * Q_HEAD_BATCH_FULL
            oi_partial = pl.slice(
                all_oi_tmp, [Q_HEAD_BATCH_FULL, HEAD_DIM], [sb_oi_row, 0],
            )
            cur_mi = pl.slice(
                all_cur_mi, [Q_HEAD_BATCH_FULL, 1], [sb_lm_row, 0],
            )
            cur_li = pl.slice(
                all_cur_li, [Q_HEAD_BATCH_FULL, 1], [sb_lm_row, 0],
            )
            mi_new = pl.maximum(mi, cur_mi)
            alpha = pl.exp(pl.sub(mi, mi_new))
            beta = pl.exp(pl.sub(cur_mi, mi_new))
            li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
            oi = pl.add(
                pl.row_expand_mul(oi, alpha),
                pl.row_expand_mul(oi_partial, beta),
            )
            mi = mi_new
        ctx = pl.row_expand_div(oi, li)
        ctx_flat_bf16 = pl.cast(
            pl.reshape(ctx, [1, Q_HEAD_BATCH_FULL * HEAD_DIM]),
            target_type=pl.BF16,
        )
        # q_base = kvh * Q_PER_KV_FULL == 0 (KV_HEADS_LOCAL=1, kvh=0).
        attn_out = pl.assemble(attn_out, ctx_flat_bf16, [fa_b, 0])

    # ----- Scope 2.5 — head-wise sigmoid gate (local heads only). -----
    # Phase 15.1: outer spmd over batch tiles only; load the full
    # gate_logits row [BATCH_TILE, NUM_HEADS_FULL_LOCAL_PAD] once as a
    # contiguous ND tile, sigmoid all heads at once, then pluck each head
    # column intra-tile.  This avoids the [BATCH_TILE, 1] DN-Vec ↔ ND-GM
    # TLOAD pair that pto-isa rejects (`isSameLayout` static_assert in
    # TLoad.hpp:459).
    attn_out_gated = pl.create_tensor([BATCH, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
    for hg_spmd_idx in pl.spmd(
        BATCH // BATCH_TILE, name_hint="full_head_gate",
    ):
        hg_b0 = hg_spmd_idx * BATCH_TILE
        gate_row_fp32 = pl.slice(
            gate_logits,
            [BATCH_TILE, NUM_HEADS_FULL_LOCAL_PAD],
            [hg_b0, 0],
        )
        sigmoid_all = pl.recip(
            pl.add(pl.exp(pl.neg(gate_row_fp32)), 1.0),
        )
        for hg_h in pl.range(NUM_HEADS_FULL_LOCAL):
            hg_col = hg_h * HEAD_DIM
            head_slice_bf16 = pl.slice(
                attn_out, [BATCH_TILE, HEAD_DIM], [hg_b0, hg_col],
            )
            hg_gate = pl.slice(sigmoid_all, [BATCH_TILE, 1], [0, hg_h])
            hg_gated_fp32 = pl.row_expand_mul(
                pl.cast(head_slice_bf16, target_type=pl.FP32), hg_gate,
            )
            gated = pl.cast(hg_gated_fp32, target_type=pl.BF16)
            attn_out_gated = pl.assemble(attn_out_gated, gated, [hg_b0, hg_col])

    # ----- Scope 3.a — local o_proj (partial result, no residual yet). -----
    # wo is column-sliced (input dim → HIDDEN_Q_FULL_LOCAL per rank); the
    # output is a partial [BATCH, HIDDEN] BF16 tensor that must be summed
    # across the TP group via the all-reduce below before the residual add.
    #
    # Phase A (2026-06-11): split the previously-mixed AIC+AIV body into two
    # sequential spmds. The original form had cube matmul → vec cast → GM
    # store all inside one `pl.spmd(... name_hint="full_out_proj")` scope,
    # which PTOAS lowered to a `MixedKernels` dispatch (`mixed_12` in
    # chip_orch.cpp). That kernel = ``aicore_kernel_0_mix_aic`` — the first
    # mixed AIC+AIV root in the program — was the deterministic 507018
    # crash site (VEC UB not-aligned, plog `hash=15033215677169261682`).
    # Splitting to pure-cube + pure-vec removes the mixed-mode dispatch.
    partial_attn_proj_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for ob in pl.spmd(
            HIDDEN // OUT_PROJ_N_CHUNK, name_hint="full_out_proj_matmul",
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
                    wo, [K_CHUNK, OUT_PROJ_N_CHUNK],
                    [layer_qhidden_base + k0, o0],
                )
                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)
            partial_attn_proj_fp32 = pl.assemble(
                partial_attn_proj_fp32, o_acc, [b0, o0],
            )

    partial_attn_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for ob in pl.spmd(
            HIDDEN // OUT_PROJ_N_CHUNK, name_hint="full_out_proj_cast",
        ):
            o0 = ob * OUT_PROJ_N_CHUNK
            fp32_chunk = pl.slice(
                partial_attn_proj_fp32,
                [BATCH_TILE, OUT_PROJ_N_CHUNK], [b0, o0],
            )
            partial_attn_proj = pl.assemble(
                partial_attn_proj,
                pl.cast(fp32_chunk, target_type=pl.BF16),
                [b0, o0],
            )

    # ----- Scope 3.b — TP all-reduce(sum) of the partial o_proj output. -----
    # Phase X.2: the pull-side ring body now lives as
    # TpAttentionFull.tp_all_reduce — see that class for the implementation.
    # After the call every TP rank holds the same fully-reduced
    # ``[BATCH, HIDDEN]`` o_proj output.
    # Phase 15.1 single-rank gate: at TP=1 the all-reduce is a no-op (no
    # peers); skip the function call entirely so the orchestration codegen
    # does not emit a stale SSA rename for the (now-empty) ring body.
    if TP_WORLD_SIZE > 1:
        partial_attn_proj = self.tp_all_reduce(
            partial_attn_proj,
            tmp_window,
            signal_window,
            my_rank,
        )

    # ----- Scope 3.c — residual add (post-all-reduce). -----
    # ``current_hidden`` is replicated across TP ranks, so each rank adds
    # the same residual to the same reduced sum — every rank ends up with
    # the same ``resid1_out``.
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for ob in pl.spmd(
            HIDDEN // OUT_PROJ_N_CHUNK, name_hint="full_out_resid_add",
        ):
            o0 = ob * OUT_PROJ_N_CHUNK
            reduced = pl.cast(
                pl.slice(partial_attn_proj, [BATCH_TILE, OUT_PROJ_N_CHUNK], [b0, o0]),
                target_type=pl.FP32,
            )
            resid = pl.cast(
                pl.slice(current_hidden, [BATCH_TILE, OUT_PROJ_N_CHUNK], [b0, o0]),
                target_type=pl.FP32,
            )
            resid_sum = pl.add(reduced, resid)
            resid1_out = pl.assemble(
                resid1_out, pl.cast(resid_sum, target_type=pl.BF16), [b0, o0],
            )

    return resid1_out


# =============================================================================
# TP wrapper — Wave-2 program scaffolding (chip_orch + host_orch).
#
# This `@pl.program` builder is the canonical entry point a Wave-3 forward
# pass uses to invoke ``attention_full`` with the TP collective threaded
# through.  It also serves as a compile-cleanness probe (importing this
# module triggers the deferred build inside the harness path).
# =============================================================================
def _build_tp_attention_full_program(tp_size: int = TP_WORLD_SIZE):
    """Return a freshly-built ``@pl.program`` class for the full-attention
    TP epilogue.

    Constructed inside a function so the module imports even on hosts that
    have not finished bringing up the pypto runtime (deferred-build
    pattern, matches the in-tree TP+EP MoE reference).
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    attention_full_inline = pl.inline(attention_full._func)
    tp_chunk = HIDDEN // tp_size

    @pl.program
    class TpAttentionFull:
        # ---------- Collective: TP all_reduce (lifted from collectives.py) ----
        # Phase X.2: pull-side ring all-reduce body, baked with
        # t_rows=BATCH, d_cols=HIDDEN, group_size=tp_size from the
        # factory closure. See ``tests/st/distributed/test_l3_allreduce.py``
        # for the canonical pattern.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, tp_chunk], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            """Pull-side ring all-reduce(sum) across the TP group."""
            group_size = tp_size
            t_rows = BATCH
            d_cols = HIDDEN
            chunk = d_cols // group_size

            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + group_size) % group_size
                recv_idx = (my_rank - step - 1 + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size

                send_tile = pl.load(
                    local, [0, send_idx * chunk], [t_rows, chunk],
                )
                pl.store(send_tile, [0, 0], tmp_window)

                pld.system.notify(
                    target=signal_window,
                    peer=next_rank,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window,
                    offsets=[prev_rank, 0],
                    expected=step + 1,
                    cmp=pld.WaitCmp.Ge,
                )

                recv_tile = pld.tile.remote_load(
                    tmp_window,
                    peer=prev_rank,
                    offsets=[0, 0],
                    shape=[t_rows, chunk],
                )
                old_tile = pl.load(
                    local, [0, recv_idx * chunk], [t_rows, chunk],
                )
                pl.store(
                    pl.add(old_tile, recv_tile),
                    [0, recv_idx * chunk],
                    local,
                )

            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + 1 + group_size) % group_size
                recv_idx = (my_rank - step + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size

                send_tile = pl.load(
                    local, [0, send_idx * chunk], [t_rows, chunk],
                )
                pl.store(send_tile, [0, 0], tmp_window)

                pld.system.notify(
                    target=signal_window,
                    peer=next_rank,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window,
                    offsets=[prev_rank, 0],
                    expected=group_size - 1 + step + 1,
                    cmp=pld.WaitCmp.Ge,
                )

                recv_tile = pld.tile.remote_load(
                    tmp_window,
                    peer=prev_rank,
                    offsets=[0, 0],
                    shape=[t_rows, chunk],
                )
                pl.store(recv_tile, [0, recv_idx * chunk], local)

            return local

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], pl.BF16],
            wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16],
            wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS], pl.BF16],
            resid1_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            tmp_window: pld.DistributedTensor[[BATCH, tp_chunk], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1_out = attention_full_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                resid1_out,
                layer_idx,
                tmp_window,
                signal_window,
                my_rank,
            )
            return resid1_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], pl.BF16],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[tp_size, ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            rope_sin: pl.Tensor[[tp_size, ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            k_cache: pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[tp_size, LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[tp_size, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS], pl.BF16],
            resid1_out: pl.Out[pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]],
            layer_idx: pl.Scalar[pl.INT32],
        ):
            tmp_buf = pld.alloc_window_buffer(BATCH * tp_chunk * 2)  # BF16
            sig_buf = pld.alloc_window_buffer(tp_size * 4)           # INT32

            for r in pl.range(pld.world_size()):
                tmp_window = pld.window(tmp_buf, [BATCH, tp_chunk], dtype=pl.BF16)
                signal_window = pld.window(sig_buf, [tp_size, 1], dtype=pl.INT32)
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    seq_lens[r], block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r],
                    k_cache[r], v_cache[r],
                    wo[r], w_g[r],
                    resid1_out[r],
                    tmp_window,
                    signal_window,
                    layer_idx,
                    r,
                    device=r,
                )

    return TpAttentionFull


# =============================================================================
# Distributed-mock torch reference and harness.
#
# The Wave-2 acceptance criterion is that this file parses clean (which is
# verified at import time by ``_build_tp_attention_full_program`` being
# constructible) and that the ``__main__`` harness validates the math via
# a pure-torch 8-rank simulation against a single-card reference.
# =============================================================================
def _torch_single_card_attention_full(
    *,
    hidden_states,
    input_rms_weight,
    wq_full,
    wk_full,
    wv_full,
    q_norm_weight,
    k_norm_weight,
    wo_full,
    w_g_full,
    seq_lens,
    block_table,
    slot_mapping,
    rope_cos,
    rope_sin,
    k_cache_full,
    v_cache_full,
    num_heads_full,
    num_kv_heads_full,
    head_dim,
    rotary_dim,
    rotary_half,
    rotary_pass,
    q_per_kv,
    eps,
    block_size,
    sliding_window=None,
):
    """Pure-torch single-card oracle (the value all 8 ranks should sum to).

    Mirrors the original single-card golden but is parametrised by
    world-level head/kv counts and an optional sliding window (re-used by
    the SWA sibling for symmetry).
    """
    import math

    import torch

    batch = hidden_states.shape[0]
    hidden_q = num_heads_full * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    def zc(x, g):
        return x * (g + 1.0)

    x = hidden_states.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    normed_bf16 = zc(x * torch.rsqrt(var + eps), input_rms_weight.float()).bfloat16()

    q_proj = normed_bf16.float() @ wq_full.float()
    k_proj = normed_bf16.float() @ wk_full.float()
    v_proj = normed_bf16.float() @ wv_full.float()

    q_h = q_proj.view(batch, num_heads_full, head_dim)
    q_h = zc(q_h * torch.rsqrt(q_h.pow(2).mean(-1, keepdim=True) + eps),
             q_norm_weight.float())
    k_h = k_proj.view(batch, num_kv_heads_full, head_dim)
    k_h = zc(k_h * torch.rsqrt(k_h.pow(2).mean(-1, keepdim=True) + eps),
             k_norm_weight.float())

    k_cache = k_cache_full.clone()
    v_cache = v_cache_full.clone()
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ
    attn_out = torch.zeros(batch, hidden_q, dtype=torch.bfloat16)
    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        eff_ctx_len = ctx_len if sliding_window is None else min(ctx_len, sliding_window)
        ctx_blocks = (eff_ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1

        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:rotary_dim]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:rotary_dim]

        kh = k_h[b]
        if rotary_pass > 0:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:rotary_dim] * s_lo,
                kh[:, rotary_half:rotary_dim] * c_hi + kh[:, :rotary_half] * s_hi,
                kh[:, rotary_dim : rotary_dim + rotary_pass]
            ], dim=-1)
        else:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:] * s_lo,
                kh[:, rotary_half:] * c_hi + kh[:, :rotary_half] * s_hi,
            ], dim=-1)

        slot = int(slot_mapping[b].item())
        sb_blk = slot // block_size
        sb_off = slot % block_size
        for ki in range(num_kv_heads_full):
            row = (sb_blk * num_kv_heads_full + ki) * block_size + sb_off
            k_cache[row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[row, :] = v_proj[
                b, ki * head_dim : (ki + 1) * head_dim,
            ].to(torch.bfloat16)

        qh = q_h[b]
        if rotary_pass > 0:
            q_rot = torch.cat([
                qh[:, :rotary_half] * c_lo - qh[:, rotary_half:rotary_dim] * s_lo,
                qh[:, rotary_half:rotary_dim] * c_hi + qh[:, :rotary_half] * s_hi,
                qh[:, rotary_dim : rotary_dim + rotary_pass]
            ], dim=-1)
        else:
            q_rot = torch.cat([
                qh[:, :rotary_half] * c_lo - qh[:, rotary_half:] * s_lo,
                qh[:, rotary_half:] * c_hi + qh[:, :rotary_half] * s_hi,
            ], dim=-1)

        attn_row = torch.zeros(1, hidden_q, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads_full):
            q_base = kvh * q_per_kv
            q_grp = q_rot[q_base : q_base + q_per_kv, :].to(torch.bfloat16)
            oi = torch.zeros(q_per_kv, head_dim)
            li = torch.zeros(q_per_kv, 1)
            mi = torch.zeros(q_per_kv, 1)
            for sb in range(ctx_blocks):
                valid_len = min(block_size, eff_ctx_len - sb * block_size)
                pbid = int(block_table[b * max_ctx_blocks + sb].item())
                cr0 = (pbid * num_kv_heads_full + kvh) * block_size
                kt = k_cache[cr0 : cr0 + block_size, :]
                vt = v_cache[cr0 : cr0 + block_size, :]
                rs = q_grp.float() @ kt.float().T
                if valid_len < block_size:
                    rs[:, valid_len:] = torch.finfo(torch.float32).min
                scores = rs * scale
                cm = scores.max(dim=-1, keepdim=True).values
                es = torch.exp(scores - cm)
                es_b = es.to(torch.bfloat16)
                cl = es_b.float().sum(dim=-1, keepdim=True)
                ot = es_b.float() @ vt.float()
                if sb == 0:
                    oi, li, mi = ot, cl, cm
                else:
                    mn = torch.maximum(mi, cm)
                    a = torch.exp(mi - mn)
                    bw = torch.exp(cm - mn)
                    li = a * li + bw * cl
                    oi = oi * a + ot * bw
                    mi = mn
            ctx = oi / li
            attn_row[
                :, q_base * head_dim : (q_base + q_per_kv) * head_dim,
            ] = ctx.reshape(1, -1).to(torch.bfloat16)
        attn_out[b : b + 1, :] = attn_row

    gate = torch.sigmoid(hidden_states.float() @ w_g_full.float())
    attn_view = attn_out.view(batch, num_heads_full, head_dim).float()
    attn_gated = (attn_view * gate.unsqueeze(-1)).to(torch.bfloat16)
    attn_gated_flat = attn_gated.view(batch, hidden_q)

    o = attn_gated_flat.float() @ wo_full.float()
    resid1 = (o + hidden_states.float()).bfloat16()
    return resid1


def _torch_per_rank_partial_full(
    *,
    rank,
    tp_world_size,
    hidden_states,
    input_rms_weight,
    wq_full,
    wk_full,
    wv_full,
    q_norm_weight,
    k_norm_weight,
    wo_full,
    w_g_full,
    seq_lens,
    block_table,
    slot_mapping,
    rope_cos,
    rope_sin,
    k_cache_full,
    v_cache_full,
    num_heads_full,
    num_kv_heads_full,
    head_dim,
    rotary_dim,
    rotary_half,
    rotary_pass,
    q_per_kv,
    eps,
    block_size,
    sliding_window=None,
):
    """Compute one rank's partial pre-all-reduce o_proj output in pure torch.

    Slices the world-level weights along the TP axis, runs the rank's
    local computation, and returns ``rank_partial_attn`` of shape
    ``[batch, HIDDEN]`` (without the residual add — the harness sums
    these across ranks first, then adds the residual once).
    """
    import math

    import torch

    batch = hidden_states.shape[0]
    heads_local = num_heads_full // tp_world_size
    kv_heads_local = num_kv_heads_full // tp_world_size
    hidden_q_local = heads_local * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    wq_local = wq_full[
        :, rank * hidden_q_local : (rank + 1) * hidden_q_local
    ]
    kv_hidden_local = kv_heads_local * head_dim
    wk_local = wk_full[
        :, rank * kv_hidden_local : (rank + 1) * kv_hidden_local
    ]
    wv_local = wv_full[
        :, rank * kv_hidden_local : (rank + 1) * kv_hidden_local
    ]
    wo_local = wo_full[
        rank * hidden_q_local : (rank + 1) * hidden_q_local, :
    ]
    w_g_local = w_g_full[:, rank * heads_local : (rank + 1) * heads_local]

    num_blocks_total = k_cache_full.shape[0] // (num_kv_heads_full * block_size)
    k_cache_full_view = k_cache_full.view(
        num_blocks_total, num_kv_heads_full, block_size, head_dim,
    )
    v_cache_full_view = v_cache_full.view(
        num_blocks_total, num_kv_heads_full, block_size, head_dim,
    )
    k_cache_local = k_cache_full_view[
        :, rank * kv_heads_local : (rank + 1) * kv_heads_local, :, :,
    ].contiguous().view(
        num_blocks_total * kv_heads_local * block_size, head_dim,
    )
    v_cache_local = v_cache_full_view[
        :, rank * kv_heads_local : (rank + 1) * kv_heads_local, :, :,
    ].contiguous().view(
        num_blocks_total * kv_heads_local * block_size, head_dim,
    )

    def zc(x, g):
        return x * (g + 1.0)

    x = hidden_states.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    normed_bf16 = zc(x * torch.rsqrt(var + eps), input_rms_weight.float()).bfloat16()

    q_proj_local = normed_bf16.float() @ wq_local.float()
    k_proj_local = normed_bf16.float() @ wk_local.float()
    v_proj_local = normed_bf16.float() @ wv_local.float()

    q_h_local = q_proj_local.view(batch, heads_local, head_dim)
    q_h_local = zc(
        q_h_local * torch.rsqrt(q_h_local.pow(2).mean(-1, keepdim=True) + eps),
        q_norm_weight.float(),
    )
    k_h_local = k_proj_local.view(batch, kv_heads_local, head_dim)
    k_h_local = zc(
        k_h_local * torch.rsqrt(k_h_local.pow(2).mean(-1, keepdim=True) + eps),
        k_norm_weight.float(),
    )

    k_cache = k_cache_local.clone()
    v_cache = v_cache_local.clone()
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ
    attn_out_local = torch.zeros(batch, hidden_q_local, dtype=torch.bfloat16)
    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        eff_ctx_len = ctx_len if sliding_window is None else min(ctx_len, sliding_window)
        ctx_blocks = (eff_ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1
        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:rotary_dim]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:rotary_dim]

        kh = k_h_local[b]
        if rotary_pass > 0:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:rotary_dim] * s_lo,
                kh[:, rotary_half:rotary_dim] * c_hi + kh[:, :rotary_half] * s_hi,
                kh[:, rotary_dim : rotary_dim + rotary_pass]
            ], dim=-1)
        else:
            k_rot = torch.cat([
                kh[:, :rotary_half] * c_lo - kh[:, rotary_half:] * s_lo,
                kh[:, rotary_half:] * c_hi + kh[:, :rotary_half] * s_hi,
            ], dim=-1)
        slot = int(slot_mapping[b].item())
        sb_blk = slot // block_size
        sb_off = slot % block_size
        for ki in range(kv_heads_local):
            row = (sb_blk * kv_heads_local + ki) * block_size + sb_off
            k_cache[row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[row, :] = v_proj_local[
                b, ki * head_dim : (ki + 1) * head_dim,
            ].to(torch.bfloat16)

        qh = q_h_local[b]
        if rotary_pass > 0:
            q_rot = torch.cat([
                qh[:, :rotary_half] * c_lo - qh[:, rotary_half:rotary_dim] * s_lo,
                qh[:, rotary_half:rotary_dim] * c_hi + qh[:, :rotary_half] * s_hi,
                qh[:, rotary_dim : rotary_dim + rotary_pass]
            ], dim=-1)
        else:
            q_rot = torch.cat([
                qh[:, :rotary_half] * c_lo - qh[:, rotary_half:] * s_lo,
                qh[:, rotary_half:] * c_hi + qh[:, :rotary_half] * s_hi,
            ], dim=-1)

        attn_row = torch.zeros(1, hidden_q_local, dtype=torch.bfloat16)
        for kvh in range(kv_heads_local):
            q_base = kvh * q_per_kv
            q_grp = q_rot[q_base : q_base + q_per_kv, :].to(torch.bfloat16)
            oi = torch.zeros(q_per_kv, head_dim)
            li = torch.zeros(q_per_kv, 1)
            mi = torch.zeros(q_per_kv, 1)
            for sb in range(ctx_blocks):
                valid_len = min(block_size, eff_ctx_len - sb * block_size)
                pbid = int(block_table[b * max_ctx_blocks + sb].item())
                cr0 = (pbid * kv_heads_local + kvh) * block_size
                kt = k_cache[cr0 : cr0 + block_size, :]
                vt = v_cache[cr0 : cr0 + block_size, :]
                rs = q_grp.float() @ kt.float().T
                if valid_len < block_size:
                    rs[:, valid_len:] = torch.finfo(torch.float32).min
                scores = rs * scale
                cm = scores.max(dim=-1, keepdim=True).values
                es = torch.exp(scores - cm)
                es_b = es.to(torch.bfloat16)
                cl = es_b.float().sum(dim=-1, keepdim=True)
                ot = es_b.float() @ vt.float()
                if sb == 0:
                    oi, li, mi = ot, cl, cm
                else:
                    mn = torch.maximum(mi, cm)
                    a = torch.exp(mi - mn)
                    bw = torch.exp(cm - mn)
                    li = a * li + bw * cl
                    oi = oi * a + ot * bw
                    mi = mn
            ctx = oi / li
            attn_row[
                :, q_base * head_dim : (q_base + q_per_kv) * head_dim,
            ] = ctx.reshape(1, -1).to(torch.bfloat16)
        attn_out_local[b : b + 1, :] = attn_row

    gate_local = torch.sigmoid(hidden_states.float() @ w_g_local.float())
    attn_view = attn_out_local.view(batch, heads_local, head_dim).float()
    attn_gated = (attn_view * gate_local.unsqueeze(-1)).to(torch.bfloat16)
    attn_gated_flat = attn_gated.view(batch, hidden_q_local)
    partial_o = (attn_gated_flat.float() @ wo_local.float()).to(torch.bfloat16)
    return partial_o


def _run_distributed_mock(
    *,
    batch,
    max_seq,
    layer_idx,
    pass_rate,
    rtol,
    atol,
    seed,
):
    """Simulate TP=TP_WORLD_SIZE ranks in a torch loop and validate.

    The reference is the single-card oracle. The TP path computes each
    rank's partial o_proj output independently, sums them across the
    8 ranks (mocking ``tp_all_reduce``), and then adds the replicated
    residual.
    """
    import torch

    torch.manual_seed(seed)

    if not is_full_attention(layer_idx):
        raise ValueError(
            f"layer_idx={layer_idx} is not a full-attention layer"
        )

    layer_rope_theta = LAYER_ROPE_THETA[layer_idx]
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    num_heads_full = NUM_HEADS_FULL_LOCAL * TP_WORLD_SIZE                 # 64
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE                    # 8
    hidden_q_full = num_heads_full * HEAD_DIM
    kv_hidden_full = num_kv_heads_full * HEAD_DIM
    cache_rows_full = num_blocks * num_kv_heads_full * BLOCK_SIZE

    rope_cos, rope_sin = build_llama3_yarn_rope_tables(
        max_seq, ROTARY_DIM, layer_rope_theta,
        factor=ROPE_SCALING["factor"],
        low=ROPE_SCALING["low_freq_factor"],
        high=ROPE_SCALING["high_freq_factor"],
        orig_max=ROPE_SCALING["original_max_position_embeddings"],
    )

    synthetic_proj_scale = 0.5
    hidden_states = (torch.rand(batch, HIDDEN) - 0.5).bfloat16()
    input_rms_weight = (torch.rand(1, HIDDEN) - 0.5).float()
    wq_full = (torch.rand(HIDDEN, hidden_q_full) / HIDDEN ** 0.5).bfloat16()
    wk_full = (torch.rand(HIDDEN, kv_hidden_full) / HIDDEN ** 0.5).bfloat16()
    wv_full = (
        synthetic_proj_scale * torch.rand(HIDDEN, kv_hidden_full) / HIDDEN ** 0.5
    ).bfloat16()
    q_norm_weight = (torch.rand(1, HEAD_DIM) - 0.5).float()
    k_norm_weight = (torch.rand(1, HEAD_DIM) - 0.5).float()
    wo_full = (
        synthetic_proj_scale * (torch.rand(hidden_q_full, HIDDEN) - 0.5)
        / hidden_q_full ** 0.5
    ).bfloat16()
    w_g_full = (
        synthetic_proj_scale * (torch.rand(HIDDEN, num_heads_full) - 0.5)
        / HIDDEN ** 0.5
    ).bfloat16()

    seq_lens = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)
    block_table = torch.arange(num_blocks, dtype=torch.int32)
    slot_mapping = torch.empty(batch, dtype=torch.int32)
    for b in range(batch):
        pos = int(seq_lens[b].item()) - 1
        logical_block = pos // BLOCK_SIZE
        page_offset = pos % BLOCK_SIZE
        phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
        slot_mapping[b] = phys_block * BLOCK_SIZE + page_offset
    k_cache_full = (torch.rand(cache_rows_full, HEAD_DIM) - 0.5).bfloat16()
    v_cache_full = (
        synthetic_proj_scale * (torch.rand(cache_rows_full, HEAD_DIM) - 0.5)
    ).bfloat16()

    expected_resid1 = _torch_single_card_attention_full(
        hidden_states=hidden_states,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        wo_full=wo_full, w_g_full=w_g_full,
        seq_lens=seq_lens, block_table=block_table,
        slot_mapping=slot_mapping,
        rope_cos=rope_cos, rope_sin=rope_sin,
        k_cache_full=k_cache_full.clone(),
        v_cache_full=v_cache_full.clone(),
        num_heads_full=num_heads_full,
        num_kv_heads_full=num_kv_heads_full,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rotary_half=ROTARY_HALF,
        rotary_pass=ROTARY_PASS,
        q_per_kv=Q_PER_KV,
        eps=EPS,
        block_size=BLOCK_SIZE,
        sliding_window=None,
    )

    summed_partial = torch.zeros(batch, HIDDEN, dtype=torch.float32)
    for r in range(TP_WORLD_SIZE):
        rank_partial = _torch_per_rank_partial_full(
            rank=r,
            tp_world_size=TP_WORLD_SIZE,
            hidden_states=hidden_states,
            input_rms_weight=input_rms_weight,
            wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            wo_full=wo_full, w_g_full=w_g_full,
            seq_lens=seq_lens, block_table=block_table,
            slot_mapping=slot_mapping,
            rope_cos=rope_cos, rope_sin=rope_sin,
            k_cache_full=k_cache_full.clone(),
            v_cache_full=v_cache_full.clone(),
            num_heads_full=num_heads_full,
            num_kv_heads_full=num_kv_heads_full,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            rotary_half=ROTARY_HALF,
            rotary_pass=ROTARY_PASS,
            q_per_kv=Q_PER_KV,
            eps=EPS,
            block_size=BLOCK_SIZE,
            sliding_window=None,
        )
        summed_partial = summed_partial + rank_partial.float()

    tp_resid1 = (summed_partial + hidden_states.float()).bfloat16()

    close = torch.isclose(tp_resid1, expected_resid1, rtol=rtol, atol=atol)
    rate = close.float().mean().item()
    n_fail = int((~close).sum().item())
    ok = rate >= pass_rate
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] attention_full distributed-mock: pass_rate={rate:.6f} "
        f"threshold={pass_rate:.6f} "
        f"{n_fail}/{tp_resid1.numel()} mismatched rtol={rtol} atol={atol}"
    )
    return ok


def build_tp_attention_full_program(tp_size: int = TP_WORLD_SIZE):
    """Public wrapper for the deferred @pl.program builder."""
    return _build_tp_attention_full_program(tp_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
                        help="Reserved for the Wave-3 real-distributed harness.")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="Reserved for the Wave-3 real-distributed harness.")
    parser.add_argument("-b", "--batch", type=int, default=BATCH)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--layer-idx", type=int, default=0,
                        help="Which full-attention layer to specialise on.")
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build-program-only", action="store_true",
                        default=False,
                        help="Just construct the @pl.program scaffold and exit.")
    args = parser.parse_args()

    if args.max_seq > MAX_SEQ_DEFAULT:
        raise ValueError(
            f"attention_full harness supports max_seq <= {MAX_SEQ_DEFAULT}"
        )

    program_cls = build_tp_attention_full_program(TP_WORLD_SIZE)
    print(
        f"[OK] built @pl.program TpAttentionFull: {program_cls.__name__} "
        f"(tp_size={TP_WORLD_SIZE})"
    )

    if args.build_program_only:
        raise SystemExit(0)

    ok = _run_distributed_mock(
        batch=args.batch,
        max_seq=args.max_seq,
        layer_idx=args.layer_idx,
        pass_rate=args.pass_rate,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
    )
    if not ok:
        raise SystemExit(1)


__all__ = [
    "attention_full",
    "build_tp_attention_full_program",
    "_build_tp_attention_full_program",
    "_torch_single_card_attention_full",
    "_torch_per_rank_partial_full",
    "_run_distributed_mock",
    "NUM_HEADS",
    "HIDDEN_Q",
    "KV_HIDDEN_DIM",
    "NUM_KV_HEADS_DIM",
    "Q_PER_KV",
    "Q_HEAD_BATCH",
    "Q_HEAD_PAD",
    "ROTARY_HALF",
    "ROTARY_DIM",
    "ROTARY_PASS",
    "Q_GROUPS",
    "TOTAL_Q_GROUPS",
    "LAYER_QHIDDEN_ROWS_DYN",
]
