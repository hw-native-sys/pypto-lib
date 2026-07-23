# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 SWA (sliding-window) attention kernel — TP=8 in-place refactor (Phase 9 Wave 2).

Each rank holds a sliding-attention shard:

  - q_proj output: NUM_HEADS_SWA_LOCAL * HEAD_DIM = 12 * 128 = 1536
  - k_proj/v_proj output: KV_HEADS_LOCAL * HEAD_DIM = 1 * 128 = 128
  - o_proj input: NUM_HEADS_SWA_LOCAL * HEAD_DIM = 1536
  - w_g output:   NUM_HEADS_SWA_LOCAL          = 12
  - q_norm / k_norm gamma [HEAD_DIM=128] — REPLICATED on every rank

Compile-time constants baked in (LOCAL means per-rank-after-TP-slicing):

  - NUM_HEADS  = NUM_HEADS_SWA_LOCAL   (12)
  - HIDDEN_Q   = HIDDEN_Q_SWA_LOCAL    (1536)
  - KV_HIDDEN_DIM = KV_HIDDEN_LOCAL    (128)
  - NUM_KV_HEADS_DIM = KV_HEADS_LOCAL  (1)
  - Q_PER_KV   = Q_PER_KV_SWA          (12 ; invariant under TP)
  - Q_HEAD_BATCH = Q_HEAD_BATCH_SWA    (12)
  - Q_HEAD_PAD = Q_HEAD_PAD_SWA        (24)
  - ROTARY_HALF = ROTARY_HALF_SWA      (64 ; partial_rotary_factor = 1.0)
  - ROTARY_DIM  = 2 * ROTARY_HALF      (128 == HEAD_DIM, no pass-through)
  - SLIDING_WINDOW = 512 (per-position mask; orthogonal to TP slicing)

TP collective epilogue
----------------------
After the local o_proj (column-sliced) each rank holds a *partial*
``[BATCH, HIDDEN]`` BF16 sum. ``tp_all_reduce`` sums these across the
TP group so every rank ends up with the fully-reduced o_proj output;
the residual add (``+ current_hidden``) happens afterwards (the
residual is replicated across ranks, so adding it post-all-reduce keeps
the math correct).

The caller (Wave-3 ``decode_layer.py`` / ``decode_fwd.py``) must provide
a per-call-site scratch ``tmp_window`` and ``signal_window`` pair with
the documented shapes — see ``attention_full.py`` for the full contract,
re-stated here for symmetry:

  - ``tmp_window``    : ``pld.DistributedTensor`` view of a
                         ``BATCH * (HIDDEN // TP_WORLD_SIZE) * 2`` byte
                         ``alloc_window_buffer`` slot (BF16).
  - ``signal_window`` : ``pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32]``,
                        zero-initialised. Each call site allocates a
                        fresh signal-window slot because the ring
                        all-reduce increments the cells across its
                        ``2 * (N - 1)`` steps; reusing a slot would
                        corrupt the wait thresholds in subsequent
                        collectives.

Per-layer ``rope_theta = 1e4`` (no yarn scaling on SWA layers per
``yarn_only_types = ["full_attention"]``). The sliding-window mask
(``eff_ctx_len = min(seq_len, SLIDING_WINDOW)``, BLOCK_SIZE=128 →
4 paged blocks per KV head) applies to the local-head slice the same
way it applies to the global heads: per-rank q rows attend only to
window-local k/v rows that the rank's KV cache shard already holds.

TODO(phase-3 SWA cache layout): linear layout retained for Wave 2; the
rotating-slot variant ``((start_pos + s) % WIN)`` is deferred until the
Wave-3 decode_fwd integration.

TODO(npu-tuning): Q_HEAD_PAD_SWA=24 / set_validshape(scores, 12, ...)
exercises the dual-AIV no-op replay path at half-pad == 12, which is
UNTESTED on hardware. Inherited from the single-card SWA draft.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import (
    build_plain_rope_tables,
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
    HIDDEN_Q_SWA_LOCAL,
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
    NUM_HEADS_SWA_LOCAL,
    NUM_HEADS_SWA_LOCAL_PAD,
    OUT_PROJ_K_CHUNK,
    OUT_PROJ_N_CHUNK,
    Q_HEAD_BATCH_SWA,
    Q_HEAD_PAD_SWA,
    Q_OUT_CHUNK,
    Q_PER_KV_SWA,
    ROPE_SEQ_DYN,
    ROTARY_HALF_SWA,
    SLIDING_WINDOW,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    is_full_attention,
)

NUM_HEADS = NUM_HEADS_SWA_LOCAL
HIDDEN_Q = HIDDEN_Q_SWA_LOCAL
KV_HIDDEN_DIM = KV_HIDDEN_LOCAL
NUM_KV_HEADS_DIM = KV_HEADS_LOCAL
Q_PER_KV = Q_PER_KV_SWA
Q_HEAD_BATCH = Q_HEAD_BATCH_SWA
Q_HEAD_PAD = Q_HEAD_PAD_SWA
ROTARY_HALF = ROTARY_HALF_SWA
ROTARY_DIM = ROTARY_HALF * 2
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH                # 1
TOTAL_Q_GROUPS = NUM_KV_HEADS_DIM * Q_GROUPS       # 1
WIN_BLOCKS = (SLIDING_WINDOW + BLOCK_SIZE - 1) // BLOCK_SIZE

# Local override for KV projection's output chunk: KV_HIDDEN_LOCAL (128) is
# below the global KV_OUT_CHUNK=256 default, so we pick the whole local KV
# hidden in a single chunk.
KV_OUT_CHUNK_LOCAL = KV_HIDDEN_LOCAL

LAYER_QHIDDEN_ROWS_DYN = pl.dynamic("LAYER_QHIDDEN_ROWS_DYN")

assert Q_HEAD_PAD % 4 == 0 and Q_HEAD_PAD // 2 >= Q_HEAD_BATCH
assert SLIDING_WINDOW % BLOCK_SIZE == 0
assert BATCH % 2 == 0, (
    "fa_fused pipelines pairs of batches under TP, so BATCH must be even"
)
assert HIDDEN_Q % OUT_PROJ_K_CHUNK == 0
assert HIDDEN % OUT_PROJ_N_CHUNK == 0
assert HIDDEN % TP_WORLD_SIZE == 0
assert KV_HIDDEN_DIM == KV_OUT_CHUNK_LOCAL


# =============================================================================
# Attention body — local compute through gated attn_out, partial o_proj,
# TP all-reduce, then residual add.
# =============================================================================
@pl.jit.inline
def attention_swa(
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_SWA_LOCAL], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_SWA * 2], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_SWA * 2], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16],
    resid1_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[
        [BATCH, HIDDEN // TP_WORLD_SIZE], pl.BF16
    ],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """Step3p5 SWA-attention layer through TP-reduced o_proj + residual."""

    decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
    kv_proj_hidden_blocks = HIDDEN // KV_PROJ_K_CHUNK_LOCAL
    out_proj_k_blocks = HIDDEN_Q_SWA_LOCAL // OUT_PROJ_K_CHUNK
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    bt_stride = pl.tensor.dim(block_table, 0) // user_batch
    batch_padded = BATCH

    layer_hidden_base = layer_idx * HIDDEN
    layer_qhidden_base = layer_idx * HIDDEN_Q_SWA_LOCAL
    layer_cache_base = layer_idx * decode_layer_cache_rows

    q_proj = pl.create_tensor([BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN_LOCAL], dtype=pl.FP32)
    normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    # alloc_tile col dim must be a multiple of 16; NUM_HEADS_SWA_LOCAL=12 is not.
    # Widen gate_logits to 16 cols; the 4 padding cols are written as zeros by
    # the padded matmul and never read (swa_head_gate iterates only heads 0-11).
    _GATE_N_PAD = 16
    gate_logits = pl.create_tensor([BATCH, _GATE_N_PAD], dtype=pl.FP32)

    # ----- Scope 1.a — zero-centred input RMSNorm. -----
    # input_rms_weight is replicated across TP ranks (HIDDEN dim is not
    # sliced); every rank computes the same normed_all tile.
    for rms_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="swa_rmsnorm_zc"):
        rms_b0 = rms_spmd_idx * BATCH_TILE
        partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(decode_scope1_hidden_blocks):
            sq_k0 = kb * INPUT_PROJ_K_CHUNK
            sq_chunk = pl.cast(
                pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [rms_b0, sq_k0]),
                target_type=pl.FP32,
            )
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(sq_chunk, sq_chunk)), [1, BATCH_TILE]),
            )
        variance = pl.reshape(
            pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH_TILE, 1],
        )
        inv_rms = pl.recip(pl.sqrt(variance))
        for kb in pl.range(decode_scope1_hidden_blocks):
            norm_k0 = kb * INPUT_PROJ_K_CHUNK
            norm_chunk = pl.cast(
                pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [rms_b0, norm_k0]),
                target_type=pl.FP32,
            )
            gamma = pl.slice(input_rms_weight, [1, INPUT_PROJ_K_CHUNK], [layer_idx, norm_k0])
            scaled = pl.row_expand_mul(norm_chunk, inv_rms)
            normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
            normed_all = pl.assemble(
                normed_all, pl.cast(normed, target_type=pl.BF16), [rms_b0, norm_k0],
            )

    # ----- Scope 1.b — Q projection. -----
    # wq is row-sliced (output dim → HIDDEN_Q_SWA_LOCAL = 1536 per rank).
    for q_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * (HIDDEN_Q_SWA_LOCAL // Q_OUT_CHUNK), name_hint="swa_q_proj",
    ):
        q_b_idx = q_spmd_idx // (HIDDEN_Q_SWA_LOCAL // Q_OUT_CHUNK)
        q_ob = q_spmd_idx % (HIDDEN_Q_SWA_LOCAL // Q_OUT_CHUNK)
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
    for k_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="swa_k_proj"):
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
    for v_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="swa_v_proj"):
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
    # q_norm / k_norm gamma [HEAD_DIM] are REPLICATED across TP ranks; only
    # the per-head loop bounds shrink (KV_HEADS_LOCAL = 1 per rank).
    # Q and K branches run sequentially in the flat spmd body. The pl.range(2)
    # sub-tiling (6 heads per sub-tile, [BATCH_TILE*6, HEAD_DIM] FP32 = 49152 B)
    # keeps the Q-chain Vec/UB peak at ~98688 B (~97 KB clear of the 192 KB
    # ceiling). After pl.range(2) exits all Q tiles have been assembled into
    # q_proj_norm, so the K branch starts with a clean slate.
    for qkn_spmd_idx in pl.spmd(
        (BATCH // BATCH_TILE) * KV_HEADS_LOCAL, name_hint="swa_qk_norm_zc",
    ):
        qkn_b_idx = qkn_spmd_idx // KV_HEADS_LOCAL
        qkn_h = qkn_spmd_idx % KV_HEADS_LOCAL
        qkn_b0 = qkn_b_idx * BATCH_TILE

        # Q branch: 2 half-head sub-tiles (Q_HEAD_BATCH_SWA//2 = 6 heads each).
        qkn_q0 = qkn_h * Q_PER_KV_SWA * HEAD_DIM
        q_gamma = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
        for qh_sub in pl.range(2):
            q_sub_col0 = qkn_q0 + qh_sub * (Q_HEAD_BATCH_SWA // 2) * HEAD_DIM
            q_chunk_sub = pl.reshape(
                pl.slice(
                    q_proj,
                    [BATCH_TILE, (Q_HEAD_BATCH_SWA // 2) * HEAD_DIM],
                    [qkn_b0, q_sub_col0],
                ),
                [BATCH_TILE * (Q_HEAD_BATCH_SWA // 2), HEAD_DIM],
            )
            q_sq = pl.row_sum(pl.mul(q_chunk_sub, q_chunk_sub))
            q_inv = pl.rsqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS))
            q_scaled = pl.row_expand_mul(q_chunk_sub, q_inv)
            q_normed = pl.col_expand_mul(q_scaled, pl.add(q_gamma, 1.0))
            q_normed_flat = pl.reshape(
                q_normed, [BATCH_TILE, (Q_HEAD_BATCH_SWA // 2) * HEAD_DIM],
            )
            q_proj_norm = pl.assemble(q_proj_norm, q_normed_flat, [qkn_b0, q_sub_col0])

        # K branch: single head per rank (KV_HEADS_LOCAL = 1).
        qkn_k0 = qkn_h * HEAD_DIM
        k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [qkn_b0, qkn_k0])
        k_gamma = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
        k_sq = pl.row_sum(pl.mul(k_chunk, k_chunk))
        k_inv = pl.rsqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS))
        k_scaled = pl.row_expand_mul(k_chunk, k_inv)
        k_normed = pl.col_expand_mul(k_scaled, pl.add(k_gamma, 1.0))
        k_proj_norm = pl.assemble(k_proj_norm, k_normed, [qkn_b0, qkn_k0])

    # ----- Scope 1.f — head-wise gate matmul (current_hidden, NOT normed). -----
    # w_g is declared with NUM_HEADS_SWA_LOCAL_PAD=16 cols (the physical weight
    # is zero-padded in the last 4 cols).  Slicing the full 16-col tile avoids
    # materialising a 12-col Vec tile (12×2=24 bytes < 32-byte row-alignment
    # requirement).  The extra 4 matmul output cols are garbage but never used:
    # swa_head_gate reads gate_logits[:, gate_h] for gate_h in 0..11 only.
    for gp_spmd_idx in pl.spmd(BATCH // BATCH_TILE, name_hint="swa_gate_proj"):
        gp_b0 = gp_spmd_idx * BATCH_TILE
        a0 = pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [gp_b0, 0])
        b0t = pl.slice(
            w_g,
            [INPUT_PROJ_K_CHUNK, NUM_HEADS_SWA_LOCAL_PAD],
            [layer_hidden_base, 0],
        )
        gp_acc = pl.matmul(a0, b0t, out_dtype=pl.FP32)
        for kb in pl.range(1, decode_scope1_hidden_blocks):
            k0 = kb * INPUT_PROJ_K_CHUNK
            a = pl.slice(current_hidden, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [gp_b0, k0])
            b = pl.slice(
                w_g,
                [INPUT_PROJ_K_CHUNK, NUM_HEADS_SWA_LOCAL_PAD],
                [layer_hidden_base + k0, 0],
            )
            gp_acc = pl.matmul_acc(gp_acc, a, b)
        gate_logits = pl.assemble(
            gate_logits,
            pl.set_validshape(gp_acc, BATCH_TILE, NUM_HEADS_SWA_LOCAL),
            [gp_b0, 0],
        )

    # ----- Scope 2 — full RoPE + paged KV cache write + fa_fused (SWA). -----
    # Full RoPE (rotary_dim == HEAD_DIM, no pass-through tail). The KV cache
    # holds KV_HEADS_LOCAL = 1 KV head per rank, so the per-rank loop over
    # local KV heads collapses to a single iteration.
    attn_out = pl.create_tensor([BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16)
    # Q_HEAD_PAD_SWA=24 is not a multiple of 16; use SWA_Q_PAD_ALIGNED=32 for
    # all alloc_tile row-dimension uses so the allocator alignment check passes.
    SWA_Q_PAD_ALIGNED = 32
    all_q_padded = pl.create_tensor(
        [BATCH * KV_HEADS_LOCAL * (Q_PER_KV_SWA // Q_HEAD_BATCH_SWA) * SWA_Q_PAD_ALIGNED, HEAD_DIM], dtype=pl.BF16,
    )

    for b in pl.parallel(user_batch):
        ctx_len = pl.tensor.read(seq_lens, [b])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE
        cos_row = pl.slice(rope_cos, [1, ROTARY_HALF_SWA * 2], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, ROTARY_HALF_SWA * 2], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, ROTARY_HALF_SWA], [0, 0])
        cos_hi = pl.slice(cos_row, [1, ROTARY_HALF_SWA], [0, ROTARY_HALF_SWA])
        sin_lo = pl.slice(sin_row, [1, ROTARY_HALF_SWA], [0, 0])
        sin_hi = pl.slice(sin_row, [1, ROTARY_HALF_SWA], [0, ROTARY_HALF_SWA])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_kv_cache"):
            for ki in pl.range(KV_HEADS_LOCAL):
                kv_col = ki * HEAD_DIM
                cache_row = (
                    layer_cache_base
                    + (slot_block * KV_HEADS_LOCAL + ki) * BLOCK_SIZE
                    + slot_offset
                )
                k_lo = pl.slice(k_proj_norm, [1, ROTARY_HALF_SWA], [b, kv_col])
                k_hi = pl.slice(
                    k_proj_norm, [1, ROTARY_HALF_SWA], [b, kv_col + ROTARY_HALF_SWA],
                )
                rot_k_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                rot_k_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_lo, target_type=pl.BF16), [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache, pl.cast(rot_k_hi, target_type=pl.BF16),
                    [cache_row, ROTARY_HALF_SWA],
                )
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(pl.slice(v_proj, [1, HEAD_DIM], [b, kv_col]),
                            target_type=pl.BF16),
                    [cache_row, 0],
                )

                q_base = ki * Q_PER_KV_SWA
                q_block = pl.reshape(
                    pl.slice(
                        q_proj_norm, [1, Q_HEAD_BATCH_SWA * HEAD_DIM],
                        [b, q_base * HEAD_DIM],
                    ),
                    [Q_HEAD_BATCH_SWA, HEAD_DIM],
                )
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH_SWA, ROTARY_HALF_SWA], [0, 0])
                q_hi = pl.slice(q_block, [Q_HEAD_BATCH_SWA, ROTARY_HALF_SWA], [0, ROTARY_HALF_SWA])
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

                pad_row_base = b * KV_HEADS_LOCAL * (Q_PER_KV_SWA // Q_HEAD_BATCH_SWA) * SWA_Q_PAD_ALIGNED + ki * SWA_Q_PAD_ALIGNED
                all_q_padded = pl.assemble(all_q_padded, rot_q_lo_bf16, [pad_row_base, 0])
                all_q_padded = pl.assemble(
                    all_q_padded, rot_q_hi_bf16, [pad_row_base, ROTARY_HALF_SWA],
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(
                        pl.full([SWA_Q_PAD_ALIGNED - Q_HEAD_BATCH_SWA, HEAD_DIM],
                                dtype=pl.FP32, value=0.0),
                        target_type=pl.BF16,
                    ),
                    [pad_row_base + Q_HEAD_BATCH_SWA, 0],
                )

    # ----- fa_fused (SWA) — Phase A (2026-06-11): qwen3/32b-style 4-spmd. -----
    # Mirror of attention_full.py's Phase A rewrite. SWA differs only in:
    #   * Q_HEAD_BATCH_SWA=12 / Q_HEAD_PAD_SWA=24 / SWA_Q_PAD_ALIGNED=32
    #     (instead of full's 8/16/16)
    #   * fa_eff_ctx_len = min(fa_ctx_len, SLIDING_WINDOW) — window clamp on
    #     the iteration count; KV-tile addressing is unchanged (the cache
    #     still spans the full ctx, the window clamp just caps how many
    #     tiles we iterate). At SLIDING_WINDOW=512 and BLOCK_SIZE=128,
    #     SWA_WIN_BLOCKS=4 caps the GM scratch size 8x below the full path.
    # See docs/step3p5/phases/15-singlerank-npu.md "Phase A route decision".
    # Localise the module-level WIN_BLOCKS constant: pypto IR's frontend does
    # not lift bare module globals computed inside the file (only ``from
    # .config import`` names round-trip through the trace).
    SWA_WIN_BLOCKS = (SLIDING_WINDOW + BLOCK_SIZE - 1) // BLOCK_SIZE
    all_raw_scores = pl.create_tensor(
        [BATCH * SWA_WIN_BLOCKS * SWA_Q_PAD_ALIGNED, BLOCK_SIZE], dtype=pl.FP32,
    )
    all_exp_padded = pl.create_tensor(
        [BATCH * SWA_WIN_BLOCKS * SWA_Q_PAD_ALIGNED, BLOCK_SIZE], dtype=pl.BF16,
    )
    all_cur_mi = pl.create_tensor(
        [BATCH * SWA_WIN_BLOCKS * Q_HEAD_BATCH_SWA, 1], dtype=pl.FP32,
    )
    all_cur_li = pl.create_tensor(
        [BATCH * SWA_WIN_BLOCKS * Q_HEAD_BATCH_SWA, 1], dtype=pl.FP32,
    )
    all_oi_tmp = pl.create_tensor(
        [BATCH * SWA_WIN_BLOCKS * SWA_Q_PAD_ALIGNED, HEAD_DIM], dtype=pl.FP32,
    )

    # Stage 1: QK matmul (cube). One core per batch.
    for fa_b in pl.spmd(BATCH, name_hint="swa_qk_matmul"):
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_eff_ctx_len = pl.min(fa_ctx_len, SLIDING_WINDOW)
        fa_ctx_blocks = (fa_eff_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        fa_block_table_base = fa_b_safe * bt_stride
        q_padded_row = fa_b * SWA_Q_PAD_ALIGNED  # KV_HEADS_LOCAL=1, Q_GROUPS=1
        q_padded = pl.slice(
            all_q_padded, [SWA_Q_PAD_ALIGNED, HEAD_DIM], [q_padded_row, 0],
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
            scratch_row = (fa_b * SWA_WIN_BLOCKS + sb) * SWA_Q_PAD_ALIGNED
            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [scratch_row, 0])

    # Stage 2: softmax (vec). pl.slice(valid_shape=) marks Q_HEAD_BATCH_SWA real
    # rows + valid_len columns; fillpad pushes -inf into the masked tail.
    for fa_b in pl.spmd(BATCH, name_hint="swa_softmax"):
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_eff_ctx_len = pl.min(fa_ctx_len, SLIDING_WINDOW)
        fa_ctx_blocks = (fa_eff_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for sb in pl.range(fa_ctx_blocks):
            s0 = sb * BLOCK_SIZE
            valid_len = pl.min(BLOCK_SIZE, fa_eff_ctx_len - s0)
            scratch_row = (fa_b * SWA_WIN_BLOCKS + sb) * SWA_Q_PAD_ALIGNED
            scratch_lm_row = (fa_b * SWA_WIN_BLOCKS + sb) * Q_HEAD_BATCH_SWA
            scores_valid = pl.slice(
                all_raw_scores,
                [Q_HEAD_BATCH_SWA, BLOCK_SIZE],
                [scratch_row, 0],
                valid_shape=[Q_HEAD_BATCH_SWA, valid_len],
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

    # Stage 3: SV matmul (cube). exp_tile uses the full SWA_Q_PAD_ALIGNED row
    # stride; matmul output's bottom (SWA_Q_PAD_ALIGNED - Q_HEAD_BATCH_SWA)
    # rows are garbage from un-initialised GM but are never read by Stage 4.
    for fa_b in pl.spmd(BATCH, name_hint="swa_sv_matmul"):
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_eff_ctx_len = pl.min(fa_ctx_len, SLIDING_WINDOW)
        fa_ctx_blocks = (fa_eff_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        fa_block_table_base = fa_b_safe * bt_stride
        for sb in pl.range(fa_ctx_blocks):
            fa_pbid = pl.cast(
                pl.tensor.read(block_table, [fa_block_table_base + sb]), pl.INDEX,
            )
            fa_cache_row = layer_cache_base + fa_pbid * BLOCK_SIZE
            v_tile = pl.slice(
                v_cache, [BLOCK_SIZE, HEAD_DIM], [fa_cache_row, 0],
            )
            scratch_row = (fa_b * SWA_WIN_BLOCKS + sb) * SWA_Q_PAD_ALIGNED
            exp_tile = pl.slice(
                all_exp_padded,
                [SWA_Q_PAD_ALIGNED, BLOCK_SIZE],
                [scratch_row, 0],
            )
            oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [scratch_row, 0])

    # Stage 4: online softmax accumulation + final normalisation + attn_out
    # write. mi/li/oi carried flat as [Q_HEAD_BATCH_SWA, 1] / [_, HEAD_DIM].
    for fa_b in pl.spmd(BATCH, name_hint="swa_online_softmax"):
        fa_b_safe = pl.min(fa_b, user_batch - 1)
        fa_ctx_len = pl.tensor.read(seq_lens, [fa_b_safe])
        fa_eff_ctx_len = pl.min(fa_ctx_len, SLIDING_WINDOW)
        fa_ctx_blocks = (fa_eff_ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        oi_row0 = fa_b * SWA_WIN_BLOCKS * SWA_Q_PAD_ALIGNED
        lm_row0 = fa_b * SWA_WIN_BLOCKS * Q_HEAD_BATCH_SWA
        oi = pl.slice(all_oi_tmp, [Q_HEAD_BATCH_SWA, HEAD_DIM], [oi_row0, 0])
        mi = pl.slice(all_cur_mi, [Q_HEAD_BATCH_SWA, 1], [lm_row0, 0])
        li = pl.slice(all_cur_li, [Q_HEAD_BATCH_SWA, 1], [lm_row0, 0])
        for sb in pl.range(1, fa_ctx_blocks):
            sb_oi_row = oi_row0 + sb * SWA_Q_PAD_ALIGNED
            sb_lm_row = lm_row0 + sb * Q_HEAD_BATCH_SWA
            oi_partial = pl.slice(
                all_oi_tmp, [Q_HEAD_BATCH_SWA, HEAD_DIM], [sb_oi_row, 0],
            )
            cur_mi = pl.slice(
                all_cur_mi, [Q_HEAD_BATCH_SWA, 1], [sb_lm_row, 0],
            )
            cur_li = pl.slice(
                all_cur_li, [Q_HEAD_BATCH_SWA, 1], [sb_lm_row, 0],
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
            pl.reshape(ctx, [1, Q_HEAD_BATCH_SWA * HEAD_DIM]),
            target_type=pl.BF16,
        )
        # q_base = kvh * Q_PER_KV_SWA == 0 (KV_HEADS_LOCAL=1, kvh=0).
        attn_out = pl.assemble(attn_out, ctx_flat_bf16, [fa_b, 0])

    # ----- Scope 2.5 — head-wise sigmoid gate (local heads only). -----
    # Phase 15.1 mirror of attention_full.py 15.A: outer spmd over batch
    # tiles only; load full gate_logits row in one ND read; intra-tile
    # per-head pluck. Avoids the [BATCH_TILE, 1] DN-Vec ↔ ND-GM TLOAD pair
    # rejected by pto-isa TLoad.hpp:459 isSameLayout static_assert.
    gated_attn_out = pl.create_tensor([BATCH, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16)
    for gate_spmd_idx in pl.spmd(
        BATCH // BATCH_TILE, name_hint="swa_head_gate",
    ):
        gate_b0 = gate_spmd_idx * BATCH_TILE
        gate_row_fp32 = pl.slice(
            gate_logits,
            [BATCH_TILE, NUM_HEADS_SWA_LOCAL_PAD],
            [gate_b0, 0],
        )
        sigmoid_all = pl.recip(
            pl.add(pl.exp(pl.neg(gate_row_fp32)), 1.0),
        )
        for gate_h in pl.range(NUM_HEADS_SWA_LOCAL):
            gate_h0 = gate_h * HEAD_DIM
            head_slice = pl.slice(
                attn_out, [BATCH_TILE, HEAD_DIM], [gate_b0, gate_h0],
            )
            hg_gate = pl.slice(sigmoid_all, [BATCH_TILE, 1], [0, gate_h])
            hg_gated_fp32 = pl.row_expand_mul(
                pl.cast(head_slice, target_type=pl.FP32), hg_gate,
            )
            gated = pl.cast(hg_gated_fp32, target_type=pl.BF16)
            gated_attn_out = pl.assemble(gated_attn_out, gated, [gate_b0, gate_h0])

    # ----- Scope 3.a — local o_proj (partial result, no residual yet). -----
    # wo is column-sliced (input dim → HIDDEN_Q_SWA_LOCAL = 1536 per rank);
    # the output is a partial [BATCH, HIDDEN] BF16 tensor that must be
    # summed across the TP group via the all-reduce below before residual.
    #
    # Phase A (2026-06-11): mirror of attention_full.py — split the cube
    # matmul + vec cast into two separate spmds so PTOAS does not lower
    # this scope to a MixedKernels dispatch (the mixed-mode AICore root is
    # the 507018 VEC UB alignment crash site; see phase-15 doc).
    partial_attn_proj_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for ob in pl.spmd(
            HIDDEN // OUT_PROJ_N_CHUNK, name_hint="swa_out_proj_matmul",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            o0 = ob * OUT_PROJ_N_CHUNK
            a_chunk_0 = pl.slice(
                gated_attn_out, [BATCH_TILE, OUT_PROJ_K_CHUNK], [b0, 0],
            )
            w_chunk_0 = pl.slice(
                wo, [OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK], [layer_qhidden_base, o0],
            )
            o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
            for kb in pl.range(1, out_proj_k_blocks):
                k0 = kb * OUT_PROJ_K_CHUNK
                a_chunk = pl.slice(
                    gated_attn_out, [BATCH_TILE, OUT_PROJ_K_CHUNK], [b0, k0],
                )
                w_chunk = pl.slice(
                    wo, [OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK],
                    [layer_qhidden_base + k0, o0],
                )
                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)
            partial_attn_proj_fp32 = pl.assemble(
                partial_attn_proj_fp32, o_acc, [b0, o0],
            )

    partial_attn_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for ob in pl.spmd(
            HIDDEN // OUT_PROJ_N_CHUNK, name_hint="swa_out_proj_cast",
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
    # The pull-side ring all-reduce body now lives as a class method on
    # the enclosing @pl.program class (see TpAttentionSwa.tp_all_reduce
    # below). Phase X.2 lifted the call out of the @pl.jit.inline wrapper
    # in collectives.py — mixing the two pypto worlds is unsupported.
    # Phase 15.1 mirror of attention_full 15.B: at TP=1 skip the call so
    # orchestration codegen does not emit a stale SSA rename for the
    # SimplifyPass-elided ring loop body.
    if TP_WORLD_SIZE > 1:
        partial_attn_proj = self.tp_all_reduce(
            partial_attn_proj,
            tmp_window,
            signal_window,
            my_rank,
        )

    # ----- Scope 3.c — residual add (post-all-reduce). -----
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        for ob in pl.spmd(
            HIDDEN // OUT_PROJ_N_CHUNK, name_hint="swa_out_resid_add",
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
# =============================================================================
def _build_tp_attention_swa_program(tp_size: int = TP_WORLD_SIZE):
    """Return a freshly-built ``@pl.program`` class for the SWA-attention
    TP epilogue.

    Constructed inside a function so the module imports even on hosts that
    have not finished bringing up the pypto runtime (deferred-build
    pattern).
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    attention_swa_inline = pl.inline(attention_swa._func)
    tp_chunk = HIDDEN // tp_size

    @pl.program
    class TpAttentionSwa:
        # ---------- Collective: TP all_reduce (lifted from collectives.py) ----
        # Phase X.2: pull-side ring all-reduce body, baked with t_rows=BATCH,
        # d_cols=HIDDEN, group_size=tp_size from the factory closure.
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
            resid1_out = attention_swa_inline(
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

    return TpAttentionSwa


# =============================================================================
# Distributed-mock torch reference and harness.
# =============================================================================
def _torch_single_card_attention_swa(
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
    rotary_half,
    q_per_kv,
    eps,
    block_size,
    sliding_window,
):
    """Pure-torch single-card oracle for the SWA path.

    SWA variant: ``rotary_dim == head_dim`` (no pass-through tail), and
    ``eff_ctx_len = min(seq_len, sliding_window)``.
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

    def per_head(x_flat, num_h, gamma):
        gef = gamma.float() + 1.0
        xh = x_flat.view(batch, num_h, head_dim).float()
        return (
            xh * torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + eps) * gef
        ).view(batch, num_h * head_dim)

    q_proj_norm = per_head(q_proj, num_heads_full, q_norm_weight[0:1, :])
    k_proj_norm = per_head(k_proj, num_kv_heads_full, k_norm_weight[0:1, :])

    gate_logits = hidden_states.float() @ w_g_full.float()
    k_cache = k_cache_full.clone()
    v_cache = v_cache_full.clone()
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ
    attn_out = torch.zeros(batch, hidden_q, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        eff_ctx_len = min(ctx_len, sliding_window)
        eff_ctx_blocks = (eff_ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1

        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:]

        slot = int(slot_mapping[b].item())
        sb_blk = slot // block_size
        sb_off = slot % block_size
        kh = k_proj_norm[b].view(num_kv_heads_full, head_dim).float()
        k_rot = torch.cat([
            kh[:, :rotary_half] * c_lo - kh[:, rotary_half:] * s_lo,
            kh[:, rotary_half:] * c_hi + kh[:, :rotary_half] * s_hi,
        ], dim=-1)
        for ki in range(num_kv_heads_full):
            row = (sb_blk * num_kv_heads_full + ki) * block_size + sb_off
            k_cache[row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[row, :] = v_proj[
                b, ki * head_dim : (ki + 1) * head_dim,
            ].to(torch.bfloat16)

        qh = q_proj_norm[b].view(num_heads_full, head_dim).float()
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
            for sb in range(eff_ctx_blocks):
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

    gate = torch.sigmoid(gate_logits).unsqueeze(-1)
    gated = (
        attn_out.view(batch, num_heads_full, head_dim).float() * gate
    ).view(batch, num_heads_full * head_dim).to(torch.bfloat16)

    o = gated.float() @ wo_full.float()
    resid1 = (o + hidden_states.float()).bfloat16()
    return resid1


def _torch_per_rank_partial_swa(
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
    rotary_half,
    q_per_kv,
    eps,
    block_size,
    sliding_window,
):
    """Compute one rank's partial pre-all-reduce o_proj output (SWA path)."""
    import math

    import torch

    batch = hidden_states.shape[0]
    heads_local = num_heads_full // tp_world_size
    kv_heads_local = num_kv_heads_full // tp_world_size
    hidden_q_local = heads_local * head_dim
    kv_hidden_local = kv_heads_local * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    wq_local = wq_full[:, rank * hidden_q_local : (rank + 1) * hidden_q_local]
    wk_local = wk_full[:, rank * kv_hidden_local : (rank + 1) * kv_hidden_local]
    wv_local = wv_full[:, rank * kv_hidden_local : (rank + 1) * kv_hidden_local]
    wo_local = wo_full[rank * hidden_q_local : (rank + 1) * hidden_q_local, :]
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
    ].contiguous().view(num_blocks_total * kv_heads_local * block_size, head_dim)
    v_cache_local = v_cache_full_view[
        :, rank * kv_heads_local : (rank + 1) * kv_heads_local, :, :,
    ].contiguous().view(num_blocks_total * kv_heads_local * block_size, head_dim)

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
        eff_ctx_len = min(ctx_len, sliding_window)
        eff_ctx_blocks = (eff_ctx_len + block_size - 1) // block_size
        pos = ctx_len - 1
        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:]

        kh = k_h_local[b]
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
            for sb in range(eff_ctx_blocks):
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
    """Simulate TP=TP_WORLD_SIZE ranks in a torch loop and validate (SWA)."""
    import torch

    torch.manual_seed(seed)

    if is_full_attention(layer_idx):
        raise ValueError(
            f"layer_idx={layer_idx} is not a sliding-attention layer"
        )

    layer_rope_theta = LAYER_ROPE_THETA[layer_idx]
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    num_heads_full = NUM_HEADS_SWA_LOCAL * TP_WORLD_SIZE                  # 96
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE                    # 8
    hidden_q_full = num_heads_full * HEAD_DIM
    kv_hidden_full = num_kv_heads_full * HEAD_DIM
    cache_rows_full = num_blocks * num_kv_heads_full * BLOCK_SIZE

    rope_cos, rope_sin = build_plain_rope_tables(
        max_seq, ROTARY_DIM, layer_rope_theta,
    )

    synthetic_proj_scale = 0.5
    hidden_states = (torch.rand(batch, HIDDEN) - 0.5).bfloat16()
    input_rms_weight = ((torch.rand(1, HIDDEN) - 0.5) * 0.1).float()
    wq_full = (torch.rand(HIDDEN, hidden_q_full) / HIDDEN ** 0.5).bfloat16()
    wk_full = (torch.rand(HIDDEN, kv_hidden_full) / HIDDEN ** 0.5).bfloat16()
    wv_full = (
        synthetic_proj_scale * torch.rand(HIDDEN, kv_hidden_full) / HIDDEN ** 0.5
    ).bfloat16()
    q_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    k_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    wo_full = (
        synthetic_proj_scale * (torch.rand(hidden_q_full, HIDDEN) - 0.5)
        / hidden_q_full ** 0.5
    ).bfloat16()
    w_g_full = (
        synthetic_proj_scale * (torch.rand(HIDDEN, num_heads_full) - 0.5)
        / HIDDEN ** 0.5
    ).bfloat16()

    # Cover the SWA seq_len regimes (single tile, multi tile, full window,
    # window+1, etc) — same pattern as the single-card SWA harness.
    seq_len_pattern = torch.tensor(
        [9, 31, 62, SLIDING_WINDOW - 1, SLIDING_WINDOW, SLIDING_WINDOW + 1,
         max_seq, max_seq // 2],
        dtype=torch.int32,
    )
    repeat = (batch + seq_len_pattern.numel() - 1) // seq_len_pattern.numel()
    seq_lens = seq_len_pattern.repeat(repeat)[:batch].clone()
    seq_lens = torch.clamp(seq_lens, min=1, max=max_seq)
    block_table = torch.arange(num_blocks, dtype=torch.int32)
    slot_mapping = torch.empty(batch, dtype=torch.int32)
    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        eff_ctx_len = min(ctx_len, SLIDING_WINDOW)
        slot_pos = eff_ctx_len - 1
        logical_block = slot_pos // BLOCK_SIZE
        page_offset = slot_pos % BLOCK_SIZE
        phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
        slot_mapping[b] = phys_block * BLOCK_SIZE + page_offset
    k_cache_full = (torch.rand(cache_rows_full, HEAD_DIM) - 0.5).bfloat16()
    v_cache_full = (
        synthetic_proj_scale * (torch.rand(cache_rows_full, HEAD_DIM) - 0.5)
    ).bfloat16()

    expected_resid1 = _torch_single_card_attention_swa(
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
        rotary_half=ROTARY_HALF,
        q_per_kv=Q_PER_KV,
        eps=EPS,
        block_size=BLOCK_SIZE,
        sliding_window=SLIDING_WINDOW,
    )

    summed_partial = torch.zeros(batch, HIDDEN, dtype=torch.float32)
    for r in range(TP_WORLD_SIZE):
        rank_partial = _torch_per_rank_partial_swa(
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
            rotary_half=ROTARY_HALF,
            q_per_kv=Q_PER_KV,
            eps=EPS,
            block_size=BLOCK_SIZE,
            sliding_window=SLIDING_WINDOW,
        )
        summed_partial = summed_partial + rank_partial.float()

    tp_resid1 = (summed_partial + hidden_states.float()).bfloat16()

    close = torch.isclose(tp_resid1, expected_resid1, rtol=rtol, atol=atol)
    rate = close.float().mean().item()
    n_fail = int((~close).sum().item())
    ok = rate >= pass_rate
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] attention_swa distributed-mock: pass_rate={rate:.6f} "
        f"threshold={pass_rate:.6f} "
        f"{n_fail}/{tp_resid1.numel()} mismatched rtol={rtol} atol={atol}"
    )
    return ok


def build_tp_attention_swa_program(tp_size: int = TP_WORLD_SIZE):
    """Public wrapper for the deferred @pl.program builder."""
    return _build_tp_attention_swa_program(tp_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
                        help="Reserved for the Wave-3 real-distributed harness.")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="Reserved for the Wave-3 real-distributed harness.")
    parser.add_argument("-b", "--batch", type=int, default=BATCH)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ_DEFAULT)
    parser.add_argument("--layer-idx", type=int, default=1,
                        help="Which sliding-attention layer to specialise on.")
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build-program-only", action="store_true",
                        default=False,
                        help="Just construct the @pl.program scaffold and exit.")
    args = parser.parse_args()

    program_cls = build_tp_attention_swa_program(TP_WORLD_SIZE)
    print(
        f"[OK] built @pl.program TpAttentionSwa: {program_cls.__name__} "
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
    "attention_swa",
    "build_tp_attention_swa_program",
    "_build_tp_attention_swa_program",
    "_torch_single_card_attention_swa",
    "_torch_per_rank_partial_swa",
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
    "Q_GROUPS",
    "TOTAL_Q_GROUPS",
    "WIN_BLOCKS",
    "LAYER_QHIDDEN_ROWS_DYN",
]
