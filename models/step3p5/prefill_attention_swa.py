# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 prefill SWA (sliding-window) attention kernel — TP=8 (Phase 6).

Sequence-major counterpart of the decode-side ``attention_swa.py``.
Per-token attention is **causal** AND **window-clamped**: query
``t`` attends to keys at positions
``[max(0, position[t] - SLIDING_WINDOW + 1), position[t]]``.

Per-rank head counts and Q/K/V slicing match the decode TP convention
(``NUM_HEADS_SWA_LOCAL = 12``, ``KV_HEADS_LOCAL = 1``). RoPE is the
partial-1.0 variant (``rotary_dim = HEAD_DIM = 128``, no pass-through
tail) and the cos/sin tables are plain (un-scaled).

Pipeline (per rank, per layer)
------------------------------
  Scope 1 (delegated to ``prefill_qkv_proj_rope.py``):
      input_rmsnorm → wq/wk/wv → q_norm/k_norm → partial RoPE
      → emit q_rot / k_rot / v_proj + gate_logits
  Scope 2 (this file):
      KV cache write at ``positions[t]`` → causal+SWA flash attention
      → head-wise gate (per-rank heads only)
  Scope 3 (this file):
      local o_proj → tp_all_reduce → residual add (post-all-reduce)

Per-card weight bundle (host weight loader contract)
----------------------------------------------------
  * ``input_rms_weight[LAYER, HIDDEN]`` FP32 (replicated)
  * ``wq[LAYER * HIDDEN, HIDDEN_Q_SWA_LOCAL=1536]`` BF16
  * ``wk[LAYER * HIDDEN, KV_HIDDEN_LOCAL=128]`` BF16
  * ``wv[LAYER * HIDDEN, KV_HIDDEN_LOCAL=128]`` BF16
  * ``q_norm_weight[LAYER, HEAD_DIM]`` FP32 (replicated)
  * ``k_norm_weight[LAYER, HEAD_DIM]`` FP32 (replicated)
  * ``w_g[LAYER * HIDDEN, NUM_HEADS_SWA_LOCAL=12]`` BF16
  * ``wo[LAYER * HIDDEN_Q_SWA_LOCAL, HIDDEN]`` BF16 (column-sliced
    by HIDDEN_Q axis; partial sum across TP group via tp_all_reduce)
  * ``rope_cos[ROPE_SEQ, ROTARY_DIM=128]`` FP32 (plain, replicated)
  * ``rope_sin[ROPE_SEQ, ROTARY_DIM=128]`` FP32 (plain, replicated)
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import (
    build_plain_rope_tables,
    head_wise_gate_apply,
    tp_all_reduce,
)
from .config import (
    ATTN_SCALE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    HEAD_DIM,
    HIDDEN,
    HIDDEN_Q_SWA_LOCAL,
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
    Q_PER_KV_SWA,
    ROPE_SEQ_DYN,
    ROTARY_HALF_SWA,
    SLIDING_WINDOW,
    TP_WORLD_SIZE,
)
from .prefill_qkv_proj_rope import (
    PREFILL_BATCH,
    PREFILL_SEQ,
    PREFILL_T,
    TOK_TILE,
    _torch_prefill_qkv_oracle_impl,
)


NUM_HEADS = NUM_HEADS_SWA_LOCAL          # 12
HIDDEN_Q = HIDDEN_Q_SWA_LOCAL            # 1536
KV_HIDDEN_DIM = KV_HIDDEN_LOCAL          # 128
NUM_KV_HEADS_DIM = KV_HEADS_LOCAL        # 1
Q_PER_KV = Q_PER_KV_SWA                  # 12
ROTARY_HALF = ROTARY_HALF_SWA            # 64
ROTARY_DIM = ROTARY_HALF * 2             # 128
WIN = SLIDING_WINDOW                     # 512

LAYER_QHIDDEN_ROWS_DYN = pl.dynamic("LAYER_QHIDDEN_ROWS_DYN_PREFILL_SWA")


assert HIDDEN_Q % OUT_PROJ_K_CHUNK == 0
assert HIDDEN % OUT_PROJ_N_CHUNK == 0
assert HIDDEN % TP_WORLD_SIZE == 0


# =============================================================================
# Prefill SWA-attention body (Scope 2 + Scope 3).
#
# The Scope-1 prelude (input RMSNorm + Q/K/V projection + per-head q/k norm
# + partial RoPE + head-wise gate matmul) is inlined directly inside the
# body — see Phase X.9. The factory in ``prefill_qkv_proj_rope.py`` is no
# longer invoked; its math is materialised in-place so the pypto frontend
# can splice the body cleanly into ``PrefillLayerDense.chip_orch`` /
# ``PrefillLayerMoE.chip_orch`` ``@pl.function`` consumers.
# =============================================================================


@pl.jit.inline
def attention_swa_prefill(
    current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_SWA_LOCAL], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_SWA * 2], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_HALF_SWA * 2], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16],
    positions: pl.Tensor[[PREFILL_T], pl.INT32],
    resid1_out: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[
        [PREFILL_T, HIDDEN // TP_WORLD_SIZE], pl.BF16
    ],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Step3p5 SWA prefill body — causal + sliding-window mask, head-wise gate.

    Phase X.9 — closure-undefined constants are inlined as numeric
    Python literals throughout the body. The pypto inline parser
    captures closure variables from ``prefill_fwd.py``'s frame at
    ``pl.inline(...)`` time, not from this module's globals; constants
    only present here therefore cannot be resolved when the body is
    spliced into ``PrefillLayerDense.chip_orch`` /
    ``PrefillLayerMoE.chip_orch``. Literals make every shape /
    arithmetic argument an unambiguous compile-time integer.
    """
    layer_qhidden_base = layer_idx * HIDDEN_Q_SWA_LOCAL
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    layer_cache_base = layer_idx * layer_cache_rows

    # ── Scope 1 — inlined prefill QKV+RoPE body (swa, Phase X.9). ────────
    # SWA variant: NUM_HEADS=12, HIDDEN_Q=1536, Q_PER_KV=12, KV_HEADS=1,
    # ROTARY_HALF=64, ROTARY_DIM=128, rotary_pass=0 (full-rotary).
    normed_tile = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
    q_rot = pl.create_tensor([PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16)
    k_rot = pl.create_tensor([PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.BF16)
    v_tile = pl.create_tensor([PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.BF16)
    gate_logits = pl.create_tensor(
        [PREFILL_T, NUM_HEADS_SWA_LOCAL_PAD], dtype=pl.FP32,
    )

    qkv_d_blocks = HIDDEN // 256
    qkv_q_blocks = HIDDEN_Q_SWA_LOCAL // 128
    layer_hidden_base = layer_idx * HIDDEN

    # ── Stage 1.a — replicated zero-centred input RMSNorm. ───────────
    for tg_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_swa_rmsnorm_zc",
    ):
        tg = tg_idx * TOK_TILE
        partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(qkv_d_blocks):
            k0 = kb * 256
            chunk = pl.cast(
                pl.slice(
                    current_hidden,
                    [TOK_TILE, 256], [tg, k0],
                ),
                target_type=pl.FP32,
            )
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(
                    pl.row_sum(pl.mul(chunk, chunk)), [1, TOK_TILE],
                ),
            )
        inv_rms = pl.reshape(
            pl.recip(
                pl.sqrt(
                    pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                ),
            ),
            [TOK_TILE, 1],
        )
        for kb in pl.range(qkv_d_blocks):
            k0 = kb * 256
            chunk = pl.cast(
                pl.slice(
                    current_hidden,
                    [TOK_TILE, 256], [tg, k0],
                ),
                target_type=pl.FP32,
            )
            gamma = pl.slice(
                input_rms_weight,
                [1, 256], [layer_idx, k0],
            )
            scaled_rms = pl.row_expand_mul(chunk, inv_rms)
            normed_rms = pl.col_expand_mul(scaled_rms, pl.add(gamma, 1.0))
            normed_tile = pl.assemble(
                normed_tile,
                pl.cast(normed_rms, target_type=pl.BF16),
                [tg, k0],
            )

    # ── Stage 1.b — Q projection (per-rank heads). ───────────────────
    q_proj = pl.create_tensor(
        [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.FP32,
    )
    for q_idx in pl.spmd(
        (PREFILL_T // TOK_TILE) * qkv_q_blocks,
        name_hint="prefill_swa_q_proj",
    ):
        qb_idx = q_idx // qkv_q_blocks
        qo_idx = q_idx % qkv_q_blocks
        tg = qb_idx * TOK_TILE
        q_o0 = qo_idx * 128
        q_a0 = pl.slice(
            normed_tile, [TOK_TILE, 256], [tg, 0],
        )
        q_w0 = pl.slice(
            wq, [256, 128],
            [layer_hidden_base, q_o0],
        )
        q_acc = pl.matmul(q_a0, q_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qkv_d_blocks):
            k0 = kb * 256
            q_a = pl.slice(
                normed_tile, [TOK_TILE, 256], [tg, k0],
            )
            q_w = pl.slice(
                wq, [256, 128],
                [layer_hidden_base + k0, q_o0],
            )
            q_acc = pl.matmul_acc(q_acc, q_a, q_w)
        q_proj = pl.assemble(q_proj, q_acc, [tg, q_o0])

    # ── Stage 1.c — K projection. ────────────────────────────────────
    k_proj = pl.create_tensor(
        [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
    )
    for tg_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_swa_k_proj",
    ):
        tg = tg_idx * TOK_TILE
        k_a0 = pl.slice(
            normed_tile, [TOK_TILE, 256], [tg, 0],
        )
        k_w0 = pl.slice(
            wk, [256, KV_HIDDEN_LOCAL],
            [layer_hidden_base, 0],
        )
        k_acc = pl.matmul(k_a0, k_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qkv_d_blocks):
            k0 = kb * 256
            k_a = pl.slice(
                normed_tile, [TOK_TILE, 256], [tg, k0],
            )
            k_w = pl.slice(
                wk, [256, KV_HIDDEN_LOCAL],
                [layer_hidden_base + k0, 0],
            )
            k_acc = pl.matmul_acc(k_acc, k_a, k_w)
        k_proj = pl.assemble(k_proj, k_acc, [tg, 0])

    # ── Stage 1.d — V projection. ────────────────────────────────────
    v_proj = pl.create_tensor(
        [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
    )
    for tg_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_swa_v_proj",
    ):
        tg = tg_idx * TOK_TILE
        v_a0 = pl.slice(
            normed_tile, [TOK_TILE, 256], [tg, 0],
        )
        v_w0 = pl.slice(
            wv, [256, KV_HIDDEN_LOCAL],
            [layer_hidden_base, 0],
        )
        v_acc = pl.matmul(v_a0, v_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qkv_d_blocks):
            k0 = kb * 256
            v_a = pl.slice(
                normed_tile, [TOK_TILE, 256], [tg, k0],
            )
            v_w = pl.slice(
                wv, [256, KV_HIDDEN_LOCAL],
                [layer_hidden_base + k0, 0],
            )
            v_acc = pl.matmul_acc(v_acc, v_a, v_w)
        v_proj = pl.assemble(v_proj, v_acc, [tg, 0])

    # ── Stage 1.e — head-wise gate matmul (on un-normed input). ──────
    for tg_idx in pl.spmd(
        PREFILL_T // TOK_TILE, name_hint="prefill_swa_gate_proj",
    ):
        tg = tg_idx * TOK_TILE
        g_a0 = pl.slice(
            current_hidden, [TOK_TILE, 256], [tg, 0],
        )
        g_w0 = pl.slice(
            w_g, [256, NUM_HEADS_SWA_LOCAL_PAD],
            [layer_hidden_base, 0],
        )
        g_acc = pl.matmul(g_a0, g_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, qkv_d_blocks):
            k0 = kb * 256
            g_a = pl.slice(
                current_hidden,
                [TOK_TILE, 256], [tg, k0],
            )
            g_w = pl.slice(
                w_g, [256, NUM_HEADS_SWA_LOCAL_PAD],
                [layer_hidden_base + k0, 0],
            )
            g_acc = pl.matmul_acc(g_acc, g_a, g_w)
        gate_logits = pl.assemble(
            gate_logits,
            pl.set_validshape(g_acc, TOK_TILE, NUM_HEADS_SWA_LOCAL),
            [tg, 0],
        )

    # ── Stage 1.f — per-head zero-centred q_norm / k_norm. ───────────
    q_proj_norm = pl.create_tensor(
        [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.FP32,
    )
    k_proj_norm = pl.create_tensor(
        [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
    )
    for qkn_idx in pl.spmd(
        (PREFILL_T // TOK_TILE) * 1,
        name_hint="prefill_swa_qk_norm_zc",
    ):
        tg_idx2 = qkn_idx // 1
        kh = qkn_idx % 1
        tg = tg_idx2 * TOK_TILE
        q_col = kh * 12 * HEAD_DIM
        q_chunk = pl.reshape(
            pl.slice(
                q_proj, [TOK_TILE, 12 * HEAD_DIM], [tg, q_col],
            ),
            [TOK_TILE * 12, HEAD_DIM],
        )
        q_gamma = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
        # Phase X.7: per_head_qk_norm body inlined.
        q_sq = pl.row_sum(pl.mul(q_chunk, q_chunk))
        q_inv = pl.rsqrt(pl.add(pl.mul(q_sq, 0.0078125), EPS))
        q_scaled = pl.row_expand_mul(q_chunk, q_inv)
        q_normed = pl.col_expand_mul(q_scaled, pl.add(q_gamma, 1.0))
        q_normed_flat = pl.reshape(
            q_normed, [TOK_TILE, 12 * HEAD_DIM],
        )
        q_proj_norm = pl.assemble(
            q_proj_norm, q_normed_flat, [tg, q_col],
        )

        k_col = kh * HEAD_DIM
        k_chunk = pl.slice(k_proj, [TOK_TILE, HEAD_DIM], [tg, k_col])
        k_gamma = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
        k_sq = pl.row_sum(pl.mul(k_chunk, k_chunk))
        k_inv = pl.rsqrt(pl.add(pl.mul(k_sq, 0.0078125), EPS))
        k_scaled = pl.row_expand_mul(k_chunk, k_inv)
        k_normed = pl.col_expand_mul(k_scaled, pl.add(k_gamma, 1.0))
        k_proj_norm = pl.assemble(k_proj_norm, k_normed, [tg, k_col])

    # ── Stage 1.g — full RoPE on Q and K (SWA: rotary_dim = HEAD_DIM). ─
    for t in pl.parallel(PREFILL_T):
        pos = pl.cast(pl.tensor.read(positions, [t]), pl.INDEX)
        cos_row = pl.slice(rope_cos, [1, 128], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, 128], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, 64], [0, 0])
        cos_hi = pl.slice(cos_row, [1, 64], [0, 64])
        sin_lo = pl.slice(sin_row, [1, 64], [0, 0])
        sin_hi = pl.slice(sin_row, [1, 64], [0, 64])

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_swa_rope_q_k",
        ):
            # K RoPE — single rank-local KV head per card under TP=8.
            # SWA: rotary_pass = HEAD_DIM - ROTARY_DIM = 0 (full rotary).
            for kh in pl.range(1):
                k_col = kh * HEAD_DIM
                k_lo = pl.slice(
                    k_proj_norm, [1, 64], [t, k_col],
                )
                k_hi = pl.slice(
                    k_proj_norm,
                    [1, 64],
                    [t, k_col + 64],
                )
                rot_k_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                rot_k_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                k_rot = pl.assemble(
                    k_rot,
                    pl.cast(rot_k_lo, target_type=pl.BF16),
                    [t, k_col],
                )
                k_rot = pl.assemble(
                    k_rot,
                    pl.cast(rot_k_hi, target_type=pl.BF16),
                    [t, k_col + 64],
                )
                # V — copy through (no rotation).
                v_slice = pl.slice(
                    v_proj, [1, HEAD_DIM], [t, k_col],
                )
                v_tile = pl.assemble(
                    v_tile,
                    pl.cast(v_slice, target_type=pl.BF16),
                    [t, k_col],
                )

            # Q RoPE — Q_PER_KV consecutive heads per KV-head bundle.
            for kh in pl.range(1):
                q_base_col = kh * 12 * HEAD_DIM
                q_block_norm = pl.reshape(
                    pl.slice(
                        q_proj_norm,
                        [1, 12 * HEAD_DIM], [t, q_base_col],
                    ),
                    [12, HEAD_DIM],
                )
                q_lo = pl.slice(
                    q_block_norm, [12, 64], [0, 0],
                )
                q_hi = pl.slice(
                    q_block_norm,
                    [12, 64],
                    [0, 64],
                )
                rot_q_lo = pl.sub(
                    pl.col_expand_mul(q_lo, cos_lo),
                    pl.col_expand_mul(q_hi, sin_lo),
                )
                rot_q_hi = pl.add(
                    pl.col_expand_mul(q_hi, cos_hi),
                    pl.col_expand_mul(q_lo, sin_hi),
                )
                for qi in pl.range(12):
                    h_col = q_base_col + qi * HEAD_DIM
                    rl = pl.slice(
                        rot_q_lo, [1, 64], [qi, 0],
                    )
                    rh = pl.slice(
                        rot_q_hi, [1, 64], [qi, 0],
                    )
                    q_rot = pl.assemble(
                        q_rot,
                        pl.cast(rl, target_type=pl.BF16),
                        [t, h_col],
                    )
                    q_rot = pl.assemble(
                        q_rot,
                        pl.cast(rh, target_type=pl.BF16),
                        [t, h_col + 64],
                    )

    # ── Scope 2.a — write KV cache. ──────────────────────────────────────
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_kv_write"):
        for t in pl.range(PREFILL_T):
            slot = pl.tensor.read(slot_mapping, [t])
            slot_block = slot // 128
            slot_offset = slot - slot_block * 128
            for kh in pl.range(1):
                cache_row = (
                    layer_cache_base
                    + (slot_block * 1 + kh) * 128
                    + slot_offset
                )
                k_row = pl.slice(k_rot, [1, HEAD_DIM], [t, kh * HEAD_DIM])
                v_row = pl.slice(v_tile, [1, HEAD_DIM], [t, kh * HEAD_DIM])
                k_cache = pl.assemble(k_cache, k_row, [cache_row, 0])
                v_cache = pl.assemble(v_cache, v_row, [cache_row, 0])

    # ── Scope 2.b — causal + sliding-window flash attention. ─────────────
    attn_out = pl.create_tensor([PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16)
    bt_stride = pl.cast(32, pl.INDEX)
    # Pad each token's 12 Q-heads to Q_HEAD_PAD_SWA=24 so the matmul
    # satisfies the Cube fractal minimum M=16. The 12 padding rows are
    # zero-filled; set_validshape(Q_HEAD_PAD_SWA // 2 = 12) masks them.
    q_rot_flat = pl.reshape(q_rot, [PREFILL_T * 12, HEAD_DIM])
    q_rot_padded = pl.create_tensor(
        [PREFILL_T * 24, HEAD_DIM], dtype=pl.BF16,
    )
    for tp in pl.parallel(PREFILL_T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_q_head_pad"):
            q_rot_padded = pl.assemble(
                q_rot_padded,
                pl.slice(q_rot_flat, [12, HEAD_DIM], [tp * 12, 0]),
                [tp * 24, 0],
            )
            q_rot_padded = pl.assemble(
                q_rot_padded,
                pl.full([12, HEAD_DIM], dtype=pl.BF16, value=0.0),
                [tp * 24 + 12, 0],
            )

    for t in pl.parallel(PREFILL_T):
        pos = pl.cast(pl.tensor.read(positions, [t]), pl.INDEX)
        ctx_len_full = pos + 1
        start_pos = pl.max(
            pl.cast(0, pl.INDEX),
            pl.cast(ctx_len_full - 512, pl.INDEX),
        )
        start_block = start_pos // 128
        end_block = (ctx_len_full + 128 - 1) // 128
        bt_base = pl.cast(0, pl.INDEX) * bt_stride

        # GM-level flash accumulators — outside InCore to avoid the Cube
        # fractal-tile reshape error on [24,1] FP32 shapes.
        mi_buf = pl.create_tensor([24, 1], dtype=pl.FP32)
        li_buf = pl.create_tensor([24, 1], dtype=pl.FP32)
        oi_buf = pl.create_tensor([24, HEAD_DIM], dtype=pl.FP32)
        # Per-KV-block intermediates; reused (overwritten) each iteration.
        exp_buf = pl.create_tensor([24, 128], dtype=pl.BF16)
        alpha_buf = pl.create_tensor([24, 1], dtype=pl.FP32)
        beta_buf = pl.create_tensor([24, 1], dtype=pl.FP32)

        # Initialise running accumulators (mi=-inf, li=0, oi=0).
        # pl.full([24, 1], FP32) fails pto.alloc_tile: cols*sizeof = 1*4 = 4 bytes,
        # not 32-byte aligned.  Use pl.full([24, HEAD_DIM], FP32) (128*4=512 bytes,
        # aligned) and derive the [24,1] init via row_max (reduction, no alloc_tile).
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_fa_init"):
            mi_buf = pl.assemble(
                mi_buf,
                pl.row_max(pl.full([24, HEAD_DIM], dtype=pl.FP32, value=-3.0e38)),
                [0, 0],
            )
            li_buf = pl.assemble(
                li_buf,
                pl.row_max(pl.full([24, HEAD_DIM], dtype=pl.FP32, value=0.0)),
                [0, 0],
            )
            oi_buf = pl.assemble(
                oi_buf,
                pl.full([24, HEAD_DIM], dtype=pl.FP32, value=0.0),
                [0, 0],
            )

        for kh in pl.range(1):
            q_base = kh * 12
            # KV loop at orchestration level — separates the dynamic loop and
            # the two matmuls (q@k, exp@v) into distinct InCore scopes.
            for sb in pl.range(start_block, end_block):
                s0 = sb * 128
                lo = pl.max(start_pos, s0)
                hi = pl.min(s0 + 128, ctx_len_full)
                valid_len = hi - lo
                bt_idx = bt_base + sb
                pbid = pl.cast(
                    pl.tensor.read(block_table, [bt_idx]), pl.INDEX,
                )
                cache_row0 = (
                    layer_cache_base
                    + (pbid * 1 + kh) * 128
                )
                k_tile = k_cache[
                    cache_row0 : cache_row0 + 128, :
                ]
                v_tile_sb = v_cache[
                    cache_row0 : cache_row0 + 128, :
                ]

                # Block 1 — QK matmul + online-softmax step.
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_swa_fa_qk",
                ):
                    # Padded 24-head block for token t: real [0:12], zero [12:24].
                    q_block = q_rot_padded[
                        t * 24 : t * 24 + 24, 0 : HEAD_DIM,
                    ]
                    raw_scores = pl.matmul(
                        q_block, k_tile, b_trans=True, out_dtype=pl.FP32,
                    )
                    scores_scaled = pl.mul(raw_scores, 0.08838834764831845)
                    scores_valid = pl.set_validshape(
                        scores_scaled, 12, valid_len,
                    )
                    scores = pl.fillpad(
                        scores_valid, pad_value=pl.PadValue.min,
                    )
                    cur_mi = pl.row_max(scores)
                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                    exp_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                    cur_li = pl.row_sum(
                        pl.cast(exp_bf16, target_type=pl.FP32)
                    )
                    # Load running mi/li from GM (MTE load, no fractal constraint).
                    mi_cur = mi_buf[0:24, 0:1]
                    li_cur = li_buf[0:24, 0:1]
                    mi_new = pl.maximum(mi_cur, cur_mi)
                    alpha = pl.exp(pl.sub(mi_cur, mi_new))
                    beta = pl.exp(pl.sub(cur_mi, mi_new))
                    li_new = pl.add(
                        pl.mul(alpha, li_cur), pl.mul(beta, cur_li)
                    )
                    mi_buf = pl.assemble(mi_buf, mi_new, [0, 0])
                    li_buf = pl.assemble(li_buf, li_new, [0, 0])
                    exp_buf = pl.assemble(exp_buf, exp_bf16, [0, 0])
                    alpha_buf = pl.assemble(alpha_buf, alpha, [0, 0])
                    beta_buf = pl.assemble(beta_buf, beta, [0, 0])

                # Block 2 — PV matmul + accumulate output.
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_swa_fa_pv",
                ):
                    exp_b2 = exp_buf[0:24, 0:128]
                    oi_tmp = pl.matmul(exp_b2, v_tile_sb, out_dtype=pl.FP32)
                    oi_cur = oi_buf[0:24, 0:HEAD_DIM]
                    alpha_b2 = alpha_buf[0:24, 0:1]
                    beta_b2 = beta_buf[0:24, 0:1]
                    oi_new = pl.add(
                        pl.row_expand_mul(oi_cur, alpha_b2),
                        pl.row_expand_mul(oi_tmp, beta_b2),
                    )
                    oi_buf = pl.assemble(oi_buf, oi_new, [0, 0])

            # Final normalisation — divide accumulated oi by li.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_swa_fa_norm",
            ):
                oi_final = oi_buf[0:24, 0:HEAD_DIM]
                li_final = li_buf[0:24, 0:1]
                ctx = pl.row_expand_div(oi_final, li_final)
                # Slice the 12 real head rows (rows 12-23 are zero-pad).
                ctx_valid = ctx[0:12, 0:HEAD_DIM]
                ctx_flat = pl.cast(
                    pl.reshape(ctx_valid, [1, 12 * HEAD_DIM]),
                    target_type=pl.BF16,
                )
                attn_out = pl.assemble(
                    attn_out, ctx_flat, [t, q_base * HEAD_DIM],
                )

    # ── Scope 2.5 — head-wise gate. ──────────────────────────────────────
    attn_out_gated = pl.create_tensor([PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16)
    for hg_idx in pl.spmd(
        (PREFILL_T // TOK_TILE) * NUM_HEADS_SWA_LOCAL,
        name_hint="prefill_swa_head_gate",
    ):
        tg_idx = hg_idx // NUM_HEADS_SWA_LOCAL
        h = hg_idx % NUM_HEADS_SWA_LOCAL
        tg = tg_idx * TOK_TILE
        h_col = h * HEAD_DIM
        head_slab = pl.slice(
            attn_out, [TOK_TILE, HEAD_DIM], [tg, h_col],
        )
        gate_col = pl.slice(gate_logits, [TOK_TILE, 1], [tg, h])
        # Phase X.7: head_wise_gate_apply body inlined.
        hg_gate = pl.recip(pl.add(pl.exp(pl.neg(gate_col)), 1.0))
        hg_gated_fp32 = pl.row_expand_mul(
            pl.cast(head_slab, target_type=pl.FP32), hg_gate,
        )
        gated = pl.cast(hg_gated_fp32, target_type=pl.BF16)
        attn_out_gated = pl.assemble(attn_out_gated, gated, [tg, h_col])

    # ── Scope 3.a — local o_proj. ────────────────────────────────────────
    out_proj_k_blocks = HIDDEN_Q_SWA_LOCAL // 256
    # Phase A (2026-06-12): mirror of decode-side `swa_out_proj` split — cube
    # matmul into FP32 GM scratch, then a separate vec spmd casts to BF16.
    # Eliminates the mixed AIC+AIV MixedKernels dispatch (see decode counterpart
    # and upstream-issues/step3p5-507018-vec-ub-align.md).
    partial_attn_proj_fp32 = pl.create_tensor(
        [PREFILL_T, HIDDEN], dtype=pl.FP32,
    )
    partial_attn_proj = pl.create_tensor(
        [PREFILL_T, HIDDEN], dtype=pl.BF16,
    )
    for op_idx in pl.spmd(
        (PREFILL_T // TOK_TILE) * (HIDDEN // 256),
        name_hint="prefill_swa_out_proj_matmul",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        tg_idx = op_idx // (HIDDEN // 256)
        ob = op_idx % (HIDDEN // 256)
        tg = tg_idx * TOK_TILE
        o0 = ob * 256
        o_a0 = pl.slice(
            attn_out_gated, [TOK_TILE, 256], [tg, 0],
        )
        o_w0 = pl.slice(
            wo, [256, 256],
            [layer_qhidden_base, o0],
        )
        o_acc = pl.matmul(o_a0, o_w0, out_dtype=pl.FP32)
        for kb in pl.range(1, out_proj_k_blocks):
            k0 = kb * 256
            o_a = pl.slice(
                attn_out_gated, [TOK_TILE, 256], [tg, k0],
            )
            o_w = pl.slice(
                wo, [256, 256],
                [layer_qhidden_base + k0, o0],
            )
            o_acc = pl.matmul_acc(o_acc, o_a, o_w)
        partial_attn_proj_fp32 = pl.assemble(
            partial_attn_proj_fp32, o_acc, [tg, o0],
        )

    for op_idx in pl.spmd(
        (PREFILL_T // TOK_TILE) * (HIDDEN // 256),
        name_hint="prefill_swa_out_proj_cast",
    ):
        tg_idx = op_idx // (HIDDEN // 256)
        ob = op_idx % (HIDDEN // 256)
        tg = tg_idx * TOK_TILE
        o0 = ob * 256
        fp32_chunk = pl.slice(
            partial_attn_proj_fp32, [TOK_TILE, 256], [tg, o0],
        )
        partial_attn_proj = pl.assemble(
            partial_attn_proj,
            pl.cast(fp32_chunk, target_type=pl.BF16),
            [tg, o0],
        )

    # ── Scope 3.b — TP all-reduce. ───────────────────────────────────────
    # Phase X.9: the pull-side ring body now lives as the consumer
    # class's ``tp_all_reduce`` ``@pl.function`` method (same pattern as
    # the decode-side ``attention_swa``). The inlined body resolves
    # ``self`` from the enclosing ``chip_orch`` method's scope.
    # Phase A (2026-06-12): mirror of decode 15.B — at TP=1 the all-reduce
    # is a no-op (no peers); skip the call so the orchestration codegen
    # does not emit a stale SSA rename for the (now-empty) ring body.
    if TP_WORLD_SIZE > 1:
        partial_attn_proj = self.tp_all_reduce(
            partial_attn_proj,
            tmp_window,
            signal_window,
            my_rank,
        )

    # ── Scope 3.c — residual add. ────────────────────────────────────────
    for ra_idx in pl.spmd(
        (PREFILL_T // TOK_TILE) * (HIDDEN // 256),
        name_hint="prefill_swa_resid_add",
    ):
        tg_idx = ra_idx // (HIDDEN // 256)
        ob = ra_idx % (HIDDEN // 256)
        tg = tg_idx * TOK_TILE
        o0 = ob * 256
        reduced = pl.cast(
            pl.slice(
                partial_attn_proj,
                [TOK_TILE, 256], [tg, o0],
            ),
            target_type=pl.FP32,
        )
        resid = pl.cast(
            pl.slice(
                current_hidden,
                [TOK_TILE, 256], [tg, o0],
            ),
            target_type=pl.FP32,
        )
        resid1_out = pl.assemble(
            resid1_out,
            pl.cast(pl.add(reduced, resid), target_type=pl.BF16),
            [tg, o0],
        )

    return resid1_out


# =============================================================================
# TP wrapper — @pl.program (chip_orch + host_orch).
# =============================================================================
def _build_tp_prefill_attention_swa_program(tp_size: int = TP_WORLD_SIZE):
    """Return a freshly-built ``@pl.program`` for the SWA prefill TP body."""
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    body_inline = pl.inline(attention_swa_prefill._func)
    tp_chunk = HIDDEN // tp_size

    @pl.program
    class PrefillAttentionSwa:
        # ---------- Collective: TP all_reduce (Phase X.9, mirrors decode). ----
        # Pull-side ring body lifted from ``collectives.tp_all_reduce``.
        # ``t_rows = PREFILL_T``, ``d_cols = HIDDEN``, ``group_size = tp_size``
        # are baked in from this factory's closure.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[
                [PREFILL_T, tp_chunk], pl.BF16
            ],
            signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            group_size = tp_size
            t_rows = PREFILL_T
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
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    target=signal_window,
                    offsets=[prev_rank, 0],
                    value=step + 1,
                    cmp=pld.WaitCmp.Ge,
                )
                recv_tile = pld.tile.remote_load(
                    tmp_window, [0, 0], [t_rows, chunk], peer=prev_rank,
                )
                local_tile = pl.load(
                    local, [0, recv_idx * chunk], [t_rows, chunk],
                )
                summed = pl.add(local_tile, recv_tile)
                pl.store(summed, [0, recv_idx * chunk], local)

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
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    target=signal_window,
                    offsets=[prev_rank, 0],
                    value=(group_size - 1) + step + 1,
                    cmp=pld.WaitCmp.Ge,
                )
                recv_tile = pld.tile.remote_load(
                    tmp_window, [0, 0], [t_rows, chunk], peer=prev_rank,
                )
                pl.store(recv_tile, [0, recv_idx * chunk], local)

            return local

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], pl.BF16],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32],
            k_cache: pl.Tensor[
                [KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            wo: pl.Tensor[
                [LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16
            ],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            resid1_out: pl.Out[
                pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]
            ],
            tmp_window: pld.DistributedTensor[
                [PREFILL_T, tp_chunk], pl.BF16
            ],
            signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            resid1_out = body_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                positions,
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
            current_hidden: pl.Tensor[
                [tp_size, PREFILL_T, HIDDEN], pl.BF16
            ],
            input_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q], pl.BF16
            ],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_DIM], pl.BF16
            ],
            q_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            k_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            block_table: pl.Tensor[
                [tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32
            ],
            slot_mapping: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32
            ],
            rope_sin: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, ROTARY_DIM], pl.FP32
            ],
            k_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            wo: pl.Tensor[
                [tp_size, LAYER_QHIDDEN_ROWS_DYN, HIDDEN], pl.BF16
            ],
            w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16
            ],
            positions: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            resid1_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]
            ],
            layer_idx: pl.Scalar[pl.INT32],
        ):
            tmp_buf = pld.alloc_window_buffer(PREFILL_T * tp_chunk * 2)
            sig_buf = pld.alloc_window_buffer(tp_size * 4)
            for r in pl.range(pld.world_size()):
                tmp_window = pld.window(
                    tmp_buf, [PREFILL_T, tp_chunk], dtype=pl.BF16,
                )
                signal_window = pld.window(
                    sig_buf, [tp_size, 1], dtype=pl.INT32,
                )
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r],
                    k_cache[r], v_cache[r],
                    wo[r], w_g[r],
                    positions[r],
                    resid1_out[r],
                    tmp_window, signal_window,
                    layer_idx,
                    r,
                    device=r,
                )

    return PrefillAttentionSwa


def _build_tp_prefill_attention_swa_program_default():
    return _build_tp_prefill_attention_swa_program(TP_WORLD_SIZE)


# =============================================================================
# Torch reference + distributed-mock harness.
# =============================================================================
def _torch_single_card_prefill_swa(
    *, hidden, input_rms_weight, wq_full, wk_full, wv_full,
    q_norm_weight, k_norm_weight, wo_full, w_g_full,
    rope_cos, rope_sin, positions,
):
    """Pure-torch single-card oracle for the SWA prefill body."""
    import math

    import torch

    num_heads_full = NUM_HEADS_SWA_LOCAL * TP_WORLD_SIZE
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE
    head_dim = HEAD_DIM
    q_per_kv = Q_PER_KV
    scale = 1.0 / math.sqrt(head_dim)

    qkv = _torch_prefill_qkv_oracle_impl(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        w_g_full=w_g_full,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
        num_heads_full=num_heads_full,
        num_kv_heads_full=num_kv_heads_full,
        rotary_half=ROTARY_HALF,
    )
    q_rot = qkv["q_rot"].float()
    k_rot = qkv["k_rot"].float()
    v_proj = qkv["v_proj"].float()
    gate_logits = qkv["gate_logits"]

    t = hidden.shape[0]
    attn_out = torch.zeros(t, num_heads_full, head_dim)
    for ti in range(t):
        start = max(0, int(positions[ti].item()) - WIN + 1)
        end = ti + 1
        for kvh in range(num_kv_heads_full):
            q_base = kvh * q_per_kv
            q_grp = q_rot[ti, q_base : q_base + q_per_kv, :]
            k_block = k_rot[start:end, kvh, :]
            v_block = v_proj[start:end, kvh, :]
            scores = (q_grp @ k_block.T) * scale
            probs = torch.softmax(scores, dim=-1)
            ctx = probs @ v_block
            attn_out[ti, q_base : q_base + q_per_kv, :] = ctx

    gate = torch.sigmoid(gate_logits).unsqueeze(-1)
    attn_gated = (attn_out * gate).to(torch.bfloat16)
    attn_gated_flat = attn_gated.view(t, num_heads_full * head_dim)
    o = attn_gated_flat.float() @ wo_full.float()
    resid1 = (o + hidden.float()).bfloat16()
    return resid1


def _torch_per_rank_partial_swa(
    *, rank, hidden, input_rms_weight, wq_full, wk_full, wv_full,
    q_norm_weight, k_norm_weight, wo_full, w_g_full,
    rope_cos, rope_sin, positions,
):
    """Per-rank partial pre-all-reduce o_proj for the SWA prefill path."""
    import math

    import torch

    num_heads_full = NUM_HEADS_SWA_LOCAL * TP_WORLD_SIZE
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE
    heads_local = num_heads_full // TP_WORLD_SIZE
    kv_heads_local = num_kv_heads_full // TP_WORLD_SIZE
    hidden_q_local = heads_local * HEAD_DIM
    kv_hidden_local = kv_heads_local * HEAD_DIM
    scale = 1.0 / math.sqrt(HEAD_DIM)

    wq_local = wq_full[
        :, rank * hidden_q_local : (rank + 1) * hidden_q_local,
    ]
    wk_local = wk_full[
        :, rank * kv_hidden_local : (rank + 1) * kv_hidden_local,
    ]
    wv_local = wv_full[
        :, rank * kv_hidden_local : (rank + 1) * kv_hidden_local,
    ]
    wo_local = wo_full[
        rank * hidden_q_local : (rank + 1) * hidden_q_local, :,
    ]
    w_g_local = w_g_full[
        :, rank * heads_local : (rank + 1) * heads_local,
    ]

    qkv = _torch_prefill_qkv_oracle_impl(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_local, wk_full=wk_local, wv_full=wv_local,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        w_g_full=w_g_local,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
        num_heads_full=heads_local,
        num_kv_heads_full=kv_heads_local,
        rotary_half=ROTARY_HALF,
    )
    q_rot = qkv["q_rot"].float()
    k_rot = qkv["k_rot"].float()
    v_proj = qkv["v_proj"].float()
    gate_logits = qkv["gate_logits"]

    t = hidden.shape[0]
    attn_local = torch.zeros(t, heads_local, HEAD_DIM)
    q_per_kv = Q_PER_KV
    for ti in range(t):
        start = max(0, int(positions[ti].item()) - WIN + 1)
        end = ti + 1
        for kvh in range(kv_heads_local):
            q_base = kvh * q_per_kv
            q_grp = q_rot[ti, q_base : q_base + q_per_kv, :]
            k_block = k_rot[start:end, kvh, :]
            v_block = v_proj[start:end, kvh, :]
            scores = (q_grp @ k_block.T) * scale
            probs = torch.softmax(scores, dim=-1)
            ctx = probs @ v_block
            attn_local[ti, q_base : q_base + q_per_kv, :] = ctx

    gate = torch.sigmoid(gate_logits).unsqueeze(-1)
    attn_gated = (attn_local * gate).to(torch.bfloat16)
    attn_flat = attn_gated.view(t, hidden_q_local)
    partial_o = (attn_flat.float() @ wo_local.float()).to(torch.bfloat16)
    return partial_o


def _run_distributed_mock(
    *, layer_idx: int = 1, pass_rate: float = 0.97,
    rtol: float = 1e-2, atol: float = 1e-2, seed: int = 0,
):
    """Mock 8-rank simulation of the prefill SWA body."""
    import torch

    torch.manual_seed(seed)
    layer_rope_theta = LAYER_ROPE_THETA[layer_idx]
    rope_cos, rope_sin = build_plain_rope_tables(
        MAX_SEQ_DEFAULT, ROTARY_DIM, layer_rope_theta,
    )

    num_heads_full = NUM_HEADS_SWA_LOCAL * TP_WORLD_SIZE
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE
    hidden_q_full = num_heads_full * HEAD_DIM
    kv_hidden_full = num_kv_heads_full * HEAD_DIM
    proj_scale = 0.5
    hidden = (torch.rand(PREFILL_T, HIDDEN) - 0.5).bfloat16()
    input_rms_weight = ((torch.rand(1, HIDDEN) - 0.5) * 0.1).float()
    wq_full = (
        (torch.rand(HIDDEN, hidden_q_full) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    wk_full = (
        (torch.rand(HIDDEN, kv_hidden_full) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    wv_full = (
        proj_scale * (torch.rand(HIDDEN, kv_hidden_full) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    q_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    k_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    wo_full = (
        proj_scale * (torch.rand(hidden_q_full, HIDDEN) - 0.5)
        / hidden_q_full ** 0.5
    ).bfloat16()
    w_g_full = (
        proj_scale * (torch.rand(HIDDEN, num_heads_full) - 0.5)
        / HIDDEN ** 0.5
    ).bfloat16()
    positions = torch.arange(PREFILL_T, dtype=torch.int32)

    expected_resid1 = _torch_single_card_prefill_swa(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        wo_full=wo_full, w_g_full=w_g_full,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
    )

    summed_partial = torch.zeros(PREFILL_T, HIDDEN, dtype=torch.float32)
    for r in range(TP_WORLD_SIZE):
        rank_partial = _torch_per_rank_partial_swa(
            rank=r,
            hidden=hidden,
            input_rms_weight=input_rms_weight,
            wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
            q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
            wo_full=wo_full, w_g_full=w_g_full,
            rope_cos=rope_cos, rope_sin=rope_sin,
            positions=positions,
        )
        summed_partial = summed_partial + rank_partial.float()
    tp_resid1 = (summed_partial + hidden.float()).bfloat16()

    close = torch.isclose(
        tp_resid1.float(), expected_resid1.float(),
        rtol=rtol, atol=atol,
    )
    rate = close.float().mean().item()
    n_fail = int((~close).sum().item())
    ok = rate >= pass_rate
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] prefill_attention_swa distributed-mock: "
        f"pass_rate={rate:.6f} threshold={pass_rate:.6f} "
        f"{n_fail}/{tp_resid1.numel()} mismatched "
        f"rtol={rtol} atol={atol}"
    )
    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build-program-only", action="store_true")
    args = parser.parse_args()

    program_cls = _build_tp_prefill_attention_swa_program(TP_WORLD_SIZE)
    print(
        f"[OK] built @pl.program PrefillAttentionSwa: {program_cls.__name__} "
        f"(tp_size={TP_WORLD_SIZE}, T={PREFILL_T})"
    )
    if args.build_program_only:
        raise SystemExit(0)
    ok = _run_distributed_mock(
        layer_idx=args.layer_idx,
        pass_rate=args.pass_rate,
        rtol=args.rtol, atol=args.atol,
        seed=args.seed,
    )
    if not ok:
        raise SystemExit(1)


__all__ = [
    "PREFILL_BATCH",
    "PREFILL_SEQ",
    "PREFILL_T",
    "TOK_TILE",
    "NUM_HEADS",
    "HIDDEN_Q",
    "KV_HIDDEN_DIM",
    "NUM_KV_HEADS_DIM",
    "Q_PER_KV",
    "ROTARY_HALF",
    "ROTARY_DIM",
    "WIN",
    "LAYER_QHIDDEN_ROWS_DYN",
    "attention_swa_prefill",
    "_build_tp_prefill_attention_swa_program",
    "_build_tp_prefill_attention_swa_program_default",
    "_torch_single_card_prefill_swa",
    "_torch_per_rank_partial_swa",
    "_run_distributed_mock",
]
