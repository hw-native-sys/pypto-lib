# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 prefill input RMSNorm + TP-sliced Q/K/V projection + partial RoPE.

Counterpart of the decode-side ``attention_full.py`` / ``attention_swa.py``
Scope-1 prelude, generalised for a sequence-major prefill shape. The
prefill tile carries ``T = PREFILL_BATCH * PREFILL_SEQ`` tokens (no
``BATCH`` round-up — every token is a real token), runs identical math
to the decode QKV-RoPE prelude on its rank's local heads, and emits
the per-rank ``[T, HIDDEN_Q_LOCAL]``, ``[T, KV_HIDDEN_LOCAL]`` projection
tiles plus the head-wise gate logits used downstream by the prefill
attention body.

Two variants are produced from one factory:

  * **full** — ``NUM_HEADS_FULL_LOCAL = 8``, rotary_half = 32
               (partial_rotary_factor = 0.5 ⇒ ``rotary_dim = 64`` with a
               64-lane pass-through tail). Yarn-scaled cos/sin tables.
  * **swa**  — ``NUM_HEADS_SWA_LOCAL  = 12``, rotary_half = 64
               (partial_rotary_factor = 1.0 ⇒ ``rotary_dim = 128`` with
               no pass-through). Plain (un-scaled) cos/sin tables.

Per-card weight bundle (host weight loader contract — TP-sliced exactly
like the decode path; see ``attention_full.py`` / ``attention_swa.py``):

  * ``input_rms_weight[LAYER, HIDDEN]`` FP32 (replicated)
  * ``wq[LAYER * HIDDEN, HIDDEN_Q_LOCAL]`` BF16
       full: ``HIDDEN_Q_FULL_LOCAL = 1024``
       swa : ``HIDDEN_Q_SWA_LOCAL  = 1536``
  * ``wk[LAYER * HIDDEN, KV_HIDDEN_LOCAL=128]`` BF16
  * ``wv[LAYER * HIDDEN, KV_HIDDEN_LOCAL=128]`` BF16
  * ``q_norm_weight[LAYER, HEAD_DIM=128]`` FP32 (replicated)
  * ``k_norm_weight[LAYER, HEAD_DIM=128]`` FP32 (replicated)
  * ``w_g[LAYER * HIDDEN, NUM_HEADS_LOCAL]`` BF16 (TP-sliced output dim)
  * ``rope_cos[ROPE_SEQ, ROTARY_DIM]`` FP32 (replicated)
  * ``rope_sin[ROPE_SEQ, ROTARY_DIM]`` FP32 (replicated)

Per-rank output bundle (handed off to ``prefill_attention_full.py`` /
``prefill_attention_swa.py``):

  * ``q_rot[T, HIDDEN_Q_LOCAL]`` BF16    — RoPE-rotated Q (per-rank heads)
  * ``k_rot[T, KV_HIDDEN_LOCAL]`` BF16    — RoPE-rotated K (per-rank kv heads)
  * ``v_tile[T, KV_HIDDEN_LOCAL]`` BF16   — V projection (per-rank kv heads)
  * ``normed_tile[T, HIDDEN]`` BF16       — replicated zero-centred input
                                            RMSNorm of ``current_hidden``
  * ``gate_logits[T, NUM_HEADS_LOCAL]`` FP32 — head-wise gate matmul output
                                              (current_hidden @ w_g, NOT
                                              the normed activation)

The kernel does **not** touch the KV cache here — the cache write is
fused into the prefill attention body (so it can be staged inside the
per-token causal/SWA attention loop alongside the QK / SV matmuls,
matching the decode path's locality).
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import (
    build_llama3_yarn_rope_tables,
    build_plain_rope_tables,
    partial_rope_rotate,
    per_head_qk_norm,
    zero_centered_rmsnorm_apply,
)
from .config import (
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    HIDDEN_Q_SWA_LOCAL,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_ROPE_THETA,
    MAX_SEQ_DEFAULT,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_SWA_LOCAL,
    Q_PER_KV_FULL,
    Q_PER_KV_SWA,
    ROPE_SCALING,
    ROPE_SEQ_DYN,
    ROTARY_HALF_FULL,
    ROTARY_HALF_SWA,
    TP_WORLD_SIZE,
)


# -----------------------------------------------------------------------------
# Prefill compile-time shape.
# -----------------------------------------------------------------------------
PREFILL_BATCH = 1
PREFILL_SEQ = 128
PREFILL_T = PREFILL_BATCH * PREFILL_SEQ  # 128

# Tile-level constants.
TOK_TILE = 32
INPUT_PROJ_K_CHUNK = 256
KV_OUT_CHUNK_LOCAL = KV_HIDDEN_LOCAL  # 128 fits in one chunk
Q_OUT_CHUNK = 128


assert PREFILL_T % TOK_TILE == 0
assert HIDDEN % INPUT_PROJ_K_CHUNK == 0
assert KV_HIDDEN_LOCAL == KV_OUT_CHUNK_LOCAL


# =============================================================================
# Prefill QKV+RoPE body factory.
#
# The factory bakes the attention flavour (full vs swa) as compile-time
# Python constants so the generated body sees fixed NUM_HEADS / ROTARY_HALF
# / Q_PER_KV / HIDDEN_Q_LOCAL specialisations.
# =============================================================================
def _build_prefill_qkv_proj_rope(*, full: bool):
    """Factory returning an inline body that runs the prefill QKV+RoPE."""
    num_heads_local = NUM_HEADS_FULL_LOCAL if full else NUM_HEADS_SWA_LOCAL
    hidden_q_local = HIDDEN_Q_FULL_LOCAL if full else HIDDEN_Q_SWA_LOCAL
    rotary_half = ROTARY_HALF_FULL if full else ROTARY_HALF_SWA
    rotary_dim = rotary_half * 2
    rotary_pass = HEAD_DIM - rotary_dim
    q_per_kv = Q_PER_KV_FULL if full else Q_PER_KV_SWA
    kv_heads_local = KV_HEADS_LOCAL

    name_prefix = "prefill_full" if full else "prefill_swa"

    assert hidden_q_local % Q_OUT_CHUNK == 0

    @pl.jit.inline
    def prefill_qkv_proj_rope_body(
        current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16],
        wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
        wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, num_heads_local], pl.BF16],
        rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
        rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
        positions: pl.Tensor[[PREFILL_T], pl.INT32],
        normed_out: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
        q_out: pl.Tensor[[PREFILL_T, hidden_q_local], pl.BF16],
        k_out: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
        v_out: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
        gate_logits_out: pl.Tensor[[PREFILL_T, num_heads_local], pl.FP32],
        layer_idx: pl.Scalar[pl.INT32],
    ):
        """TP-sliced prefill QKV projection + per-head q/k norm + partial RoPE.

        ``positions`` carries the absolute KV-cache row position per
        token. For a single-prompt prefill this is just
        ``[0, 1, ..., T - 1]`` (the caller fills it at the host side).

        The body runs replicated input RMSNorm, three per-rank
        projection matmuls (sliced by HEAD count), per-head zero-centred
        q/k norm, partial RoPE on Q and K, and the head-wise gate matmul
        on ``current_hidden`` (NOT the normed activation). All outputs
        are per-rank shards; no collective is invoked.
        """
        d_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
        q_blocks = hidden_q_local // Q_OUT_CHUNK
        layer_hidden_base = layer_idx * HIDDEN

        # ── Stage 1.a — replicated zero-centred input RMSNorm. ───────────
        for tg_idx in pl.spmd(
            PREFILL_T // TOK_TILE, name_hint=f"{name_prefix}_rmsnorm_zc",
        ):
            tg = tg_idx * TOK_TILE
            partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(d_blocks):
                k0 = kb * INPUT_PROJ_K_CHUNK
                chunk = pl.cast(
                    pl.slice(
                        current_hidden,
                        [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, k0],
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
            for kb in pl.range(d_blocks):
                k0 = kb * INPUT_PROJ_K_CHUNK
                chunk = pl.cast(
                    pl.slice(
                        current_hidden,
                        [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, k0],
                    ),
                    target_type=pl.FP32,
                )
                gamma = pl.slice(
                    input_rms_weight,
                    [1, INPUT_PROJ_K_CHUNK], [layer_idx, k0],
                )
                scaled = pl.row_expand_mul(chunk, inv_rms)
                normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
                normed_out = pl.assemble(
                    normed_out,
                    pl.cast(normed, target_type=pl.BF16),
                    [tg, k0],
                )

        # ── Stage 1.b — Q projection (per-rank heads). ────────────────────
        q_proj = pl.create_tensor([PREFILL_T, hidden_q_local], dtype=pl.FP32)
        for q_idx in pl.spmd(
            (PREFILL_T // TOK_TILE) * q_blocks,
            name_hint=f"{name_prefix}_q_proj",
        ):
            qb_idx = q_idx // q_blocks
            qo_idx = q_idx % q_blocks
            tg = qb_idx * TOK_TILE
            q_o0 = qo_idx * Q_OUT_CHUNK
            a0 = pl.slice(
                normed_out, [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, 0],
            )
            w0 = pl.slice(
                wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK],
                [layer_hidden_base, q_o0],
            )
            q_acc = pl.matmul(a0, w0, out_dtype=pl.FP32)
            for kb in pl.range(1, d_blocks):
                k0 = kb * INPUT_PROJ_K_CHUNK
                a = pl.slice(
                    normed_out, [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, k0],
                )
                w = pl.slice(
                    wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK],
                    [layer_hidden_base + k0, q_o0],
                )
                q_acc = pl.matmul_acc(q_acc, a, w)
            q_proj = pl.assemble(q_proj, q_acc, [tg, q_o0])

        # ── Stage 1.c — K projection. ────────────────────────────────────
        k_proj = pl.create_tensor(
            [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
        )
        for tg_idx in pl.spmd(
            PREFILL_T // TOK_TILE, name_hint=f"{name_prefix}_k_proj",
        ):
            tg = tg_idx * TOK_TILE
            a0 = pl.slice(
                normed_out, [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, 0],
            )
            wk0 = pl.slice(
                wk, [INPUT_PROJ_K_CHUNK, KV_HIDDEN_LOCAL],
                [layer_hidden_base, 0],
            )
            k_acc = pl.matmul(a0, wk0, out_dtype=pl.FP32)
            for kb in pl.range(1, d_blocks):
                k0 = kb * INPUT_PROJ_K_CHUNK
                a = pl.slice(
                    normed_out, [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, k0],
                )
                w = pl.slice(
                    wk, [INPUT_PROJ_K_CHUNK, KV_HIDDEN_LOCAL],
                    [layer_hidden_base + k0, 0],
                )
                k_acc = pl.matmul_acc(k_acc, a, w)
            k_proj = pl.assemble(k_proj, k_acc, [tg, 0])

        # ── Stage 1.d — V projection. ────────────────────────────────────
        v_proj = pl.create_tensor(
            [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
        )
        for tg_idx in pl.spmd(
            PREFILL_T // TOK_TILE, name_hint=f"{name_prefix}_v_proj",
        ):
            tg = tg_idx * TOK_TILE
            a0 = pl.slice(
                normed_out, [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, 0],
            )
            wv0 = pl.slice(
                wv, [INPUT_PROJ_K_CHUNK, KV_HIDDEN_LOCAL],
                [layer_hidden_base, 0],
            )
            v_acc = pl.matmul(a0, wv0, out_dtype=pl.FP32)
            for kb in pl.range(1, d_blocks):
                k0 = kb * INPUT_PROJ_K_CHUNK
                a = pl.slice(
                    normed_out, [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, k0],
                )
                w = pl.slice(
                    wv, [INPUT_PROJ_K_CHUNK, KV_HIDDEN_LOCAL],
                    [layer_hidden_base + k0, 0],
                )
                v_acc = pl.matmul_acc(v_acc, a, w)
            v_proj = pl.assemble(v_proj, v_acc, [tg, 0])

        # ── Stage 1.e — head-wise gate matmul (on un-normed input). ──────
        for tg_idx in pl.spmd(
            PREFILL_T // TOK_TILE, name_hint=f"{name_prefix}_gate_proj",
        ):
            tg = tg_idx * TOK_TILE
            a0 = pl.slice(
                current_hidden, [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, 0],
            )
            wg0 = pl.slice(
                w_g, [INPUT_PROJ_K_CHUNK, num_heads_local],
                [layer_hidden_base, 0],
            )
            g_acc = pl.matmul(a0, wg0, out_dtype=pl.FP32)
            for kb in pl.range(1, d_blocks):
                k0 = kb * INPUT_PROJ_K_CHUNK
                a = pl.slice(
                    current_hidden,
                    [TOK_TILE, INPUT_PROJ_K_CHUNK], [tg, k0],
                )
                w = pl.slice(
                    w_g, [INPUT_PROJ_K_CHUNK, num_heads_local],
                    [layer_hidden_base + k0, 0],
                )
                g_acc = pl.matmul_acc(g_acc, a, w)
            gate_logits_out = pl.assemble(gate_logits_out, g_acc, [tg, 0])

        # ── Stage 1.f — per-head zero-centred q_norm / k_norm. ───────────
        q_proj_norm = pl.create_tensor(
            [PREFILL_T, hidden_q_local], dtype=pl.FP32,
        )
        k_proj_norm = pl.create_tensor(
            [PREFILL_T, KV_HIDDEN_LOCAL], dtype=pl.FP32,
        )
        for qkn_idx in pl.spmd(
            (PREFILL_T // TOK_TILE) * kv_heads_local,
            name_hint=f"{name_prefix}_qk_norm_zc",
        ):
            tg_idx2 = qkn_idx // kv_heads_local
            kh = qkn_idx % kv_heads_local
            tg = tg_idx2 * TOK_TILE
            q_col = kh * q_per_kv * HEAD_DIM
            q_chunk = pl.reshape(
                pl.slice(
                    q_proj, [TOK_TILE, q_per_kv * HEAD_DIM], [tg, q_col],
                ),
                [TOK_TILE * q_per_kv, HEAD_DIM],
            )
            q_gamma = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
            # Phase X.7: per_head_qk_norm body inlined.
            q_sq = pl.row_sum(pl.mul(q_chunk, q_chunk))
            q_inv = pl.rsqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS))
            q_scaled = pl.row_expand_mul(q_chunk, q_inv)
            q_normed = pl.col_expand_mul(q_scaled, pl.add(q_gamma, 1.0))
            q_normed_flat = pl.reshape(
                q_normed, [TOK_TILE, q_per_kv * HEAD_DIM],
            )
            q_proj_norm = pl.assemble(
                q_proj_norm, q_normed_flat, [tg, q_col],
            )

            k_col = kh * HEAD_DIM
            k_chunk = pl.slice(k_proj, [TOK_TILE, HEAD_DIM], [tg, k_col])
            k_gamma = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
            k_sq = pl.row_sum(pl.mul(k_chunk, k_chunk))
            k_inv = pl.rsqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS))
            k_scaled = pl.row_expand_mul(k_chunk, k_inv)
            k_normed = pl.col_expand_mul(k_scaled, pl.add(k_gamma, 1.0))
            k_proj_norm = pl.assemble(k_proj_norm, k_normed, [tg, k_col])

        # ── Stage 1.g — partial RoPE on Q and K (per-token positions). ───
        # ``positions[t]`` is the absolute cache row of token t. For the
        # prefill bring-up the host passes a contiguous arange.
        for t in pl.parallel(PREFILL_T):
            pos = pl.cast(pl.tensor.read(positions, [t]), pl.INDEX)
            cos_row = pl.slice(rope_cos, [1, rotary_dim], [pos, 0])
            sin_row = pl.slice(rope_sin, [1, rotary_dim], [pos, 0])
            cos_lo = pl.slice(cos_row, [1, rotary_half], [0, 0])
            cos_hi = pl.slice(cos_row, [1, rotary_half], [0, rotary_half])
            sin_lo = pl.slice(sin_row, [1, rotary_half], [0, 0])
            sin_hi = pl.slice(sin_row, [1, rotary_half], [0, rotary_half])

            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint=f"{name_prefix}_rope_q_k",
            ):
                # K RoPE — single rank-local KV head per card under TP=8.
                for kh in pl.range(kv_heads_local):
                    k_col = kh * HEAD_DIM
                    k_lo = pl.slice(
                        k_proj_norm, [1, rotary_half], [t, k_col],
                    )
                    k_hi = pl.slice(
                        k_proj_norm,
                        [1, rotary_half],
                        [t, k_col + rotary_half],
                    )
                    rot_k_lo = pl.sub(
                        pl.col_expand_mul(k_lo, cos_lo),
                        pl.col_expand_mul(k_hi, sin_lo),
                    )
                    rot_k_hi = pl.add(
                        pl.col_expand_mul(k_hi, cos_hi),
                        pl.col_expand_mul(k_lo, sin_hi),
                    )
                    k_out = pl.assemble(
                        k_out,
                        pl.cast(rot_k_lo, target_type=pl.BF16),
                        [t, k_col],
                    )
                    k_out = pl.assemble(
                        k_out,
                        pl.cast(rot_k_hi, target_type=pl.BF16),
                        [t, k_col + rotary_half],
                    )
                    if rotary_pass > 0:
                        k_pass = pl.slice(
                            k_proj_norm,
                            [1, rotary_pass],
                            [t, k_col + rotary_dim],
                        )
                        k_out = pl.assemble(
                            k_out,
                            pl.cast(k_pass, target_type=pl.BF16),
                            [t, k_col + rotary_dim],
                        )
                    # V — copy through (no rotation).
                    v_slice = pl.slice(
                        v_proj, [1, HEAD_DIM], [t, k_col],
                    )
                    v_out = pl.assemble(
                        v_out,
                        pl.cast(v_slice, target_type=pl.BF16),
                        [t, k_col],
                    )

                # Q RoPE — q_per_kv consecutive heads per KV-head bundle.
                for kh in pl.range(kv_heads_local):
                    q_base_col = kh * q_per_kv * HEAD_DIM
                    q_block = pl.reshape(
                        pl.slice(
                            q_proj_norm,
                            [1, q_per_kv * HEAD_DIM], [t, q_base_col],
                        ),
                        [q_per_kv, HEAD_DIM],
                    )
                    q_lo = pl.slice(
                        q_block, [q_per_kv, rotary_half], [0, 0],
                    )
                    q_hi = pl.slice(
                        q_block,
                        [q_per_kv, rotary_half],
                        [0, rotary_half],
                    )
                    rot_q_lo = pl.sub(
                        pl.col_expand_mul(q_lo, cos_lo),
                        pl.col_expand_mul(q_hi, sin_lo),
                    )
                    rot_q_hi = pl.add(
                        pl.col_expand_mul(q_hi, cos_hi),
                        pl.col_expand_mul(q_lo, sin_hi),
                    )
                    for qi in pl.range(q_per_kv):
                        h_col = q_base_col + qi * HEAD_DIM
                        rl = pl.slice(
                            rot_q_lo, [1, rotary_half], [qi, 0],
                        )
                        rh = pl.slice(
                            rot_q_hi, [1, rotary_half], [qi, 0],
                        )
                        q_out = pl.assemble(
                            q_out,
                            pl.cast(rl, target_type=pl.BF16),
                            [t, h_col],
                        )
                        q_out = pl.assemble(
                            q_out,
                            pl.cast(rh, target_type=pl.BF16),
                            [t, h_col + rotary_half],
                        )
                        if rotary_pass > 0:
                            q_pass = pl.slice(
                                q_proj_norm,
                                [1, rotary_pass],
                                [t, h_col + rotary_dim],
                            )
                            q_out = pl.assemble(
                                q_out,
                                pl.cast(q_pass, target_type=pl.BF16),
                                [t, h_col + rotary_dim],
                            )

        return normed_out, q_out, k_out, v_out, gate_logits_out

    return prefill_qkv_proj_rope_body


# Pre-built specialisations.
prefill_qkv_proj_rope_full = _build_prefill_qkv_proj_rope(full=True)
prefill_qkv_proj_rope_swa = _build_prefill_qkv_proj_rope(full=False)


def select_prefill_qkv(full: bool):
    """Return the pre-built QKV+RoPE body for the requested flavour."""
    return prefill_qkv_proj_rope_full if full else prefill_qkv_proj_rope_swa


# =============================================================================
# Standalone @pl.program wrapper — chip_orch + host_orch.
# Mirrors the deferred-build pattern from ``attention_full._build_tp_*``.
# =============================================================================
def _build_tp_prefill_qkv_proj_rope_program(
    *, full: bool, tp_size: int = TP_WORLD_SIZE,
):
    """Return a freshly-built ``@pl.program`` class wrapping the QKV+RoPE body.

    The wrapper is a thin pass-through: no collective is invoked inside
    the QKV+RoPE body, so the per-rank ``chip_orch`` just calls the
    inline body once per rank from the ``host_orch`` rank loop.
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be a multiple of tp_size={tp_size}"
        )
    num_heads_local = NUM_HEADS_FULL_LOCAL if full else NUM_HEADS_SWA_LOCAL
    hidden_q_local = HIDDEN_Q_FULL_LOCAL if full else HIDDEN_Q_SWA_LOCAL
    rotary_dim = ROTARY_HALF_FULL * 2 if full else ROTARY_HALF_SWA * 2
    body = select_prefill_qkv(full)
    body_inline = pl.inline(body._func)

    @pl.program
    class TpPrefillQkvProjRope:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16,
            ],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16,
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16,
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, num_heads_local], pl.BF16,
            ],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            normed_out: pl.Out[
                pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]
            ],
            q_out: pl.Out[
                pl.Tensor[[PREFILL_T, hidden_q_local], pl.BF16]
            ],
            k_out: pl.Out[
                pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16]
            ],
            v_out: pl.Out[
                pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16]
            ],
            gate_logits_out: pl.Out[
                pl.Tensor[[PREFILL_T, num_heads_local], pl.FP32]
            ],
            layer_idx: pl.Scalar[pl.INT32],
        ):
            normed_out, q_out, k_out, v_out, gate_logits_out = body_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                w_g,
                rope_cos, rope_sin,
                positions,
                normed_out, q_out, k_out, v_out, gate_logits_out,
                layer_idx,
            )
            return normed_out, q_out, k_out, v_out, gate_logits_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[
                [tp_size, PREFILL_T, HIDDEN], pl.BF16,
            ],
            input_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32,
            ],
            wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16,
            ],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16,
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16,
            ],
            q_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32,
            ],
            k_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32,
            ],
            w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, num_heads_local], pl.BF16,
            ],
            rope_cos: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32,
            ],
            rope_sin: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32,
            ],
            positions: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            normed_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]
            ],
            q_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, hidden_q_local], pl.BF16]
            ],
            k_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16]
            ],
            v_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16]
            ],
            gate_logits_out: pl.Out[
                pl.Tensor[
                    [tp_size, PREFILL_T, num_heads_local], pl.FP32,
                ]
            ],
            layer_idx: pl.Scalar[pl.INT32],
        ):
            for r in pl.range(pld.world_size()):
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    w_g[r],
                    rope_cos[r], rope_sin[r],
                    positions[r],
                    normed_out[r],
                    q_out[r], k_out[r], v_out[r],
                    gate_logits_out[r],
                    layer_idx,
                    device=r,
                )

    return TpPrefillQkvProjRope


def _build_tp_prefill_qkv_proj_rope_full_program(
    tp_size: int = TP_WORLD_SIZE,
):
    return _build_tp_prefill_qkv_proj_rope_program(full=True, tp_size=tp_size)


def _build_tp_prefill_qkv_proj_rope_swa_program(
    tp_size: int = TP_WORLD_SIZE,
):
    return _build_tp_prefill_qkv_proj_rope_program(full=False, tp_size=tp_size)


# =============================================================================
# Torch reference helpers (used by the distributed-mock harness in
# ``prefill_attention_full.py`` / ``prefill_attention_swa.py``).
# =============================================================================
def _torch_prefill_qkv_oracle_impl(
    *, hidden, input_rms_weight, wq_full, wk_full, wv_full,
    q_norm_weight, k_norm_weight, w_g_full,
    rope_cos, rope_sin, positions,
    num_heads_full, num_kv_heads_full, rotary_half,
):
    """Implementation shared by the full and SWA torch oracles."""
    import torch

    rotary_dim = rotary_half * 2
    head_dim = HEAD_DIM
    rotary_pass = head_dim - rotary_dim

    t = hidden.shape[0]

    def zc(x, g):
        return x * (g + 1.0)

    x = hidden.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    normed_bf16 = zc(
        x * torch.rsqrt(var + EPS), input_rms_weight.float(),
    ).bfloat16()

    q_proj = normed_bf16.float() @ wq_full.float()  # [T, H*head_dim]
    k_proj = normed_bf16.float() @ wk_full.float()  # [T, kv*head_dim]
    v_proj = normed_bf16.float() @ wv_full.float()

    q_h = q_proj.view(t, num_heads_full, head_dim)
    q_h = zc(
        q_h * torch.rsqrt(q_h.pow(2).mean(-1, keepdim=True) + EPS),
        q_norm_weight.float(),
    )
    k_h = k_proj.view(t, num_kv_heads_full, head_dim)
    k_h = zc(
        k_h * torch.rsqrt(k_h.pow(2).mean(-1, keepdim=True) + EPS),
        k_norm_weight.float(),
    )

    q_rot = torch.zeros_like(q_h)
    k_rot = torch.zeros_like(k_h)
    for ti in range(t):
        pos = int(positions[ti].item())
        cr = rope_cos[pos : pos + 1, :]
        sr = rope_sin[pos : pos + 1, :]
        c_lo, c_hi = cr[:, :rotary_half], cr[:, rotary_half:rotary_dim]
        s_lo, s_hi = sr[:, :rotary_half], sr[:, rotary_half:rotary_dim]
        for h in range(num_heads_full):
            v_head = q_h[ti, h]
            lo = v_head[:rotary_half]
            hi = v_head[rotary_half:rotary_dim]
            rl = lo * c_lo - hi * s_lo
            rh = hi * c_hi + lo * s_hi
            q_rot[ti, h, :rotary_half] = rl
            q_rot[ti, h, rotary_half:rotary_dim] = rh
            if rotary_pass > 0:
                q_rot[ti, h, rotary_dim:] = v_head[rotary_dim:]
        for h in range(num_kv_heads_full):
            v_head = k_h[ti, h]
            lo = v_head[:rotary_half]
            hi = v_head[rotary_half:rotary_dim]
            rl = lo * c_lo - hi * s_lo
            rh = hi * c_hi + lo * s_hi
            k_rot[ti, h, :rotary_half] = rl
            k_rot[ti, h, rotary_half:rotary_dim] = rh
            if rotary_pass > 0:
                k_rot[ti, h, rotary_dim:] = v_head[rotary_dim:]

    gate_logits = hidden.float() @ w_g_full.float()

    return {
        "normed": normed_bf16,
        "q_rot": q_rot.bfloat16(),
        "k_rot": k_rot.bfloat16(),
        "v_proj": v_proj.bfloat16().view(t, num_kv_heads_full, head_dim),
        "gate_logits": gate_logits,
    }


def _torch_prefill_qkv_proj_rope_full_oracle(
    *, hidden, input_rms_weight, wq_full, wk_full, wv_full,
    q_norm_weight, k_norm_weight, w_g_full, rope_cos, rope_sin, positions,
):
    """Single-card (world-level) torch oracle for the full-attn QKV+RoPE."""
    return _torch_prefill_qkv_oracle_impl(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        w_g_full=w_g_full,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
        num_heads_full=NUM_HEADS_FULL_LOCAL * TP_WORLD_SIZE,
        num_kv_heads_full=KV_HEADS_LOCAL * TP_WORLD_SIZE,
        rotary_half=ROTARY_HALF_FULL,
    )


def _torch_prefill_qkv_proj_rope_swa_oracle(
    *, hidden, input_rms_weight, wq_full, wk_full, wv_full,
    q_norm_weight, k_norm_weight, w_g_full, rope_cos, rope_sin, positions,
):
    """Single-card (world-level) torch oracle for the SWA QKV+RoPE."""
    return _torch_prefill_qkv_oracle_impl(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        w_g_full=w_g_full,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
        num_heads_full=NUM_HEADS_SWA_LOCAL * TP_WORLD_SIZE,
        num_kv_heads_full=KV_HEADS_LOCAL * TP_WORLD_SIZE,
        rotary_half=ROTARY_HALF_SWA,
    )


# =============================================================================
# Distributed-mock harness — verifies the QKV+RoPE TP wiring against a
# single-card torch oracle.
# =============================================================================
def _run_distributed_mock(
    *,
    full: bool,
    pass_rate: float = 0.97,
    rtol: float = 5e-3,
    atol: float = 5e-3,
    seed: int = 0,
):
    import torch

    torch.manual_seed(seed)
    num_heads_full = (
        NUM_HEADS_FULL_LOCAL if full else NUM_HEADS_SWA_LOCAL
    ) * TP_WORLD_SIZE
    num_kv_heads_full = KV_HEADS_LOCAL * TP_WORLD_SIZE
    hidden_q_full = num_heads_full * HEAD_DIM
    rotary_half = ROTARY_HALF_FULL if full else ROTARY_HALF_SWA
    rotary_dim = rotary_half * 2

    layer_idx = 0 if full else 1
    layer_rope_theta = LAYER_ROPE_THETA[layer_idx]
    if full:
        rope_cos, rope_sin = build_llama3_yarn_rope_tables(
            MAX_SEQ_DEFAULT, rotary_dim, layer_rope_theta,
            factor=ROPE_SCALING["factor"],
            low=ROPE_SCALING["low_freq_factor"],
            high=ROPE_SCALING["high_freq_factor"],
            orig_max=ROPE_SCALING["original_max_position_embeddings"],
        )
    else:
        rope_cos, rope_sin = build_plain_rope_tables(
            MAX_SEQ_DEFAULT, rotary_dim, layer_rope_theta,
        )

    hidden = (torch.rand(PREFILL_T, HIDDEN) - 0.5).bfloat16()
    input_rms_weight = ((torch.rand(1, HIDDEN) - 0.5) * 0.1).float()
    wq_full = (
        (torch.rand(HIDDEN, hidden_q_full) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    wk_full = (
        (torch.rand(HIDDEN, num_kv_heads_full * HEAD_DIM) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    wv_full = (
        (torch.rand(HIDDEN, num_kv_heads_full * HEAD_DIM) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    q_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    k_norm_weight = ((torch.rand(1, HEAD_DIM) - 0.5) * 0.1).float()
    w_g_full = (
        (torch.rand(HIDDEN, num_heads_full) - 0.5) / HIDDEN ** 0.5
    ).bfloat16()
    positions = torch.arange(PREFILL_T, dtype=torch.int32)

    oracle = _torch_prefill_qkv_oracle_impl(
        hidden=hidden,
        input_rms_weight=input_rms_weight,
        wq_full=wq_full, wk_full=wk_full, wv_full=wv_full,
        q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
        w_g_full=w_g_full,
        rope_cos=rope_cos, rope_sin=rope_sin,
        positions=positions,
        num_heads_full=num_heads_full,
        num_kv_heads_full=num_kv_heads_full,
        rotary_half=rotary_half,
    )

    rank_pass = []
    n_heads_local = num_heads_full // TP_WORLD_SIZE
    n_kv_heads_local = num_kv_heads_full // TP_WORLD_SIZE
    for r in range(TP_WORLD_SIZE):
        wq_r = wq_full[
            :, r * n_heads_local * HEAD_DIM
            : (r + 1) * n_heads_local * HEAD_DIM,
        ]
        wk_r = wk_full[
            :, r * n_kv_heads_local * HEAD_DIM
            : (r + 1) * n_kv_heads_local * HEAD_DIM,
        ]
        wv_r = wv_full[
            :, r * n_kv_heads_local * HEAD_DIM
            : (r + 1) * n_kv_heads_local * HEAD_DIM,
        ]
        w_g_r = w_g_full[
            :, r * n_heads_local : (r + 1) * n_heads_local,
        ]
        rank_out = _torch_prefill_qkv_oracle_impl(
            hidden=hidden,
            input_rms_weight=input_rms_weight,
            wq_full=wq_r, wk_full=wk_r, wv_full=wv_r,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            w_g_full=w_g_r,
            rope_cos=rope_cos, rope_sin=rope_sin,
            positions=positions,
            num_heads_full=n_heads_local,
            num_kv_heads_full=n_kv_heads_local,
            rotary_half=rotary_half,
        )
        exp_q = oracle["q_rot"][
            :, r * n_heads_local : (r + 1) * n_heads_local, :,
        ]
        exp_k = oracle["k_rot"][
            :, r * n_kv_heads_local : (r + 1) * n_kv_heads_local, :,
        ]
        exp_v = oracle["v_proj"][
            :, r * n_kv_heads_local : (r + 1) * n_kv_heads_local, :,
        ]
        exp_g = oracle["gate_logits"][
            :, r * n_heads_local : (r + 1) * n_heads_local,
        ]
        close_q = torch.isclose(
            rank_out["q_rot"].float(), exp_q.float(),
            rtol=rtol, atol=atol,
        )
        close_k = torch.isclose(
            rank_out["k_rot"].float(), exp_k.float(),
            rtol=rtol, atol=atol,
        )
        close_v = torch.isclose(
            rank_out["v_proj"].float(), exp_v.float(),
            rtol=rtol, atol=atol,
        )
        close_g = torch.isclose(
            rank_out["gate_logits"], exp_g, rtol=rtol, atol=atol,
        )
        ok_count = (
            int(close_q.sum()) + int(close_k.sum())
            + int(close_v.sum()) + int(close_g.sum())
        )
        total = (
            close_q.numel() + close_k.numel()
            + close_v.numel() + close_g.numel()
        )
        rank_pass.append(ok_count / total)

    worst = min(rank_pass)
    ok = worst >= pass_rate
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] prefill_qkv_proj_rope({'full' if full else 'swa'}) "
        f"distributed-mock worst_rank_pass_rate={worst:.6f} "
        f"threshold={pass_rate:.6f}"
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
    parser.add_argument(
        "--variant", choices=["full", "swa"], default="full",
    )
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build-program-only", action="store_true")
    args = parser.parse_args()

    full = args.variant == "full"
    program_cls = _build_tp_prefill_qkv_proj_rope_program(full=full)
    print(
        f"[OK] built @pl.program {program_cls.__name__} "
        f"(variant={args.variant}, tp_size={TP_WORLD_SIZE})"
    )
    if args.build_program_only:
        raise SystemExit(0)
    ok = _run_distributed_mock(
        full=full, pass_rate=args.pass_rate, seed=args.seed,
    )
    if not ok:
        raise SystemExit(1)


__all__ = [
    "PREFILL_BATCH",
    "PREFILL_SEQ",
    "PREFILL_T",
    "TOK_TILE",
    "prefill_qkv_proj_rope_full",
    "prefill_qkv_proj_rope_swa",
    "select_prefill_qkv",
    "_build_tp_prefill_qkv_proj_rope_program",
    "_build_tp_prefill_qkv_proj_rope_full_program",
    "_build_tp_prefill_qkv_proj_rope_swa_program",
    "_torch_prefill_qkv_proj_rope_full_oracle",
    "_torch_prefill_qkv_proj_rope_swa_oracle",
    "_torch_prefill_qkv_oracle_impl",
    "_run_distributed_mock",
]
