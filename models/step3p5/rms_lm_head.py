# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 final RMSNorm + LM-head projection — TP vocab-sliced (Phase 9 Wave 3).

Each TP rank owns a contiguous ``VOCAB_LOCAL = VOCAB // TP_WORLD_SIZE``
vocabulary slab of ``lm_head_weight``. The kernel emits the per-rank
logits shard ``[USER_BATCH, VOCAB_LOCAL]`` FP32 and does NOT all-gather
or all-reduce — the caller is responsible for whatever downstream gather
makes sense (e.g. per-rank argmax followed by a tiny INT32 + FP32
``(arg, val)`` pair all-gather to pick the global argmax). This split
keeps the heaviest matmul (rank-local) decoupled from the cheap
cross-rank tie-break.

Final-RMSNorm staging:
  The residual stream arriving at ``rms_lm_head`` has already been
  homogenised by the last layer's TP all-reduce (every rank holds the
  same fully-reduced hidden), so the per-row zero-centred RMSNorm runs
  REPLICATED on every rank — no extra all-reduce is needed. The
  ``final_norm_weight`` gamma is also replicated.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import zero_centered_rmsnorm_apply
from .config import (
    BATCH,
    BATCH_TILE,
    EPS,
    FINAL_RMS_K_CHUNK,
    HIDDEN,
    HIDDEN_INV,
    LM_HEAD_K_CHUNK,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    VOCAB,
    VOCAB_CHUNK,
    VOCAB_LOCAL,
)


assert VOCAB_LOCAL % VOCAB_CHUNK == 0, (
    f"VOCAB_LOCAL={VOCAB_LOCAL} must be a multiple of VOCAB_CHUNK={VOCAB_CHUNK}"
)
assert HIDDEN % LM_HEAD_K_CHUNK == 0
assert HIDDEN % FINAL_RMS_K_CHUNK == 0


# =============================================================================
# TP vocab-sliced final RMSNorm + LM-head matmul (inline body).
# =============================================================================
@pl.jit.inline
def rms_lm_head(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB_LOCAL, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]:
    """Per-rank zero-centred RMSNorm + vocab-sliced LM head matmul.

    Steps:
      1. Per-row zero-centred RMSNorm of ``hidden_states`` against the
         REPLICATED ``final_norm_weight``. Emits BF16 ``final_normed``
         of shape ``[BATCH, HIDDEN]``.
      2. Tiled matmul ``final_normed @ lm_head_weight.T`` into the rank-
         local shard ``out [USER_BATCH_DYN, VOCAB_LOCAL]`` FP32. The
         trailing rows past the dynamic ``user_batch`` are masked off
         via ``valid_shape``.

    The kernel does not produce a full-vocab tensor and does not run a
    cross-rank gather; the caller plumbs the per-rank shards as needed
    (e.g. per-rank argmax + cross-rank (idx, val) gather).
    """
    user_batch = pl.tensor.dim(seq_lens, 0)
    rms_blocks = HIDDEN // FINAL_RMS_K_CHUNK
    k_blocks = HIDDEN // LM_HEAD_K_CHUNK
    vocab_blocks = VOCAB_LOCAL // VOCAB_CHUNK

    # ── Step 1: replicated zero-centred RMSNorm. ───────────────────────
    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm_zc"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(rms_blocks):
                final_sq_k0 = kb * FINAL_RMS_K_CHUNK
                final_sq_chunk = pl.cast(
                    pl.slice(
                        hidden_states,
                        [BATCH_TILE, FINAL_RMS_K_CHUNK],
                        [b0, final_sq_k0],
                    ),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(
                        pl.row_sum(pl.mul(final_sq_chunk, final_sq_chunk)),
                        [1, BATCH_TILE],
                    ),
                )
            inv_rms_final = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )

            for kb in pl.range(rms_blocks):
                final_norm_k0 = kb * FINAL_RMS_K_CHUNK
                final_hidden_chunk = pl.cast(
                    pl.slice(
                        hidden_states,
                        [BATCH_TILE, FINAL_RMS_K_CHUNK],
                        [b0, final_norm_k0],
                    ),
                    target_type=pl.FP32,
                )
                final_gamma = pl.slice(
                    final_norm_weight,
                    [1, FINAL_RMS_K_CHUNK],
                    [0, final_norm_k0],
                )
                scaled = pl.row_expand_mul(final_hidden_chunk, inv_rms_final)
                # Inlined zero_centered_rmsnorm_apply: gamma_eff = gamma + 1.0,
                # then col-broadcast multiply. pypto frontend rejects calling
                # the @pl.jit.inline helper from inside a @pl.program method
                # body (Phase X.7 lift recipe), so we expand it here.
                final_normed_chunk = pl.col_expand_mul(
                    scaled, pl.add(final_gamma, 1.0),
                )
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(final_normed_chunk, target_type=pl.BF16),
                    [b0, final_norm_k0],
                )

    # ── Step 2: per-rank LM-head matmul into the VOCAB_LOCAL shard. ────
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        lm_valid_rows = pl.min(BATCH_TILE, user_batch - b0)
        for ob in pl.parallel(vocab_blocks):
            lm_o0 = ob * VOCAB_CHUNK
            lm_acc_gm = pl.create_tensor(
                [BATCH_TILE, VOCAB_CHUNK], dtype=pl.FP32,
            )
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_tp"):
                lm_hidden_chunk = pl.slice(
                    final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, 0],
                )
                lm_weight_chunk = pl.slice(
                    lm_head_weight,
                    [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                    [lm_o0, 0],
                )
                lm_acc = pl.matmul(
                    lm_hidden_chunk, lm_weight_chunk,
                    out_dtype=pl.FP32, b_trans=True,
                )
                for kb in pl.range(1, k_blocks):
                    lm_k0 = kb * LM_HEAD_K_CHUNK
                    lm_hidden_chunk = pl.slice(
                        final_normed,
                        [BATCH_TILE, LM_HEAD_K_CHUNK],
                        [b0, lm_k0],
                    )
                    lm_weight_chunk = pl.slice(
                        lm_head_weight,
                        [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                        [lm_o0, lm_k0],
                    )
                    lm_acc = pl.matmul_acc(
                        lm_acc, lm_hidden_chunk, lm_weight_chunk,
                        b_trans=True,
                    )
                lm_acc_gm = pl.assemble(lm_acc_gm, lm_acc, [0, 0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_store_tp"):
                lm_acc_chunk = pl.slice(
                    lm_acc_gm, [BATCH_TILE, VOCAB_CHUNK], [0, 0],
                )
                lm_acc_trimmed = pl.slice(
                    lm_acc_chunk,
                    [BATCH_TILE, VOCAB_CHUNK],
                    [0, 0],
                    valid_shape=[lm_valid_rows, VOCAB_CHUNK],
                )
                out = pl.assemble(out, lm_acc_trimmed, [b0, lm_o0])

    return out


# =============================================================================
# TP wrapper — Wave-3 program scaffolding (chip_orch + host_orch).
# =============================================================================
def _build_tp_rms_lm_head_program(tp_size: int = TP_WORLD_SIZE):
    """Return a freshly-built ``@pl.program`` class for the TP-sliced head.

    Deferred-build pattern (mirrors ``attention_full._build_tp_*``): the
    class is constructed inside a Python factory so the module imports
    cleanly even without a pypto runtime.
    """
    if VOCAB % tp_size != 0:
        raise ValueError(
            f"VOCAB={VOCAB} must be divisible by tp_size={tp_size}"
        )
    rms_lm_head_inline = pl.inline(rms_lm_head._func)
    vocab_per_tp = VOCAB // tp_size

    @pl.program
    class TpRmsLmHead:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[[vocab_per_tp, HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[[USER_BATCH_DYN, vocab_per_tp], pl.FP32]
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[USER_BATCH_DYN, vocab_per_tp], pl.FP32]:
            # ``my_rank`` is reserved for future cross-rank gather hooks;
            # the per-rank LM-head body is rank-symmetric so the parameter
            # is currently unused in the matmul itself. (Note: pypto's
            # frontend AST parser rejects Python ``del`` statements inside
            # @pl.function bodies, so we leave the unused parameter rather
            # than ``del`` it — pypto won't warn on unused params.)
            logits_shard_out = rms_lm_head_inline(
                hidden_states,
                final_norm_weight,
                lm_head_weight,
                seq_lens,
                logits_shard_out,
            )
            return logits_shard_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            hidden_states: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],
            final_norm_weight: pl.Tensor[[tp_size, 1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[
                [tp_size, vocab_per_tp, HIDDEN], pl.BF16
            ],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[[tp_size, USER_BATCH_DYN, vocab_per_tp], pl.FP32]
            ],
        ):
            for r in pl.range(pld.world_size()):
                self.chip_orch(
                    hidden_states[r],
                    final_norm_weight[r],
                    lm_head_weight[r],
                    seq_lens[r],
                    logits_shard_out[r],
                    r,
                    device=r,
                )

    return TpRmsLmHead


# Pre-built default. Importers can construct a different size with
# ``_build_tp_rms_lm_head_program(tp_size)``.
TpRmsLmHead = _build_tp_rms_lm_head_program(TP_WORLD_SIZE)


def golden_rms_lm_head(hidden_states, final_norm_weight, lm_head_weight):
    """Torch reference for the TP-sliced final RMSNorm + LM-head matmul.

    The reference accepts EITHER a per-rank shard
    ``[VOCAB_LOCAL, HIDDEN]`` (returning ``[batch, VOCAB_LOCAL]``) OR
    the full ``[VOCAB, HIDDEN]`` weight (returning ``[batch, VOCAB]``);
    callers reshape as needed. RMSNorm itself is replicated, so this
    function does not need a tp_size argument.
    """
    import torch

    h = hidden_states.float()
    variance = h.pow(2).mean(dim=-1, keepdim=True)
    gamma_eff = final_norm_weight.float() + 1.0
    final_normed = (h * torch.rsqrt(variance + EPS) * gamma_eff).bfloat16()

    vocab_out = lm_head_weight.shape[0]
    out = torch.zeros(h.shape[0], vocab_out, dtype=torch.float32)
    rhs = lm_head_weight.float()
    lhs = final_normed.float()
    for k0 in range(0, HIDDEN, LM_HEAD_K_CHUNK):
        out = out + lhs[:, k0:k0 + LM_HEAD_K_CHUNK] @ rhs[
            :, k0:k0 + LM_HEAD_K_CHUNK,
        ].T
    return out


__all__ = [
    "rms_lm_head",
    "TpRmsLmHead",
    "_build_tp_rms_lm_head_program",
    "golden_rms_lm_head",
    "VOCAB_LOCAL",
]
