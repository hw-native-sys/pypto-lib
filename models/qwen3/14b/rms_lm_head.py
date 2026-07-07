# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B final RMSNorm and LM head projection."""

# pyright: reportUndefinedVariable=false

import pypto.language as pl

from config import (
    BATCH,
    BATCH_TILE,
    EPS,
    FINAL_RMS_K_CHUNK,
    HIDDEN,
    HIDDEN_INV,
    USER_BATCH_DYN,
    VOCAB,
)

# Local overrides — config defaults (64/128) made cube tasks too small,
# leaving cube cores at ~45% utilisation behind dispatch bubbles. Wider
# N+K amortises per-task dispatch overhead and lifts the innermost K dim
# to one L2 cache line (512 B, perf_hint PH001).
#
# VOCAB_CHUNK is capped by the L1 Mat-buffer limit (512 KiB): the stage=2
# K-pipeline below double-buffers both operands, so the buffer needs
# (BATCH_TILE*K + VOCAB_CHUNK*K) * 2 bytes * 2. With K=512 that forces
# VOCAB_CHUNK <= 240; 192 is the largest 16-multiple that divides VOCAB
# (152064 / 192 = 792 chunks) and keeps the wide K cache line.
LM_HEAD_K_CHUNK = 512
VOCAB_CHUNK = 192
LM_HEAD_CORES = 24


@pl.jit.inline
def rms_lm_head(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    user_batch = pl.tensor.dim(seq_lens, 0)

    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_sq_k0 = kb * FINAL_RMS_K_CHUNK
                final_sq_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_sq_k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(final_sq_chunk, final_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_final = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )

            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_norm_k0 = kb * FINAL_RMS_K_CHUNK
                final_hidden_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_norm_k0]),
                    target_type=pl.FP32,
                )
                final_gamma = pl.slice(final_norm_weight, [1, FINAL_RMS_K_CHUNK], [0, final_norm_k0])
                final_normed_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(final_hidden_chunk, inv_rms_final),
                    final_gamma,
                )
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(final_normed_chunk, target_type=pl.BF16),
                    [b0, final_norm_k0],
                )

    # LM-head matmul: ONE pl.spmd grid-stride over the VOCAB//VOCAB_CHUNK (792)
    # output chunks across LM_HEAD_CORES persistent blocks (see LM_HEAD_CORES above
    # for why one spmd dispatch beats the old per-chunk pl.parallel fan-out). Like
    # q_proj / fa_fused / out_proj. Each block owns a DISJOINT set of vocab columns,
    # so there is no split-K / atomic-add and the output is bit-identical to the
    # fan-out. M tiles (b0) are walked serially inside each block, mirroring the
    # final_rmsnorm loop above (BATCH == BATCH_TILE today, so it runs once).
    for lm_core in pl.spmd(LM_HEAD_CORES, name_hint="lm_head"):
        for b0 in pl.range(0, BATCH, BATCH_TILE):
            lm_valid_rows = pl.min(BATCH_TILE, user_batch - b0)
            for ob in pl.range(lm_core, VOCAB // VOCAB_CHUNK, LM_HEAD_CORES):
                lm_o0 = ob * VOCAB_CHUNK
                # matmul (cube), then trim the L0C result to the dynamic user-batch rows
                # via set_validshape and store straight to `out` (no GM scratch + separate
                # vector store). The valid-row trim bounds the write into the
                # dynamic-shaped `out` [USER_BATCH_DYN, VOCAB].
                lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, 0])
                lm_weight_chunk = pl.slice(lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [lm_o0, 0])
                lm_acc = pl.matmul(lm_hidden_chunk, lm_weight_chunk, out_dtype=pl.FP32, b_trans=True)
                # Pipeline the K-accumulation (stage=2) so the next K-tile load overlaps
                # the current matmul_acc — same idiom as q_proj / out_proj / gate_proj.
                # The leading matmul (kb=0) stays peeled: it initializes the L0C
                # accumulator, so it can't sit inside the pipelined accumulate loop.
                for kb in pl.pipeline(1, HIDDEN // LM_HEAD_K_CHUNK, stage=2):
                    lm_k0 = kb * LM_HEAD_K_CHUNK
                    lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, lm_k0])
                    lm_weight_chunk = pl.slice(
                        lm_head_weight,
                        [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                        [lm_o0, lm_k0],
                    )
                    lm_acc = pl.matmul_acc(lm_acc, lm_hidden_chunk, lm_weight_chunk, b_trans=True)
                lm_acc_trimmed = pl.tensor.set_validshape(lm_acc, lm_valid_rows, VOCAB_CHUNK)
                out = pl.assemble(out, lm_acc_trimmed, [b0, lm_o0])

    return out


@pl.jit.inline
def rms_lm_head_fp32(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    user_batch = pl.tensor.dim(seq_lens, 0)

    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_sq_k0 = kb * FINAL_RMS_K_CHUNK
                final_sq_chunk = pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_sq_k0])
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(final_sq_chunk, final_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_final = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )

            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_norm_k0 = kb * FINAL_RMS_K_CHUNK
                final_hidden_chunk = pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_norm_k0])
                final_gamma = pl.slice(final_norm_weight, [1, FINAL_RMS_K_CHUNK], [0, final_norm_k0])
                final_normed_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(final_hidden_chunk, inv_rms_final),
                    final_gamma,
                )
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(final_normed_chunk, target_type=pl.BF16),
                    [b0, final_norm_k0],
                )

    for lm_core in pl.spmd(LM_HEAD_CORES, name_hint="lm_head"):
        for b0 in pl.range(0, BATCH, BATCH_TILE):
            lm_valid_rows = pl.min(BATCH_TILE, user_batch - b0)
            for ob in pl.range(lm_core, VOCAB // VOCAB_CHUNK, LM_HEAD_CORES):
                lm_o0 = ob * VOCAB_CHUNK
                lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, 0])
                lm_weight_chunk = pl.slice(lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [lm_o0, 0])
                lm_acc = pl.matmul(lm_hidden_chunk, lm_weight_chunk, out_dtype=pl.FP32, b_trans=True)
                for kb in pl.pipeline(1, HIDDEN // LM_HEAD_K_CHUNK, stage=2):
                    lm_k0 = kb * LM_HEAD_K_CHUNK
                    lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, lm_k0])
                    lm_weight_chunk = pl.slice(
                        lm_head_weight,
                        [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                        [lm_o0, lm_k0],
                    )
                    lm_acc = pl.matmul_acc(lm_acc, lm_hidden_chunk, lm_weight_chunk, b_trans=True)
                lm_acc_trimmed = pl.set_validshape(lm_acc, lm_valid_rows, VOCAB_CHUNK)
                out = pl.assemble(out, lm_acc_trimmed, [b0, lm_o0])

    return out


@pl.jit.inline
def rms_only(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    """Variant of rms_lm_head that performs the final RMSNorm but skips the
    LM-head matmul entirely. ``out`` is returned untouched (the harness
    zero-inits it), and ``lm_head_weight`` is accepted but unused so the
    function signature stays interchangeable with ``rms_lm_head``.

    TODO: ``final_normed`` is written but never read or returned (the LM-head
    matmul that consumes it is gone in this skip variant). Each pl.assemble
    is a GM store with side effects, so the pypto JIT should preserve the
    RMSNorm loop, but the value is dead-data from the IR's perspective and
    a future DCE pass could elide it (Gemini review on PR #387). If the
    measured RMSNorm cost collapses to ~0 in perf traces, force ``final_normed``
    live by writing a small slice (e.g. ``out[:, :HIDDEN]`` cast to FP32) and
    mirroring the same dummy slice in ``golden_decode_layer_no_lm_head``."""
    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_sq_k0 = kb * FINAL_RMS_K_CHUNK
                final_sq_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_sq_k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(final_sq_chunk, final_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_final = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )

            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_norm_k0 = kb * FINAL_RMS_K_CHUNK
                final_hidden_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_norm_k0]),
                    target_type=pl.FP32,
                )
                final_gamma = pl.slice(final_norm_weight, [1, FINAL_RMS_K_CHUNK], [0, final_norm_k0])
                final_normed_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(final_hidden_chunk, inv_rms_final),
                    final_gamma,
                )
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(final_normed_chunk, target_type=pl.BF16),
                    [b0, final_norm_k0],
                )

    return out


@pl.jit.inline
def rms_lm_head_single_chunk(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    """Variant of rms_lm_head that runs only the first ``VOCAB_CHUNK`` of the
    LM-head matmul (one outer ``ob`` iteration). Rows past ``VOCAB_CHUNK``
    stay zero-initialised by the harness."""
    user_batch = pl.tensor.dim(seq_lens, 0)

    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_sq_k0 = kb * FINAL_RMS_K_CHUNK
                final_sq_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_sq_k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(final_sq_chunk, final_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_final = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )

            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                final_norm_k0 = kb * FINAL_RMS_K_CHUNK
                final_hidden_chunk = pl.cast(
                    pl.slice(hidden_states, [BATCH_TILE, FINAL_RMS_K_CHUNK], [b0, final_norm_k0]),
                    target_type=pl.FP32,
                )
                final_gamma = pl.slice(final_norm_weight, [1, FINAL_RMS_K_CHUNK], [0, final_norm_k0])
                final_normed_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(final_hidden_chunk, inv_rms_final),
                    final_gamma,
                )
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(final_normed_chunk, target_type=pl.BF16),
                    [b0, final_norm_k0],
                )

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        lm_valid_rows = pl.min(BATCH_TILE, user_batch - b0)
        lm_o0 = 0
        lm_acc_gm = pl.create_tensor([BATCH_TILE, VOCAB_CHUNK], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head"):
            lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, 0])
            lm_weight_chunk = pl.slice(lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [lm_o0, 0])
            lm_acc = pl.matmul(lm_hidden_chunk, lm_weight_chunk, out_dtype=pl.FP32, b_trans=True)
            for kb in pl.range(1, HIDDEN // LM_HEAD_K_CHUNK):
                lm_k0 = kb * LM_HEAD_K_CHUNK
                lm_hidden_chunk = pl.slice(final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, lm_k0])
                lm_weight_chunk = pl.slice(
                    lm_head_weight,
                    [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                    [lm_o0, lm_k0],
                )
                lm_acc = pl.matmul_acc(lm_acc, lm_hidden_chunk, lm_weight_chunk, b_trans=True)
            lm_acc_gm = pl.assemble(lm_acc_gm, lm_acc, [0, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_store"):
            lm_acc_chunk = pl.slice(lm_acc_gm, [BATCH_TILE, VOCAB_CHUNK], [0, 0])
            lm_acc_trimmed = pl.slice(
                lm_acc_chunk,
                [BATCH_TILE, VOCAB_CHUNK],
                [0, 0],
                valid_shape=[lm_valid_rows, VOCAB_CHUNK],
            )
            out = pl.assemble(out, lm_acc_trimmed, [b0, lm_o0])

    return out
