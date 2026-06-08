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
# N+K plus OB_CHUNK amortises per-task dispatch overhead and lifts the
# innermost K dim to one L2 cache line (512 B, perf_hint PH001).
LM_HEAD_K_CHUNK = 512
VOCAB_CHUNK = 256
LM_HEAD_OB_CHUNK = 2


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

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        lm_valid_rows = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk], name_hint="lm_head"):
            for ob in pl.parallel(VOCAB // VOCAB_CHUNK, chunk=LM_HEAD_OB_CHUNK):
                lm_o0 = ob * VOCAB_CHUNK
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
                # pl.slice(..., valid_shape=...) on the acc-resident lm_acc
                # forces a tmov to a new acc tile with a different slayout,
                # which ptoas rejects ("expects a supported tmov address-space
                # pair"). set_validshape is metadata-only — no data movement.
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


@pl.jit
def test_rms_lm_head(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    user_batch = pl.tensor.dim(hidden_states, 0)
    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                copy_k0 = kb * FINAL_RMS_K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, FINAL_RMS_K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, FINAL_RMS_K_CHUNK],
                )
                current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, copy_k0])

    out = rms_lm_head(current_hidden, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    vocab_size: int = VOCAB,
):
    import torch
    from golden import TensorSpec

    # Host allocates every batch-dependent tensor at the user-visible batch
    # (no host pad / no host trim). The kernel internally rounds up to
    # BATCH_TILE, zero-pads via valid_shape on input loads, and trims via
    # valid_shape on the FP32 output store. A single compiled program serves
    # any batch <= host capacity (USER_BATCH_DYN is a pl.dynamic dim).
    synthetic_proj_scale = 0.5

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_final_norm_weight():
        return torch.ones(1, hidden_size)

    def init_lm_head_weight():
        return synthetic_proj_scale * (torch.rand(vocab_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_seq_lens():
        # seq_lens values are not consumed by rms_lm_head; only the leading
        # dim (= user_batch) matters. Use ones for determinism.
        return torch.ones(batch, dtype=torch.int32)

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("final_norm_weight", [1, hidden_size], torch.float32,
                   init_value=init_final_norm_weight),
        TensorSpec("lm_head_weight", [vocab_size, hidden_size], torch.bfloat16,
                   init_value=init_lm_head_weight),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("out", [batch, vocab_size], torch.float32, is_output=True),
    ]


def golden_rms_lm_head(tensors):
    """PyTorch reference: final RMSNorm + LM head projection."""
    import torch

    hidden_states = tensors["hidden_states"]
    final_norm_weight = tensors["final_norm_weight"]
    lm_head_weight = tensors["lm_head_weight"]

    hidden_size = hidden_states.shape[1]
    eps = EPS

    def chunked_row_sq_sum(x, k_chunk):
        acc = torch.zeros(x.shape[0], 1, dtype=torch.float32)
        for k0 in range(0, x.shape[1], k_chunk):
            x_chunk = x[:, k0 : k0 + k_chunk]
            acc = acc + (x_chunk * x_chunk).sum(dim=-1, keepdim=True)
        return acc

    def tiled_lm_head(lhs, rhs_t, k_chunk, vocab_chunk):
        out = torch.zeros(lhs.shape[0], rhs_t.shape[0], dtype=torch.float32)
        for k0 in range(0, lhs.shape[1], k_chunk):
            out = out + lhs[:, k0 : k0 + k_chunk].float() @ rhs_t[:, k0 : k0 + k_chunk].float().T
        return out

    variance = chunked_row_sq_sum(hidden_states.float(), FINAL_RMS_K_CHUNK) / hidden_size
    inv_rms = torch.rsqrt(variance + eps)
    final_normed = (hidden_states.float() * inv_rms * final_norm_weight.float()).bfloat16()

    tensors["out"][:] = tiled_lm_head(
        final_normed,
        lm_head_weight,
        LM_HEAD_K_CHUNK,
        VOCAB_CHUNK,
    )


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=BATCH,
                        help=("User-visible batch size. Host allocates every "
                              "batch-dependent tensor at exactly this size; "
                              "the kernel internally rounds up to BATCH_TILE "
                              "(%d), zero-pads input loads via valid_shape, "
                              "and trims the FP32 output via valid_shape on "
                              "the store. A single compiled program serves "
                              "any batch <= host capacity. Default: "
                              "%%(default)s" % BATCH_TILE))
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=test_rms_lm_head,
        specs=build_tensor_specs(batch=args.batch),
        golden_fn=golden_rms_lm_head,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=3e-3,
        atol=3e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
