# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared RMSNorm primitives: ``@pl.inline`` with CT kwargs.

Two variants:

- ``rmsnorm`` — full normalize-and-write: accumulates sq-sum in FP32, then
  normalizes via ``inv_rms`` and writes ``(x / rms(x)) * gamma`` to ``out``
  via ``pl.assemble``. Returns the assembled ``out``.

  Use case: the triplicated RMSNorm body in ``rms_lm_head.py`` (sites 1-3),
  ``post_rmsnorm`` in ``prefill_fwd.py`` (site 5), and any loop that follows
  the compute-normalize → write-to-output pattern.

- ``rmsnorm_recip`` — reciprocal only: computes ``inv_rms = 1 / sqrt(...)``
  without writing the normalized output. Returns ``inv_rms``.

  Use case: the ``rms_recip`` snippet in ``decode_layer.py`` (site 6) where
  the caller applies gamma multiplication and output routing separately.

Both are ``@pl.inline`` with keyword-only CT kwargs (``rows``, ``k_chunk``,
``eps``, ``hidden``) so callers can set per-call-site values. The caller MUST
capture the return value — the body is scope-isolated (variables do not leak
to the caller).

See ``~/workspace/subprojects/pypto/docs/guide-ct-args.md`` for the full
context on ``@pl.inline`` + CT kwargs (in the compiler subproject, not here).
"""

import pypto.language as pl


@pl.inline
def rmsnorm(
    x, gamma, out, b0,
    *, rows: int, k_chunk: int, eps: float, hidden: int,
    stages: int = 2,
    cast_input: bool = True,
):
    """Full RMSNorm: accumulate sq-sum in FP32, then normalize+write to ``out``.

    Args:
        x: Input tensor ``[rows, hidden]`` (BF16 expected).
        gamma: Weight tensor ``[1, hidden]`` (FP32).
        out: Output tensor that ``pl.assemble`` writes into.
        b0: Row offset into ``x``, ``gamma``, and ``out``.
        rows: Number of rows in this tile.
        k_chunk: K-dimension chunk size for the reduction loops.
        eps: Epsilon for numerical stability.
        hidden: Hidden dimension size.
        stages: Software pipeline depth for the reduction/write loops.

    Returns:
        The assembled ``out`` tensor (caller MUST capture — scope isolation).
    """
    sq_sum = pl.full([1, rows], dtype=pl.FP32, value=0.0)
    for kb in pl.pipeline(hidden // k_chunk, stage=stages):
        k0 = kb * k_chunk
        if cast_input:
            chunk = pl.cast(
                pl.slice(x, [rows, k_chunk], [b0, k0]),
                target_type=pl.FP32,
            )
        else:
            chunk = pl.slice(x, [rows, k_chunk], [b0, k0])
        sq_sum = pl.add(
            sq_sum,
            pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, rows]),
        )
    inv_rms = pl.reshape(
        pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / hidden), eps)),
        [rows, 1],
    )
    for kb in pl.pipeline(hidden // k_chunk, stage=stages):
        k0 = kb * k_chunk
        if cast_input:
            h = pl.cast(pl.slice(x, [rows, k_chunk], [b0, k0]), target_type=pl.FP32)
        else:
            h = pl.slice(x, [rows, k_chunk], [b0, k0])
        g = pl.slice(gamma, [1, k_chunk], [0, k0])
        out = pl.assemble(
            out,
            pl.cast(pl.col_expand_mul(pl.row_expand_mul(h, inv_rms), g), target_type=pl.BF16),
            [b0, k0],
        )
    return out


@pl.inline
def rmsnorm_recip(
    x, *, rows: int, k_chunk: int, eps: float, hidden: int,
    stages: int = 2,
    cast_input: bool = True,
):
    """Compute only the RMSNorm reciprocal ``inv_rms`` (no normalize-and-write).

    Args:
        x: Input tensor ``[rows, hidden]`` (BF16 expected).
        rows: Number of rows in this tile.
        k_chunk: K-dimension chunk size for the sq-sum reduction.
        eps: Epsilon for numerical stability.
        hidden: Hidden dimension size.
        stages: Software pipeline depth for the sq-sum reduction loop.
        cast_input: Cast input chunks to FP32 for accumulation. Set to
            ``False`` when the input is already FP32.

    Returns:
        ``inv_rms``, a ``[rows, 1]`` FP32 reciprocal.
    """
    sq_sum = pl.full([1, rows], dtype=pl.FP32, value=0.0)
    for kb in pl.pipeline(hidden // k_chunk, stage=stages):
        k0 = kb * k_chunk
        if cast_input:
            chunk = pl.cast(
                pl.slice(x, [rows, k_chunk], [0, k0]),
                target_type=pl.FP32,
            )
        else:
            chunk = pl.slice(x, [rows, k_chunk], [0, k0])
        sq_sum = pl.add(
            sq_sum,
            pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, rows]),
        )
    inv_rms = pl.reshape(
        pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / hidden), eps)),
        [rows, 1],
    )
    return inv_rms
