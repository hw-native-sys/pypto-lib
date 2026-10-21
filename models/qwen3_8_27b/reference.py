# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared host helpers for the Gated DeltaNet operators: the drawn inputs, the two layout
conversions and the acceptance criterion.

Each stage's float64 reference lives in its own operator file. The references follow the
model's natural layout ([T, H, D] values, [T, H] per-token scalars, [NCHUNK, H, D, D]
states); the kernels take some of those transposed or flattened, and :func:`to_hT` and
:func:`flat_state` convert.

Grouped-query attention is implemented: `hg` QK heads against `h` value heads, value head
`i` reading key head `i // (h // hg)`. `hg` defaults to `h`, which makes the mapping an
identity. Single sequence (B = 1); packed variable-length batches are a pipeline feature
the kernels do not implement yet.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

REF_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Inputs and layout helpers
# ---------------------------------------------------------------------------


def make_inputs(t: int, h: int, d: int, hg: int | None = None, seed: int = 42) -> dict[str, torch.Tensor]:
    """The model's own input distribution, in the model's dtypes.

    `hg` is the number of QK heads; `h` the number of value heads. They differ
    under GQA (Qwen3.8-27B is 48 against 16), and each value head reads key head
    `h // (H // Hg)`. Defaults to `h`.

    q and k are L2-normalised along the head dimension, v is unnormalised noise,
    beta is uniform on [0, 1) and the gate logits are log-sigmoid, so `g` is
    negative and its chunk-local prefix sum decays.

    Each tensor draws from its own generator, so changing one shape does not
    reshuffle the others -- `g` at a given (t, h) is the same whatever D is.
    """
    hg = h if hg is None else hg

    def gen(offset):
        return torch.Generator().manual_seed(seed + offset)

    return dict(
        q=F.normalize(torch.randn(t, hg, d, dtype=torch.float16, generator=gen(1)), dim=-1, p=2),
        k=F.normalize(torch.randn(t, hg, d, dtype=torch.float16, generator=gen(2)), dim=-1, p=2),
        v=torch.randn(t, h, d, dtype=torch.float16, generator=gen(3)),
        beta=torch.rand(t, h, dtype=torch.float16, generator=gen(4)),
        g=F.logsigmoid(torch.randn(t, h, dtype=torch.float32, generator=gen(5))),
    )


def to_hT(x: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """[T, H] per-token scalars to the [H, T] head-major layout the kernels read."""
    return x.t().contiguous().to(dtype)


def flat_state(state: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    """[NCHUNK, H, D, D] snapshots to the [NCHUNK * H * D, D] layout chunk_h writes."""
    nc, h, d, _ = state.shape
    return state.reshape(nc * h * d, d).contiguous().to(dtype)


# ---------------------------------------------------------------------------
# Acceptance criterion
# ---------------------------------------------------------------------------

# Measured on a2a3 at T = 512 and T = 8192, every output of the six stages and the layer:
# not one point exceeds DIFF_THD, and the worst relative Frobenius norm is 3.6e-4.
DIFF_THD = 1e-3         # per-point floored relative difference
PCT_THD = 1e-6          # share of points allowed to exceed it
MAX_DIFF_HD = 1.0       # one point this far out fails whatever the share
FTOL = 1e-3             # relative Frobenius norm over the whole output


def frob(actual: torch.Tensor, expected: torch.Tensor) -> float | None:
    """Relative Frobenius norm, or None when the reference is all zeros."""
    act = actual.cpu().to(REF_DTYPE)
    exp = expected.cpu().to(REF_DTYPE)
    denom = float(torch.sqrt((exp ** 2).sum()))
    if denom == 0.0:
        return None
    return float(torch.sqrt(((act - exp) ** 2).sum())) / denom


def frob_ok(ftol: float = FTOL, diff_thd: float = DIFF_THD, pct_thd: float = PCT_THD):
    """`ratio_reldiff` per point, plus the relative Frobenius norm over the whole output.

    Every per-point check is the harness comparator's: the floored relative difference,
    its bad-point budget, the single-point cap and the non-finite rejection. The Frobenius
    norm is the one criterion it does not carry, and a chunk-length reduction's
    accumulated rounding shows up there and nowhere else.

    An all-zero reference has no relative norm, so there the per-point check stands alone.
    That case is reachable: `chunk_h`'s state snapshot entering a single chunk is all
    zeros, and a criterion that scored it would accept any output at all.
    """
    from golden import ratio_reldiff

    per_point = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd, max_diff_hd=MAX_DIFF_HD)

    def cmp(actual, expected, **kwargs):
        ok, detail = per_point(actual, expected, **kwargs)
        value = frob(actual, expected)
        if value is not None and value > ftol:
            ok = False
        if ok:
            return True, ""
        act = actual.cpu().to(REF_DTYPE)
        exp = expected.cpu().to(REF_DTYPE)
        if value is None:
            stats = "frob n/a (the reference is all zeros)"
        else:
            # The relation, not a fixed ">": a per-point failure can carry a healthy norm.
            stats = f"frob={value:.4g} ({'>' if value > ftol else '<='}{ftol})"
        head = (f"    {stats}, max diff={float((act - exp).abs().max()):.4g}, "
                f"peak |ref|={float(exp.abs().max()):.4g}")
        return False, f"{head}\n{detail}"

    cmp.__name__ = "frob_ok"
    return cmp

