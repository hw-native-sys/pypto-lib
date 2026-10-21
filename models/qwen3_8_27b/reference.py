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


def make_inputs(t: int, h: int, d: int, hg: int | None = None,
                seed: int = 42) -> dict[str, torch.Tensor]:
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
        q=F.normalize(torch.randn(t, hg, d, dtype=torch.float16, generator=gen(1)),
                      dim=-1, p=2),
        k=F.normalize(torch.randn(t, hg, d, dtype=torch.float16, generator=gen(2)),
                      dim=-1, p=2),
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

# rtol scales with the chunk size: that is how many rounding steps a
# chunk-length reduction accumulates.
RTOL = 5e-3
ATOL = 1.5e-4
FTOL = 1e-3


def stats_ok(actual: torch.Tensor, expected: torch.Tensor,
             chunk: int = 1) -> tuple[bool, str]:
    """The acceptance test, and the numbers behind it."""
    act = actual.to(REF_DTYPE)
    exp = expected.to(REF_DTYPE)
    diff = (act - exp).abs()
    denom = torch.sqrt((exp ** 2).sum())
    frob = float(torch.sqrt((diff ** 2).sum()) / denom) if float(denom) > 0 else 0.0
    bound = ATOL + min(0.5, RTOL * chunk) * exp.abs()
    elementwise = not bool((diff > bound).all())
    detail = (f"frob={frob:.4g} (<={FTOL}), max diff={float(diff.max()):.4g}, "
              f"peak |ref|={float(exp.abs().max()):.4g}")
    if not elementwise:
        return False, "every element outside the relative bound; " + detail
    return frob <= FTOL, detail

