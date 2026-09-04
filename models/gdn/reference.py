# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Float64 CPU reference for the Gated DeltaNet forward pass.

One function per pipeline stage, plus :func:`pipeline` which chains all six and
returns every intermediate. Each stage kernel validates against the matching
function here, and every stage's test input is the reference output of the stage
before it -- so no stage is ever fed data the pipeline could not produce.

The stage signatures follow the model's natural layout ([T, H, D] values,
[T, H] per-token scalars, [NCHUNK, H, D, D] states). The kernels take some of
those transposed or flattened; :func:`to_hT` and :func:`flat_state` convert.

Single sequence (B = 1), no GQA. Packed variable-length batches and separate
key-head counts are pipeline features the kernels do not implement yet.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

REF_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Inputs and layout helpers
# ---------------------------------------------------------------------------


def make_inputs(t: int, h: int, d: int, seed: int = 42) -> dict[str, torch.Tensor]:
    """The model's own input distribution, in the model's dtypes.

    Same draw as the reference harness (`megagdn-pto/tests/utils.py:
    generate_random_inputs`): q and k L2-normalised along the head dimension, v
    unnormalised noise, beta uniform on [0, 1) and the gate logits log-sigmoid,
    so `g` is negative and its chunk-local prefix sum decays.

    Each tensor draws from its own generator, so changing one shape does not
    reshuffle the others -- `g` at a given (t, h) is the same whatever D is.
    """
    def gen(offset):
        return torch.Generator().manual_seed(seed + offset)

    return dict(
        q=F.normalize(torch.randn(t, h, d, dtype=torch.float16, generator=gen(1)),
                      dim=-1, p=2),
        k=F.normalize(torch.randn(t, h, d, dtype=torch.float16, generator=gen(2)),
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
# Stages
# ---------------------------------------------------------------------------


def cumsum(g: torch.Tensor, chunk: int) -> torch.Tensor:
    """S1: chunk-local prefix sum of the gate logits. [T, H] -> [T, H]."""
    out = torch.zeros_like(g, dtype=REF_DTYPE)
    gf = g.to(REF_DTYPE)
    for t0 in range(0, g.shape[0], chunk):
        out[t0 : t0 + chunk] = gf[t0 : t0 + chunk].cumsum(dim=0)
    return out


def kkt(k: torch.Tensor, beta: torch.Tensor, g_sum: torch.Tensor,
        chunk: int) -> torch.Tensor:
    """S3: the gated key-key matrix, strictly lower triangular. -> [T, H, chunk].

    The decay is `exp(g_i - g_j)` clamped to zero above the diagonal rather than
    `exp(min(g_i - g_j, 0))`: on the strict lower triangle the two agree, and the
    masked-out entries are discarded either way.
    """
    t, h, _ = k.shape
    kf, bf, gf = k.to(REF_DTYPE), beta.to(REF_DTYPE), g_sum.to(REF_DTYPE)
    out = torch.zeros(t, h, chunk, dtype=REF_DTYPE)
    rows = torch.arange(chunk)[:, None]
    cols = torch.arange(chunk)[None, :]
    strict_lower = (rows > cols).to(REF_DTYPE)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            kc = kf[t0 : t0 + chunk, hh, :]
            gc = gf[t0 : t0 + chunk, hh]
            diff = gc[:, None] - gc[None, :]
            decay = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
            out[t0 : t0 + chunk, hh, :] = (
                (kc @ kc.T) * decay * bf[t0 : t0 + chunk, hh, None] * strict_lower)
    return out


def solve_tril(a: torch.Tensor, chunk: int) -> torch.Tensor:
    """S4: `(I + A)^-1` per (chunk, head), exact in float64. -> [T, H, chunk]."""
    t, h, _ = a.shape
    af = a.to(REF_DTYPE)
    out = torch.zeros_like(af)
    eye = torch.eye(chunk, dtype=REF_DTYPE)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            out[t0 : t0 + chunk, hh, :] = torch.linalg.inv(
                eye + af[t0 : t0 + chunk, hh, :])
    return out


def wy_fast(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
            a_inv: torch.Tensor, g_sum: torch.Tensor,
            chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """S5: the WY representation. -> (W, U), both [T, H, D]."""
    t, h, d = k.shape
    kf, vf, bf, af, gf = (x.to(REF_DTYPE) for x in (k, v, beta, a_inv, g_sum))
    w = torch.zeros(t, h, d, dtype=REF_DTYPE)
    u = torch.zeros(t, h, d, dtype=REF_DTYPE)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            ab = af[t0 : t0 + chunk, hh, :]
            bc = bf[t0 : t0 + chunk, hh, None]
            gc = gf[t0 : t0 + chunk, hh, None]
            u[t0 : t0 + chunk, hh, :] = ab @ (vf[t0 : t0 + chunk, hh, :] * bc)
            w[t0 : t0 + chunk, hh, :] = ab @ (
                kf[t0 : t0 + chunk, hh, :] * bc * torch.exp(gc))
    return w, u


def chunk_h(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor,
            g_sum: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """S6: the inter-chunk state recurrence.

    Returns the state snapshot ENTERING each chunk, [NCHUNK, H, D, D], and the
    residual-corrected values V_new, [T, H, D].
    """
    t, h, d = k.shape
    nc = t // chunk
    kf, wf, uf, gf = (x.to(REF_DTYPE) for x in (k, w, u, g_sum))
    state = torch.zeros(nc, h, d, d, dtype=REF_DTYPE)
    v_new = torch.zeros(t, h, d, dtype=REF_DTYPE)
    for hh in range(h):
        s = torch.zeros(d, d, dtype=REF_DTYPE)
        for ci in range(nc):
            t0 = ci * chunk
            gc = gf[t0 : t0 + chunk, hh]
            g_last = gc[-1]
            state[ci, hh] = s
            vc = uf[t0 : t0 + chunk, hh, :] - wf[t0 : t0 + chunk, hh, :] @ s
            v_new[t0 : t0 + chunk, hh, :] = vc
            kv = kf[t0 : t0 + chunk, hh, :].T @ (vc * torch.exp(g_last - gc)[:, None])
            s = torch.exp(g_last) * s + kv
    return state, v_new


def chunk_o(q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor,
            state: torch.Tensor, g_sum: torch.Tensor, chunk: int) -> torch.Tensor:
    """S7: the chunk output, inter-chunk plus intra-chunk. -> [T, H, D].

    The causal mask includes the diagonal here, unlike :func:`kkt`'s.
    """
    t, h, d = q.shape
    qf, kf, vf, sf, gf = (x.to(REF_DTYPE) for x in (q, k, v_new, state, g_sum))
    out = torch.zeros(t, h, d, dtype=REF_DTYPE)
    rows = torch.arange(chunk)[:, None]
    cols = torch.arange(chunk)[None, :]
    causal = (rows >= cols).to(REF_DTYPE)
    zero = torch.zeros(chunk, chunk, dtype=REF_DTYPE)
    for ci in range(t // chunk):
        t0 = ci * chunk
        for hh in range(h):
            qc = qf[t0 : t0 + chunk, hh, :]
            kc = kf[t0 : t0 + chunk, hh, :]
            gc = gf[t0 : t0 + chunk, hh]
            inter = (qc @ sf[ci, hh]) * torch.exp(gc)[:, None]
            gate = torch.exp(torch.minimum(gc[:, None] - gc[None, :], zero))
            intra = ((qc @ kc.T) * gate * causal) @ vf[t0 : t0 + chunk, hh, :]
            out[t0 : t0 + chunk, hh, :] = inter + intra
    return out


# Pipeline order. Each stage consumes the outputs of the ones before it.
STAGES = ("chunk_cumsum", "scaled_dot_kkt", "solve_tril", "wy_fast",
          "chunk_h", "chunk_o")

_CACHE: dict[tuple, dict] = {}


def compute(upto: str, t: int, h: int, d: int, chunk: int,
            seed: int = 42) -> dict[str, torch.Tensor]:
    """Reference inputs plus every stage output through *upto*, cached and extended.

    Values crossing a stage boundary are narrowed to the dtype the kernels
    exchange -- FP16 for A, A_inv, W, U, V_new and the state snapshots -- so a
    stage's reference input is bit-identical to what the preceding kernel would
    have handed it, and a comparison measures that stage alone.

    *upto* is a stage name or ``"inputs"``. Calling it twice on the same shape
    only computes the stages that are missing.
    """
    want = 0 if upto == "inputs" else STAGES.index(upto) + 1
    key = (t, h, d, chunk, seed)
    st = _CACHE.get(key)
    if st is None:
        st = _CACHE[key] = dict(make_inputs(t, h, d, seed), _done=0)
    while st["_done"] < want:
        stage = STAGES[st["_done"]]
        if stage == "chunk_cumsum":
            st["g_sum"] = cumsum(st["g"], chunk)
        elif stage == "scaled_dot_kkt":
            st["a"] = kkt(st["k"], st["beta"], st["g_sum"], chunk)
            st["a16"] = st["a"].to(torch.float16)
        elif stage == "solve_tril":
            st["a_inv"] = solve_tril(st["a16"], chunk)
            st["a_inv16"] = st["a_inv"].to(torch.float16)
        elif stage == "wy_fast":
            st["w"], st["u"] = wy_fast(st["k"], st["v"], st["beta"],
                                       st["a_inv16"], st["g_sum"], chunk)
            st["w16"] = st["w"].to(torch.float16)
            st["u16"] = st["u"].to(torch.float16)
        elif stage == "chunk_h":
            st["state"], st["v_new"] = chunk_h(st["k"], st["w16"], st["u16"],
                                               st["g_sum"], chunk)
            st["state16"] = st["state"].to(torch.float16)
            st["v_new16"] = st["v_new"].to(torch.float16)
        elif stage == "chunk_o":
            st["o"] = chunk_o(st["q"], st["k"], st["v_new16"], st["state16"],
                              st["g_sum"], chunk)
        st["_done"] += 1
    return st


def stage_inputs(stage: str, t: int, h: int, d: int, chunk: int,
                 seed: int = 42) -> dict[str, torch.Tensor]:
    """Everything *stage* consumes: the reference chain up to its predecessor."""
    i = STAGES.index(stage)
    return compute("inputs" if i == 0 else STAGES[i - 1], t, h, d, chunk, seed)


def lazy(stage: str, key: str, t: int, h: int, d: int, chunk: int,
         transform=None, seed: int = 42):
    """A no-argument callable returning one reference tensor, computed on first use.

    `TensorSpec(init_value=...)` takes a callable, and deferring the chain this
    way keeps spec construction free -- the benchmark builds specs only to read
    their shapes and dtypes, and never pays for the host reference.
    """
    def load():
        value = stage_inputs(stage, t, h, d, chunk, seed)[key]
        return transform(value) if transform is not None else value

    return load


# ---------------------------------------------------------------------------
# Acceptance criterion
# ---------------------------------------------------------------------------

# megagdn-pto/tests/utils.py: NumericalAccuracy. rtol is scaled by the chunk
# size because a chunk-length reduction accumulates that many rounding steps.
RTOL = 5e-3
ATOL = 1.5e-4
FTOL = 1e-3


def stats_ok(actual: torch.Tensor, expected: torch.Tensor,
             chunk: int = 1) -> tuple[bool, str]:
    """The reference harness's acceptance test, and the numbers behind it."""
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
