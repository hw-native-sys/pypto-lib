# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host tensor operations for the Gated DeltaNet chunked delta rule."""

import torch

REF_DTYPE = torch.float64


def cumsum_ref(g: torch.Tensor, chunk: int) -> torch.Tensor:
    """S1: chunk-local prefix sum of the gate logits. [T, H] -> [T, H]."""
    out = torch.zeros_like(g, dtype=REF_DTYPE)
    gf = g.to(REF_DTYPE)
    for t0 in range(0, g.shape[0], chunk):
        out[t0 : t0 + chunk] = gf[t0 : t0 + chunk].cumsum(dim=0)
    return out


def kkt_ref(k: torch.Tensor, beta: torch.Tensor, g_sum: torch.Tensor, chunk: int) -> torch.Tensor:
    """S3: the gated key-key matrix, strictly lower triangular. -> [T, H, chunk].

    The decay is `exp(g_i - g_j)` clamped to zero above the diagonal rather than
    `exp(min(g_i - g_j, 0))`: on the strict lower triangle the two agree, and the
    masked-out entries are discarded either way.
    """
    t, hg, _ = k.shape
    h = beta.shape[1]
    grp = h // hg
    kf, bf, gf = k.to(REF_DTYPE), beta.to(REF_DTYPE), g_sum.to(REF_DTYPE)
    out = torch.zeros(t, h, chunk, dtype=REF_DTYPE)
    rows = torch.arange(chunk)[:, None]
    cols = torch.arange(chunk)[None, :]
    strict_lower = (rows > cols).to(REF_DTYPE)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            kc = kf[t0 : t0 + chunk, hh // grp, :]
            gc = gf[t0 : t0 + chunk, hh]
            diff = gc[:, None] - gc[None, :]
            decay = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
            gated = (kc @ kc.T) * decay * bf[t0 : t0 + chunk, hh, None]
            out[t0 : t0 + chunk, hh, :] = gated * strict_lower
    return out


def solve_tril_ref(a: torch.Tensor, chunk: int) -> torch.Tensor:
    """S4: `(I + A)^-1` per (chunk, head), exact in float64. -> [T, H, chunk]."""
    t, h, _ = a.shape
    af = a.to(REF_DTYPE)
    out = torch.zeros_like(af)
    eye = torch.eye(chunk, dtype=REF_DTYPE)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            out[t0 : t0 + chunk, hh, :] = torch.linalg.inv( eye + af[t0 : t0 + chunk, hh, :])
    return out


def wy_fast_ref(
    k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
    a_inv: torch.Tensor, g_sum: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """S5: the WY representation. -> (W, U), both [T, H, D]."""
    t, hg, d = k.shape
    h = v.shape[1]
    grp = h // hg
    kf, vf, bf, af, gf = (x.to(REF_DTYPE) for x in (k, v, beta, a_inv, g_sum))
    w = torch.zeros(t, h, d, dtype=REF_DTYPE)
    u = torch.zeros(t, h, d, dtype=REF_DTYPE)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            ab = af[t0 : t0 + chunk, hh, :]
            bc = bf[t0 : t0 + chunk, hh, None]
            gc = gf[t0 : t0 + chunk, hh, None]
            kg = kf[t0 : t0 + chunk, hh // grp, :] * bc * torch.exp(gc)
            u[t0 : t0 + chunk, hh, :] = ab @ (vf[t0 : t0 + chunk, hh, :] * bc)
            w[t0 : t0 + chunk, hh, :] = ab @ kg
    return w, u


def chunk_h_ref(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_sum: torch.Tensor,
            chunk: int, initial_state: torch.Tensor | None = None,
            row_base: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S6: the inter-chunk state recurrence.

    Returns the state snapshot ENTERING each chunk, [NCHUNK, H, D, D]; the
    residual-corrected values V_new, [T, H, D]; and the state LEAVING the last
    chunk, [H, D, D], which is what an inference cache carries forward.
    """
    t, hg, d = k.shape
    h = w.shape[1]
    grp = h // hg
    nc = t // chunk
    kf, wf, uf, gf = (x.to(REF_DTYPE) for x in (k, w, u, g_sum))
    state = torch.zeros(nc, h, d, d, dtype=REF_DTYPE)
    v_new = torch.zeros(t, h, d, dtype=REF_DTYPE)
    final_state = torch.zeros(h, d, d, dtype=REF_DTYPE)
    for hh in range(h):
        s = (torch.zeros(d, d, dtype=REF_DTYPE) if initial_state is None
             else initial_state[hh].to(REF_DTYPE).clone())
        for ci in range(nc):
            t0 = ci * chunk
            if ci and row_base is not None and int(row_base[t0]) == t0:
                s.zero_()
            gc = gf[t0 : t0 + chunk, hh]
            g_last = gc[-1]
            state[ci, hh] = s
            vc = uf[t0 : t0 + chunk, hh, :] - wf[t0 : t0 + chunk, hh, :] @ s
            v_new[t0 : t0 + chunk, hh, :] = vc
            kv = (kf[t0 : t0 + chunk, hh // grp, :].T
                  @ (vc * torch.exp(g_last - gc)[:, None]))
            s = torch.exp(g_last) * s + kv
        final_state[hh] = s
    return state, v_new, final_state


def chunk_o_ref(
    q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor,
    state: torch.Tensor, g_sum: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """S7: the chunk output, inter-chunk plus intra-chunk. -> [T, H, D].

    The causal mask includes the diagonal here, unlike :func:`scaled_dot_kkt.kkt_ref`'s.
    """
    t, hg, d = q.shape
    h = v_new.shape[1]
    grp = h // hg
    qf, kf, vf, sf, gf = (x.to(REF_DTYPE) for x in (q, k, v_new, state, g_sum))
    out = torch.zeros(t, h, d, dtype=REF_DTYPE)
    rows = torch.arange(chunk)[:, None]
    cols = torch.arange(chunk)[None, :]
    causal = (rows >= cols).to(REF_DTYPE)
    zero = torch.zeros(chunk, chunk, dtype=REF_DTYPE)
    for ci in range(t // chunk):
        t0 = ci * chunk
        for hh in range(h):
            qc = qf[t0 : t0 + chunk, hh // grp, :]
            kc = kf[t0 : t0 + chunk, hh // grp, :]
            gc = gf[t0 : t0 + chunk, hh]
            inter = (qc @ sf[ci, hh]) * torch.exp(gc)[:, None]
            gate = torch.exp(torch.minimum(gc[:, None] - gc[None, :], zero))
            intra = ((qc @ kc.T) * gate * causal) @ vf[t0 : t0 + chunk, hh, :]
            out[t0 : t0 + chunk, hh, :] = inter + intra
    return out
