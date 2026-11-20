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
identity. The fixtures use a single sequence (B = 1).

Block fixture generation, quantization, and projection math are shared here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from config import GDN_QUANT, Qwen38Config

REF_DTYPE = torch.float64
MAX_THREADS = 32


def host_torch_setup() -> None:
    """Limit CPU reference threads and disable oneDNN."""
    torch.set_num_threads(min(torch.get_num_threads(), MAX_THREADS))
    torch.backends.mkldnn.enabled = False


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
# The block around the delta rule
# ---------------------------------------------------------------------------
#
# `Qwen3_5GatedDeltaNet.forward` (transformers 5.17.0, `models/qwen3_5/
# modeling_qwen3_5.py`), read line by line, for one sequence of T tokens:
#
#   qkv     = x @ Wqkv^T                          in_proj_qkv, hidden -> 2*Hg*D + H*D
#   z       = x @ Wz^T                            in_proj_z,   hidden -> H*D
#   a, b    = x @ Wa^T, x @ Wb^T                  in_proj_a/b, hidden -> H each
#   qkv     = silu(causal_conv(qkv))              depthwise, K taps, no bias
#   q, k, v = split(qkv)                          [T, Hg, D], [T, Hg, D], [T, H, D]
#   q       = l2norm(q) * D^-0.5;  k = l2norm(k)  eps = 1e-6, inside the sqrt
#   beta    = sigmoid(b)
#   g       = -exp(A_log) * softplus(a + dt_bias)
#   o       = delta_rule(q, k, v, beta, g)        `gdn_layer.delta_rule_ref`
#   y       = o * rsqrt(mean(o^2) + eps) * w_norm * silu(z)     per head row
#   out     = y @ Wout^T                          out_proj, H*D -> hidden
#
# Weights carry the checkpoint's names under `linear_attn.`, so one dict serves
# this chain and the module's `load_state_dict` alike. `nn.Linear` stores
# `[out, in]`; `conv1d.weight` is `[C, 1, K]`.

WEIGHT_NAMES = ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
                "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias",
                "norm.weight", "out_proj.weight")

# The model's dtype: what the checkpoint stores and what its activations carry.
MODEL_DTYPE = torch.bfloat16

def make_block_inputs(t: int, cfg: Qwen38Config, seed: int = 42) -> torch.Tensor:
    """Hidden states entering the block, `[T, hidden]`, unit normal in bf16.

    What the layer's input RMSNorm hands over is unit-scale per channel, and this
    is the stand-in for it; nothing here depends on a real prompt.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(t, cfg.hidden_size, generator=gen).to(MODEL_DTYPE)


def make_block_weights(cfg: Qwen38Config, seed: int = 42) -> dict[str, torch.Tensor]:
    """Random weights from the module's own init, in the checkpoint's bf16.

    `nn.Linear` and `nn.Conv1d` default to kaiming-uniform, which for their
    default gain is `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`; the conv's fan-in is its
    K taps. `A_log = log(U(0.01, 16))`, `dt_bias = 1` and `norm.weight = 1` are
    `Qwen3_5GatedDeltaNet.__init__` verbatim. Fine for the fast loop; the decay
    range a trained layer reaches is what the real weights are for.

    Each tensor draws from its own generator, as :func:`make_inputs` does.
    """
    h, hg, d = (cfg.linear_num_value_heads, cfg.linear_num_key_heads,
                cfg.linear_value_head_dim)
    hidden, k = cfg.hidden_size, cfg.linear_conv_kernel_dim

    def uniform(offset, *shape, bound):
        gen = torch.Generator().manual_seed(seed + offset)
        return ((torch.rand(*shape, generator=gen) * 2 - 1) * bound).to(MODEL_DTYPE)

    lin = hidden ** -0.5
    a_gen = torch.Generator().manual_seed(seed + 6)
    return {
        "in_proj_qkv.weight": uniform(1, cfg.qkv_width, hidden, bound=lin),
        "in_proj_z.weight": uniform(2, h * d, hidden, bound=lin),
        "in_proj_a.weight": uniform(3, h, hidden, bound=lin),
        "in_proj_b.weight": uniform(4, h, hidden, bound=lin),
        "conv1d.weight": uniform(5, cfg.qkv_width, 1, k, bound=k ** -0.5),
        "A_log": torch.log(0.01 + (16 - 0.01) * torch.rand(h, generator=a_gen)).to(MODEL_DTYPE),
        "dt_bias": torch.ones(h, dtype=MODEL_DTYPE),
        "norm.weight": torch.ones(d, dtype=MODEL_DTYPE),
        "out_proj.weight": uniform(7, hidden, h * d, bound=(h * d) ** -0.5),
    }


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 with one fp32 scale per row, in the kernels' arithmetic.

    Per output channel of an `[out, in]` weight, per token of a `[T, C]`
    activation -- the same chain either way: in fp32, `amax` over the row clamped
    at `amax_eps`, the row times `scale_max / amax`, round to nearest even, clamp
    to +-scale_max, int8; the dequant scale is `amax / scale_max`. Mirrors
    `quant_int8_per_out_channel` (deepseek_v4_pro) and the `*_act_quant` regions
    of `qwen3_14b/prefill_fwd_a8w8.py`, multiply by the reciprocal included.
    """
    xf = x.to(torch.float32)
    amax = xf.abs().amax(dim=-1).clamp_min(GDN_QUANT.amax_eps)
    q = torch.round(xf * (GDN_QUANT.scale_max / amax)[:, None])
    q = q.clamp(-GDN_QUANT.scale_max, GDN_QUANT.scale_max).to(torch.int8)
    return q, amax / GDN_QUANT.scale_max


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8 rows times their fp32 scale, in float64: exact, the integers being small."""
    return q.to(REF_DTYPE) * scale.to(REF_DTYPE)[:, None]


def weight(w: dict[str, torch.Tensor], module: str) -> torch.Tensor:
    """A projection's `[out, in]` weight in float64; int8 ones dequantised by their scale."""
    value = w[module + ".weight"]
    if value.dtype == torch.int8:
        return dequantize(value, w[module + ".weight_scale"])
    return value.to(REF_DTYPE)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`nn.Linear` without bias: `x @ W^T`, W stored `[out, in]`."""
    return x.to(REF_DTYPE) @ weight.to(REF_DTYPE).t()


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
        max_diff = float((act - exp).abs().max())
        peak = float(exp.abs().max())
        head = f"    {stats}, max diff={max_diff:.4g}, peak |ref|={peak:.4g}"
        return False, f"{head}\n{detail}"

    cmp.__name__ = "frob_ok"
    return cmp
