# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Float64 CPU reference for the Gated DeltaNet block.

Two layers. The delta rule: one function per pipeline stage, plus :func:`compute`
which chains all six and returns every intermediate. Each stage kernel validates
against the matching function here, and every stage's test input is the
reference output of the stage before it -- so no stage is ever fed data the
pipeline could not produce. Around it the block: the projections, the short
convolution, the qk-norm and gate, the gated RMSNorm and `out_proj`, chained by
:func:`block` from hidden states to hidden states. That chain is checked against
the model's own `Qwen3_5GatedDeltaNet` by `test_block_reference.py`.

:func:`block` is two references in one. On bf16 weights with no activation
quantisation it is the float64 truth: how far a result is from the model. On
int8 weights (:func:`quantize_weights`) with `quant_x` / `quant_y` it is the
W8A8 chain the projection kernels implement, with their exact quantisation
arithmetic -- the correctness gate for those kernels, since quantisation alone
puts the block ~3e-2 from the truth, ten times the model's own bf16 rounding.

The stage signatures follow the model's natural layout ([T, H, D] values,
[T, H] per-token scalars, [NCHUNK, H, D, D] states). The kernels take some of
those transposed or flattened; :func:`to_hT` and :func:`flat_state` convert.

Grouped-query attention is implemented: `hg` QK heads against `h` value heads,
value head `i` reading key head `i // (h // hg)`. `hg` defaults to `h`, which
makes the mapping an identity.

Single sequence (B = 1). Packed variable-length batches are a pipeline feature
the kernels do not implement yet.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from config import GDN_QUANT, Qwen38Config

REF_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Inputs and layout helpers
# ---------------------------------------------------------------------------


def make_inputs(t: int, h: int, d: int, hg: int | None = None,
                seed: int = 42) -> dict[str, torch.Tensor]:
    """The model's own input distribution, in the model's dtypes.

    `hg` is the number of QK heads; `h` the number of value heads. They differ
    under GQA (Qwen3.8-27B is 48 against 16), and each value head reads key head
    `h // (H // Hg)` -- the model's own implementation reaches the same place by
    `repeat_interleave`, megagdn's reference by that index. Defaults to `h`.

    Same draw as the reference harness (`megagdn-pto/tests/utils.py:
    generate_random_inputs`): q and k L2-normalised along the head dimension, v
    unnormalised noise, beta uniform on [0, 1) and the gate logits log-sigmoid,
    so `g` is negative and its chunk-local prefix sum decays.

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
            u[t0 : t0 + chunk, hh, :] = ab @ (vf[t0 : t0 + chunk, hh, :] * bc)
            w[t0 : t0 + chunk, hh, :] = ab @ (
                kf[t0 : t0 + chunk, hh // grp, :] * bc * torch.exp(gc))
    return w, u


def chunk_h(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_sum: torch.Tensor,
            chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        s = torch.zeros(d, d, dtype=REF_DTYPE)
        for ci in range(nc):
            t0 = ci * chunk
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


def chunk_o(q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor,
            state: torch.Tensor, g_sum: torch.Tensor, chunk: int) -> torch.Tensor:
    """S7: the chunk output, inter-chunk plus intra-chunk. -> [T, H, D].

    The causal mask includes the diagonal here, unlike :func:`kkt`'s.
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


# Pipeline order. Each stage consumes the outputs of the ones before it.
STAGES = ("chunk_cumsum", "scaled_dot_kkt", "solve_tril", "wy_fast",
          "chunk_h", "chunk_o")

_CACHE: dict[tuple, dict] = {}


def _run_stage(st: dict, stage: str, chunk: int, narrow: bool) -> None:
    """Advance the chain in *st* by one stage.

    With *narrow*, values crossing a stage boundary are also kept narrowed to the
    dtype the kernels exchange -- FP16 for A, A_inv, W, U, V_new and the state
    snapshots, under the key with a `16` suffix -- and that copy is what the next
    stage reads, so a stage's reference input is bit-identical to what the
    preceding kernel would have handed it. Without it every stage reads float64.
    """
    def put(key, value):
        st[key] = value
        if narrow:
            st[key + "16"] = value.to(torch.float16)

    def src(key):
        return st[key + "16"] if narrow else st[key]

    if stage == "chunk_cumsum":
        st["g_sum"] = cumsum(st["g"], chunk)
    elif stage == "scaled_dot_kkt":
        put("a", kkt(st["k"], st["beta"], st["g_sum"], chunk))
    elif stage == "solve_tril":
        put("a_inv", solve_tril(src("a"), chunk))
    elif stage == "wy_fast":
        w, u = wy_fast(st["k"], st["v"], st["beta"], src("a_inv"), st["g_sum"], chunk)
        put("w", w)
        put("u", u)
    elif stage == "chunk_h":
        state, v_new, final_state = chunk_h(st["k"], src("w"), src("u"),
                                            st["g_sum"], chunk)
        put("state", state)
        put("v_new", v_new)
        st["final_state"] = final_state
    elif stage == "chunk_o":
        st["o"] = chunk_o(st["q"], st["k"], src("v_new"), src("state"),
                          st["g_sum"], chunk)


def compute(upto: str, t: int, h: int, d: int, chunk: int,
            hg: int | None = None, seed: int = 42) -> dict[str, torch.Tensor]:
    """Reference inputs plus every stage output through *upto*, cached and extended.

    Values crossing a stage boundary are narrowed to the dtype the kernels
    exchange -- FP16 for A, A_inv, W, U, V_new and the state snapshots -- so a
    stage's reference input is bit-identical to what the preceding kernel would
    have handed it, and a comparison measures that stage alone.

    *upto* is a stage name or ``"inputs"``. Calling it twice on the same shape
    only computes the stages that are missing.
    """
    hg = h if hg is None else hg
    want = 0 if upto == "inputs" else STAGES.index(upto) + 1
    key = (t, h, d, chunk, hg, seed)
    st = _CACHE.get(key)
    if st is None:
        st = _CACHE[key] = dict(make_inputs(t, h, d, hg, seed), _done=0)
    while st["_done"] < want:
        _run_stage(st, STAGES[st["_done"]], chunk, narrow=True)
        st["_done"] += 1
    return st


def delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               beta: torch.Tensor, g: torch.Tensor, chunk: int,
               narrow: bool = False) -> dict[str, torch.Tensor]:
    """All six stages on the given inputs; every intermediate, `o` the output.

    The block's core. Float64 throughout unless *narrow*, which narrows the
    stage boundaries exactly as :func:`compute` does.
    """
    st = dict(q=q, k=k, v=v, beta=beta, g=g)
    for stage in STAGES:
        _run_stage(st, stage, chunk, narrow)
    return st


def stage_inputs(stage: str, t: int, h: int, d: int, chunk: int,
                 hg: int | None = None, seed: int = 42) -> dict[str, torch.Tensor]:
    """Everything *stage* consumes: the reference chain up to its predecessor."""
    i = STAGES.index(stage)
    return compute("inputs" if i == 0 else STAGES[i - 1], t, h, d, chunk, hg, seed)


def lazy(stage: str, key: str, t: int, h: int, d: int, chunk: int,
         transform=None, hg: int | None = None, seed: int = 42):
    """A no-argument callable returning one reference tensor, computed on first use.

    `TensorSpec(init_value=...)` takes a callable, and deferring the chain this
    way keeps spec construction free -- the benchmark builds specs only to read
    their shapes and dtypes, and never pays for the host reference.
    """
    def load():
        value = stage_inputs(stage, t, h, d, chunk, hg, seed)[key]
        return transform(value) if transform is not None else value

    return load


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
#   o       = delta_rule(q, k, v, beta, g)        the six stages above
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

QK_NORM_EPS = 1e-6


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


def quantize_weights(w: dict[str, torch.Tensor],
                     names: tuple[str, ...] = GDN_QUANT.weights) -> dict[str, torch.Tensor]:
    """The int8 form of a weight set: *names* replaced by int8, `<module>.weight_scale` added.

    Everything else passes through. The scale's name follows the W8A8 checkpoint
    convention (`weight_scale` beside `weight`), so a converted checkpoint's
    tensors take the same keys.
    """
    out = dict(w)
    for name in names:
        assert name.endswith(".weight"), name
        out[name], out[name[: -len("weight")] + "weight_scale"] = quantize_rows(w[name])
    return out


def weight(w: dict[str, torch.Tensor], module: str) -> torch.Tensor:
    """A projection's `[out, in]` weight in float64; int8 ones dequantised by their scale."""
    value = w[module + ".weight"]
    if value.dtype == torch.int8:
        return dequantize(value, w[module + ".weight_scale"])
    return value.to(REF_DTYPE)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`nn.Linear` without bias: `x @ W^T`, W stored `[out, in]`."""
    return x.to(REF_DTYPE) @ weight.to(REF_DTYPE).t()


def short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution over the token axis, then silu. [T, C] -> [T, C].

    `out[t, c] = sum_j w[c, j] * x[t - (K - 1) + j, c]`, taps before the start of
    the sequence reading zero -- `nn.Conv1d(padding=K-1, groups=C, bias=False)`
    kept to its first T outputs. So tap `K-1` reads the token itself and tap `0`
    the token `K-1` back: the filter is written newest-last.
    """
    t, c = x.shape
    k = weight.shape[-1]
    xf = x.to(REF_DTYPE)
    wf = weight.reshape(c, k).to(REF_DTYPE)
    padded = torch.cat([torch.zeros(k - 1, c, dtype=REF_DTYPE), xf])
    out = torch.zeros(t, c, dtype=REF_DTYPE)
    for j in range(k):
        out += padded[j : j + t] * wf[:, j]
    return F.silu(out)


def l2norm(x: torch.Tensor, eps: float = QK_NORM_EPS) -> torch.Tensor:
    """`x * rsqrt(sum(x^2) + eps)` over the last dim: the eps sits inside the sqrt."""
    xf = x.to(REF_DTYPE)
    return xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)


def qk_norm_gate(q: torch.Tensor, k: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                 a_log: torch.Tensor, dt_bias: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """What the delta rule reads: q and k L2-normalised, q scaled by D^-0.5, beta, g.

    `beta = sigmoid(b)`; `g = -exp(A_log) * softplus(a + dt_bias)`, so g <= 0 and
    the state only ever decays. Both are per value head, `[T, H]`.
    """
    d = q.shape[-1]
    q_out = l2norm(q) * d ** -0.5
    k_out = l2norm(k)
    beta = torch.sigmoid(b.to(REF_DTYPE))
    g = -torch.exp(a_log.to(REF_DTYPE)) * F.softplus(a.to(REF_DTYPE) + dt_bias.to(REF_DTYPE))
    return q_out, k_out, beta, g


def gated_rmsnorm(o: torch.Tensor, z: torch.Tensor, weight: torch.Tensor,
                  eps: float) -> torch.Tensor:
    """RMSNorm over the head dim, times the weight, times `silu(z)`. [T, H, D] -> same.

    The gate multiplies after the norm (`Qwen3_5RMSNormGated`: "norm before
    gate"), so z does not enter the variance.
    """
    of = o.to(REF_DTYPE)
    var = (of * of).mean(dim=-1, keepdim=True)
    return of * torch.rsqrt(var + eps) * weight.to(REF_DTYPE) * F.silu(z.to(REF_DTYPE))


def block(x: torch.Tensor, w: dict[str, torch.Tensor], cfg: Qwen38Config,
          chunk: int, quant_x: bool = False, quant_y: bool = False
          ) -> dict[str, torch.Tensor]:
    """The whole block in float64, hidden states `[T, hidden]` to `out` of the same shape.

    Returns every intermediate: the projections `qkv`, `z`, `a_proj`, `b_proj`
    (`a` is the delta rule's key-key matrix, as everywhere else here); the conv
    output `qkv_conv`; the delta rule's inputs `q`, `k`, `v`, `beta`, `g` and
    its output `o` (plus everything :func:`delta_rule` keeps); the normed `y`;
    and `out`. Nothing is narrowed anywhere.

    What you pass is what you get. bf16 weights compute the truth; int8 ones
    (:func:`quantize_weights`) are used at their dequantised values. *quant_x*
    quantises the hidden states per token before every projection reads them,
    *quant_y* the normed output per token before `out_proj` -- the two
    activation quantisations a W8A8 block performs, at the places the kernels
    do them. The int8 tensors and fp32 scales come back as `x_q`, `x_scale`,
    `y_q`, `y_scale`, so a kernel can be checked at the quantised hand-off too.
    `y` is quantised from its fp32 rounding, as the norm kernel that produces
    it will hold it.
    """
    h, hg, d = (cfg.linear_num_value_heads, cfg.linear_num_key_heads,
                cfg.linear_value_head_dim)
    t = x.shape[0]
    key_width = hg * d

    st = {}
    xf = x
    if quant_x:
        st["x_q"], st["x_scale"] = quantize_rows(x)
        xf = dequantize(st["x_q"], st["x_scale"])
    st["qkv"] = linear(xf, weight(w, "in_proj_qkv"))
    st["z"] = linear(xf, weight(w, "in_proj_z")).reshape(t, h, d)
    st["a_proj"] = linear(xf, weight(w, "in_proj_a"))
    st["b_proj"] = linear(xf, weight(w, "in_proj_b"))
    st["qkv_conv"] = short_conv(st["qkv"], w["conv1d.weight"])
    q, k, v = torch.split(st["qkv_conv"], [key_width, key_width, h * d], dim=-1)
    q, k, beta, g = qk_norm_gate(q.reshape(t, hg, d), k.reshape(t, hg, d),
                                 st["a_proj"], st["b_proj"], w["A_log"], w["dt_bias"])
    st.update(delta_rule(q, k, v.reshape(t, h, d), beta, g, chunk))
    st["y"] = gated_rmsnorm(st["o"], st["z"], w["norm.weight"], cfg.rms_norm_eps)
    y = st["y"].reshape(t, h * d)
    if quant_y:
        st["y_q"], st["y_scale"] = quantize_rows(y)
        y = dequantize(st["y_q"], st["y_scale"])
    st["out"] = linear(y, weight(w, "out_proj"))
    return st


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
