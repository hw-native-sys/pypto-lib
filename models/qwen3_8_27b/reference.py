# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Float64 CPU reference for the Qwen3.8-27B text prefill, and the host helpers the
operators share.

The delta-rule references use the shared Torch computation for each stage.
This module also provides inputs, layout conversions, and acceptance criteria.

The block: the projections, the short convolution, the qk-norm and gate, the gated
RMSNorm and `out_proj`, chained by :func:`block` from hidden states to hidden states and
checked against the model's own `Qwen3_5GatedDeltaNet` by `test_block_reference.py`.
:func:`block` is two references in one -- on bf16 weights with no activation quantisation
it is the float64 truth, how far a result is from the model; on int8 weights
(:func:`quantize_weights`) with `quant_x` it is the fake-quant reference the kernels are
gated against; `quant_y` prices quantising the block's last hand-off, which they do not.

The model: :func:`text_prefill`, the embedding, both decoder-layer variants, dense
SwiGLU, the residual paths, the final RMSNorm and the LM head, checked against
`Qwen3_5ForConditionalGeneration` by `test_model_reference.py`.

The references follow the model's natural layout ([T, H, D] values, [T, H] per-token
scalars, [NCHUNK, H, D, D] states); the kernels take some of those transposed or
flattened, and :func:`to_hT` and :func:`flat_state` convert.

Grouped-query attention is implemented: `hg` QK heads against `h` value heads, value head
`i` reading key head `i // (h // hg)`. `hg` defaults to `h`, which makes the mapping an
identity. Packed validation runs the reference separately for each cache owner.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

if __package__:
    from .config import GDN_QUANT, Qwen38Config
    from .delta_rule_reference import REF_DTYPE, cumsum_ref, kkt_ref, solve_tril_ref, wy_fast_ref, chunk_h_ref, chunk_o_ref
else:
    from config import GDN_QUANT, Qwen38Config
    from delta_rule_reference import REF_DTYPE, cumsum_ref, kkt_ref, solve_tril_ref, wy_fast_ref, chunk_h_ref, chunk_o_ref

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





# Pipeline order. Each stage consumes the outputs of the ones before it.
_STAGES = ("chunk_cumsum", "scaled_dot_kkt", "solve_tril", "wy_fast",
           "chunk_h", "chunk_o")


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
        st["g_sum"] = cumsum_ref(st["g"], chunk)
    elif stage == "scaled_dot_kkt":
        put("a", kkt_ref(st["k"], st["beta"], st["g_sum"], chunk))
    elif stage == "solve_tril":
        put("a_inv", solve_tril_ref(src("a"), chunk))
    elif stage == "wy_fast":
        w, u = wy_fast_ref(st["k"], st["v"], st["beta"], src("a_inv"), st["g_sum"], chunk)
        put("w", w)
        put("u", u)
    elif stage == "chunk_h":
        state, v_new, final_state = chunk_h_ref(st["k"], src("w"), src("u"),
                                            st["g_sum"], chunk, st.get("initial_state"))
        put("state", state)
        put("v_new", v_new)
        st["final_state"] = final_state
    elif stage == "chunk_o":
        st["o"] = chunk_o_ref(st["q"], st["k"], src("v_new"), src("state"),
                          st["g_sum"], chunk)


def delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               beta: torch.Tensor, g: torch.Tensor, chunk: int,
               narrow: bool = False,
               initial_state: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """All six stages on the given inputs; every intermediate, `o` the output.

    The block's core. Float64 throughout unless *narrow*, which narrows the
    stage boundaries exactly as :func:`compute` does.
    """
    if chunk <= 0 or q.shape[0] <= 0 or q.shape[0] % chunk:
        raise ValueError("chunked delta rule needs a positive multiple of the chunk size")
    st = dict(q=q, k=k, v=v, beta=beta, g=g, initial_state=initial_state)
    for stage in _STAGES:
        _run_stage(st, stage, chunk, narrow)
    return st


def recurrent_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         beta: torch.Tensor, g: torch.Tensor,
                         initial_state: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode form: one token at a time, carrying the recurrent state.

    `torch_recurrent_gated_delta_rule` in transformers' `modeling_qwen3_5`, the
    branch the layer dispatches at `seq_len == 1`. Per value head, with `s` of
    `[D_k, D_v]`::

        s   = s * exp(g_t)
        kv  = s^T k_t
        s   = s + k_t outer ((v_t - kv) * beta_t)
        o_t = s^T q_t

    Takes the same arguments as :func:`delta_rule` -- raw per-token `g`, not its
    chunk prefix sum, and q and k already L2-normalised -- minus `chunk`, which
    has no meaning here. `initial_state` is `[H, D, D]`, the layout
    :func:`chunk_h` returns as `final_state`, so a prefill's last state feeds
    straight in. Returns `o` as `[T, H, D]` and the state leaving the last
    token.

    Float64 throughout: this is the oracle the device kernel is gated against,
    and the state accumulates over every token, so a narrow reference would
    drift from the thing being measured rather than bound it.
    """
    t, hg, d = k.shape
    h = v.shape[1]
    grp = h // hg
    qf, kf, vf, bf, gf = (x.to(REF_DTYPE) for x in (q, k, v, beta, g))
    state = (torch.zeros(h, d, d, dtype=REF_DTYPE) if initial_state is None
             else initial_state.to(REF_DTYPE).clone())
    o = torch.zeros(t, h, d, dtype=REF_DTYPE)
    for hh in range(h):
        s = state[hh]
        kh = hh // grp
        for i in range(t):
            k_t = kf[i, kh, :]
            s = s * torch.exp(gf[i, hh])
            delta = (vf[i, hh, :] - s.T @ k_t) * bf[i, hh]
            s = s + k_t[:, None] * delta[None, :]
            o[i, hh, :] = s.T @ qf[i, kh, :]
        state[hh] = s
    return o, state





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
        if not name.endswith(".weight"):
            raise ValueError(f"expected a projection weight name, got {name}")
        scale_name = name.removesuffix(".weight") + ".weight_scale"
        if w[name].dtype == torch.int8:
            out[scale_name] = w[scale_name].float().reshape(-1)
        else:
            out[name], out[scale_name] = quantize_rows(w[name])
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


def short_conv(x: torch.Tensor, weight: torch.Tensor,
               initial_state: torch.Tensor | None = None) -> torch.Tensor:
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
    history = (torch.zeros(k - 1, c, dtype=REF_DTYPE) if initial_state is None
               else initial_state.to(REF_DTYPE))
    padded = torch.cat([history, xf])
    out = torch.zeros(t, c, dtype=REF_DTYPE)
    for j in range(k):
        out += padded[j : j + t] * wf[:, j]
    return F.silu(out)


def recurrent_short_conv(x: torch.Tensor, weight: torch.Tensor,
                         initial_state: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode form of :func:`short_conv`: one token at a time, carrying the window.

    `causal_conv1d_update` in transformers' `modeling_qwen3_5`, the branch the layer
    takes at `seq_len == 1`. Per token, with `s` the last `K - 1` stream rows::

        window = concat(s, x_t)                 # [K, C], newest last
        out_t  = silu(sum_j w[:, j] * window[j])
        s      = window[1:]                     # drop the oldest

    The state holds rows of the **pre-convolution, pre-activation** stream -- not the
    conv output, not the silu output. So a prefill's state is just the last `K - 1`
    rows of the conv's own input, and no prefill kernel has to compute it.

    `initial_state` is `[K - 1, C]`, **time-major**, which is the layout the device
    kernel uses; transformers keeps it channel-major `[C, K - 1]`, so a caller
    crossing that boundary transposes. `None` means the start of a sequence, which
    matches :func:`short_conv`'s zero padding.

    Float64 throughout: this is the oracle the device kernel is gated against.
    """
    t, c = x.shape
    k = weight.shape[-1]
    xf = x.to(REF_DTYPE)
    wf = weight.reshape(c, k).to(REF_DTYPE)
    state = (torch.zeros(k - 1, c, dtype=REF_DTYPE) if initial_state is None
             else initial_state.to(REF_DTYPE).clone())
    acc = torch.zeros(t, c, dtype=REF_DTYPE)
    for i in range(t):
        window = torch.cat([state, xf[i : i + 1]])          # [K, C]
        acc[i] = (window * wf.t()).sum(dim=0)
        state = window[1:]
    return F.silu(acc), state


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
          chunk: int, quant_x: bool = False, quant_y: bool = False,
          conv_state: torch.Tensor | None = None,
          state: torch.Tensor | None = None,
          ) -> dict[str, torch.Tensor]:
    """The whole block in float64, hidden states `[T, hidden]` to `out` of the same shape.

    Returns every intermediate: the projections `qkv`, `z`, `a_proj`, `b_proj`
    (`a` is the delta rule's key-key matrix, as everywhere else here); the conv
    output `qkv_conv`; the delta rule's inputs `q`, `k`, `v`, `beta`, `g` and
    its output `o` (plus everything :func:`delta_rule` keeps); the normed `y`;
    and `out`. Nothing is narrowed anywhere.

    What you pass is what you get. bf16 weights compute the truth; int8 ones
    (:func:`quantize_weights`) are used at their dequantised values. *quant_x*
    quantises the hidden states per token for the QKV and Z projections,
    *quant_y* the normed output per token before `out_proj`. The kernels
    quantise the hidden states only, `out_proj` being BF16 both sides, so
    *quant_x* alone is the fake-quant reference they are gated against and
    *quant_y* prices the alternative. The int8 tensors and fp32 scales come back as `x_q`, `x_scale`,
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
    st["a_proj"] = linear(x, weight(w, "in_proj_a"))
    st["b_proj"] = linear(x, weight(w, "in_proj_b"))
    st["qkv_conv"] = short_conv(st["qkv"], w["conv1d.weight"], conv_state)
    st["conv_state_out"] = torch.cat([
        torch.zeros(cfg.linear_conv_kernel_dim - 1, cfg.qkv_width, dtype=REF_DTYPE)
        if conv_state is None else conv_state.to(REF_DTYPE), st["qkv"]
    ])[-(cfg.linear_conv_kernel_dim - 1):]
    q, k, v = torch.split(st["qkv_conv"], [key_width, key_width, h * d], dim=-1)
    q, k, beta, g = qk_norm_gate(q.reshape(t, hg, d), k.reshape(t, hg, d),
                                 st["a_proj"], st["b_proj"], w["A_log"], w["dt_bias"])
    st.update(delta_rule(q, k, v.reshape(t, h, d), beta, g, chunk,
                         initial_state=state))
    st["y"] = gated_rmsnorm(st["o"], st["z"], w["norm.weight"], cfg.rms_norm_eps)
    y = st["y"].reshape(t, h * d)
    if quant_y:
        st["y_q"], st["y_scale"] = quantize_rows(y)
        y = dequantize(st["y_q"], st["y_scale"])
    st["out"] = linear(y, weight(w, "out_proj"))
    return st


def recurrent_block(x: torch.Tensor, w: dict[str, torch.Tensor], cfg: Qwen38Config,
                    conv_state: torch.Tensor | None = None,
                    state: torch.Tensor | None = None,
                    quant_x: bool = False, quant_y: bool = False
                    ) -> dict[str, torch.Tensor]:
    """:func:`block` with the two recurrences in their decode form, one sequence.

    Same arithmetic as :func:`block` and the same returned keys, with
    :func:`short_conv` replaced by :func:`recurrent_short_conv` and
    :func:`delta_rule` by :func:`recurrent_delta_rule`, so the `chunk` argument
    disappears and two states enter and leave instead. `x` is `[T, hidden]` and its
    rows are consumed in order, carrying both states; `T = 1` is the single decode
    step, and a longer `T` is the rollout that
    :func:`test_decode_reference.check_against_chunked` compares against the chunked
    path.

    Returns `block`'s keys plus `conv_state_out` and `state_out`. The split of work
    differs from the kernels -- `qk_norm_gate` here both normalises q and k and forms
    beta and g, where at decode `decode_gate` forms beta and g and
    `decode_delta_rule` normalises -- because an oracle owes the arithmetic, not the
    division of labour.
    """
    t = x.shape[0]
    h, hg = cfg.linear_num_value_heads, cfg.linear_num_key_heads
    d = cfg.linear_value_head_dim
    key_width = hg * d

    st: dict[str, torch.Tensor] = {}
    xf = x
    if quant_x:
        st["x_q"], st["x_scale"] = quantize_rows(x)
        xf = dequantize(st["x_q"], st["x_scale"])
    st["qkv"] = linear(xf, weight(w, "in_proj_qkv"))
    st["z"] = linear(xf, weight(w, "in_proj_z")).reshape(t, h, d)
    st["a_proj"] = linear(x, weight(w, "in_proj_a"))
    st["b_proj"] = linear(x, weight(w, "in_proj_b"))
    st["qkv_conv"], st["conv_state_out"] = recurrent_short_conv(
        st["qkv"], w["conv1d.weight"], conv_state)
    q, k, v = torch.split(st["qkv_conv"], [key_width, key_width, h * d], dim=-1)
    q, k, beta, g = qk_norm_gate(q.reshape(t, hg, d), k.reshape(t, hg, d),
                                 st["a_proj"], st["b_proj"], w["A_log"], w["dt_bias"])
    st["q"], st["k"], st["v"] = q, k, v.reshape(t, h, d)
    st["beta"], st["g"] = beta, g
    st["o"], st["state_out"] = recurrent_delta_rule(q, k, v.reshape(t, h, d), beta, g, state)
    st["y"] = gated_rmsnorm(st["o"], st["z"], w["norm.weight"], cfg.rms_norm_eps)
    y = st["y"].reshape(t, h * d)
    if quant_y:
        st["y_q"], st["y_scale"] = quantize_rows(y)
        y = dequantize(st["y_q"], st["y_scale"])
    st["out"] = linear(y, weight(w, "out_proj"))
    return st


_BLOCK_CACHE: dict[tuple, dict] = {}


def block_inputs(key: str, t: int, cfg: Qwen38Config, chunk: int,
                 transform=None, seed: int = 42, weights: str | None = None,
                 quant_x: bool = False, quant_y: bool = False):
    """A no-argument callable returning one tensor of the block chain, computed once.

    What :func:`lazy` is for a stage, this is for a kernel of the block: the
    chain runs on first use and is cached, so several specs drawing from it pay
    for it once. *weights* is a path to a layer `weights.py` wrote, or None for
    the random set.
    """
    def load():
        cache_key = (t, cfg, chunk, seed, weights, quant_x, quant_y)
        st = _BLOCK_CACHE.get(cache_key)
        if st is None:
            if weights is None:
                w = make_block_weights(cfg, seed)
            else:
                w = torch.load(weights, weights_only=True)
            st = _BLOCK_CACHE[cache_key] = block(
                make_block_inputs(t, cfg, seed), w, cfg, chunk,
                quant_x=quant_x, quant_y=quant_y)
        value = st[key]
        return transform(value) if transform is not None else value

    return load


# ---------------------------------------------------------------------------
# The complete text prefill
# ---------------------------------------------------------------------------


def make_text_inputs(t: int, cfg: Qwen38Config, seed: int = 42) -> torch.Tensor:
    """Deterministic text token ids for the full-stack reference, shape ``[T]``."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (t,), generator=gen, dtype=torch.int64)


def rmsnorm(x: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5's zero-centred RMSNorm: ``norm(x) * (1 + weight)``.

    This is distinct from the GDN block's :func:`gated_rmsnorm`, whose weight is
    centred at one and whose gate follows the norm.
    """
    xf = x.to(REF_DTYPE)
    variance = (xf * xf).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps) * (1.0 + norm_weight.to(REF_DTYPE))


def swiglu_act(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`silu(gate) * up`, the MLP's activation on its own, in float64.

    Split out from :func:`swiglu_mlp` because the kernels fuse it with the
    per-token quantiser that feeds `down_proj`, so it needs its own gate.
    """
    gf = gate.to(REF_DTYPE)
    return F.silu(gf) * up.to(REF_DTYPE)


_MLP_CACHE: dict = {}


MLP_MODULES = (("w_gate", "gate_proj"), ("w_up", "up_proj"), ("w_down", "down_proj"))


def mlp_inputs(key: str, t: int, hidden: int, inter: int, seed: int = 42,
               weights: str | None = None):
    """One tensor of the W8A8 MLP's input set, drawn or loaded once and cached.

    The three weights are int8 with one fp32 scale per output channel, which is
    the deployed checkpoint's layout; the kernel reads them unchanged. With
    *weights* they come from one layer's BF16 `mlp` tensors, quantised with the
    checkpoint's own recipe rather than this module's fp32 one.

    `x_q` and `x_scale` are what `quant_x` hands the projections, so a kernel
    that starts after the quantiser can be fed the same activation the composed
    MLP would give it. `x_scale` is the `[1, T]` row the kernels read.
    """
    cache_key = (t, hidden, inter, seed, weights)
    st = _MLP_CACHE.get(cache_key)
    if st is None:
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(t, hidden, generator=gen).to(MODEL_DTYPE)
        st = {"x": x}
        x_q, x_scale = quantize_rows(x)
        st["x_q"], st["x_scale"] = x_q, x_scale.float().reshape(1, -1)
        if weights is None:
            for name, out_features, in_features in (
                ("w_gate", inter, hidden),
                ("w_up", inter, hidden),
                ("w_down", hidden, inter),
            ):
                dense = torch.randn(out_features, in_features, generator=gen) * (in_features ** -0.5)
                q, scale = quantize_rows(dense.to(MODEL_DTYPE))
                st[name] = q
                st[name + "_scale"] = scale
        else:
            if __package__:
                from .weights import modelslim_quantize_rows
            else:
                from weights import modelslim_quantize_rows

            layer = torch.load(weights, weights_only=True)
            for name, module in MLP_MODULES:
                q, scale, _ = modelslim_quantize_rows(layer[module + ".weight"])
                st[name] = q
                st[name + "_scale"] = scale.reshape(-1).float()
        _MLP_CACHE[cache_key] = st
    return st[key]


def gate_up_swiglu(a_q: torch.Tensor, a_scale: torch.Tensor,
                   w_gate: torch.Tensor, w_gate_scale: torch.Tensor,
                   w_up: torch.Tensor, w_up_scale: torch.Tensor) -> torch.Tensor:
    """The fused projection pair and activation, in the kernel's dtypes.

    Both projections are narrowed to BF16 before the activation, because that
    is where the unfused pair stored them and the fused kernel keeps the same
    narrowing. `a_scale` is the `[1, M]` row the kernels read.
    """
    af = dequantize(a_q, a_scale.reshape(-1))
    gate = (af @ dequantize(w_gate, w_gate_scale).t()).to(MODEL_DTYPE)
    up = (af @ dequantize(w_up, w_up_scale).t()).to(MODEL_DTYPE)
    return swiglu_act(gate, up)


def mlp_w8a8(x: torch.Tensor, w_gate: torch.Tensor, w_gate_scale: torch.Tensor,
             w_up: torch.Tensor, w_up_scale: torch.Tensor,
             w_down: torch.Tensor, w_down_scale: torch.Tensor,
             act_bf16: bool = False) -> torch.Tensor:
    """The MLP as the kernels compute it: int8 both sides, float64 in between.

    The fake-quant reference rather than the float64 truth. The activation
    entering `down_proj` is quantised per token exactly as `swiglu_quant`
    does, because that hand-off is where the W8A8 chain loses most of what it
    loses, and a gate against the truth would be measuring the scheme rather
    than the kernel.

    The **dtypes are modelled, not idealised**: `a8w8_linear` dequantises its
    INT32 accumulator through fp32 and writes BF16, so `gate` and `up` reach
    the activation with seven mantissa bits. Leaving them in float64 here puts
    56% of the output more than 1% away from the kernel and says nothing about
    whether the kernel is right.

    *act_bf16* models the **fused** chain, where one kernel computes both
    projections and writes a single BF16 `act` tensor for the quantiser to
    read. The unfused chain materialises no activation at all -- `swiglu_quant`
    holds a token's row and goes from the two BF16 projections straight to
    int8 -- so the fused form adds a rounding that does not exist today, and a
    reference that skipped it would be scoring a chain the kernel no longer
    implements.
    """
    x_q, x_scale = quantize_rows(x)
    xf = dequantize(x_q, x_scale)
    gate = (xf @ dequantize(w_gate, w_gate_scale).t()).to(MODEL_DTYPE)
    up = (xf @ dequantize(w_up, w_up_scale).t()).to(MODEL_DTYPE)
    act = swiglu_act(gate, up)
    if act_bf16:
        act = act.to(MODEL_DTYPE).to(REF_DTYPE)
    act_q, act_scale = quantize_rows(act)
    out = dequantize(act_q, act_scale) @ dequantize(w_down, w_down_scale).t()
    return out.to(MODEL_DTYPE).to(REF_DTYPE)


def mlp_truth(x: torch.Tensor, w_gate: torch.Tensor, w_gate_scale: torch.Tensor,
              w_up: torch.Tensor, w_up_scale: torch.Tensor,
              w_down: torch.Tensor, w_down_scale: torch.Tensor) -> torch.Tensor:
    """The same MLP in float64 throughout: the weights as stored, nothing else quantised.

    The reference :func:`mlp_w8a8` is scored against this, and so is the
    kernel. What the ratio of the two isolates is the cost of the activation
    path -- the per-token quantiser and the BF16 hand-offs -- with the
    checkpoint's own int8 weights held common to both.
    """
    xf = x.to(REF_DTYPE)
    gate = xf @ dequantize(w_gate, w_gate_scale).t()
    up = xf @ dequantize(w_up, w_up_scale).t()
    return swiglu_act(gate, up) @ dequantize(w_down, w_down_scale).t()


def swiglu_mlp(x: torch.Tensor, w: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Dense Qwen SwiGLU MLP under ``prefix``, including all three projections."""
    gate = linear(x, weight(w, prefix + ".gate_proj"))
    up = linear(x, weight(w, prefix + ".up_proj"))
    activated = F.silu(gate) * up
    return {
        "gate": gate,
        "up": up,
        "activated": activated,
        "out": linear(activated, weight(w, prefix + ".down_proj")),
    }


def rotary_cos_sin(position_ids: torch.Tensor, cfg: Qwen38Config
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-only partial-RoPE tables for ``position_ids``, both ``[T, rotary_dim]``.

    Qwen3.8 uses multimodal RoPE, but all three spatial position streams equal
    the text positions when no image or video tokens are present. The frequency
    recomposition therefore reduces to ordinary RoPE over the first
    ``cfg.rotary_dim`` dimensions.
    """
    rotary_dim = cfg.rotary_dim
    if rotary_dim <= 0 or rotary_dim > cfg.head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even and <= head_dim; got {rotary_dim} and {cfg.head_dim}"
        )
    exponent = torch.arange(0, rotary_dim, 2, dtype=REF_DTYPE) / rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** exponent)
    frequencies = position_ids.to(REF_DTYPE)[:, None] * inv_freq[None, :]
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rotated * cos[:, None, :] + _rotate_half(rotated) * sin[:, None, :]
    return torch.cat((rotated, passthrough), dim=-1)


def attn_qk_norm_rope(qg: torch.Tensor, k: torch.Tensor, q_norm_w: torch.Tensor,
                      k_norm_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      heads: int, kv_heads: int, head_dim: int, eps: float
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention q/k norm and partial RoPE on the kernels' flat layout, in float64.

    `qg` is the `q_proj` output, `[T, heads * 2 * head_dim]` with head `h`'s
    query in the first half of its `2 * head_dim` columns and its gate in the
    second; `k` is `[T, kv_heads * head_dim]`. The query also takes the score
    scale `head_dim ** -0.5`, which the attention core then omits. Returns
    `q` as `[T, heads * head_dim]` and `k` as `[T, kv_heads * head_dim]`.
    """
    t = qg.shape[0]
    q = qg.to(REF_DTYPE).reshape(t, heads, 2 * head_dim)[..., :head_dim]
    kk = k.to(REF_DTYPE).reshape(t, kv_heads, head_dim)
    q = rmsnorm(q, q_norm_w, eps) * head_dim ** -0.5
    kk = rmsnorm(kk, k_norm_w, eps)
    cos, sin = cos.to(REF_DTYPE), sin.to(REF_DTYPE)
    q = _apply_partial_rope(q, cos, sin).reshape(t, heads * head_dim)
    kk = _apply_partial_rope(kk, cos, sin).reshape(t, kv_heads * head_dim)
    return q, kk


def attn_gate(attended: torch.Tensor, qg: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """`attended * sigmoid(gate)` on the kernels' flat layout, in float64.

    `gate` is the second half of each head's `2 * head_dim` columns of the
    `q_proj` output `qg`; `attended` is `[T, heads * head_dim]`.
    """
    t = qg.shape[0]
    gate = qg.to(REF_DTYPE).reshape(t, heads, 2 * head_dim)[..., head_dim:].reshape(t, heads * head_dim)
    return attended.to(REF_DTYPE) * torch.sigmoid(gate)


ATTN_WEIGHTS = ("q_proj", "k_proj", "v_proj", "o_proj")
_ATTN_CACHE: dict = {}


def attn_inputs(key: str, t: int, cfg: Qwen38Config, weights: str | None = None, seed: int = 42):
    """One tensor of the W8A8 gated-attention input set, drawn or loaded once and cached.

    Keys: `x`, `w_q`, `w_k`, `w_v`, `w_o` (int8, `[out, in]`), each with a
    `_scale` (fp32 `[out]`), `q_norm_w` and `k_norm_w`. With *weights* the
    projections and norms come from one layer's BF16 `self_attn` tensors,
    quantised with the Eco-Tech checkpoint's own recipe; `x` is drawn either way.
    """
    cache_key = (t, cfg.name, weights, seed)
    st = _ATTN_CACHE.get(cache_key)
    if st is None:
        gen = torch.Generator().manual_seed(seed)
        st = {"x": torch.randn(t, cfg.hidden_size, generator=gen).to(MODEL_DTYPE)}
        if weights is None:
            shapes = {
                "q_proj": (2 * cfg.attention_q_width, cfg.hidden_size),
                "k_proj": (cfg.attention_kv_width, cfg.hidden_size),
                "v_proj": (cfg.attention_kv_width, cfg.hidden_size),
                "o_proj": (cfg.hidden_size, cfg.attention_q_width),
            }
            dense = {name: (torch.randn(*shape, generator=gen) * shape[1] ** -0.5).to(MODEL_DTYPE)
                     for name, shape in shapes.items()}
            for name in ("q_norm", "k_norm"):
                dense[name] = (torch.randn(cfg.head_dim, generator=gen) * 0.5).to(MODEL_DTYPE)
            quantize = quantize_rows
        else:
            if __package__:
                from .weights import modelslim_quantize_rows
            else:
                from weights import modelslim_quantize_rows

            layer = torch.load(weights, weights_only=True)
            dense = {name: layer[name + ".weight"] for name in (*ATTN_WEIGHTS, "q_norm", "k_norm")}

            # the deployed checkpoint's recipe, which reproduces its int8 exactly
            def quantize(weight):
                q, scale, _ = modelslim_quantize_rows(weight)
                return q, scale.reshape(-1)
        for name in ATTN_WEIGHTS:
            q, scale = quantize(dense[name])
            short = "w_" + name[0]
            st[short], st[short + "_scale"] = q, scale.float()
        st["q_norm_w"], st["k_norm_w"] = dense["q_norm"], dense["k_norm"]
        _ATTN_CACHE[cache_key] = st
    return st[key]


def attn_w8a8(x: torch.Tensor, w: dict[str, torch.Tensor], cos: torch.Tensor, sin: torch.Tensor,
              cfg: Qwen38Config, prefix_k: torch.Tensor | None = None,
              prefix_v: torch.Tensor | None = None,
              return_cache: bool = False):
    """The gated-attention mixer as the kernels compute it: int8 both sides, float64 between.

    The fake-quant reference, not the truth. Every hand-off the kernels make
    in BF16 is rounded to BF16 here -- the three projections, the normed and
    rotated q and k, the attention output -- and the two int8 hand-offs use
    the kernels' per-token quantiser. *w* holds the `attn_inputs` keys.
    """
    h, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    x_q, x_scale = quantize_rows(x)
    xf = dequantize(x_q, x_scale)
    qg = (xf @ dequantize(w["w_q"], w["w_q_scale"]).t()).to(MODEL_DTYPE)
    k = (xf @ dequantize(w["w_k"], w["w_k_scale"]).t()).to(MODEL_DTYPE)
    v = (xf @ dequantize(w["w_v"], w["w_v_scale"]).t()).to(MODEL_DTYPE)
    q_rot, k_rot = attn_qk_norm_rope(qg, k, w["q_norm_w"], w["k_norm_w"], cos, sin, h, hkv, d, cfg.rms_norm_eps)
    k_attn = k_rot.to(MODEL_DTYPE)
    v_attn = v
    if prefix_k is not None:
        k_attn = torch.cat([prefix_k, k_attn])
        v_attn = torch.cat([prefix_v, v_attn])
    attended = causal_gqa_attention(q_rot.to(MODEL_DTYPE), k_attn, v_attn, h, hkv, d).to(MODEL_DTYPE)
    act_q, act_scale = quantize_rows(attn_gate(attended, qg, h, d))
    out = dequantize(act_q, act_scale) @ dequantize(w["w_o"], w["w_o_scale"]).t()
    result = out.to(MODEL_DTYPE).to(REF_DTYPE)
    return (result, k_rot.to(MODEL_DTYPE), v) if return_cache else result


def attn_truth(x: torch.Tensor, w: dict[str, torch.Tensor], cos: torch.Tensor, sin: torch.Tensor,
               cfg: Qwen38Config, prefix_k: torch.Tensor | None = None,
               prefix_v: torch.Tensor | None = None,
               return_cache: bool = False):
    """The same mixer in float64 throughout: the weights as stored, nothing else quantised.

    Isolates what the activation path costs -- the two per-token quantisers
    and the BF16 hand-offs -- with the checkpoint's own int8 weights common to
    both this and :func:`attn_w8a8`.
    """
    h, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    xf = x.to(REF_DTYPE)
    qg = xf @ dequantize(w["w_q"], w["w_q_scale"]).t()
    k = xf @ dequantize(w["w_k"], w["w_k_scale"]).t()
    v = xf @ dequantize(w["w_v"], w["w_v_scale"]).t()
    q_rot, k_rot = attn_qk_norm_rope(qg, k, w["q_norm_w"], w["k_norm_w"], cos, sin, h, hkv, d, cfg.rms_norm_eps)
    k_current, v_current = k_rot, v
    if prefix_k is not None:
        k_rot = torch.cat([prefix_k, k_rot])
        v = torch.cat([prefix_v, v])
    attended = causal_gqa_attention(q_rot, k_rot, v, h, hkv, d)
    result = attn_gate(attended, qg, h, d) @ dequantize(w["w_o"], w["w_o_scale"]).t()
    return (result, k_current, v_current) if return_cache else result


ATTN_BAND = 1024        # query rows per band of the causal attention below


def causal_gqa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         heads: int, kv_heads: int, head_dim: int,
                         scale: float = 1.0) -> torch.Tensor:
    """Causal grouped-query attention on flat `[T, H * D]` operands, in float64.

    The kernels' layout: `q` is `[T, heads * head_dim]`, `k` and `v` are
    `[prefix + T, kv_heads * head_dim]`, and query head `h` reads KV head
    `h // (heads // kv_heads)`. Returns `[T, heads * head_dim]`.

    Banded over query rows, and each band stops at its last key: the whole
    score tensor is 12.9 GB at T = 8192 in float64, and half of it is masked.
    """
    t = q.shape[0]
    kv_t = k.shape[0]
    prefix = kv_t - t
    if prefix < 0 or v.shape[0] != kv_t:
        raise ValueError("K/V must contain the query chunk and any cached prefix")
    group = heads // kv_heads
    qh = q.to(REF_DTYPE).reshape(t, heads, head_dim).transpose(0, 1)
    kh = k.to(REF_DTYPE).reshape(kv_t, kv_heads, head_dim).transpose(0, 1)
    vh = v.to(REF_DTYPE).reshape(kv_t, kv_heads, head_dim).transpose(0, 1)
    out = torch.empty(heads, t, head_dim, dtype=REF_DTYPE)
    for r0 in range(0, t, ATTN_BAND):
        r1 = min(r0 + ATTN_BAND, t)
        k1 = prefix + r1
        causal = torch.arange(k1)[None, :] > (prefix + torch.arange(r0, r1)[:, None])
        for h in range(heads):
            scores = (qh[h, r0:r1] @ kh[h // group, :k1].t()) * scale
            probs = torch.softmax(scores.masked_fill(causal, -torch.inf), dim=-1)
            out[h, r0:r1] = probs @ vh[h // group, :k1]
    return out.transpose(0, 1).reshape(t, heads * head_dim)


def kv_page_rows(block_table: torch.Tensor, seq: int, length: int,
                 page_size: int) -> torch.Tensor:
    """The physical rows holding sequence `seq`'s first `length` cached tokens, in order.

    The paged cache's index arithmetic, in one place: logical page `p` starts at
    `block_table[seq, p] * page_size`, and the last page carries only the tokens the
    length leaves in it. Every reader and writer of the cache derives its rows from
    here, so a block-table or page-size disagreement between them cannot exist.
    """
    pages = (length + page_size - 1) // page_size
    if pages > block_table.shape[1]:
        raise ValueError(f"sequence {seq} at length {length} needs {pages} pages, "
                         f"past the block table's {block_table.shape[1]}")
    rows = []
    for p in range(pages):
        lo = int(block_table[seq, p]) * page_size
        take = min(page_size, length - p * page_size)
        rows.append(torch.arange(lo, lo + take))
    return torch.cat(rows)


def decode_gqa_attention(q: torch.Tensor, k_pages: torch.Tensor, v_pages: torch.Tensor,
                         block_table: torch.Tensor, seq_lens: torch.Tensor,
                         heads: int, kv_heads: int, head_dim: int, page_size: int,
                         scale: float = 1.0) -> torch.Tensor:
    """One decode step of grouped-query attention over a **paged** KV cache, float64.

    The decode form of :func:`causal_gqa_attention`: one query row per sequence,
    attending to every key that sequence has cached. There is no causal mask --
    the query is the newest position, so every cached key is in its past.

    `q` is `[B, heads * head_dim]`, one row per sequence. `k_pages` and `v_pages`
    are the paged cache, `[pages * page_size, kv_heads * head_dim]`, and
    `block_table[b, i]` is the physical page holding sequence `b`'s logical page
    `i`. `seq_lens[b]` is how many of those tokens are real; the tail of the last
    page is not. Query head `h` reads KV head `h // (heads // kv_heads)`.

    The paging is modelled rather than flattened away on purpose: a block-table or
    page-size error is the failure this reference exists to catch, and a contiguous
    cache would hide it.
    """
    b = q.shape[0]
    group = heads // kv_heads
    qh = q.to(REF_DTYPE).reshape(b, heads, head_dim)
    out = torch.zeros(b, heads, head_dim, dtype=REF_DTYPE)
    for i in range(b):
        length = int(seq_lens[i])
        idx = kv_page_rows(block_table, i, length, page_size)
        kh = k_pages[idx].to(REF_DTYPE).reshape(length, kv_heads, head_dim)
        vh = v_pages[idx].to(REF_DTYPE).reshape(length, kv_heads, head_dim)
        for h in range(heads):
            scores = (qh[i, h] @ kh[:, h // group].t()) * scale
            probs = torch.softmax(scores, dim=-1)
            out[i, h] = probs @ vh[:, h // group]
    return out.reshape(b, heads * head_dim)


def pages_from_contiguous(k: torch.Tensor, v: torch.Tensor, page_size: int,
                          first_page: int = 0) -> tuple:
    """Lay a contiguous `[T, kv_heads * head_dim]` cache out in pages.

    Returns `(k_pages, v_pages, block_table)` for one sequence, with its logical
    pages placed at physical `first_page`, `first_page + 1`, ... so a kernel that
    ignored the block table reads the wrong rows unless `first_page` is zero.
    """
    t, width = k.shape
    pages = (t + page_size - 1) // page_size
    total = first_page + pages
    k_pages = torch.zeros(total * page_size, width, dtype=k.dtype)
    v_pages = torch.zeros(total * page_size, width, dtype=v.dtype)
    for p in range(pages):
        lo, hi = p * page_size, min((p + 1) * page_size, t)
        dst = (first_page + p) * page_size
        k_pages[dst : dst + hi - lo] = k[lo:hi]
        v_pages[dst : dst + hi - lo] = v[lo:hi]
    table = torch.arange(first_page, first_page + pages, dtype=torch.int32).reshape(1, pages)
    return k_pages, v_pages, table


def append_kv_pages(k_pages: torch.Tensor, v_pages: torch.Tensor,
                    block_table: torch.Tensor, seq_lens: torch.Tensor,
                    k_new: torch.Tensor, v_new: torch.Tensor, page_size: int) -> None:
    """Write one decode token's K and V into each sequence's slot, in place.

    The inverse of the read :func:`decode_gqa_attention` performs, and derived the same
    way: `seq_lens[i]` **counts the token being written**, so its position is
    `seq_lens[i] - 1` and its physical row is
    `block_table[i, pos // page_size] * page_size + pos % page_size`. A length that
    excluded the new token would place it one slot early and the reader would skip it.
    """
    batch = k_new.shape[0]
    if block_table.shape[0] != batch or seq_lens.shape[0] != batch:
        raise ValueError(f"block_table {tuple(block_table.shape)} and seq_lens "
                         f"{tuple(seq_lens.shape)} must both start with batch {batch}")
    for i in range(batch):
        pos = int(seq_lens[i]) - 1
        if pos < 0:
            raise ValueError(f"seq_lens[{i}] counts the token being written; it must be >= 1")
        page, offset = divmod(pos, page_size)
        if page >= block_table.shape[1]:
            raise ValueError(f"sequence {i} at position {pos} needs page {page}, "
                             f"past the block table's {block_table.shape[1]}")
        row = int(block_table[i, page]) * page_size + offset
        k_pages[row] = k_new[i].to(k_pages.dtype)
        v_pages[row] = v_new[i].to(v_pages.dtype)


def gated_attention(x: torch.Tensor, w: dict[str, torch.Tensor], cfg: Qwen38Config,
                    prefix: str, position_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """One causal Qwen3.8 gated-attention mixer on a single text sequence.

    Q has 24 heads in the real model and shares four KV heads through GQA. The
    Q projection is twice the usual width: its second half is a sigmoid output
    gate applied after attention and before ``o_proj``.

    **This is the module-shaped reference, and it is dense on purpose.** It keeps the
    model's natural `[T, H, D]` layout and scores every key against every query, as the
    module does, so the two can be read side by side where `test_model_reference` checks
    `text_prefill` against `Qwen3_5ForConditionalGeneration`. The kernel-shaped
    references -- `attn_truth` and `attn_w8a8` -- take the flat layout and go through
    `causal_gqa_attention`, which is banded; `test_attention_reference` ties the two
    together, and that check is worth something because the two sides compute attention
    differently.

    **Long sequences do not reach here.** Every caller is a reference check at a short
    one: `test_attention_reference` at 67 tokens, `test_model_reference` at 7 on a
    2-layer, 32-wide config, and `test_weights_contract` at 6. Prefill-length runs go
    through the kernel-shaped references instead. The `[heads, T, T]` score tensor is
    therefore bounded by how this function is used, not by what it could cost -- it would
    be 12.9 GB at T = 8192 in float64, which is why the banded form exists for the side
    that needs it.
    """
    if cfg.attention_bias:
        raise ValueError("the Qwen3.8 reference currently supports bias-free attention only")
    t = x.shape[0]
    h, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    if h % hkv:
        raise ValueError(f"num_attention_heads ({h}) must be divisible by num_key_value_heads ({hkv})")

    q_gate = linear(x, weight(w, prefix + ".q_proj")).reshape(t, h, 2 * d)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    k = linear(x, weight(w, prefix + ".k_proj")).reshape(t, hkv, d)
    v = linear(x, weight(w, prefix + ".v_proj")).reshape(t, hkv, d)
    q = rmsnorm(q, w[prefix + ".q_norm.weight"], cfg.rms_norm_eps)
    k = rmsnorm(k, w[prefix + ".k_norm.weight"], cfg.rms_norm_eps)

    cos, sin = rotary_cos_sin(position_ids, cfg)
    q = _apply_partial_rope(q, cos, sin)
    k = _apply_partial_rope(k, cos, sin)

    groups = h // hkv
    q_heads = q.transpose(0, 1)
    k_heads = k.transpose(0, 1).repeat_interleave(groups, dim=0)
    v_heads = v.transpose(0, 1).repeat_interleave(groups, dim=0)
    scores = (q_heads @ k_heads.transpose(-1, -2)) * (d ** -0.5)
    causal = torch.ones(t, t, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(causal, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    attended = (probabilities @ v_heads).transpose(0, 1).reshape(t, h * d)
    gated = attended * torch.sigmoid(gate.reshape(t, h * d).to(REF_DTYPE))
    out = linear(gated, weight(w, prefix + ".o_proj"))
    # `scores` and `probabilities` stay local: nothing in this tree reads them, and they
    # are the two values that grow with T squared.
    return {
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "attended": attended,
        "gated": gated,
        "out": out,
    }


def _subweights(w: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Strip ``prefix`` from one GDN block's checkpoint-native tensor names."""
    out = {}
    for name, value in w.items():
        if name.startswith(prefix):
            out[name[len(prefix):]] = value
    missing = [name for name in WEIGHT_NAMES if name not in out]
    if missing:
        raise KeyError(f"missing GDN weights below {prefix}: {missing}")
    return out


def attn_w8a8_decode(x: torch.Tensor, w: dict[str, torch.Tensor], cos: torch.Tensor,
                     sin: torch.Tensor, cfg: Qwen38Config, k_pages: torch.Tensor,
                     v_pages: torch.Tensor, block_table: torch.Tensor,
                     seq_lens: torch.Tensor, page_size: int) -> torch.Tensor:
    """:func:`attn_w8a8` for one decode step per sequence, over a paged cache.

    The fake-quant reference, not the truth, and rounded exactly where
    :func:`attn_w8a8` rounds: the three projections, the normed and rotated q and k and
    the attention output all pass through BF16, and the two int8 hand-offs use the
    kernels' per-token quantiser.

    `x` is `[B, hidden]`, one token per sequence. This step's K and V are appended to the
    slot `block_table` names for position `seq_lens[b] - 1` before the read, which is the
    convention :func:`append_kv_pages` and :func:`decode_gqa_attention` share. The score
    scale rides on the query inside :func:`attn_qk_norm_rope`, as in the kernels, so the
    attention call takes `scale = 1.0`. `k_pages` and `v_pages` are updated in place.
    """
    h, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    x_q, x_scale = quantize_rows(x)
    xf = dequantize(x_q, x_scale)
    qg = (xf @ dequantize(w["w_q"], w["w_q_scale"]).t()).to(MODEL_DTYPE)
    k = (xf @ dequantize(w["w_k"], w["w_k_scale"]).t()).to(MODEL_DTYPE)
    v = (xf @ dequantize(w["w_v"], w["w_v_scale"]).t()).to(MODEL_DTYPE)
    q_rot, k_rot = attn_qk_norm_rope(qg, k, w["q_norm_w"], w["k_norm_w"], cos, sin,
                                     h, hkv, d, cfg.rms_norm_eps)
    append_kv_pages(k_pages, v_pages, block_table, seq_lens,
                    k_rot.to(MODEL_DTYPE), v, page_size)
    attended = decode_gqa_attention(q_rot.to(MODEL_DTYPE), k_pages, v_pages, block_table,
                                    seq_lens, h, hkv, d, page_size,
                                    scale=1.0).to(MODEL_DTYPE)
    act_q, act_scale = quantize_rows(attn_gate(attended, qg, h, d))
    out = dequantize(act_q, act_scale) @ dequantize(w["w_o"], w["w_o_scale"]).t()
    return out.to(MODEL_DTYPE).to(REF_DTYPE)


def attn_truth_decode(x: torch.Tensor, w: dict[str, torch.Tensor], cos: torch.Tensor,
                      sin: torch.Tensor, cfg: Qwen38Config, k_pages: torch.Tensor,
                      v_pages: torch.Tensor, block_table: torch.Tensor,
                      seq_lens: torch.Tensor, page_size: int) -> torch.Tensor:
    """:func:`attn_truth` for one decode step: float64 throughout, same int8 weights.

    Isolates what the decode activation path costs. `k_pages` and `v_pages` are updated in
    place, so pass a copy if both this and :func:`attn_w8a8_decode` are to read the same
    starting cache.
    """
    h, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    xf = x.to(REF_DTYPE)
    qg = xf @ dequantize(w["w_q"], w["w_q_scale"]).t()
    k = xf @ dequantize(w["w_k"], w["w_k_scale"]).t()
    v = xf @ dequantize(w["w_v"], w["w_v_scale"]).t()
    q_rot, k_rot = attn_qk_norm_rope(qg, k, w["q_norm_w"], w["k_norm_w"], cos, sin,
                                     h, hkv, d, cfg.rms_norm_eps)
    append_kv_pages(k_pages, v_pages, block_table, seq_lens, k_rot, v, page_size)
    attended = decode_gqa_attention(q_rot, k_pages, v_pages, block_table, seq_lens,
                                    h, hkv, d, page_size, scale=1.0)
    return attn_gate(attended, qg, h, d) @ dequantize(w["w_o"], w["w_o_scale"]).t()


def decoder_layer(x: torch.Tensor, w: dict[str, torch.Tensor], cfg: Qwen38Config,
                  layer_idx: int, chunk: int, position_ids: torch.Tensor
                  ) -> dict[str, torch.Tensor]:
    """One complete decoder layer, including both residual additions and the MLP."""
    prefix = f"model.language_model.layers.{layer_idx}"
    input_norm = rmsnorm(x, w[prefix + ".input_layernorm.weight"], cfg.rms_norm_eps)
    layer_type = cfg.layer_types[layer_idx]
    if layer_type == "linear_attention":
        # The stage reference is chunk-shaped, while the model accepts any text
        # length. Padding only the causal suffix cannot affect real-token rows.
        pad = (-input_norm.shape[0]) % chunk
        gdn_input = input_norm
        if pad:
            gdn_input = torch.cat(
                (input_norm, torch.zeros(pad, input_norm.shape[1], dtype=input_norm.dtype)),
                dim=0,
            )
        mixer = block(
            gdn_input,
            _subweights(w, prefix + ".linear_attn."),
            cfg,
            chunk,
        )["out"][: input_norm.shape[0]]
    elif layer_type == "full_attention":
        mixer = gated_attention(
            input_norm,
            w,
            cfg,
            prefix + ".self_attn",
            position_ids,
        )["out"]
    else:
        raise ValueError(f"unsupported layer type at index {layer_idx}: {layer_type!r}")

    post_mixer = x.to(REF_DTYPE) + mixer
    post_attention_norm = rmsnorm(
        post_mixer,
        w[prefix + ".post_attention_layernorm.weight"],
        cfg.rms_norm_eps,
    )
    mlp = swiglu_mlp(post_attention_norm, w, prefix + ".mlp")["out"]
    return {
        "input_norm": input_norm,
        "mixer": mixer,
        "post_mixer": post_mixer,
        "post_attention_norm": post_attention_norm,
        "mlp": mlp,
        "out": post_mixer + mlp,
    }


def residual_layer(residual: torch.Tensor, delta: torch.Tensor, norm_in_w: torch.Tensor,
                   norm_post_w: torch.Tensor, mixer: Callable[[torch.Tensor], torch.Tensor],
                   mlp_w: dict[str, torch.Tensor], cfg: Qwen38Config, quant: bool = True
                   ) -> dict[str, torch.Tensor]:
    """One decoder layer: two residual adds, two norms, a mixer and the MLP.

    `mixer` is a callable taking the normed hidden states and returning that mixer's
    output, so this function carries the layer's own arithmetic and nothing else, and
    serves both paths. Any of the four mixer oracles fits -- :func:`block` or
    :func:`attn_w8a8` over a prefill chunk, :func:`recurrent_block` or
    :func:`attn_w8a8_decode` at a decode batch -- closed over its own weights and its own
    state, which only the caller can supply.

    **The layer boundary is one add behind the module boundary.** `delta` is the previous
    layer's MLP output and `residual` the stream it adds to, so the layer begins by
    completing that addition and ends with its own MLP output still unadded. That is what
    a stacked loop computes per layer, on either path: two fused add-and-norm stages, not
    two norms and two separate adds. `out` is the complete layer output for a caller that
    wants it.

    *quant* picks the chain. With it, the dtypes are modelled where the kernels hold them:
    the running residual in fp32, every norm output and the mixer output in bf16, and the
    MLP as :func:`mlp_w8a8` computes it with a bf16 activation. Without it, everything is
    float64 and the MLP is :func:`mlp_truth` -- the same weights, nothing else narrowed.
    """
    def resid(x: torch.Tensor) -> torch.Tensor:
        """The running residual, which the kernels keep in fp32 and never in bf16."""
        return x.float().to(REF_DTYPE) if quant else x

    def narrow(x: torch.Tensor) -> torch.Tensor:
        return x.to(MODEL_DTYPE).to(REF_DTYPE) if quant else x

    st = {}
    st["residual"] = resid(residual.to(REF_DTYPE) + delta.to(REF_DTYPE))
    st["input_norm"] = narrow(rmsnorm(st["residual"], norm_in_w, cfg.rms_norm_eps))
    st["mixer"] = narrow(mixer(st["input_norm"]))
    st["residual_out"] = resid(st["residual"] + st["mixer"])
    st["post_attention_norm"] = narrow(
        rmsnorm(st["residual_out"], norm_post_w, cfg.rms_norm_eps))
    args = (st["post_attention_norm"], mlp_w["w_gate"], mlp_w["w_gate_scale"],
            mlp_w["w_up"], mlp_w["w_up_scale"], mlp_w["w_down"], mlp_w["w_down_scale"])
    st["mlp_out"] = mlp_w8a8(*args, act_bf16=True) if quant else mlp_truth(*args)
    st["out"] = st["residual_out"] + st["mlp_out"]
    return st


def text_prefill(input_ids: torch.Tensor, w: dict[str, torch.Tensor], cfg: Qwen38Config,
                 chunk: int, position_ids: torch.Tensor | None = None,
                 capture_intermediates: bool = False) -> dict[str, torch.Tensor]:
    """Complete text-only prefill for one sequence, from token ids to logits.

    ``w`` uses the checkpoint's native ``Qwen3_5ForConditionalGeneration`` key
    names. The returned ``hidden`` and ``logits`` are always present. With
    ``capture_intermediates``, the dictionary also contains the embedding and
    every decoder-layer boundary used by the Transformers equivalence test.
    """
    if input_ids.ndim != 1:
        raise ValueError(f"text_prefill expects one packed sequence [T], got {tuple(input_ids.shape)}")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
    t = input_ids.numel()
    if t == 0:
        raise ValueError("text_prefill requires at least one token")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if position_ids is None:
        position_ids = torch.arange(t, dtype=torch.int64)
    if position_ids.shape != (t,):
        raise ValueError(f"position_ids must have shape ({t},), got {tuple(position_ids.shape)}")

    embedding_weight = w["model.language_model.embed_tokens.weight"].to(REF_DTYPE)
    hidden = F.embedding(input_ids.to(torch.int64), embedding_weight)
    state = {"embedding": hidden} if capture_intermediates else {}
    for layer_idx in range(cfg.num_hidden_layers):
        layer = decoder_layer(hidden, w, cfg, layer_idx, chunk, position_ids)
        hidden = layer["out"]
        if capture_intermediates:
            state.update({f"layers.{layer_idx}.{name}": value for name, value in layer.items()})

    hidden = rmsnorm(hidden, w["model.language_model.norm.weight"], cfg.rms_norm_eps)
    logits = linear(hidden, weight(w, "lm_head"))
    state.update(hidden=hidden, logits=logits)
    return state





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
