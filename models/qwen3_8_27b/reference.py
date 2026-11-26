# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Float64 CPU reference for the Qwen3.8-27B GDN block and its operators.

The delta-rule references use the shared Torch computation for each stage.
This module also provides inputs, layout conversions, and acceptance criteria.

The block: the projections, the short convolution, the qk-norm and gate, the gated
RMSNorm and `out_proj`, chained by :func:`block` from hidden states to hidden states and
checked against the model's own `Qwen3_5GatedDeltaNet` by `test_block_reference.py`.
:func:`block` is two references in one -- on bf16 weights with no activation quantisation
it is the float64 truth, how far a result is from the model; on int8 weights
(:func:`quantize_weights`) with `quant_x` it is the fake-quant reference the kernels are
gated against; `quant_y` prices quantising the block's last hand-off, which they do not.

The references follow the model's natural layout ([T, H, D] values, [T, H] per-token
scalars, [NCHUNK, H, D, D] states); the kernels take some of those transposed or
flattened, and :func:`to_hT` and :func:`flat_state` convert.

Grouped-query attention is implemented: `hg` QK heads against `h` value heads, value head
`i` reading key head `i // (h // hg)`. `hg` defaults to `h`, which makes the mapping an
identity. Packed validation runs the reference separately for each cache owner.
"""

from __future__ import annotations

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
