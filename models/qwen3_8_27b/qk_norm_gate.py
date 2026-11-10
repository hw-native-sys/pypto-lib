# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet qk-norm and gate: what the delta rule reads, from the conv output and a/b.

    q_out = q * rsqrt(sum(q^2) + eps) * D^-0.5        per (token, head) over the head dim
    k_out = k * rsqrt(sum(k^2) + eps)
    beta  = sigmoid(b)                                 [H, T], head-major
    g     = -exp(A_log) * softplus(a + dt_bias)        [T, H], <= 0

The eps sits inside the sqrt, not outside it. fp32 throughout, fp16 out for q
and k because that is what the delta-rule stages read; beta and g stay fp32.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# model shape
H = QWEN3_8_27B.linear_num_value_heads      # value heads: a, b, beta and g
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads: q and k
D = QWEN3_8_27B.linear_value_head_dim       # head dimension, the norm's axis
EPS = 1e-6                                  # inside the sqrt

# case shape
T = 8192                # tokens (single sequence, B = 1)

# tiling
ROW_TILE = 64           # (token, head) rows per pipelined step of the norm
GROUP = 8               # row-tiles per block, the pipelined inner loop
TOK_TILE = 256          # tokens per block of the gate

# softplus(x) = log1p(exp(x)) overflows fp32 at x ~ 88 and is exactly x well
# before that; the model's own dt_bias already reaches 19.25.
SOFTPLUS_LINEAR = 20.0


def build_kernel(t: int = T, h: int = H, hg: int = HG, d: int = D, eps: float = EPS,
                 inline: bool = False, row_tile: int = ROW_TILE, group: int = GROUP,
                 tok_tile: int = TOK_TILE):
    """The operator at one shape. `inline` makes it a callee for `gdn_block`."""
    rows = row_tile * group
    if (t * hg) % rows or t % tok_tile:
        raise ValueError(f"t*hg={t * hg} must be a multiple of row_tile*group={rows} "
                         f"and t={t} a multiple of tok_tile={tok_tile}")

    @(pl.jit.inline if inline else pl.jit)
    def gdn_qk_norm_gate(
        q: pl.Tensor[[t, hg, d], pl.BF16],
        k: pl.Tensor[[t, hg, d], pl.BF16],
        a_proj: pl.Tensor[[t, h], pl.BF16],
        b_proj: pl.Tensor[[t, h], pl.BF16],
        a_log: pl.Tensor[[h], pl.BF16],
        dt_bias: pl.Tensor[[h], pl.BF16],
        q_out: pl.Out[pl.Tensor[[t, hg, d], pl.FP16]],
        k_out: pl.Out[pl.Tensor[[t, hg, d], pl.FP16]],
        beta: pl.Out[pl.Tensor[[h, t], pl.FP32]],
        g: pl.Out[pl.Tensor[[t, h], pl.FP32]],
    ):
        q_rows = pl.reshape(q, [t * hg, d])
        k_rows = pl.reshape(k, [t * hg, d])
        qo_rows = pl.reshape(q_out, [t * hg, d])
        ko_rows = pl.reshape(k_out, [t * hg, d])
        scale = d ** -0.5
        for blk in pl.spmd((t * hg) // rows, name_hint="qk_norm"):
            r00 = blk * rows
            for i in pl.pipeline(group, stage=2):
                r0 = r00 + i * row_tile
                qf = pl.cast(q_rows[r0 : r0 + row_tile, :], target_type=pl.FP32)
                q_inv = pl.rsqrt(pl.add(pl.row_sum(pl.mul(qf, qf)), eps), high_precision=True)
                q_normed = pl.mul(pl.row_expand_mul(qf, q_inv), scale)
                qo_rows[r0 : r0 + row_tile, :] = pl.cast(q_normed, target_type=pl.FP16, mode="rint")

                kf = pl.cast(k_rows[r0 : r0 + row_tile, :], target_type=pl.FP32)
                k_inv = pl.rsqrt(pl.add(pl.row_sum(pl.mul(kf, kf)), eps), high_precision=True)
                k_normed = pl.row_expand_mul(kf, k_inv)
                ko_rows[r0 : r0 + row_tile, :] = pl.cast(k_normed, target_type=pl.FP16, mode="rint")

        for blk in pl.spmd(t // tok_tile, name_hint="gate"):
            t0 = blk * tok_tile
            log_row = pl.cast(pl.reshape(a_log[:], [1, h]), target_type=pl.FP32)
            bias_row = pl.cast(pl.reshape(dt_bias[:], [1, h]), target_type=pl.FP32)
            decay_row = pl.neg(pl.exp(log_row))

            bf = pl.cast(b_proj[t0 : t0 + tok_tile, :], target_type=pl.FP32)
            # sigmoid, where recip is right: there is no numerator to fold into a divide
            beta_tile = pl.recip(pl.add(pl.exp(pl.neg(bf)), 1.0))
            beta[:, t0 : t0 + tok_tile] = pl.transpose(beta_tile, 0, 1)

            af = pl.cast(a_proj[t0 : t0 + tok_tile, :], target_type=pl.FP32)
            shifted = pl.col_expand_add(af, bias_row)
            # softplus, guarded: log1p(exp(x)) overflows fp32 well above the model's
            # range, and above SOFTPLUS_LINEAR the two agree to fp32 anyway, so take
            # the larger of the two branches instead of choosing between them
            soft = pl.log(pl.add(pl.exp(pl.minimum(shifted, SOFTPLUS_LINEAR)), 1.0))
            soft = pl.maximum(soft, shifted)
            g[t0 : t0 + tok_tile, :] = pl.col_expand_mul(soft, decay_row)
        return q_out

    return gdn_qk_norm_gate


gdn_qk_norm_gate = build_kernel()

OUTPUTS = ("q_out", "k_out", "beta", "g")       # bench.py reads it


def golden_parts(q, k, a_proj, b_proj, a_log, dt_bias, d: int = D, eps: float = EPS):
    """The kernel's arithmetic in fp32, in its order."""
    import torch

    qf = q.float()
    q_out = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + eps) * (d ** -0.5)
    kf = k.float()
    k_out = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + eps)
    beta = torch.reciprocal(torch.exp(-b_proj.float()) + 1.0)
    shifted = a_proj.float() + dt_bias.float()
    soft = torch.maximum(torch.log(torch.exp(shifted.clamp(max=SOFTPLUS_LINEAR)) + 1.0), shifted)
    g = soft * -torch.exp(a_log.float())
    return q_out, k_out, beta, g


def golden_gdn_qk_norm_gate(tensors):
    import torch

    q_out, k_out, beta, g = golden_parts(tensors["q"], tensors["k"], tensors["a_proj"],
                                         tensors["b_proj"], tensors["a_log"], tensors["dt_bias"])
    tensors["q_out"].copy_(q_out.to(torch.float16))
    tensors["k_out"].copy_(k_out.to(torch.float16))
    tensors["beta"].copy_(beta.t())
    tensors["g"].copy_(g)


def _split(which: int, t: int, hg: int, d: int, chunk: int, weights: str | None):
    """q (0) or k (1) out of the conv output: the chain keeps only their normed forms."""
    import torch

    import reference

    src = reference.block_inputs("qkv_conv", t, QWEN3_8_27B, chunk, weights=weights)
    width = hg * d

    def load():
        col = which * width
        return src()[:, col : col + width].reshape(t, hg, d).to(torch.bfloat16)

    return load


def _proj(key: str, t: int, chunk: int, weights: str | None):
    import torch

    import reference

    src = reference.block_inputs(key, t, QWEN3_8_27B, chunk, weights=weights)
    return lambda: src().to(torch.bfloat16)


def _param(name: str, weights: str | None):
    import torch

    import reference

    def load():
        w = (reference.make_block_weights(QWEN3_8_27B) if weights is None
             else torch.load(weights, weights_only=True))
        return w[name].to(torch.bfloat16)

    return load


def build_tensor_specs(t: int = T, h: int = H, hg: int = HG, d: int = D,
                       chunk: int = 128, weights: str | None = None):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("q", [t, hg, d], torch.bfloat16, init_value=_split(0, t, hg, d, chunk, weights)),
        TensorSpec("k", [t, hg, d], torch.bfloat16, init_value=_split(1, t, hg, d, chunk, weights)),
        TensorSpec("a_proj", [t, h], torch.bfloat16, init_value=_proj("a_proj", t, chunk, weights)),
        TensorSpec("b_proj", [t, h], torch.bfloat16, init_value=_proj("b_proj", t, chunk, weights)),
        TensorSpec("a_log", [h], torch.bfloat16, init_value=_param("A_log", weights)),
        TensorSpec("dt_bias", [h], torch.bfloat16, init_value=_param("dt_bias", weights)),
        TensorSpec("q_out", [t, hg, d], torch.float16),
        TensorSpec("k_out", [t, hg, d], torch.float16),
        TensorSpec("beta", [h, t], torch.float32),
        TensorSpec("g", [t, h], torch.float32),
    ]


def compare_against_reference(t: int, chunk: int, weights: str | None):
    """Report each output's distance from the float64 reference beside the golden gate."""
    import reference

    def make(key, transform=None):
        def compare(actual, expected, **kw):
            from golden import ratio_allclose

            gate = ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)
            ok, detail = gate(actual, expected, **kw)
            ref = reference.block_inputs(key, t, QWEN3_8_27B, chunk, weights=weights)()
            if transform is not None:
                ref = transform(ref)
            dev = actual.double().reshape(ref.shape)
            print(f"[stats] {key}: max abs {float((dev - ref).abs().max()):.3e}, "
                  f"rel frob {float((dev - ref).norm() / ref.norm()):.3e} vs float64", flush=True)
            return ok, detail

        return compare

    return {"q_out": make("q"), "k_out": make("k"),
            "beta": make("beta", lambda x: x.t().contiguous()), "g": make("g")}


if __name__ == "__main__":
    import argparse
    import os
    import statistics

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=T)
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
    parser.add_argument("--row-tile", type=int, default=ROW_TILE)
    parser.add_argument("--group", type=int, default=GROUP)
    parser.add_argument("--tok-tile", type=int, default=TOK_TILE)
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the kernel over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    result = run(
        fn=build_kernel(t=args.seq_len, row_tile=args.row_tile, group=args.group,
                        tok_tile=args.tok_tile),
        specs=build_tensor_specs(t=args.seq_len, weights=args.weights),
        golden_fn=golden_gdn_qk_norm_gate,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn=compare_against_reference(args.seq_len, 128, args.weights),
        compile_only=args.compile_only,
    )
    if args.rounds and result.bench is not None:
        samples = [s for s in result.bench.per_round("effective") if s > 0]
        if samples:
            print(f"[bench] qk_norm_gate T={args.seq_len} row={args.row_tile} grp={args.group} "
                  f"tok={args.tok_tile} rounds={len(samples)} "
                  f"mean={statistics.fmean(samples):.1f} us "
                  f"median={statistics.median(samples):.1f} min={min(samples):.1f}", flush=True)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
