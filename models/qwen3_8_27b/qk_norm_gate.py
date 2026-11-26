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

The eps sits inside the sqrt, not outside it. fp16 out for q and k, fp32 for
beta and g.

The tile is a `build_kernel` argument because the task count is a quotient of it, and
at a decode batch the prefill value makes that quotient zero. `decode_vector.py`
carries the decode value.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads: a, b, beta and g
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads: q and k
D = QWEN3_8_27B.linear_value_head_dim       # head dimension, the norm's axis
EPS = 1e-6                                  # inside the sqrt
# validation
# one TOK_TILE of the gate, then two. The token count must be a multiple of 256.
DEFAULT_CASES = (256, 512)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
ROW_TILE = 64           # (token, head) rows per pipelined step of the norm
GROUP = 8               # row-tiles per block, the pipelined inner loop
TOK_TILE = 256          # tokens per block of the gate

# softplus(x) = log1p(exp(x)) overflows fp32 at x ~ 88 and is exactly x well
# before that; the model's own dt_bias already reaches 19.25.
SOFTPLUS_LINEAR = 20.0


SCALE = D ** -0.5


def build_kernel(row_tile: int = ROW_TILE, group: int = GROUP, tok_tile: int = TOK_TILE,
                 inline: bool = False):
    """Norm and gate at given tiles; `t * HG` must divide by `row_tile * group` and `t` by `tok_tile`."""

    rows_tile = row_tile * group            # (token, head) rows one block walks

    @(pl.jit.inline if inline else pl.jit)
    def _gdn_qk_norm_gate(
        q: pl.Tensor[[T_DYN, HG, D], pl.BF16],
        k: pl.Tensor[[T_DYN, HG, D], pl.BF16],
        a_proj: pl.Tensor[[T_DYN, H], pl.BF16],
        b_proj: pl.Tensor[[T_DYN, H], pl.BF16],
        a_log: pl.Tensor[[H], pl.BF16],
        dt_bias: pl.Tensor[[H], pl.BF16],
        q_out: pl.Out[pl.Tensor[[T_DYN, HG, D], pl.FP16]],
        k_out: pl.Out[pl.Tensor[[T_DYN, HG, D], pl.FP16]],
        beta: pl.Out[pl.Tensor[[H, T_DYN], pl.FP32]],
        g: pl.Out[pl.Tensor[[T_DYN, H], pl.FP32]],
    ):
        """T * HG must be a multiple of rows_tile, and T of tok_tile."""
        q.bind_dynamic(0, T_DYN)
        k.bind_dynamic(0, T_DYN)
        a_proj.bind_dynamic(0, T_DYN)
        b_proj.bind_dynamic(0, T_DYN)
        q_out.bind_dynamic(0, T_DYN)
        k_out.bind_dynamic(0, T_DYN)
        beta.bind_dynamic(1, T_DYN)
        g.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(q, 0)
        qk_rows = t_dim * HG
        q_rows = pl.reshape(q, [qk_rows, D])
        k_rows = pl.reshape(k, [qk_rows, D])
        qo_rows = pl.reshape(q_out, [qk_rows, D])
        ko_rows = pl.reshape(k_out, [qk_rows, D])
        for blk in pl.spmd(qk_rows // rows_tile, name_hint="qk_norm"):
            r00 = blk * rows_tile
            for i in pl.pipeline(group, stage=2):
                r0 = r00 + i * row_tile
                qf = pl.cast(q_rows[r0 : r0 + row_tile, :], target_type=pl.FP32)
                q_inv = pl.rsqrt(pl.add(pl.row_sum(pl.mul(qf, qf)), EPS), high_precision=True)
                q_normed = pl.mul(pl.row_expand_mul(qf, q_inv), SCALE)
                qo_rows[r0 : r0 + row_tile, :] = pl.cast(q_normed, target_type=pl.FP16, mode="rint")

                kf = pl.cast(k_rows[r0 : r0 + row_tile, :], target_type=pl.FP32)
                k_inv = pl.rsqrt(pl.add(pl.row_sum(pl.mul(kf, kf)), EPS), high_precision=True)
                k_normed = pl.row_expand_mul(kf, k_inv)
                ko_rows[r0 : r0 + row_tile, :] = pl.cast(k_normed, target_type=pl.FP16, mode="rint")

        for blk in pl.spmd(t_dim // tok_tile, name_hint="gate"):
            t0 = blk * tok_tile
            log_row = pl.cast(pl.reshape(a_log[:], [1, H]), target_type=pl.FP32)
            bias_row = pl.cast(pl.reshape(dt_bias[:], [1, H]), target_type=pl.FP32)
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

    return _gdn_qk_norm_gate


gdn_qk_norm_gate = build_kernel(inline=True)
gdn_qk_norm_gate_test = build_kernel()



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


def build_tensor_specs(t: int = BENCH_SEQ, h: int = H, hg: int = HG, d: int = D,
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
            if ok:
                return True, ""
            ref = reference.block_inputs(key, t, QWEN3_8_27B, chunk, weights=weights)()
            if transform is not None:
                ref = transform(ref)
            dev = actual.double().reshape(ref.shape)
            head = (f"    {key}: max abs {float((dev - ref).abs().max()):.3e}, "
                    f"rel frob {float((dev - ref).norm() / ref.norm()):.3e} vs float64")
            return False, f"{head}\n{detail}"

        return compare

    return {"q_out": make("q"), "k_out": make("k"),
            "beta": make("beta", lambda x: x.t().contiguous()), "g": make("g")}


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
    parser.add_argument("--group", type=int, default=GROUP)
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the kernel over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    rows_tile = ROW_TILE * args.group
    if args.group <= 0 or any(t <= 0 or (t * HG) % rows_tile or t % TOK_TILE for t in cases):
        parser.error(f"--seq-len*HG must be a positive multiple of {rows_tile} and --seq-len of {TOK_TILE}")

    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    kernel = gdn_qk_norm_gate_test if args.group == GROUP else build_kernel(group=args.group)
    for t in cases:
        print(f"[case] qk_norm_gate T={t}", flush=True)
        result = run(
            fn=kernel,
            specs=build_tensor_specs(t=t, weights=args.weights),
            golden_fn=golden_gdn_qk_norm_gate,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-5,
            compare_fn=compare_against_reference(t, 128, args.weights),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
