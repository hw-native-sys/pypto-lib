# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The Gated DeltaNet gate at decode: the two per-head scalars the recurrence reads.

    beta = sigmoid(b_proj)                              [T, H]
    g    = -exp(A_log) * softplus(a_proj + dt_bias)     [T, H], <= 0

This is `qk_norm_gate`'s second pass and nothing else. Two differences from it:

* **`beta` comes out `[T, H]`, not `[H, T]`.** `decode_delta_rule` reads a token's row of
  per-head betas; the prefill chunk path reads a head's row of per-token betas and
  `qk_norm_gate` transposes on the way out.
* **The q/k L2 norm is not here.** `decode_delta_rule` normalises q and k itself, inside
  the task that consumes them.

The arithmetic is `qk_norm_gate`'s, unchanged, including the guarded softplus:
`log1p(exp(x))` overflows fp32 near x = 88 and the model's own `dt_bias` already reaches
19.25, so above `SOFTPLUS_LINEAR` take the larger of the two branches rather than
choosing between them.

One task per `tok_tile` tokens, and the tile defaults to the padded decode batch.
"""
import pypto.language as pl

from config import QWEN3_8_27B
from decode_projection import M_FRACTAL

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # decode tokens, one per sequence

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads: a, b, beta and g

# softplus(x) = log1p(exp(x)) overflows fp32 at x ~ 88 and is exactly x well before that;
# the model's own dt_bias already reaches 19.25.
SOFTPLUS_LINEAR = 20.0
# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True


# validation
# one padded decode batch, then two.
DEFAULT_CASES = (16, 32)

# tiling
TOK_TILE = M_FRACTAL    # tokens per task; the padded decode batch is a multiple of it


def build_kernel(tok_tile: int = TOK_TILE, inline: bool = False):
    """The gate at a given token tile; `t` must be a multiple of `tok_tile`."""

    @(pl.jit.inline if inline else pl.jit)
    def _gdn_decode_gate(
        a_proj: pl.Tensor[[T_DYN, H], pl.BF16],
        b_proj: pl.Tensor[[T_DYN, H], pl.BF16],
        a_log: pl.Tensor[[H], pl.BF16],
        dt_bias: pl.Tensor[[H], pl.BF16],
        beta: pl.Out[pl.Tensor[[T_DYN, H], pl.FP32]],
        g: pl.Out[pl.Tensor[[T_DYN, H], pl.FP32]],
    ):
        """The token count must be a multiple of tok_tile; a block walks one whole tile."""
        a_proj.bind_dynamic(0, T_DYN)
        b_proj.bind_dynamic(0, T_DYN)
        beta.bind_dynamic(0, T_DYN)
        g.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(a_proj, 0)
        for blk in pl.spmd(t_dim // tok_tile, name_hint="decode_gate",
                           allow_early_resolve=EARLY_RESOLVE):
            t0 = blk * tok_tile
            log_row = pl.cast(pl.reshape(a_log[:], [1, H]), target_type=pl.FP32)
            bias_row = pl.cast(pl.reshape(dt_bias[:], [1, H]), target_type=pl.FP32)
            decay_row = pl.neg(pl.exp(log_row))

            # sigmoid, where recip is right: there is no numerator to fold into a divide
            bf = pl.cast(b_proj[t0 : t0 + tok_tile, :], target_type=pl.FP32)
            beta[t0 : t0 + tok_tile, :] = pl.recip(pl.add(pl.exp(pl.neg(bf)), 1.0))

            af = pl.cast(a_proj[t0 : t0 + tok_tile, :], target_type=pl.FP32)
            shifted = pl.col_expand_add(af, bias_row)
            soft = pl.log(pl.add(pl.exp(pl.minimum(shifted, SOFTPLUS_LINEAR)), 1.0))
            soft = pl.maximum(soft, shifted)
            g[t0 : t0 + tok_tile, :] = pl.col_expand_mul(soft, decay_row)
        return beta, g

    return _gdn_decode_gate


gdn_decode_gate = build_kernel(inline=True)
gdn_decode_gate_test = build_kernel()


def build_tensor_specs(t: int = M_FRACTAL, h: int = H):
    import torch
    from golden import TensorSpec

    def draw(seed, *shape, scale=1.0):
        gen = torch.Generator().manual_seed(seed)
        return lambda: (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    return [
        TensorSpec("a_proj", [t, h], torch.bfloat16, init_value=draw(42, t, h)),
        TensorSpec("b_proj", [t, h], torch.bfloat16, init_value=draw(43, t, h)),
        TensorSpec("a_log", [h], torch.bfloat16, init_value=draw(44, h)),
        TensorSpec("dt_bias", [h], torch.bfloat16, init_value=draw(45, h)),
        TensorSpec("beta", [t, h], torch.float32),
        TensorSpec("g", [t, h], torch.float32),
    ]


def golden_gdn_decode_gate(tensors):
    import torch

    import reference

    # The shared oracle, not a local copy of the two formulas: `qk_norm_gate` also
    # normalises q and k, which this kernel does not compute, so they are passed as a
    # one-element dummy head and the returned pair dropped. Gating both kernels against
    # one reference is what makes a drift in it visible in both.
    dummy = torch.ones(tensors["a_proj"].shape[0], 1, 1, dtype=torch.float32)
    _, _, beta_ref, g_ref = reference.qk_norm_gate(
        dummy, dummy, tensors["a_proj"], tensors["b_proj"],
        tensors["a_log"], tensors["dt_bias"],
    )
    tensors["beta"].copy_(beta_ref.to(torch.float32))
    tensors["g"].copy_(g_ref.to(torch.float32))


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.batch] if args.batch else list(DEFAULT_CASES)
    if args.batch and args.batch % TOK_TILE:
        parser.error(f"--batch must be a multiple of TOK_TILE={TOK_TILE}")

    for batch in cases:
        print(f"[case] decode_gate batch={batch}", flush=True)
        result = run(
            fn=gdn_decode_gate_test,
            specs=build_tensor_specs(t=batch),
            golden_fn=golden_gdn_decode_gate,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device),
            rtol=1e-2,
            atol=1e-5,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at batch={batch}")
