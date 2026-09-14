# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet short convolution: depthwise causal conv over the token axis, then silu.

    y[t, c] = silu(sum_j w[j, c] * x_pad[t + j, c])        j = 0 .. K-1, fp32 accumulate

`x_pad` carries the K-1 tokens preceding the sequence in its first rows, so
`x_pad[t + j]` is `x[t - (K - 1) + j]` and tap K-1 reads the token itself. For
prefill those rows are zero; they are where a decode cache would go. Passing
them in the tensor is what keeps every tap offset non-negative, so no block
needs a head-padding special case. No bias: the model's conv1d has none.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# model shape
C = QWEN3_8_27B.qkv_width                   # q and k at Hg heads, v at H
K = QWEN3_8_27B.linear_conv_kernel_dim      # taps, newest last

# case shape
T = 8192                # tokens (single sequence, B = 1)

# tiling
TOK_TILE = 16           # output rows per pipelined step
CH_TILE = 256           # channels per block
GROUP = 16              # row-tiles per block, the pipelined inner loop


def build_kernel(t: int = T, c: int = C, k: int = K, inline: bool = False,
                 tok_tile: int = TOK_TILE, ch_tile: int = CH_TILE, group: int = GROUP):
    """The operator at one shape. `inline` makes it a callee for `gdn_block`.

    A block walks `group` row-tiles of one channel tile, so there is an inner
    loop to pipeline; one tile per block leaves the loads nothing to hide behind.
    """
    rows = tok_tile * group
    # A tile that does not divide its axis silently leaves the tail unwritten --
    # the kernel returns a correct-looking result missing t % rows tokens.
    if t % rows or c % ch_tile:
        raise ValueError(f"t={t} must be a multiple of tok_tile*group={rows} and "
                         f"c={c} a multiple of ch_tile={ch_tile}")

    @(pl.jit.inline if inline else pl.jit)
    def gdn_short_conv(
        x_pad: pl.Tensor[[t + k - 1, c], pl.BF16],
        w: pl.Tensor[[k, c], pl.FP32],
        y: pl.Out[pl.Tensor[[t, c], pl.BF16]],
    ):
        for blk in pl.spmd((t // rows) * (c // ch_tile), name_hint="short_conv"):
            r0 = (blk // (c // ch_tile)) * rows
            c0 = (blk % (c // ch_tile)) * ch_tile
            w_taps = w[:, c0 : c0 + ch_tile]
            for i in pl.pipeline(group, stage=2):
                t0 = r0 + i * tok_tile
                acc = pl.full([tok_tile, ch_tile], dtype=pl.FP32, value=0.0)
                # one load per tap; the windows overlap by tok_tile - 1 rows
                for j in pl.unroll(k):
                    src = pl.cast(x_pad[t0 + j : t0 + j + tok_tile, c0 : c0 + ch_tile],
                                  target_type=pl.FP32)
                    acc = pl.add(acc, pl.col_expand_mul(src, w_taps[j : j + 1, :]))
                # silu as acc / (1 + exp(-acc)): one pass fewer than recip + mul,
                # which is 11% of this kernel. The coding guide prefers recip + mul
                # on hot paths, and that holds when the reciprocal is reused; here
                # it is used once, so the divide simply removes an op.
                den = pl.add(pl.exp(pl.neg(acc)), 1.0)
                y[t0 : t0 + tok_tile, c0 : c0 + ch_tile] = pl.cast(pl.div(acc, den),
                                                                   target_type=pl.BF16, mode="rint")
        return y

    return gdn_short_conv


gdn_short_conv = build_kernel()

OUTPUTS = ("y",)                    # bench.py reads it


def golden_y(x_pad, w, k: int = K):
    """The kernel's arithmetic in fp32, in its order. -> fp32 [T, C]."""
    import torch

    t = x_pad.shape[0] - (k - 1)
    xf = x_pad.float()
    wf = w.float()
    acc = sum(xf[j : j + t] * wf[j] for j in range(k))
    return acc / (torch.exp(-acc) + 1.0)


def golden_gdn_short_conv(tensors):
    import torch

    y = golden_y(tensors["x_pad"], tensors["w"])
    tensors["y"].copy_(y.to(torch.bfloat16))


def _pad_input(t: int, c: int, k: int, chunk: int, weights: str | None):
    """The projection output with k-1 zero rows in front, bf16. -> [T + k - 1, C]."""
    import torch

    import reference

    src = reference.block_inputs("qkv", t, QWEN3_8_27B, chunk, weights=weights)

    def load():
        x = src().to(torch.bfloat16)
        return torch.cat([torch.zeros(k - 1, c, dtype=torch.bfloat16), x])

    return load


def _taps(c: int, k: int, weights: str | None):
    """`conv1d.weight` as [K, C] fp32: a row per tap, which is what the tile loop reads."""
    import torch

    import reference

    def load():
        if weights is None:
            w = reference.make_block_weights(QWEN3_8_27B)["conv1d.weight"]
        else:
            w = torch.load(weights, weights_only=True)["conv1d.weight"]
        return w.reshape(c, k).t().contiguous().float()

    return load


def build_tensor_specs(t: int = T, c: int = C, k: int = K, chunk: int = 128,
                       weights: str | None = None):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x_pad", [t + k - 1, c], torch.bfloat16, init_value=_pad_input(t, c, k, chunk, weights)),
        TensorSpec("w", [k, c], torch.float32, init_value=_taps(c, k, weights)),
        TensorSpec("y", [t, c], torch.bfloat16),
    ]


def compare_y(t: int, chunk: int, weights: str | None):
    """The bf16 gate, plus the distance from the float64 reference the roadmap set."""
    from golden import ratio_allclose

    import reference

    gate = ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)

    def compare(actual, expected, **kw):
        ok, detail = gate(actual, expected, **kw)
        ref = reference.block_inputs("qkv_conv", t, QWEN3_8_27B, chunk, weights=weights)()
        dev = actual.double()
        max_abs = float((dev - ref).abs().max())
        frob = float((dev - ref).norm() / ref.norm())
        print(f"[stats] y vs float64 reference: max abs {max_abs:.3e} (bf16 tol 6e-2), "
              f"rel frob {frob:.3e}; vs golden max abs {float((dev - expected.double()).abs().max()):.3e}",
              flush=True)
        return ok, detail

    return compare


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
    parser.add_argument("--tok-tile", type=int, default=TOK_TILE)
    parser.add_argument("--ch-tile", type=int, default=CH_TILE)
    parser.add_argument("--group", type=int, default=GROUP)
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
        fn=build_kernel(t=args.seq_len, tok_tile=args.tok_tile, ch_tile=args.ch_tile, group=args.group),
        specs=build_tensor_specs(t=args.seq_len, weights=args.weights),
        golden_fn=golden_gdn_short_conv,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn={"y": compare_y(args.seq_len, 128, args.weights)},
        compile_only=args.compile_only,
    )
    if args.rounds and result.bench is not None:
        samples = [s for s in result.bench.per_round("effective") if s > 0]
        if samples:
            print(f"[bench] short_conv T={args.seq_len} C={C} "
                  f"tok={args.tok_tile} ch={args.ch_tile} grp={args.group} "
                  f"rounds={len(samples)} mean={statistics.fmean(samples):.1f} us "
                  f"median={statistics.median(samples):.1f} min={min(samples):.1f}", flush=True)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
