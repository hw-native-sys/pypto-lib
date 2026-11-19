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

# validation
DEFAULT_TEST_SEQ_LEN = 512

# tiling
TOK_TILE = 16           # output rows per pipelined step
CH_TILE = 256           # channels per block
GROUP = 16              # row-tiles per block, the pipelined inner loop


def build_kernel(t: int = DEFAULT_TEST_SEQ_LEN, c: int = C, k: int = K, inline: bool = False,
                 tok_tile: int = TOK_TILE, ch_tile: int = CH_TILE, group: int = GROUP,
                 c_off: int = 0, c_out: int | None = None, out_heads: int | None = None,
                 out_dtype=pl.BF16, name: str = "short_conv"):
    """The operator at one shape. `inline` makes it a callee for `gdn_block`.

    A block walks `group` row-tiles of one channel tile, so there is an inner
    loop to pipeline; one tile per block leaves the loads nothing to hide behind.

    `c_off` and `c_out` convolve one contiguous run of channels instead of all
    of them, and `out_heads` declares the output `[t, out_heads, d]` rather than
    `[t, c_out]`. The block builds this three times, once per consumer: q and k
    go to the qk-norm and v goes straight to the delta rule, which wants it in
    fp16, and each consumer reads the rows of one contiguous tensor. Splitting
    at the write is what makes that layout free -- the alternative is one wide
    output that every consumer then has to gather from.
    """
    c_out = c if c_out is None else c_out
    rows = tok_tile * group
    # A tile that does not divide its axis silently leaves the tail unwritten --
    # the kernel returns a correct-looking result missing t % rows tokens.
    if t % rows or c_out % ch_tile:
        raise ValueError(f"t={t} must be a multiple of tok_tile*group={rows} and "
                         f"c_out={c_out} a multiple of ch_tile={ch_tile}")
    if c_off + c_out > c:
        raise ValueError(f"channels {c_off}..{c_off + c_out} run past c={c}")
    if out_heads is not None and c_out % out_heads:
        raise ValueError(f"c_out={c_out} must divide by out_heads={out_heads}")
    out_shape = [t, c_out] if out_heads is None else [t, out_heads, c_out // out_heads]

    @(pl.jit.inline if inline else pl.jit)
    def gdn_short_conv(
        x_pad: pl.Tensor[[t + k - 1, c], pl.BF16],
        w: pl.Tensor[[k, c], pl.FP32],
        y: pl.Out[pl.Tensor[out_shape, out_dtype]],
    ):
        y_flat = pl.reshape(y, [t, c_out])
        for blk in pl.spmd((t // rows) * (c_out // ch_tile), name_hint=name):
            r0 = (blk // (c_out // ch_tile)) * rows
            yc = (blk % (c_out // ch_tile)) * ch_tile
            c0 = c_off + yc
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
                y_flat[t0 : t0 + tok_tile, yc : yc + ch_tile] = pl.cast(
                    pl.div(acc, den), target_type=out_dtype, mode="rint")
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


def short_conv_ref(x, weight):
    """Float64 causal convolution and SiLU on unpadded tokens and checkpoint taps."""
    import torch
    from torch.nn.functional import silu

    from reference import REF_DTYPE

    t, c = x.shape
    k = weight.shape[-1]
    xf = x.to(REF_DTYPE)
    wf = weight.reshape(c, k).to(REF_DTYPE)
    padded = torch.cat([torch.zeros(k - 1, c, dtype=REF_DTYPE), xf])
    out = torch.zeros(t, c, dtype=REF_DTYPE)
    for j in range(k):
        out += padded[j : j + t] * wf[:, j]
    return silu(out)


def build_tensor_specs(t: int = DEFAULT_TEST_SEQ_LEN, c: int = C, k: int = K, weights: str | None = None):
    import torch
    from golden import TensorSpec

    from weights import load_layer

    def padded_input():
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(t, c, generator=generator).to(torch.bfloat16)
        return torch.cat([torch.zeros(k - 1, c, dtype=x.dtype), x])

    def taps():
        if weights is None:
            generator = torch.Generator().manual_seed(47)
            w = ((2 * torch.rand(c, k, generator=generator) - 1) * k ** -0.5).to(torch.bfloat16)
        else:
            w = load_layer(weights)["conv1d.weight"].reshape(c, k)
        return w.t().contiguous().float()

    return [
        TensorSpec("x_pad", [t + k - 1, c], torch.bfloat16, init_value=padded_input),
        TensorSpec("w", [k, c], torch.float32, init_value=taps),
        TensorSpec("y", [t, c], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    import os
    import statistics

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_TEST_SEQ_LEN)
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
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn={"y": ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)},
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
