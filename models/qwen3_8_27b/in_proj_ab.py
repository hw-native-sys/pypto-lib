# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""`in_proj_a` and `in_proj_b`, fused: the two per-head scalars the gate is made of.

    a_proj, b_proj = split(x @ W_ab^T)     [T, 5120] -> two [T, 48], bf16 throughout

The two projections share an input and differ only in their weight rows, so
they are one matmul against a `[96, 5120]` weight built by stacking them. N = 48
alone is legal under the multiple-of-16 tile rule but a poor cube tile, and two
passes over x would cost twice what one does.

bf16, not A8W8: at 0.5 MB of weights and 0.4% of the block's multiply-adds
there is nothing to gain, and INT8 here was measured to move the block output
by 3e-5 (`test_block_reference.py --quant`). The kernel is memory-bound on x.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# model shape
K = QWEN3_8_27B.hidden_size                 # contraction: the block's input width
H = QWEN3_8_27B.linear_num_value_heads      # one scalar per value head, twice over

MODULES = ("in_proj_a", "in_proj_b")

# case shape
T = 8192                # tokens (single sequence, B = 1)

# tiling
M_TILE = 128            # token rows per block
K_TILE = 256            # contraction step; bf16 halves the useful K tile

OUTPUTS = ("a_proj", "b_proj")      # bench.py reads it


def build_kernel(t: int = T, k: int = K, h: int = H, inline: bool = False,
                 m_tile: int = M_TILE, k_tile: int = K_TILE):
    """The operator at one shape. `inline` makes it a callee for `gdn_block`."""
    if t % m_tile or k % k_tile:
        raise ValueError(f"t={t} must be a multiple of m_tile={m_tile} and "
                         f"k={k} of k_tile={k_tile}")

    @(pl.jit.inline if inline else pl.jit)
    def gdn_in_proj_ab(
        x: pl.Tensor[[t, k], pl.BF16],
        w_ab: pl.Tensor[[2 * h, k], pl.BF16],
        a_proj: pl.Out[pl.Tensor[[t, h], pl.BF16]],
        b_proj: pl.Out[pl.Tensor[[t, h], pl.BF16]],
    ):
        # slot_num=1 for the same reason as a8w8_linear: the ring is sized by the
        # tile crossing cube to vector, and the default depth of two does not fit.
        for blk in pl.spmd(t // m_tile, name_hint="in_proj_ab",
                           optimizations=[pl.cross_core_slot(slot_num=1)]):
            m0 = blk * m_tile
            acc = pl.matmul(x[m0 : m0 + m_tile, 0:k_tile], w_ab[:, 0:k_tile],
                            b_trans=True, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, k // k_tile, stage=2):
                k0 = kb * k_tile
                acc = pl.matmul_acc(acc, x[m0 : m0 + m_tile, k0 : k0 + k_tile],
                                    w_ab[:, k0 : k0 + k_tile], b_trans=True)
            a_proj[m0 : m0 + m_tile, :] = pl.cast(acc[:, 0:h], target_type=pl.BF16,
                                                  mode="rint")
            b_proj[m0 : m0 + m_tile, :] = pl.cast(acc[:, h : 2 * h], target_type=pl.BF16,
                                                  mode="rint")
        return a_proj, b_proj

    return gdn_in_proj_ab


gdn_in_proj_ab = build_kernel()


def golden_fn(tensors):
    import torch

    import a8w8_linear

    a8w8_linear.host_torch_setup()
    h = tensors["a_proj"].shape[1]
    acc = tensors["x"].float() @ tensors["w_ab"].float().t()
    tensors["a_proj"].copy_(acc[:, :h].to(torch.bfloat16))
    tensors["b_proj"].copy_(acc[:, h:].to(torch.bfloat16))


def _stacked_weight(weights: str | None):
    """`in_proj_a.weight` on top of `in_proj_b.weight`: the [2H, K] the kernel reads."""
    import torch

    import a8w8_linear

    def load():
        w = a8w8_linear.load_weights(QWEN3_8_27B, weights)
        return torch.cat([w[m + ".weight"] for m in MODULES])

    return load


def build_tensor_specs(t: int = T, k: int = K, h: int = H, weights: str | None = None,
                       **_):
    import torch
    from golden import TensorSpec

    import reference

    return [
        TensorSpec("x", [t, k], torch.bfloat16,
                   init_value=lambda: reference.make_block_inputs(t, QWEN3_8_27B)),
        TensorSpec("w_ab", [2 * h, k], torch.bfloat16, init_value=_stacked_weight(weights)),
        TensorSpec("a_proj", [t, h], torch.bfloat16),
        TensorSpec("b_proj", [t, h], torch.bfloat16),
    ]


def compare(name: str, t: int, weights: str | None):
    """The bf16 gate, plus the distance from the float64 projection.

    Nothing is quantized here, so there is one reference and not three: the gap
    is bf16 rounding on the output and nothing else.
    """
    from golden import ratio_allclose

    import a8w8_linear
    import reference

    module = "in_proj_a" if name == "a_proj" else "in_proj_b"
    gate = ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)

    def compare_fn(actual, expected, **kw):
        ok, detail = gate(actual, expected, **kw)
        a8w8_linear.host_torch_setup()
        w = a8w8_linear.load_weights(QWEN3_8_27B, weights)[module + ".weight"]
        ref = reference.linear(reference.make_block_inputs(t, QWEN3_8_27B), w)
        dev = actual.double()
        print(f"[stats] {name}: rel frob {float((dev - ref).norm() / ref.norm()):.3e} "
              f"vs the float64 projection", flush=True)
        return ok, detail

    return compare_fn


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
    parser.add_argument("--m-tile", type=int, default=M_TILE)
    parser.add_argument("--k-tile", type=int, default=K_TILE)
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
        fn=build_kernel(t=args.seq_len, m_tile=args.m_tile, k_tile=args.k_tile),
        specs=build_tensor_specs(t=args.seq_len, weights=args.weights),
        golden_fn=golden_fn,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn={"a_proj": compare("a_proj", args.seq_len, args.weights),
                    "b_proj": compare("b_proj", args.seq_len, args.weights)},
        compile_only=args.compile_only,
    )
    if args.rounds and result.bench is not None:
        samples = [s for s in result.bench.per_round("effective") if s > 0]
        if samples:
            print(f"[bench] in_proj_ab M={args.seq_len} K={K} N={2 * H} "
                  f"m={args.m_tile} k={args.k_tile} rounds={len(samples)} "
                  f"mean={statistics.fmean(samples):.1f} us "
                  f"median={statistics.median(samples):.1f} min={min(samples):.1f}",
                  flush=True)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
