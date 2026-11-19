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

One matmul against a `[96, 5120]` weight built by stacking the two.

The tile is a `build_kernel` argument because the task count is a quotient of it, and
at a decode batch the prefill value makes that quotient zero. `decode_vector.py`
carries the decode value.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
K = QWEN3_8_27B.hidden_size                 # contraction: the block's input width
H = QWEN3_8_27B.linear_num_value_heads      # one scalar per value head, twice over

MODULES = ("in_proj_a", "in_proj_b")
# validation
# one M_TILE of tokens, then two. The token count must be a multiple of 128.
DEFAULT_CASES = (128, 256)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
M_TILE = 128            # token rows per block
K_TILE = 256            # contraction step; bf16 halves the useful K tile



AB_WIDTH = 2 * H


def build_kernel(m_tile: int = M_TILE, k_tile: int = K_TILE, inline: bool = False):
    """The a/b projection at a given token tile; `t` must be a multiple of `m_tile`."""
    if k_tile <= 0 or K % k_tile:
        raise ValueError(f"k_tile must be a positive divisor of {K}")

    @(pl.jit.inline if inline else pl.jit)
    def _gdn_in_proj_ab(
        x: pl.Tensor[[T_DYN, K], pl.BF16],
        w_ab: pl.Tensor[[AB_WIDTH, K], pl.BF16],
        a_proj: pl.Out[pl.Tensor[[T_DYN, H], pl.BF16]],
        b_proj: pl.Out[pl.Tensor[[T_DYN, H], pl.BF16]],
    ):
        """The token count must be a multiple of m_tile; a block walks one whole tile."""
        x.bind_dynamic(0, T_DYN)
        a_proj.bind_dynamic(0, T_DYN)
        b_proj.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(x, 0)
        # slot_num=1: the ring is sized by the tile crossing cube to vector, and
        # two slots of it do not fit the vector buffer (pypto#2903).
        for blk in pl.spmd(t_dim // m_tile, name_hint="in_proj_ab",
                           optimizations=[pl.cross_core_slot(slot_num=1)]):
            m0 = blk * m_tile
            acc = pl.matmul(x[m0 : m0 + m_tile, 0:k_tile], w_ab[:, 0:k_tile],
                            b_trans=True, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, K // k_tile, stage=2):
                k0 = kb * k_tile
                acc = pl.matmul_acc(acc, x[m0 : m0 + m_tile, k0 : k0 + k_tile],
                                    w_ab[:, k0 : k0 + k_tile], b_trans=True)
            a_proj[m0 : m0 + m_tile, :] = pl.cast(acc[:, 0:H], target_type=pl.BF16,
                                                  mode="rint")
            b_proj[m0 : m0 + m_tile, :] = pl.cast(acc[:, H:AB_WIDTH], target_type=pl.BF16,
                                                  mode="rint")
        return a_proj, b_proj

    return _gdn_in_proj_ab


gdn_in_proj_ab = build_kernel(inline=True)
gdn_in_proj_ab_test = build_kernel()


def golden_fn(tensors):
    import torch

    from reference import host_torch_setup

    host_torch_setup()
    h = tensors["a_proj"].shape[1]
    acc = tensors["x"].float() @ tensors["w_ab"].float().t()
    tensors["a_proj"].copy_(acc[:, :h].to(torch.bfloat16))
    tensors["b_proj"].copy_(acc[:, h:].to(torch.bfloat16))


def _stacked_weight(weights: str | None):
    """`in_proj_a.weight` on top of `in_proj_b.weight`: the [2H, K] the kernel reads."""
    from weights import load_weights, stacked_ab_weight

    def load():
        w = load_weights(QWEN3_8_27B, weights)
        return stacked_ab_weight(w)

    return load


def build_tensor_specs(t: int = BENCH_SEQ, k: int = K, h: int = H, weights: str | None = None,
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

    import reference
    from weights import load_weights

    module = "in_proj_a" if name == "a_proj" else "in_proj_b"
    gate = ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)

    def compare_fn(actual, expected, **kw):
        ok, detail = gate(actual, expected, **kw)
        if ok:
            return True, ""
        reference.host_torch_setup()
        w = load_weights(QWEN3_8_27B, weights)[module + ".weight"]
        ref = reference.linear(reference.make_block_inputs(t, QWEN3_8_27B), w)
        dev = actual.double()
        head = (f"    {name}: rel frob {float((dev - ref).norm() / ref.norm()):.3e} "
                  f"vs the float64 projection")
        return False, f"{head}\n{detail}"

    return compare_fn


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
    parser.add_argument("--k-tile", type=int, default=K_TILE)
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
    if args.seq_len and args.seq_len % M_TILE:
        parser.error(f"--seq-len must be a multiple of M_TILE={M_TILE}")

    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    kernel = gdn_in_proj_ab_test if args.k_tile == K_TILE else build_kernel(k_tile=args.k_tile)
    for t in cases:
        print(f"[case] in_proj_ab T={t}", flush=True)
        result = run(
            fn=kernel,
            specs=build_tensor_specs(t=t, weights=args.weights),
            golden_fn=golden_fn,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-5,
            compare_fn={"a_proj": compare("a_proj", t, args.weights),
                        "b_proj": compare("b_proj", t, args.weights)},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
