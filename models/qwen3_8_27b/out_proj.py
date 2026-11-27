# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GDN output projection with BF16 activations and weights and FP32 accumulation."""
from math import gcd

import pypto.language as pl

from config import QWEN3_8_27B

# model config
K = QWEN3_8_27B.value_width                 # contraction: H value heads of D
N = QWEN3_8_27B.hidden_size                 # back to the residual stream
# validation
DEFAULT_TEST_SEQ_LEN = 512

# tiling defaults; each instance overrides them
M_TILE = 128            # token rows per output tile
N_TILE = 128            # output channels per output tile
K_TILE = 256            # contraction step; 256 BF16 columns is one L2 line
M_GROUP = 8             # output tiles a block walks down the token axis
N_GROUP = 2             # and across the channel axis
K_STAGGER = 1           # rotate each block's K start; see `a8w8_linear`
RING_SLOTS = 2          # GM transport depth; the on-chip reserve stays at 1


def build_kernel(t: int = DEFAULT_TEST_SEQ_LEN, k: int = K, n: int = N,
                 inline: bool = False, m_tile: int = M_TILE, n_tile: int = N_TILE,
                 k_tile: int = K_TILE, m_group: int | None = None, n_group: int = N_GROUP,
                 *, name: str = "out_proj", k_stagger: int = K_STAGGER,
                 ring_slots: int = RING_SLOTS, workers: int = 0):
    """The projection at one shape.

    `workers` splits the block count from the patch count; see `a8w8_linear.build_kernel`.

    A block walks an `m_group` by `n_group` patch of output tiles rather than
    one, for the reasons `a8w8_linear` gives: it keeps the operands off the bus
    and it keeps the task count inside the runtime's ring heap. The inner loop
    takes two n-tiles against one activation tile, so `n_group` must be even.
    """
    if t % m_tile or n % n_tile or k % k_tile:
        raise ValueError(f"t={t} must be a multiple of m_tile={m_tile}, "
                         f"n={n} of n_tile={n_tile}, k={k} of k_tile={k_tile}")
    mb, nb = t // m_tile, n // n_tile
    if m_group is None:
        m_group = gcd(mb, M_GROUP)
    if mb % m_group or nb % n_group:
        raise ValueError(f"the {mb}x{nb} output tiles must divide by the patch "
                         f"{m_group}x{n_group}")
    if n_group % 2:
        raise ValueError(f"n_group={n_group} must be even: the inner loop pairs n-tiles")
    if workers < 0:
        raise ValueError(f"workers={workers} must be zero or a positive block count")
    mg, ng = mb // m_group, nb // n_group
    patches = mg * ng
    blocks = min(patches, workers) if workers else patches

    @(pl.jit.inline if inline else pl.jit)
    def gdn_out_proj(
        a: pl.Tensor[[t, k], pl.BF16],
        w: pl.Tensor[[n, k], pl.BF16],
        y: pl.Out[pl.Tensor[[t, n], pl.BF16]],
    ):
        # The tile crossing cube to vector is the FP32 accumulator, and two slots of
        # it do not fit the vector buffer (pypto#2903). `local_slot_num` holds that
        # reserve at one slot while `ring_slots` buys depth in GM, so the cube need
        # not wait for the vector to free a slot before pushing the next tile.
        for blk in pl.spmd(blocks, name_hint=name,
                           optimizations=[pl.cross_core_slot(slot_num=ring_slots,
                                                             local_slot_num=1)]):
            for patch in pl.range(blk, patches, blocks):
                pm = (patch % mg) * m_group
                pn = (patch // mg) * n_group
                for gm in pl.range(m_group):
                    m0 = (pm + gm) * m_tile
                    # n innermost: the activation tile is the one held across the
                    # inner loop, and it is the operand every tile in the row reads
                    for gn in pl.range(n_group // 2):
                        n0 = (pn + 2 * gn) * n_tile
                        n1 = n0 + n_tile
                        acc0 = pl.create_tensor([m_tile, n_tile], dtype=pl.FP32)
                        acc1 = pl.create_tensor([m_tile, n_tile], dtype=pl.FP32)
                        for kb in pl.pipeline(0, k // k_tile, stage=2):
                            # `k_stagger` rotates where this block starts its K sweep, so the
                            # resident blocks are not all asking for the same slice of the
                            # weight at once. `init_cond` still means this block's first step,
                            # and the sum over K is commutative.
                            k0 = ((kb + patch * k_stagger) % (k // k_tile)) * k_tile
                            at = a[m0 : m0 + m_tile, k0 : k0 + k_tile]
                            acc0 = pl.matmul_acc(acc0, at, w[n0 : n0 + n_tile, k0 : k0 + k_tile],
                                                 b_trans=True, init_cond=(kb == 0))
                            acc1 = pl.matmul_acc(acc1, at, w[n1 : n1 + n_tile, k0 : k0 + k_tile],
                                                 b_trans=True, init_cond=(kb == 0))
                        y[m0 : m0 + m_tile, n0 : n0 + n_tile] = pl.cast(
                            acc0, target_type=pl.BF16, mode="rint")
                        y[m0 : m0 + m_tile, n1 : n1 + n_tile] = pl.cast(
                            acc1, target_type=pl.BF16, mode="rint")
        return y

    return gdn_out_proj


# ---------------------------------------------------------------------------
# Golden and references
# ---------------------------------------------------------------------------


def golden_y(a, w):
    """The kernel's arithmetic in float64, in its order. -> float64 [M, N]."""
    from reference import host_torch_setup

    host_torch_setup()
    return a.double() @ w.double().t()


def golden_fn(tensors):
    import torch

    tensors["y"].copy_(golden_y(tensors["a"], tensors["w"]).to(torch.bfloat16))


# acceptance: the kernel accumulates in FP32 and writes BF16, the golden
# accumulates in float64 and writes BF16, so the two differ by the accumulation
# order over K and one rounding. A relative Frobenius bound is the right shape;
# an element-wise one would measure cancellation in the contraction.
MAX_REL = 5e-3
MAX_ROW_REL = 2e-2


def compare_y(actual, expected, **_kwargs):
    """Whole-tensor and worst-row relative distance to the float64 golden."""
    import torch

    dev, ref = actual.double(), expected.double()
    minimum = torch.finfo(ref.dtype).tiny
    difference = dev - ref
    rel = float(difference.norm() / ref.norm().clamp_min(minimum))
    row = float((difference.norm(dim=-1) / ref.norm(dim=-1).clamp_min(minimum)).max())
    detail = (f"[stats] y vs float64 golden: rel frob {rel:.3e} (gate {MAX_REL}), "
              f"worst row {row:.3e} (gate {MAX_ROW_REL})")
    ok = rel <= MAX_REL and row <= MAX_ROW_REL
    return ok, "" if ok else detail


# ---------------------------------------------------------------------------
# The GDN `out_proj` shape, which is the only place the model needs this kernel
# ---------------------------------------------------------------------------


def build_tensor_specs(t: int = DEFAULT_TEST_SEQ_LEN, k: int = K, n: int = N, weights: str | None = None):
    """The projection's inputs: a real layer's BF16 `out_proj` when given one.

    The activation is drawn rather than taken from the block chain. This file
    measures and validates the projection itself; what the gated norm hands it
    is validated by the composed `gdn_block` entry.
    """
    import torch
    from golden import TensorSpec

    from weights import load_weights

    def draw(shape, seed, scale):
        def make():
            gen = torch.Generator().manual_seed(seed)
            return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)
        return make

    def weight():
        if weights is None:
            return draw((n, k), 2, k ** -0.5)()
        return load_weights(QWEN3_8_27B, weights)["out_proj.weight"]

    return [
        TensorSpec("a", [t, k], torch.bfloat16, init_value=draw((t, k), 1, k ** -0.5)),
        TensorSpec("w", [n, k], torch.bfloat16, init_value=weight),
        TensorSpec("y", [t, n], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_TEST_SEQ_LEN)
    parser.add_argument("--weights", type=str, default=None,
                        help="a real GDN layer from weights.py; default: random weights")
    parser.add_argument("--m-tile", type=int, default=M_TILE)
    parser.add_argument("--n-tile", type=int, default=N_TILE)
    parser.add_argument("--k-tile", type=int, default=K_TILE)
    parser.add_argument("--k-stagger", type=int, default=K_STAGGER)
    parser.add_argument("--ring-slots", type=int, default=RING_SLOTS)
    parser.add_argument("--m-group", type=int, default=None)
    parser.add_argument("--n-group", type=int, default=N_GROUP)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    t = args.seq_len
    print(f"[case] out_proj T={t}", flush=True)
    result = run(
        fn=build_kernel(t, K, N, m_tile=args.m_tile, n_tile=args.n_tile, k_tile=args.k_tile,
                        m_group=args.m_group, n_group=args.n_group,
                        k_stagger=args.k_stagger, ring_slots=args.ring_slots),
        specs=build_tensor_specs(t=t, weights=args.weights),
        golden_fn=golden_fn,
        compare_fn={"y": compare_y},
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-2,
        atol=1e-2,
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(f"FAILED at T={t}")
