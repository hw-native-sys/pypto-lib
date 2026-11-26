# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The A8W8 projection shared by `in_proj_qkv`, `in_proj_z` and `out_proj`:

    y[m, n] = bf16( sum_k a_q[m, k] * w_q[n, k] * a_scale[m] * w_scale[n] )

INT8 both sides, INT32 accumulation over a pipelined K loop, dequantized by the
token's scale and the output channel's scale, bf16 out. The three projections
are the same arithmetic at three shapes, so they are one kernel with three sets
of tile constants -- unrelated matmuls get their own knobs, which is what makes
a tile sweep readable.

The weight keeps `nn.Linear`'s `[out, in]` layout and is read with
`b_trans=True`, so the contraction axis is contiguous. That sets the floor on
the K tile: an INT8 row of `k_tile` bytes should cover a 512-byte L2 line, so
`k_tile >= 512`.

The two instances live in `in_proj_qkv.py` and `in_proj_z.py`, which supply
the shape, the tiles, and where the weight and the reference come from; the
attention projections and the MLP build their own. The block's `out_proj` is
BF16 both sides in `out_proj.py`.
"""
import pypto.language as pl

# tiling defaults; each instance overrides them
M_TILE = 128            # token rows per output tile
N_TILE = 128            # output channels per output tile
K_TILE = 512            # contraction step; 512 INT8 columns is one L2 line
M_GROUP = 4             # output tiles a block walks down the token axis
N_GROUP = 4             # and across the channel axis


# scheduling: see `rmsnorm.EARLY_RESOLVE`. This kernel is shared with prefill, where a
# saturated engine leaves no early window for the hint to win, so it should be inert there;
# that is measured rather than assumed.
EARLY_RESOLVE = True


def build_kernel(m: int, k: int, n: int, name: str = "a8w8_linear",
                 inline: bool = False, m_tile: int = M_TILE, n_tile: int = N_TILE,
                 k_tile: int = K_TILE, m_group: int = M_GROUP, n_group: int = N_GROUP,
                 row_off: int = 0, k_stagger: int = 0, ring_slots: int = 1):
    """The projection at one shape.

    One output tile is capped near 128x128 by the cube-to-vector crossing: the
    ring reserve is sized by the INT32 accumulator and the vector buffer must
    hold it twice over. A block therefore walks an `m_group` by `n_group` patch
    of output tiles rather than one, for two reasons. It is what keeps the
    operands off the bus -- a patch reads
    `(m_group * m_tile + n_group * n_tile) * K` bytes for
    `m_group * n_group * m_tile * n_tile` outputs, so squaring the patch is
    what a bigger tile would have bought. And 5120 one-tile blocks is more
    tasks than the runtime's ring heap carries: `in_proj_qkv` at 128x128 with
    no patch dies with HEAP_RING_DEADLOCK before it computes anything.

    The inner loop takes two n-tiles at a time against one activation tile, so
    `n_group` must be even. Two accumulators keep each crossing at one tile
    while the cube works a 128x256 region from a single L1 operand.

    `k_stagger` rotates each block's K start by its own block index, so the blocks
    resident on the 24 cube cores are not all asking for the same slice of the weight at
    the same moment. It is the cheapest of the two L2-contention levers CANN's own GEMM
    uses (its other one, an L2-tiled serpentine block walk, measured smaller here and does
    not add to this one). Default off: it is measured on the three GDN projections and
    nowhere else, and this kernel also serves the attention and MLP projections and the
    decode shapes.

    `row_off` makes the output `[m + row_off, n]` and writes the result below
    that many rows. The convolution downstream of `in_proj_qkv` reads the K-1
    tokens before each of its windows, so in the block the projection writes
    straight into the padded buffer the convolution wants and no pass copies
    `[m, n]` into `[m + K - 1, n]`.
    """
    if m % m_tile or n % n_tile or k % k_tile:
        raise ValueError(f"m={m} must be a multiple of m_tile={m_tile}, "
                         f"n={n} of n_tile={n_tile}, k={k} of k_tile={k_tile}")
    mb, nb = m // m_tile, n // n_tile
    if mb % m_group or nb % n_group:
        raise ValueError(f"the {mb}x{nb} output tiles must divide by the patch "
                         f"{m_group}x{n_group}")
    if n_group % 2:
        raise ValueError(f"n_group={n_group} must be even: the inner loop pairs n-tiles")
    mg, ng = mb // m_group, nb // n_group

    @(pl.jit.inline if inline else pl.jit)
    def gdn_a8w8_linear(
        a_q: pl.Tensor[[m, k], pl.INT8],
        a_scale: pl.Tensor[[1, m], pl.FP32],
        w_q: pl.Tensor[[n, k], pl.INT8],
        w_scale: pl.Tensor[[n], pl.FP32],
        y: pl.Out[pl.Tensor[[m + row_off, n], pl.BF16]],
    ):
        # The ring is sized by the tile crossing cube to vector, here the INT32
        # accumulator, and two on-chip slots of it exceed the vector buffer
        # (pypto#2903). `local_slot_num` keeps that reserve at one slot while
        # `ring_slots` buys transport depth in GM, so the cube need not wait for
        # the vector to free a slot before pushing the next tile.
        for blk in pl.spmd(mg * ng, name_hint=name,
                           optimizations=[pl.cross_core_slot(slot_num=ring_slots,
                                                             local_slot_num=1)],
                           allow_early_resolve=EARLY_RESOLVE):
            pm = (blk % mg) * m_group
            pn = (blk // mg) * n_group
            for gm in pl.range(m_group):
                m0 = (pm + gm) * m_tile
                yr = m0 + row_off
                xs = pl.reshape(a_scale[0:1, m0 : m0 + m_tile], [m_tile, 1])
                # n innermost: the activation tile is the one held across the
                # inner loop, and it is the operand every tile in the row reads
                for gn in pl.range(n_group // 2):
                    n0 = (pn + 2 * gn) * n_tile
                    n1 = n0 + n_tile
                    # init_cond, not a peeled first block: peeling leaves block 0's
                    # operand load outside the software pipeline, the one load that
                    # then has nothing to overlap with.
                    acc0 = pl.create_tensor([m_tile, n_tile], dtype=pl.INT32)
                    acc1 = pl.create_tensor([m_tile, n_tile], dtype=pl.INT32)
                    for kb in pl.pipeline(0, k // k_tile, stage=2):
                        # `k_stagger` rotates where this block starts its K sweep. Without
                        # it every block asks for the same K slice of the weight at the same
                        # moment; `init_cond` still holds, because it means this block's
                        # first step rather than K block zero, and the sum is commutative.
                        k0 = ((kb + blk * k_stagger) % (k // k_tile)) * k_tile
                        at = a_q[m0 : m0 + m_tile, k0 : k0 + k_tile]
                        acc0 = pl.matmul_acc(acc0, at, w_q[n0 : n0 + n_tile, k0 : k0 + k_tile],
                                             b_trans=True, init_cond=(kb == 0))
                        acc1 = pl.matmul_acc(acc1, at, w_q[n1 : n1 + n_tile, k0 : k0 + k_tile],
                                             b_trans=True, init_cond=(kb == 0))
                    # dequant: the channel's scale along the row, the token's down
                    # the column
                    ws0 = pl.reshape(w_scale[n0 : n0 + n_tile], [1, n_tile])
                    deq0 = pl.row_expand_mul(
                        pl.col_expand_mul(pl.cast(acc0, target_type=pl.FP32), ws0), xs)
                    y[yr : yr + m_tile, n0 : n0 + n_tile] = pl.cast(
                        deq0, target_type=pl.BF16, mode="rint")
                    ws1 = pl.reshape(w_scale[n1 : n1 + n_tile], [1, n_tile])
                    deq1 = pl.row_expand_mul(
                        pl.col_expand_mul(pl.cast(acc1, target_type=pl.FP32), ws1), xs)
                    y[yr : yr + m_tile, n1 : n1 + n_tile] = pl.cast(
                        deq1, target_type=pl.BF16, mode="rint")
        return y

    return gdn_a8w8_linear


# ---------------------------------------------------------------------------
# Golden and references
# ---------------------------------------------------------------------------


def golden_y(a_q, a_scale, w_q, w_scale):
    """The kernel's arithmetic in fp32, in its order. -> fp32 [M, N].

    fp32 rather than int32 for the contraction: the products are exact in fp32
    and the accumulation error over K terms is ~1e-6 relative, a thousand times
    under the bf16 rounding the output takes, and an int32 matmul of this size
    has no BLAS path on the host.
    """
    from reference import host_torch_setup

    host_torch_setup()
    acc = a_q.float() @ w_q.float().t()
    return acc * w_scale.float()[None, :] * a_scale.float().reshape(-1, 1)


def golden_fn(tensors):
    import torch

    y = golden_y(tensors["a_q"], tensors["a_scale"], tensors["w_q"], tensors["w_scale"])
    tensors["y"].copy_(y.to(torch.bfloat16))


def compare_y(truth_fn, fakequant_fn, label: str):
    """The bf16 gate against the kernel's own golden, plus the two distances.

    *fakequant_fn* returns the float64 W8A8 chain -- the same arithmetic the
    kernel implements -- and *truth_fn* the float64 bf16-weight projection. The
    gap between them is what quantization costs this projection; the gap between
    the kernel and *fakequant_fn* is what the kernel costs on top.
    """
    from golden import ratio_allclose

    gate = ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)

    def compare(actual, expected, **kw):
        ok, detail = gate(actual, expected, **kw)
        if ok:
            return True, ""
        dev = actual.double()
        fq = fakequant_fn()
        truth = truth_fn()
        head = (f"    {label}: rel frob {float((dev - fq).norm() / fq.norm()):.3e} vs the "
                  f"float64 W8A8 chain, {float((dev - truth).norm() / truth.norm()):.3e} vs the "
                f"float64 truth; quantization alone costs "
                f"{float((fq - truth).norm() / truth.norm()):.3e}")
        return False, f"{head}\n{detail}"

    return compare


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


_ACT_CACHE: dict[tuple, dict] = {}


def hidden_activation(cfg, t: int, weights: str | None = None):
    """The quantized hidden states entering the block: what `quant_x` produces.

    Returns a callable giving a dict with `x` (the bf16 input), `q` and `scale`
    as a `[1, T]` row -- the layout `quant_x` writes and the kernel reads. The
    cache is module-level so the specs and the comparison share one draw.
    """
    import reference

    def load():
        key = (cfg.name, t, weights)
        cache = _ACT_CACHE.get(key)
        if cache is None:
            x = reference.make_block_inputs(t, cfg)
            q, scale = reference.quantize_rows(x)
            cache = _ACT_CACHE[key] = dict(x=x, q=q, scale=scale.float().reshape(1, -1))
        return cache

    return load


def references(cfg, module: str, act_fn, weights: str | None):
    """The float64 W8A8 chain and the float64 bf16-weight truth, each computed once.

    *act_fn* returns the same cache :func:`hidden_activation` does: `x` for the
    truth, `q` and `scale` for the chain the kernel implements.
    """
    import reference
    from weights import load_weights

    cache = {}

    def truth():
        if "truth" not in cache:
            reference.host_torch_setup()
            w = reference.weight(load_weights(cfg, weights), module)
            cache["truth"] = reference.linear(act_fn()["x"], w)
        return cache["truth"]

    def fakequant():
        if "fq" not in cache:
            reference.host_torch_setup()
            wq = reference.quantize_weights(load_weights(cfg, weights),
                                            names=(module + ".weight",))
            xf = reference.dequantize(act_fn()["q"], act_fn()["scale"].reshape(-1))
            cache["fq"] = reference.linear(xf, reference.weight(wq, module))
        return cache["fq"]

    return truth, fakequant


def build_specs(m: int, k: int, n: int, a_q_fn, a_scale_fn, w_q_fn, w_scale_fn):
    """The five tensors, with the inputs supplied as lazy callables."""
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("a_q", [m, k], torch.int8, init_value=a_q_fn),
        TensorSpec("a_scale", [1, m], torch.float32, init_value=a_scale_fn),
        TensorSpec("w_q", [n, k], torch.int8, init_value=w_q_fn),
        TensorSpec("w_scale", [n], torch.float32, init_value=w_scale_fn),
        TensorSpec("y", [m, n], torch.bfloat16),
    ]


def add_args(parser, m_tile: int, n_tile: int, k_tile: int,
             m_group: int | None = M_GROUP, n_group: int = N_GROUP):
    """The CLI every instance shares. The shape comes from the entry's DEFAULT_CASES."""
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of the entry's DEFAULT_CASES")
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
    parser.add_argument("--m-tile", type=int, default=m_tile)
    parser.add_argument("--n-tile", type=int, default=n_tile)
    parser.add_argument("--k-tile", type=int, default=k_tile)
    parser.add_argument("--m-group", type=int, default=m_group)
    parser.add_argument("--n-group", type=int, default=n_group)
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the kernel over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    return parser


def report(label: str, t: int, args, result, k: int, n: int, *, m_group: int | None = None):
    """The achieved rate for the shape and tiles, which the harness cannot derive.

    The harness already prints the per-round statistics, so this adds only what it
    does not know: the matmul's shape, the tiles it ran with, and the rate those imply.
    """
    import statistics

    if not args.rounds or result.bench is None:
        return
    samples = [s for s in result.bench.per_round("effective") if s > 0]
    if not samples:
        return
    # 2 ops per MAC, the convention every TOPS figure uses
    tops = 2 * t * k * n / (statistics.fmean(samples) * 1e-6) / 1e12
    m_group = args.m_group if m_group is None else m_group
    print(f"[bench] {label} M={t} K={k} N={n} "
          f"m={args.m_tile} n={args.n_tile} k={args.k_tile} "
          f"patch={m_group}x{args.n_group} -> {tops:.1f} TOPS", flush=True)
