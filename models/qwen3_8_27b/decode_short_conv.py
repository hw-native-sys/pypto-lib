# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet short convolution, decode: one token per state row, then silu.

Per token and channel, with `s` the last `K - 1` rows of the stream,

    window = concat(s, x_t)                     # [K, C], newest last
    out    = silu(sum_j w[j] * window[j])       # fp32 accumulate
    s      = window[1:]                         # drop the oldest

and the q, k and v slices of the channel axis are written to their own outputs.

**The state holds rows of the pre-convolution, pre-activation stream** -- not the conv
output, not the silu output. A row leaves the window after `K - 1` steps, so nothing
accumulates in it and BF16 is faithful to the module.

`conv_state` is **time-major**, `[rows, K - 1, C]`, which is the layout the AscendC
operator's ABI fixes. transformers keeps it channel-major, so the boundary transposes --
`reference.recurrent_short_conv` documents which side is which.

The weight is `[K, C]`, tap-major: tap *j* is one contiguous per-channel row.

The taps are written out straight rather than looped. A loop index is a `Scalar`, so it
cannot drive the trace-time choice between reading the conv state and reading the
incoming token.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                     # decode tokens, one per sequence
CSTATE_DYN = pl.dynamic("CSTATE_DYN")           # rows of the conv-state pool

# model config
C = QWEN3_8_27B.qkv_width                       # q and k at HG heads, v at H
K = QWEN3_8_27B.linear_conv_kernel_dim          # taps, newest last
H = QWEN3_8_27B.linear_num_value_heads
HG = QWEN3_8_27B.linear_num_key_heads
D = QWEN3_8_27B.linear_value_head_dim
STATE_LEN = K - 1
# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True
                               # stream rows the window carries
KEY_WIDTH = HG * D                              # q, then k, then v along the channel axis
VAL_WIDTH = H * D

# validation
# one token, then several; a token is one state row of the pool.
DEFAULT_CASES = (1, 5)

# tiling
C_TILE = 2048           # channels per task: the widest that does not straddle q/k/v
TOK_TILE = 4            # tokens one task walks; see the table above


def build_kernel(c_tile: int = C_TILE, tok_tile: int = TOK_TILE):
    """The decode convolution at a given tiling; returns `(inline, jit)` forms."""
    if C % c_tile or KEY_WIDTH % c_tile:
        raise ValueError(f"c_tile={c_tile} must divide C={C} and KEY_WIDTH={KEY_WIDTH}, "
                         "so that no tile straddles the q/k/v boundaries")
    lanes = C // c_tile

    def _gdn_decode_short_conv(
        x: pl.Tensor[[T_DYN, C], pl.BF16],
        w: pl.Tensor[[K, C], pl.FP32],
        conv_state: pl.InOut[pl.Tensor[[CSTATE_DYN, STATE_LEN, C], pl.BF16]],
        state_rows: pl.Tensor[[T_DYN], pl.INT32],
        q: pl.Out[pl.Tensor[[T_DYN, HG, D], pl.BF16]],
        k: pl.Out[pl.Tensor[[T_DYN, HG, D], pl.BF16]],
        v: pl.Out[pl.Tensor[[T_DYN, H, D], pl.BF16]],
    ):
        """`state_rows` must name a distinct pool row per token; the tasks are independent."""
        x.bind_dynamic(0, T_DYN)
        conv_state.bind_dynamic(0, CSTATE_DYN)
        state_rows.bind_dynamic(0, T_DYN)
        q.bind_dynamic(0, T_DYN)
        k.bind_dynamic(0, T_DYN)
        v.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(x, 0)
        q_flat = pl.reshape(q, [t_dim, KEY_WIDTH])
        k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
        v_flat = pl.reshape(v, [t_dim, VAL_WIDTH])
        # A store whose source is a tile reshaped to two leading unit axes lowers to a
        # tile with a single-element row, which ptoas rejects on its 32-byte row
        # alignment. Addressing the pool through a flat view of the same buffer keeps
        # the stored tile the shape the arithmetic already produced.
        cs2 = pl.reshape(conv_state, [pl.tensor.dim(conv_state, 0) * STATE_LEN, C])

        tok_tasks = (t_dim + tok_tile - 1) // tok_tile
        for task in pl.spmd(tok_tasks * lanes, name_hint="decode_short_conv",
                            allow_early_resolve=EARLY_RESOLVE):
            t0 = (task // lanes) * tok_tile
            c0 = (task % lanes) * c_tile
            w0 = w[0:1, c0 : c0 + c_tile]
            w1 = w[1:2, c0 : c0 + c_tile]
            w2 = w[2:3, c0 : c0 + c_tile]
            w3 = w[3:4, c0 : c0 + c_tile]
            for i in pl.unroll(tok_tile):
                t = t0 + i
                # A batch that is not a multiple of tok_tile leaves the tail of the last
                # task idle rather than reading past the end. The trip count is a
                # compile-time constant, so the loop is unrolled, and this is the only
                # runtime test in the body.
                if t < t_dim:
                    row = pl.cast(pl.read(state_rows, [t]), pl.INDEX)
                    srow = row * STATE_LEN
                    st0 = pl.cast(pl.slice(cs2, [1, c_tile], [srow, c0]), target_type=pl.FP32)
                    st1 = pl.cast(pl.slice(cs2, [1, c_tile], [srow + 1, c0]), target_type=pl.FP32)
                    st2 = pl.cast(pl.slice(cs2, [1, c_tile], [srow + 2, c0]), target_type=pl.FP32)
                    xt_bf = pl.slice(x, [1, c_tile], [t, c0])
                    xt = pl.cast(xt_bf, target_type=pl.FP32)
                    acc = pl.add(pl.add(pl.mul(st0, w0), pl.mul(st1, w1)),
                                 pl.add(pl.mul(st2, w2), pl.mul(xt, w3)))
                    # silu as acc / (1 + exp(-acc)): one pass fewer than recip + mul,
                    # and the reciprocal is used once, so the divide simply removes an
                    # op. Same spelling as the prefill kernel.
                    den = pl.add(pl.exp(pl.neg(acc)), 1.0)
                    out_tile = pl.cast(pl.div(acc, den), target_type=pl.BF16, mode="rint")
                    if c0 < KEY_WIDTH:
                        q_flat[t : t + 1, c0 : c0 + c_tile] = out_tile
                    elif c0 < 2 * KEY_WIDTH:
                        qk = c0 - KEY_WIDTH
                        k_flat[t : t + 1, qk : qk + c_tile] = out_tile
                    else:
                        vc = c0 - 2 * KEY_WIDTH
                        v_flat[t : t + 1, vc : vc + c_tile] = out_tile

                    # Shift the window. Every source is one of the FP32 tiles above, so
                    # all three rows are loaded before any is overwritten -- row
                    # srow+1 is both a source and a destination, so the order is
                    # load-bearing. Casting FP32 back to BF16 is exact here: these
                    # values came from BF16, and BF16 -> FP32 -> BF16 round-trips.
                    cs2[srow : srow + 1, c0 : c0 + c_tile] = pl.cast(
                        st1, target_type=pl.BF16, mode="rint")
                    cs2[srow + 1 : srow + 2, c0 : c0 + c_tile] = pl.cast(
                        st2, target_type=pl.BF16, mode="rint")
                    cs2[srow + 2 : srow + 3, c0 : c0 + c_tile] = xt_bf
        return q, k, v, conv_state

    return pl.jit.inline(_gdn_decode_short_conv), pl.jit(_gdn_decode_short_conv)


gdn_decode_short_conv, gdn_decode_short_conv_test = build_kernel()


def golden_state_from_channel_major(state):
    """`[..., C, K-1]`, transformers' layout, to this kernel's time-major `[..., K-1, C]`."""
    return state.transpose(-1, -2).contiguous()


def golden_state_to_channel_major(state):
    """This kernel's `[..., K-1, C]` back to transformers' `[..., C, K-1]`."""
    return state.transpose(-1, -2).contiguous()


def golden_step(x, w, state, rows):
    """One decode launch, in float64, against a `[rows, K-1, C]` pool.

    Returns `(out, state)` with `out` the full `[tokens, C]` activated stream and
    `state` updated for the rows `rows` names, untouched elsewhere.
    """
    import torch

    tokens = x.shape[0]
    s = state.double().clone()
    wf = w.double()
    out = torch.zeros(tokens, C, dtype=torch.float64)
    for t in range(tokens):
        r = int(rows[t])
        window = torch.cat([s[r], x[t : t + 1].double()])       # [K, C], newest last
        out[t] = (window * wf).sum(dim=0)
        s[r] = window[1:]
    return torch.nn.functional.silu(out), s


def golden_gdn_decode_short_conv(tensors):
    out, s = golden_step(tensors["x"], tensors["w"], tensors["conv_state"],
                         tensors["state_rows"])
    tokens = out.shape[0]
    tensors["q"].copy_(out[:, 0:KEY_WIDTH].reshape(tokens, HG, D).to(tensors["q"].dtype))
    tensors["k"].copy_(out[:, KEY_WIDTH : 2 * KEY_WIDTH].reshape(tokens, HG, D)
                       .to(tensors["k"].dtype))
    tensors["v"].copy_(out[:, 2 * KEY_WIDTH :].reshape(tokens, H, D).to(tensors["v"].dtype))
    tensors["conv_state"].copy_(s.to(tensors["conv_state"].dtype))


POOL_ROWS = 6


def build_tensor_specs(tokens: int = 5, pool_rows: int = POOL_ROWS):
    import torch
    from golden import TensorSpec

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32

    def gen(offset):
        return torch.Generator().manual_seed(42 + offset)

    # Distinct rows, and deliberately not the identity mapping. Distinctness is the
    # kernel's contract and this test cannot prove it: at these sizes the blocks run
    # in task-index order, which is token order, so a fixture mapping two tokens to
    # one row would agree with a sequential golden by accident.
    rows = torch.randperm(pool_rows, generator=gen(7))[:tokens].to(i32)
    if len(set(rows.tolist())) != tokens:
        raise ValueError("state_rows must name a distinct pool row per token")

    return [
        TensorSpec("x", [tokens, C], bf,
                   init_value=lambda: torch.randn(tokens, C, generator=gen(1)).to(bf)),
        TensorSpec("w", [K, C], f32,
                   init_value=lambda: torch.randn(K, C, generator=gen(2)) * 0.3),
        # A non-zero window, so a kernel that silently starts from zero fails.
        TensorSpec("conv_state", [pool_rows, STATE_LEN, C], bf,
                   init_value=lambda: torch.randn(pool_rows, STATE_LEN, C,
                                                  generator=gen(3)).to(bf)),
        TensorSpec("state_rows", [tokens], i32, init_value=lambda: rows),
        TensorSpec("q", [tokens, HG, D], bf),
        TensorSpec("k", [tokens, HG, D], bf),
        TensorSpec("v", [tokens, H, D], bf),
    ]


if __name__ == "__main__":
    import argparse

    import torch
    from golden import run

    from utils import rel_frobenius

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--pool-rows", type=int, default=POOL_ROWS)
    parser.add_argument("--c-tile", type=int, default=C_TILE)
    parser.add_argument("--tok-tile", type=int, default=TOK_TILE)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.tokens] if args.tokens else list(DEFAULT_CASES)
    if max(cases) > args.pool_rows:
        parser.error(f"--tokens {max(cases)} must not exceed --pool-rows {args.pool_rows}; "
                     "one token per state row")
    torch.set_num_threads(min(torch.get_num_threads(), 32))
    torch.backends.mkldnn.enabled = False

    _, fn = build_kernel(args.c_tile, args.tok_tile)
    for tokens in cases:
        print(f"[case] decode_short_conv tokens={tokens}", flush=True)
        result = run(
            fn=fn,
            specs=build_tensor_specs(tokens=tokens, pool_rows=args.pool_rows),
            golden_fn=golden_gdn_decode_short_conv,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            compile_only=args.compile_only,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=2e-2,
            atol=2e-2,
            # The window shift must be exact: it moves BF16 rows, so anything but 0 is a
            # layout or ordering bug, not rounding.
            compare_fn={"conv_state": rel_frobenius(0.0, "conv_state"),
                        "q": rel_frobenius(6e-3, "q"), "k": rel_frobenius(6e-3, "k"),
                        "v": rel_frobenius(6e-3, "v")},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at tokens={tokens}")
