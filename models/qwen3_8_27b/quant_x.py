# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-token INT8 quantization of the hidden states, the A8W8 activation side:

    x_scale[t] = max(amax(|x[t]|), amax_eps) / 127
    x_q[t]     = rint(x[t] * 127 / max(amax(|x[t]|), amax_eps))

One pass over the block's input, shared by every projection that reads it --
`in_proj_qkv`, `in_proj_z` and, at bf16, nothing else. `in_proj_ab` stays bf16
and reads `x` directly. The gated RMSNorm does the same arithmetic on its own
output for `out_proj`, in its epilogue rather than here, because the value it
quantizes never reaches GM in bf16.

Purely memory-bound: 84 MB in, 42 MB out at T = 8192.
"""
import pypto.language as pl

from config import GDN_QUANT, GDN_TILING, QWEN3_8_27B

# model shape
C = QWEN3_8_27B.hidden_size                 # the block's input width
CHUNK = GDN_TILING.chunk                    # only picks the reference chain to draw from

# quantization
SCALE_MAX = GDN_QUANT.scale_max
AMAX_EPS = GDN_QUANT.amax_eps

# case shape
T = 8192                # tokens (single sequence, B = 1)

# gate
MIN_EXACT_VS_REFERENCE = 0.999  # of x_q elements identical to the fake-quant reference

# tiling
TOK_TILE = 16           # tokens per pipelined step
CH_TILE = 1024          # columns per reduction and per quantization step
GROUP = 4               # pipelined steps per block

OUTPUTS = ("x_q", "x_scale")        # bench.py reads it


def build_kernel(t: int = T, c: int = C, inline: bool = False,
                 tok_tile: int = TOK_TILE, ch_tile: int = CH_TILE, group: int = GROUP):
    """The operator at one shape. `inline` makes it a callee for `gdn_block`.

    The amax pass and the quantization pass both read x from GM: a resident
    [tok, 5120] row does not fit the vector buffer beside a pipelined step's
    working tiles at any tile worth having, and the second read of a tile the
    block just touched is an L2 hit rather than a second trip to memory.

    `tok_tile * ch_tile` is what the vector buffer bounds, near 16000 elements;
    within that, the widest column tile wins.

    `x_scale` is a `[1, T]` row, not a `[T]` vector, because a tile store must
    be 2-D; a consumer reshapes its slice to the `[M, 1]` column it multiplies by.
    """
    rows = tok_tile * group
    # A tile that does not divide its axis silently leaves the tail unwritten.
    if t % rows or c % ch_tile:
        raise ValueError(f"t={t} must be a multiple of tok_tile*group={rows} and "
                         f"c={c} a multiple of ch_tile={ch_tile}")

    @(pl.jit.inline if inline else pl.jit)
    def gdn_quant_x(
        x: pl.Tensor[[t, c], pl.BF16],
        x_q: pl.Out[pl.Tensor[[t, c], pl.INT8]],
        x_scale: pl.Out[pl.Tensor[[1, t], pl.FP32]],
    ):
        for blk in pl.spmd(t // rows, name_hint="quant_x"):
            r0 = blk * rows
            for i in pl.pipeline(group, stage=2):
                t0 = r0 + i * tok_tile
                # amax as max(max, -min), not max(abs): the two reductions replace an
                # abs pass over the whole tile with one negate of a [tok, 1] column.
                # The fp32 cast is not optional -- ptoas takes no bf16 reduction --
                # but it is exact, so this amax is the reference's amax.
                xf0 = pl.cast(x[t0 : t0 + tok_tile, 0:ch_tile], target_type=pl.FP32)
                mx0 = pl.row_max(xf0)
                mn0 = pl.row_min(xf0)
                # pl.range, not pl.unroll: an unrolled column loop keeps every
                # iteration's temporaries live at once and overflows the vector
                # buffer by 3x.
                for cb, (mx, mn) in pl.range(1, c // ch_tile, init_values=(mx0, mn0)):
                    c0 = cb * ch_tile
                    xbf = pl.cast(x[t0 : t0 + tok_tile, c0 : c0 + ch_tile],
                                  target_type=pl.FP32)
                    mx_end, mn_end = pl.yield_(pl.maximum(mx, pl.row_max(xbf)),
                                               pl.minimum(mn, pl.row_min(xbf)))
                amax = pl.maximum(pl.maximum(mx_end, pl.neg(mn_end)), AMAX_EPS)
                x_scale[0:1, t0 : t0 + tok_tile] = pl.reshape(
                    pl.mul(amax, 1.0 / SCALE_MAX), [1, tok_tile])

                # recip, not div: one reciprocal feeds the whole row's multiply. It
                # costs a last-bit difference against the reference's divide, which
                # the exact-match gate measures.
                inv = pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX)
                for cb in pl.range(c // ch_tile):
                    c0 = cb * ch_tile
                    xf = pl.cast(x[t0 : t0 + tok_tile, c0 : c0 + ch_tile], target_type=pl.FP32)
                    # No clamp: amax is the row's own maximum, so |x| * 127 / amax
                    # <= 127 and a +-127 clamp cannot bind.
                    # INT32 and then INT8, not fp32 straight to INT8: the direct
                    # cast rounds the value to FP16 on the way, whose spacing at 127
                    # is a sixteenth of an INT8 step, and that moves 0.9% of the
                    # output by one step. Rounding to INT32 first makes the last cast
                    # a truncation of an exact integer.
                    q32 = pl.cast(pl.row_expand_mul(xf, inv), target_type=pl.INT32,
                                  mode="rint")
                    x_q[t0 : t0 + tok_tile, c0 : c0 + ch_tile] = pl.cast(
                        q32, target_type=pl.INT8, mode="trunc")
        return x_q, x_scale

    return gdn_quant_x


gdn_quant_x = build_kernel()


def golden_gdn_quant_x(tensors):
    """`reference.quantize_rows` verbatim: it is already the kernel's arithmetic."""
    import reference

    q, scale = reference.quantize_rows(tensors["x"])
    tensors["x_q"].copy_(q)
    tensors["x_scale"].copy_(scale.reshape(1, -1))


def build_tensor_specs(t: int = T, c: int = C, chunk: int = CHUNK,
                       weights: str | None = None):
    import torch
    from golden import TensorSpec

    import reference

    return [
        TensorSpec("x", [t, c], torch.bfloat16,
                   init_value=lambda: reference.make_block_inputs(t, QWEN3_8_27B)),
        TensorSpec("x_q", [t, c], torch.int8),
        TensorSpec("x_scale", [1, t], torch.float32),
    ]


def compare_x_q(t: int, chunk: int, weights: str | None):
    """Exact-match rate plus a +-1-step bound, the gate a quantized output needs.

    The reference and the kernel differ only in how they form `127 / amax` -- a
    divide against a high-precision reciprocal and a multiply -- so all but a
    handful of elements must be identical. A step bound alone would accept a
    rounding-mode change, which moves a large fraction of the output by exactly
    one step.
    """
    import torch

    from golden import ratio_allclose

    gate = ratio_allclose(atol=1, rtol=0, max_error_ratio=1e-4)

    def compare(actual, expected, actual_outputs=None, **kw):
        ok, detail = gate(actual, expected, actual_outputs=actual_outputs, **kw)
        step = (actual.to(torch.int32) - expected.to(torch.int32)).abs()
        exact = float((step == 0).float().mean())
        if exact < MIN_EXACT_VS_REFERENCE:
            ok = False
            detail += (f"; only {exact * 100:.4f}% exact "
                       f"(< {MIN_EXACT_VS_REFERENCE * 100:.1f}%): a systematic "
                       f"rounding change, not boundary flips")
        print(f"[stats] x_q: {exact * 100:.4f}% exact "
              f"(gate {MIN_EXACT_VS_REFERENCE * 100:.1f}%), max {int(step.max())} step(s) off",
              flush=True)
        return ok, detail

    return compare


def compare_x_scale(actual, expected, **kw):
    """The scale carries the whole dequantization, so it is held to fp32 rounding."""
    rel = float(((actual.float() - expected.float()).abs() / expected.float()).max())
    print(f"[stats] x_scale: max rel diff {rel:.3e}", flush=True)
    return rel <= 1e-6, f"max rel diff {rel:.3e} (<=1e-6)"


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
        fn=build_kernel(t=args.seq_len, tok_tile=args.tok_tile, ch_tile=args.ch_tile,
                        group=args.group),
        specs=build_tensor_specs(t=args.seq_len),
        golden_fn=golden_gdn_quant_x,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn={"x_q": compare_x_q(args.seq_len, CHUNK, None),
                    "x_scale": compare_x_scale},
        compile_only=args.compile_only,
    )
    if args.rounds and result.bench is not None:
        samples = [s for s in result.bench.per_round("effective") if s > 0]
        if samples:
            print(f"[bench] quant_x T={args.seq_len} C={C} tok={args.tok_tile} "
                  f"ch={args.ch_tile} grp={args.group} rounds={len(samples)} "
                  f"mean={statistics.fmean(samples):.1f} us "
                  f"median={statistics.median(samples):.1f} min={min(samples):.1f}", flush=True)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
