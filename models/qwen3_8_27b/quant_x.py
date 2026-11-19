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

The token tile is a build argument because the task count is a quotient of it, and at a
decode batch that quotient is zero -- which aborts the launch rather than quietly doing
nothing. `decode_vector.py` carries the decode values.
"""
import pypto.language as pl

from chunk_cumsum import CHUNK_TILE
from config import GDN_QUANT, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
C = QWEN3_8_27B.hidden_size                 # the block's input width

# quantization
SCALE_MAX = GDN_QUANT.scale_max
AMAX_EPS = GDN_QUANT.amax_eps
# validation
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# gate
MIN_EXACT_VS_REFERENCE = 0.999  # of x_q elements identical to the fake-quant reference

# tiling
TOK_TILE = 16           # tokens per pipelined step
CH_TILE = 1024          # columns per reduction and per quantization step
GROUP = 4               # pipelined steps per block
STORE_GROUP = 128       # elements per INT8 store group
DEFAULT_CASES = (128, 1024, TOK_TILE * GROUP + 1, TOK_TILE * (GROUP + 1) - 1)


def build_kernel(tok_tile: int = TOK_TILE, ch_tile: int = CH_TILE, group: int = GROUP,
                 inline: bool = False):
    """Quantize token rows with disjoint scale cache lines and a masked tail."""
    if tok_tile <= 0 or tok_tile % TOK_TILE or group <= 0:
        raise ValueError(f"tok_tile must be a positive multiple of {TOK_TILE} and group must be positive")
    if ch_tile <= 0 or C % ch_tile or ch_tile % STORE_GROUP:
        raise ValueError(f"ch_tile must divide {C} and be a positive multiple of {STORE_GROUP}")
    rows_tile = tok_tile * group

    @(pl.jit.inline if inline else pl.jit)
    def _gdn_quant_x(
        x: pl.Tensor[[T_DYN, C], pl.BF16],
        x_q: pl.Out[pl.Tensor[[T_DYN, C], pl.INT8]],
        x_scale: pl.Out[pl.Tensor[[1, T_DYN], pl.FP32]],
    ):
        x.bind_dynamic(0, T_DYN)
        x_q.bind_dynamic(0, T_DYN)
        x_scale.bind_dynamic(1, T_DYN)
        t_dim = pl.tensor.dim(x, 0)
        for blk in pl.spmd((t_dim + rows_tile - 1) // rows_tile, name_hint="quant_x"):
            r0 = blk * rows_tile
            tile_count = pl.min(group, (t_dim - r0 + tok_tile - 1) // tok_tile)
            for i in pl.pipeline(tile_count, stage=2):
                t0 = r0 + i * tok_tile
                live = pl.min(tok_tile, t_dim - t0)
                # amax as max(max, -min), not max(abs): the two reductions replace an
                # abs pass over the whole tile with one negate of a [tok, 1] column.
                # The fp32 cast is not optional -- ptoas takes no bf16 reduction --
                # but it is exact, so this amax is the reference's amax.
                xf0 = pl.cast(pl.slice(x, [tok_tile, ch_tile], [t0, 0],
                                       valid_shape=[live, ch_tile]), target_type=pl.FP32)
                mx0 = pl.row_max(xf0)
                mn0 = pl.row_min(xf0)
                # pl.range, not pl.unroll: an unrolled column loop keeps every
                # iteration's temporaries live at once and overflows the vector
                # buffer by 3x.
                for cb, (mx, mn) in pl.range(1, C // ch_tile, init_values=(mx0, mn0)):
                    c0 = cb * ch_tile
                    xbf = pl.cast(pl.slice(x, [tok_tile, ch_tile], [t0, c0],
                                           valid_shape=[live, ch_tile]), target_type=pl.FP32)
                    mx_end, mn_end = pl.yield_(pl.maximum(mx, pl.row_max(xbf)),
                                               pl.minimum(mn, pl.row_min(xbf)))
                amax = pl.maximum(pl.maximum(mx_end, pl.neg(mn_end)), AMAX_EPS)
                scale_row = pl.reshape(pl.mul(amax, 1.0 / SCALE_MAX), [1, tok_tile])
                x_scale[0:1, t0 : t0 + tok_tile] = pl.tensor.set_validshape(scale_row, 1, live)

                # The numerator shares the row reduction's layout and valid rows.
                zero_scale = pl.mul(amax, 0.0)
                scale_max = pl.add(zero_scale, SCALE_MAX)
                inv = pl.div(scale_max, amax)
                for cb in pl.range(C // ch_tile):
                    c0 = cb * ch_tile
                    xf = pl.cast(pl.slice(x, [tok_tile, ch_tile], [t0, c0],
                                          valid_shape=[live, ch_tile]), target_type=pl.FP32)
                    # No clamp: amax is the row's own maximum, so |x| * 127 / amax
                    # <= 127 and a +-127 clamp cannot bind.
                    # INT32 and then INT8, not fp32 straight to INT8: the direct
                    # cast rounds the value to FP16 on the way, whose spacing at 127
                    # is a sixteenth of an INT8 step, and that moves 0.9% of the
                    # output by one step. Rounding to INT32 first makes the last cast
                    # a truncation of an exact integer.
                    q32 = pl.cast(pl.row_expand_mul(xf, inv), target_type=pl.INT32,
                                  mode="rint")
                    q_i8 = pl.cast(q32, target_type=pl.INT8, mode="trunc")
                    x_q[t0 : t0 + tok_tile, c0 : c0 + ch_tile] = pl.tensor.set_validshape(
                        q_i8, live, ch_tile)
        return x_q, x_scale

    return _gdn_quant_x


# Column groups the decode form splits the reduction over; see `build_fanned_kernel`.
COL_GROUPS = 4
VEC_BUFFER_BYTES = 188416       # a2a3's vector buffer, which bounds the column range

# Speculative early dispatch on the decode chain; `rmsnorm.EARLY_RESOLVE` carries the full
# note. It only pays when every direct producer of a consumer has it, so the flag travels
# along a chain rather than sitting on one stage.
EARLY_RESOLVE = True


def build_fanned_kernel(m: int, col_groups: int = COL_GROUPS, tok_tile: int = TOK_TILE,
                        inline: bool = False):
    """The decode form: the per-token reduction split across the column axis.

    Same shape of fix as `act_quant.build_fanned_kernel`, and the same reason. The body
    above gives one block a run of token rows and walks the column axis inside it; at a
    decode batch that is one task walking `C / ch_tile` reduction steps and as many
    quantisation steps, with nothing to overlap. The token axis is exhausted, so the fan-out
    moves to the columns: a partial pass over a `[col_groups, m]` FP32 scratch, a single-task
    combine that owns the whole `x_scale` cache line, and an apply pass.

    The combine must be one task because 16 FP32 scales are exactly one 64-byte line and a
    line written from two blocks loses one of them silently. The apply pass re-folds the
    scratch rather than carrying `amax` forward, which costs `col_groups * m` elements and
    removes a second scratch.

    **Bit-identical to the body above.** Maximum and minimum are associative and exact, so
    splitting the reduction by column group and combining cannot move `amax`; the epsilon
    clamp commutes with it; `x_scale` keeps the same multiply by `1 / SCALE_MAX` and the
    apply keeps the same `recip` and the same INT32-then-truncate cast.

    `col_groups` must divide `C / STORE_GROUP` so that no INT8 store is a partial
    128-element group, which is miscompiled (see `act_quant.build_fanned_kernel`). The apply
    pass also holds its column range as FP32 twice, INT32 once and INT8 once, so the range
    is capacity-bound from the other side: 4 and 5 groups do not fit, which leaves 8, 10, 20
    and 40.

    **The win is modest and it shrinks with the batch**, which is the honest reason to read
    this entry before reusing the pattern. a2a3, median of 30 rounds, every variant in
    one process:

    | col_groups | shipped | 4 | 5 | 8 | 10 | 20 |
    |---|---:|---:|---:|---:|---:|---:|
    | column width | 5120 | 1280 | 1024 | 640 | 512 | 256 |
    | us at M = 16 | 23.9 | **18.2** | 18.6 | 20.8 | 21.8 | 28.0 |
    | us at M = 32 | 24.7 | **22.5** | 22.5 | 23.0 | 23.8 | 29.0 |

    1.31x at M = 16 and 1.10x at M = 32, so 5.7 and 2.3 us per invocation. `quant_x` runs
    **128 times per decode step** -- `gdn_block` x48, `gated_attention` x16, the MLP x64 --
    which is 45.2 us/token at batch 16 and 9.0 at batch 32. Against `act_quant`'s 203.6
    us/token for the same structural change, this stage was already near the launch floor
    that `decode_kv_append` puts around 14 us: there is only about 10 us of serial work here
    to divide, where `act_quant` had 64.

    **4 groups is the edge, and this time the bound is the compiler's**: 2 groups needs
    327808 B of the 188416 B vector buffer, which it says so itself. An earlier version of
    this file put the floor at 8 groups because the guard above estimated the working set at
    13 bytes per element; the real figure is 8, so 4 and 5 were refused by arithmetic rather
    than by capacity and the default was set one step inside a boundary of its own making.
    Four groups is four tasks on a 48-core machine and still wins, which is
    `decode_short_conv`'s rule again: a task that carries enough work to amortise its own
    cost should be big, not numerous.
    """
    if m % tok_tile:
        raise ValueError(f"m={m} must be a multiple of tok_tile={tok_tile}")
    if C % col_groups:
        raise ValueError(f"col_groups={col_groups} must divide C={C}")
    width = C // col_groups
    if width % STORE_GROUP:
        raise ValueError(
            f"col_groups={col_groups} leaves a {width}-column range, which is not a multiple "
            f"of {STORE_GROUP}; an INT8 tile store of a partial group writes wrong values. "
            f"col_groups must divide {C // STORE_GROUP}")
    # 8 bytes per element, calibrated against the compiler's own reported usage rather than
    # counted off the source: the apply pass names four tiles (FP32 in, FP32 scaled, INT32,
    # INT8) but the scaled buffer is reused, so a naive 13 refuses configurations that build.
    # The compiler holds the real number -- "Vec buffer usage (N bytes) exceeds platform
    # limit" -- and this check only moves that error earlier.
    live = tok_tile * width * 8
    if live > VEC_BUFFER_BYTES:
        raise ValueError(f"col_groups={col_groups} needs about {live} B of the "
                         f"{VEC_BUFFER_BYTES} B vector buffer in the apply pass; use at least "
                         f"{C * tok_tile * 8 // VEC_BUFFER_BYTES + 1} groups")

    @(pl.jit.inline if inline else pl.jit)
    def _gdn_quant_x_fanned(
        x: pl.Tensor[[m, C], pl.BF16],
        x_q: pl.Out[pl.Tensor[[m, C], pl.INT8]],
        x_scale: pl.Out[pl.Tensor[[1, m], pl.FP32]],
    ):
        """One task per column group; `m` is the padded decode batch."""
        part = pl.create_tensor([col_groups, m], dtype=pl.FP32)

        for cg in pl.spmd(col_groups, name_hint="quant_x_amax",
                          allow_early_resolve=EARLY_RESOLVE):
            c0 = cg * width
            for t0 in pl.range(0, m, tok_tile):
                xf = pl.cast(x[t0 : t0 + tok_tile, c0 : c0 + width], target_type=pl.FP32)
                group_amax = pl.maximum(pl.row_max(xf), pl.neg(pl.row_min(xf)))
                part[cg : cg + 1, t0 : t0 + tok_tile] = pl.reshape(group_amax, [1, tok_tile])

        for _solo in pl.spmd(1, name_hint="quant_x_scale",
                             allow_early_resolve=EARLY_RESOLVE):
            for t0 in pl.range(0, m, tok_tile):
                folded_row = pl.col_max(part[0:col_groups, t0 : t0 + tok_tile])
                amax_row = pl.maximum(folded_row, AMAX_EPS)
                x_scale[0:1, t0 : t0 + tok_tile] = pl.mul(amax_row, 1.0 / SCALE_MAX)

        # `amax_col` is a distinct name from the scale pass's `amax_row`: one body is one
        # scope, and the DSL refuses a name reassigned at a different tile shape.
        for cg in pl.spmd(col_groups, name_hint="quant_x_apply",
                          allow_early_resolve=EARLY_RESOLVE):
            c0 = cg * width
            for t0 in pl.range(0, m, tok_tile):
                folded_col = pl.col_max(part[0:col_groups, t0 : t0 + tok_tile])
                amax_col = pl.reshape(pl.maximum(folded_col, AMAX_EPS), [tok_tile, 1])
                inv = pl.mul(pl.recip(amax_col, high_precision=True), SCALE_MAX)
                xf = pl.cast(x[t0 : t0 + tok_tile, c0 : c0 + width], target_type=pl.FP32)
                q32 = pl.cast(pl.row_expand_mul(xf, inv), target_type=pl.INT32, mode="rint")
                x_q[t0 : t0 + tok_tile, c0 : c0 + width] = pl.cast(q32, target_type=pl.INT8,
                                                                   mode="trunc")
        return x_q, x_scale

    return _gdn_quant_x_fanned


gdn_quant_x = build_kernel(inline=True)
gdn_quant_x_test = build_kernel()


def golden_gdn_quant_x(tensors):
    """`reference.quantize_rows` verbatim: it is already the kernel's arithmetic."""
    import reference

    q, scale = reference.quantize_rows(tensors["x"])
    tensors["x_q"].copy_(q)
    tensors["x_scale"].copy_(scale.reshape(1, -1))


def build_tensor_specs(t: int = BENCH_SEQ, c: int = C, chunk: int = CHUNK_TILE,
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
        if ok:
            return True, ""
        head = (f"    x_q: {exact * 100:.4f}% exact "
                f"(gate {MIN_EXACT_VS_REFERENCE * 100:.1f}%), max {int(step.max())} step(s) off")
        return False, f"{head}\n{detail}"

    return compare


def compare_x_scale(actual, expected, **kw):
    """The scale carries the whole dequantization, so it is held to fp32 rounding.

    `max_error_ratio=0` because there is no outlier budget here: every scale has to be
    right, and the comparator's own non-finite rejection comes with it.
    """
    from golden import ratio_allclose

    return ratio_allclose(atol=0.0, rtol=1e-6, max_error_ratio=0.0)(actual, expected, **kw)


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--ch-tile", type=int, default=CH_TILE)
    parser.add_argument("--fanned", action="store_true", default=False,
                        help="only the per-token form, one task per token")
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the kernel over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len is not None else list(DEFAULT_CASES[:2] if args.fanned else DEFAULT_CASES)
    if any(t <= 0 or (args.fanned and t % TOK_TILE) for t in cases):
        parser.error(f"--seq-len must be positive and a multiple of {TOK_TILE} for --fanned")
    if args.fanned and args.ch_tile != CH_TILE:
        parser.error("--ch-tile is only supported by the tiled kernel")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    # Two programs live here, the token-tiled form and the per-token one the decode path
    # reads, so the default set runs both rather than leaving the second uncovered.
    plan = [("fanned", t) for t in cases] if args.fanned else [("tiled", t) for t in cases]
    if not args.fanned and not args.seq_len:
        plan += [("fanned", TOK_TILE)]

    tiled_kernel = gdn_quant_x_test if args.ch_tile == CH_TILE else build_kernel(ch_tile=args.ch_tile)
    for label, t in plan:
        print(f"[case] quant_x {label} T={t}", flush=True)
        result = run(
            fn=build_fanned_kernel(t) if label == "fanned" else tiled_kernel,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_gdn_quant_x,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-5,
            compare_fn={"x_q": compare_x_q(t, CHUNK_TILE, None),
                        "x_scale": compare_x_scale},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at {label} T={t}")
