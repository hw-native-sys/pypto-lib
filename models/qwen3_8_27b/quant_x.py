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
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the kernel over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len is not None else list(DEFAULT_CASES)
    if any(t <= 0 for t in cases):
        parser.error("--seq-len must be positive")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    plan = [("tiled", t) for t in cases]

    tiled_kernel = gdn_quant_x_test if args.ch_tile == CH_TILE else build_kernel(ch_tile=args.ch_tile)
    for label, t in plan:
        print(f"[case] quant_x {label} T={t}", flush=True)
        result = run(
            fn=tiled_kernel,
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
