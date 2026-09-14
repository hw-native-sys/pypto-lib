# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet gated RMSNorm, and the per-token INT8 quantization out_proj reads:

    n = bf16(o * rsqrt(mean(o^2) + eps))          per (token, head) over the head dim
    y = bf16(n * norm_w) * silu(z)                gate applied after the norm, in fp32
    y_q[t] = rint(y[t] * 127 / amax(|y[t]|))      per token over all H*D values
    y_scale[t] = amax(|y[t]|) / 127

The two bf16 narrowings are the model's own cast sequence
(`Qwen3_5RMSNormGated`): fp32 variance and rsqrt, narrow to the model dtype,
weight applied there, gate applied in fp32. `quant=False` ends in that dtype
instead of the INT8 epilogue.
"""
import pypto.language as pl

from config import GDN_QUANT, GDN_TILING, QWEN3_8_27B

# model shape
H = QWEN3_8_27B.linear_num_value_heads      # value heads
D = QWEN3_8_27B.linear_value_head_dim       # head dimension, the norm's axis
EPS = QWEN3_8_27B.rms_norm_eps
CHUNK = GDN_TILING.chunk                    # only picks the reference chain to draw from

# quantization
SCALE_MAX = GDN_QUANT.scale_max
AMAX_EPS = GDN_QUANT.amax_eps

# case shape
T = 8192                # tokens (single sequence, B = 1)

# gate
MIN_EXACT_VS_REFERENCE = 0.95   # of y_q elements identical to the fake-quant reference

# tiling
TOK_TILE = 16           # tokens per block; 16 scalar scale writes fill one 64-byte line

OUTPUTS = ("y_q", "y_scale")        # int8 variant; bench.py reads it


def build_kernel(t: int = T, h: int = H, d: int = D, eps: float = EPS,
                 inline: bool = False, quant: bool = True):
    """The operator at one shape. `inline` makes it a callee for `gdn_block`.

    The two variants are two functions because a `@pl.jit` body can neither
    branch on a Python flag nor call a helper; the per-token arithmetic is the
    same text in both, up to the last cast.
    """
    jit = pl.jit.inline if inline else pl.jit

    @jit
    def gdn_gated_rmsnorm(
        o: pl.Tensor[[t, h, d], pl.FP16],
        z: pl.Tensor[[t, h, d], pl.BF16],
        norm_w: pl.Tensor[[d], pl.BF16],
        y_q: pl.Out[pl.Tensor[[t, h * d], pl.INT8]],
        y_scale: pl.Out[pl.Tensor[[t], pl.FP32]],
    ):
        o_rows = pl.reshape(o, [t * h, d])
        z_rows = pl.reshape(z, [t * h, d])
        q_rows = pl.reshape(y_q, [t * h, d])
        for blk in pl.spmd(t // TOK_TILE, name_hint="gated_rmsnorm"):
            t0 = blk * TOK_TILE
            w_row = pl.cast(pl.reshape(norm_w[:], [1, d]), target_type=pl.FP32)
            ones_h8 = pl.full([h, 8], dtype=pl.FP32, value=1.0)   # broadcasts the amax partials
            for i in pl.unroll(TOK_TILE):
                r0 = (t0 + i) * h
                of = pl.cast(o_rows[r0 : r0 + h, :], target_type=pl.FP32)
                inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(of, of)), 1.0 / d), eps),
                                   high_precision=True)
                # the module narrows the normed values, then the weighted ones
                normed = pl.cast(pl.cast(pl.row_expand_mul(of, inv_rms),
                                         target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
                weighted = pl.cast(pl.cast(pl.col_expand_mul(normed, w_row),
                                           target_type=pl.BF16, mode="rint"), target_type=pl.FP32)

                zf = pl.cast(z_rows[r0 : r0 + h, :], target_type=pl.FP32)
                gate = pl.mul(zf, pl.recip(pl.add(pl.exp(pl.neg(zf)), 1.0)))
                y = pl.mul(weighted, gate)

                # The token's amax must reach every row of the [h, d] tile, and it
                # cannot be carried through a scalar: a [1, 1] reduction result is a
                # 4-byte tile ptoas refuses to allocate (32-byte row alignment), and
                # folding the last values with scalar `pl.max` builds a kernel the
                # DEVICE compiler then rejects -- ptoas emits its `ptoas_bitcast`
                # helper (the signed-zero path of a scalar max) without `__aicore__`,
                # so both simulators accept the kernel and only a hardware build
                # fails. Reduce on tiles the whole way instead: columns to [1, 8]
                # partials, clamped there while the layout is still row-major, then
                # broadcast down the head axis so `row_max` lands the same maximum in
                # every row of an [h, 1] column. One `pl.read` of row 0 is left, for
                # the scale, and it is a plain load.
                part_8 = pl.maximum(pl.col_max(pl.reshape(pl.col_max(pl.abs(y)), [d // 8, 8])),
                                    pl.full([1, 8], dtype=pl.FP32, value=AMAX_EPS))
                amax = pl.row_max(pl.col_expand_mul(ones_h8, part_8))
                pl.write(y_scale, [t0 + i], pl.read(amax, [0, 0]) / SCALE_MAX)

                # No clamp: amax is the tile's own maximum, so |y| * SCALE_MAX / amax
                # <= 127 and a +-127 clamp cannot bind. The cast rounds to nearest;
                # an FP16 hop before INT8 would truncate instead, which is a silent
                # one-step shift on 39% of the output.
                q = pl.row_expand_mul(y, pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX))
                q_rows[r0 : r0 + h, :] = pl.cast(q, target_type=pl.INT8, mode="rint")
        return y_q

    @jit
    def gdn_gated_rmsnorm_bf16(
        o: pl.Tensor[[t, h, d], pl.FP16],
        z: pl.Tensor[[t, h, d], pl.BF16],
        norm_w: pl.Tensor[[d], pl.BF16],
        y: pl.Out[pl.Tensor[[t, h * d], pl.BF16]],
    ):
        o_rows = pl.reshape(o, [t * h, d])
        z_rows = pl.reshape(z, [t * h, d])
        y_rows = pl.reshape(y, [t * h, d])
        for blk in pl.spmd(t // TOK_TILE, name_hint="gated_rmsnorm_bf16"):
            t0 = blk * TOK_TILE
            w_row = pl.cast(pl.reshape(norm_w[:], [1, d]), target_type=pl.FP32)
            for i in pl.unroll(TOK_TILE):
                r0 = (t0 + i) * h
                of = pl.cast(o_rows[r0 : r0 + h, :], target_type=pl.FP32)
                inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(of, of)), 1.0 / d), eps),
                                   high_precision=True)
                normed = pl.cast(pl.cast(pl.row_expand_mul(of, inv_rms),
                                         target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
                weighted = pl.cast(pl.cast(pl.col_expand_mul(normed, w_row),
                                           target_type=pl.BF16, mode="rint"), target_type=pl.FP32)

                zf = pl.cast(z_rows[r0 : r0 + h, :], target_type=pl.FP32)
                gate = pl.mul(zf, pl.recip(pl.add(pl.exp(pl.neg(zf)), 1.0)))
                y_rows[r0 : r0 + h, :] = pl.cast(pl.mul(weighted, gate),
                                                 target_type=pl.BF16, mode="rint")
        return y

    return gdn_gated_rmsnorm if quant else gdn_gated_rmsnorm_bf16


gdn_gated_rmsnorm = build_kernel()


def golden_y(o, z, norm_w, eps: float = EPS):
    """The kernel's fp32 arithmetic in its order, both bf16 narrowings included. -> fp32 [T, H*D]."""
    import torch

    t, h, d = o.shape
    of = o.float()
    inv_rms = torch.rsqrt((of * of).sum(dim=-1, keepdim=True) * (1.0 / d) + eps)
    normed = (of * inv_rms).to(torch.bfloat16).float()
    weighted = (normed * norm_w.float()).to(torch.bfloat16).float()
    zf = z.float()
    return (weighted * (zf * torch.reciprocal(torch.exp(-zf) + 1.0))).reshape(t, h * d)


def golden_quant(y):
    """The int8 epilogue on an fp32 [T, C]: -> (int8 [T, C], fp32 scale [T]).

    The kernel reaches the same scale by a high-precision reciprocal and a
    multiply, torch by a divide; the gate's one-step allowance covers the
    difference, and the simulator and hardware runs measure it.
    """
    import torch

    amax = y.abs().amax(dim=-1).clamp_min(AMAX_EPS)
    q = torch.round(y * (SCALE_MAX / amax)[:, None]).clamp(-SCALE_MAX, SCALE_MAX)
    return q.to(torch.int8), (amax / SCALE_MAX).float()


def golden_gdn_gated_rmsnorm(tensors):
    q, scale = golden_quant(golden_y(tensors["o"], tensors["z"], tensors["norm_w"]))
    tensors["y_q"].copy_(q)
    tensors["y_scale"].copy_(scale)


def golden_gdn_gated_rmsnorm_bf16(tensors):
    import torch

    y = golden_y(tensors["o"], tensors["z"], tensors["norm_w"])
    tensors["y"].copy_(y.to(torch.bfloat16))


def _draw(key: str, t: int, chunk: int, weights: str | None):
    """A block-chain tensor as a lazy spec value; one chain serves every spec and the compare."""
    import reference

    # quant_y=True so the same cached chain also carries y_q / y_scale for the gate
    return reference.block_inputs(key, t, QWEN3_8_27B, chunk, weights=weights, quant_y=True)


def _norm_weight(weights: str | None):
    """`norm.weight` of the weight set the inputs were drawn from."""
    import torch

    import reference

    if weights is None:
        return reference.make_block_weights(QWEN3_8_27B)["norm.weight"]
    return torch.load(weights, weights_only=True)["norm.weight"]


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK,
                       weights: str | None = None, quant: bool = True):
    import torch
    from golden import TensorSpec

    def draw(key, dtype):
        src = _draw(key, t, chunk, weights)
        return lambda: src().to(dtype)

    specs = [
        TensorSpec("o", [t, h, d], torch.float16, init_value=draw("o", torch.float16)),
        TensorSpec("z", [t, h, d], torch.bfloat16, init_value=draw("z", torch.bfloat16)),
        TensorSpec("norm_w", [d], torch.bfloat16, init_value=lambda: _norm_weight(weights)),
    ]
    if quant:
        specs += [TensorSpec("y_q", [t, h * d], torch.int8),
                  TensorSpec("y_scale", [t], torch.float32)]
    else:
        specs += [TensorSpec("y", [t, h * d], torch.bfloat16)]
    return specs


def compare_y_q(t: int, chunk: int, weights: str | None):
    """Two gates on `y_q`, plus the numbers the gates do not say.

    Against the kernel's own golden: +-1 INT8 step (one step is 0.8% of full
    scale, and an fp32 rsqrt or exp differing in the last bit moves a value
    across a rounding boundary) for all but 0.5% of elements. Against the
    fake-quant reference: the same step bound for all but 1e-4 of elements,
    `MIN_EXACT_VS_REFERENCE` of them identical, and the per-token scale within
    2e-2. The exact-match rate is a gate and not a statistic -- a rounding-mode
    change moves a large fraction of the output by exactly one step, which a
    step bound alone accepts.

    Printed beside the verdict: the dequantized output's distance from the
    float64 `y` and from the fake-quant reference's dequantized `y`.
    """
    import torch

    from golden import ratio_allclose

    import reference

    golden_gate = ratio_allclose(atol=1, rtol=0, max_error_ratio=0.005)
    ref_gate = ratio_allclose(atol=1, rtol=0, max_error_ratio=1e-4)

    def compare(actual, expected, actual_outputs=None, **kw):
        ok, detail = golden_gate(actual, expected, actual_outputs=actual_outputs, **kw)
        step = (actual.to(torch.int32) - expected.to(torch.int32)).abs()
        lines = [f"[stats] y_q vs golden: {float((step == 0).float().mean()) * 100:.4f}% exact, "
                 f"max {int(step.max())} step(s) off"]
        if actual_outputs is not None and "y_scale" in actual_outputs:
            scale_dev = actual_outputs["y_scale"].cpu().float()
            q_ref = _draw("y_q", t, chunk, weights)()
            scale_ref = _draw("y_scale", t, chunk, weights)().float()
            ref_ok, ref_detail = ref_gate(actual, q_ref, actual_outputs=actual_outputs, **kw)
            step = (actual.to(torch.int32) - q_ref.to(torch.int32)).abs()
            exact_ref = float((step == 0).float().mean())
            scale_rel = float(((scale_dev - scale_ref).abs() / scale_ref).max())
            scale_ok = scale_rel <= 2e-2
            exact_ok = exact_ref >= MIN_EXACT_VS_REFERENCE
            ref_ok = ref_ok and exact_ok
            lines.append(f"[stats] y_q vs fake-quant reference: "
                         f"{exact_ref * 100:.4f}% exact (gate {MIN_EXACT_VS_REFERENCE * 100:.0f}%), "
                         f"max {int(step.max())} step(s) off, "
                         f"y_scale max rel diff {scale_rel:.2e} "
                         f"-> {'PASS' if ref_ok and scale_ok else 'FAIL'}")
            y_dev = reference.dequantize(actual, scale_dev)
            y_ref = _draw("y", t, chunk, weights)().reshape(t, -1)
            y_fq = reference.dequantize(q_ref, scale_ref)
            lines.append(f"[stats] dequantised y: rel frob {float((y_dev - y_ref).norm() / y_ref.norm()):.3e} "
                         f"vs float64 y, {float((y_dev - y_fq).norm() / y_fq.norm()):.3e} "
                         f"vs fake-quant reference")
            if not (ref_ok and scale_ok):
                why = [] if exact_ok else [f"only {exact_ref * 100:.2f}% exact "
                                           f"(< {MIN_EXACT_VS_REFERENCE * 100:.0f}%): "
                                           f"a systematic rounding change, not boundary flips"]
                if not scale_ok:
                    why.append(f"y_scale off by {scale_rel:.2e}")
                ok, detail = False, "; ".join([detail, f"vs fake-quant reference: {ref_detail}", *why])
        print("\n".join(lines), flush=True)
        return ok, detail

    return compare


def compare_y(t: int, chunk: int, weights: str | None):
    """The bf16 variant: the bf16 gate the roadmap sets for bf16 outputs, plus the truth distance."""
    from golden import ratio_allclose

    gate = ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)

    def compare(actual, expected, **kw):
        ok, detail = gate(actual, expected, **kw)
        y_ref = _draw("y", t, chunk, weights)().reshape(t, -1)
        y_dev = actual.double()
        print(f"[stats] bf16 y: rel frob {float((y_dev - y_ref).norm() / y_ref.norm()):.3e} vs float64 y, "
              f"{float((y_dev - expected.double()).norm() / y_ref.norm()):.3e} vs golden", flush=True)
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
    parser.add_argument("--heads", type=int, default=H, help="value heads H")
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
    parser.add_argument("--bf16-out", action="store_true", default=False,
                        help="the bf16-output variant, to price the int8 epilogue")
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

    quant = not args.bf16_out
    result = run(
        fn=build_kernel(t=args.seq_len, h=args.heads, quant=quant),
        specs=build_tensor_specs(t=args.seq_len, h=args.heads, weights=args.weights, quant=quant),
        golden_fn=golden_gdn_gated_rmsnorm if quant else golden_gdn_gated_rmsnorm_bf16,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn=({"y_q": compare_y_q(args.seq_len, CHUNK, args.weights)} if quant
                    else {"y": compare_y(args.seq_len, CHUNK, args.weights)}),
        compile_only=args.compile_only,
    )
    if args.rounds and result.bench is not None:
        samples = [s for s in result.bench.per_round("effective") if s > 0]
        if samples:
            print(f"[bench] gated_rmsnorm{'' if quant else '_bf16'} T={args.seq_len} H={args.heads} "
                  f"rounds={len(samples)} mean={statistics.fmean(samples):.1f} us "
                  f"median={statistics.median(samples):.1f} min={min(samples):.1f} "
                  f"max={max(samples):.1f}", flush=True)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
