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
weight applied there, gate applied in fp32.
"""
import pypto.language as pl

from config import GDN_QUANT, QWEN3_8_27B

# model shape
H = QWEN3_8_27B.linear_num_value_heads      # value heads
D = QWEN3_8_27B.linear_value_head_dim       # head dimension, the norm's axis
EPS = QWEN3_8_27B.rms_norm_eps

# quantization
SCALE_MAX = GDN_QUANT.scale_max
AMAX_EPS = GDN_QUANT.amax_eps

# validation
DEFAULT_TEST_SEQ_LEN = 512

# gate
MIN_EXACT_VS_REFERENCE = 0.95   # of y_q elements identical to the fake-quant reference

# tiling
TOK_TILE = 16           # tokens per block; 16 scalar scale writes fill one 64-byte line

# `y_scale` is a [1, T] row and not a [T] vector: `out_proj` reshapes its slice
# to the [M, 1] column it multiplies by, and a tile store must be 2-D anyway.
OUTPUTS = ("y_q", "y_scale")


def build_kernel(t: int = DEFAULT_TEST_SEQ_LEN, h: int = H, d: int = D, eps: float = EPS,
                 inline: bool = False):
    """Build gated RMSNorm with per-token INT8 quantization."""
    # A block walks one TOK_TILE of rows, so a token count that does not divide it
    # leaves the tail silently unwritten -- the same trap `quant_x` and `short_conv`
    # refuse. The composed block rejects such shapes earlier; this covers the
    # standalone `--seq-len` path.
    if t % TOK_TILE:
        raise ValueError(f"t={t} must be a multiple of TOK_TILE={TOK_TILE}")

    jit = pl.jit.inline if inline else pl.jit

    @jit
    def gdn_gated_rmsnorm(
        o: pl.Tensor[[t, h, d], pl.FP16],
        z: pl.Tensor[[t, h * d], pl.BF16],
        norm_w: pl.Tensor[[d], pl.BF16],
        y_q: pl.Out[pl.Tensor[[t, h * d], pl.INT8]],
        y_scale: pl.Out[pl.Tensor[[1, t], pl.FP32]],
    ):
        o_flat = pl.reshape(o, [t, h * d])
        for blk in pl.spmd(t // TOK_TILE, name_hint="gated_rmsnorm"):
            t0 = blk * TOK_TILE
            w_row = pl.cast(pl.reshape(norm_w[:], [1, d]), target_type=pl.FP32)
            ones_h8 = pl.full([h, 8], dtype=pl.FP32, value=1.0)   # broadcasts the amax partials
            # stage=2: the tokens are independent, so the next one's loads overlap this
            # one's arithmetic. The kernel is latency-bound on the per-token chain.
            for i in pl.pipeline(TOK_TILE, stage=2):
                r0 = t0 + i
                # one token is h*d contiguous values: load the whole run, then take the
                # [h, d] view the norm reduces over. Loading [h, d] directly issues h
                # separate row transfers of d elements.
                of_run = pl.cast(o_flat[r0 : r0 + 1, :], target_type=pl.FP32)
                of = pl.reshape(of_run, [h, d])
                inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(of, of)), 1.0 / d), eps),
                                   high_precision=True)
                # the module narrows the normed values, then the weighted ones; both stay
                # narrow until used, the fp32 copy of a bf16-rounded value being only buffer
                normed_bf = pl.cast(pl.row_expand_mul(of, inv_rms), target_type=pl.BF16, mode="rint")
                normed = pl.cast(normed_bf, target_type=pl.FP32)
                weighted_bf = pl.cast(pl.col_expand_mul(normed, w_row), target_type=pl.BF16, mode="rint")

                zf_run = pl.cast(z[r0 : r0 + 1, :], target_type=pl.FP32)
                zf = pl.reshape(zf_run, [h, d])
                gate = pl.div(zf, pl.add(pl.exp(pl.neg(zf)), 1.0))
                y = pl.mul(pl.cast(weighted_bf, target_type=pl.FP32), gate)

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
                pl.write(y_scale, [0, r0], pl.read(amax, [0, 0]) / SCALE_MAX)

                # No clamp: amax is the tile's own maximum, so |y| * SCALE_MAX / amax
                # <= 127 and a +-127 clamp cannot bind. The cast rounds to nearest;
                # an FP16 hop before INT8 would truncate instead, which is a silent
                # one-step shift on 39% of the output.
                # recip, not div, here: one reciprocal feeds a whole tile's multiply
                q = pl.row_expand_mul(y, pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX))
                q_i8 = pl.cast(q, target_type=pl.INT8, mode="rint")
                y_q[r0 : r0 + 1, :] = pl.reshape(q_i8, [1, h * d])
        return y_q

    return gdn_gated_rmsnorm


gdn_gated_rmsnorm = build_kernel()


def golden_y(o, z, norm_w, eps: float = EPS):
    """The kernel's fp32 arithmetic in its order, both bf16 narrowings included. -> fp32 [T, H*D]."""
    import torch

    t, h, d = o.shape
    of = o.float()
    inv_rms = torch.rsqrt((of * of).sum(dim=-1, keepdim=True) * (1.0 / d) + eps)
    normed = (of * inv_rms).to(torch.bfloat16).float()
    weighted = (normed * norm_w.float()).to(torch.bfloat16).float()
    # z arrives as the [T, H*D] row `in_proj_z` writes; the norm is over the head
    zf = z.float().reshape(t, h, d)
    return (weighted * (zf / (torch.exp(-zf) + 1.0))).reshape(t, h * d)


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
    tensors["y_scale"].copy_(scale.reshape(1, -1))


def gated_rmsnorm_ref(o, z, weight, eps: float = EPS):
    """Float64 RMSNorm per head followed by the SiLU gate."""
    import torch
    from torch.nn.functional import silu

    from reference import REF_DTYPE

    of = o.to(REF_DTYPE)
    var = (of * of).mean(dim=-1, keepdim=True)
    return of * torch.rsqrt(var + eps) * weight.to(REF_DTYPE) * silu(z.to(REF_DTYPE))


def build_tensor_specs(t: int = DEFAULT_TEST_SEQ_LEN, h: int = H, d: int = D,
                       weights: str | None = None):
    import torch
    from golden import TensorSpec

    from weights import load_layer

    def output():
        generator = torch.Generator().manual_seed(42)
        return torch.randn(t, h, d, generator=generator).to(torch.float16)

    def gate_input():
        generator = torch.Generator().manual_seed(43)
        return (torch.randn(t, h * d, generator=generator) * 0.5).to(torch.bfloat16)

    def norm_weight():
        if weights is None:
            return torch.ones(d, dtype=torch.bfloat16)
        return load_layer(weights)["norm.weight"]

    return [
        TensorSpec("o", [t, h, d], torch.float16, init_value=output),
        TensorSpec("z", [t, h * d], torch.bfloat16, init_value=gate_input),
        TensorSpec("norm_w", [d], torch.bfloat16, init_value=norm_weight),
        TensorSpec("y_q", [t, h * d], torch.int8),
        TensorSpec("y_scale", [1, t], torch.float32),
    ]


def compare_y_q():
    """Check INT8 steps, exact matches and scale against the operator references."""
    import torch
    from golden import ratio_allclose

    from reference import quantize_rows

    golden_gate = ratio_allclose(atol=1, rtol=0, max_error_ratio=0.005)
    ref_gate = ratio_allclose(atol=1, rtol=0, max_error_ratio=1e-4)

    def compare(actual, expected, *, inputs, actual_outputs, **kwargs):
        ok, detail = golden_gate(actual, expected, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
        o = inputs["o"]
        z = inputs["z"].reshape(o.shape)
        y_ref = gated_rmsnorm_ref(o, z, inputs["norm_w"]).flatten(1)
        q_ref, scale_ref = quantize_rows(y_ref)
        ref_ok, ref_detail = ref_gate(actual, q_ref, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
        step = (actual.to(torch.int32) - q_ref.to(torch.int32)).abs()
        exact_ref = float((step == 0).float().mean())
        scale_dev = actual_outputs["y_scale"].cpu().float().reshape(-1)
        scale_rel = float(((scale_dev - scale_ref).abs() / scale_ref).max())
        passed = ok and ref_ok and exact_ref >= MIN_EXACT_VS_REFERENCE and scale_rel <= 2e-2
        stats = (f"{exact_ref:.4%} exact vs float64 quantized reference (minimum {MIN_EXACT_VS_REFERENCE:.0%}), "
                 f"max {int(step.max())} INT8 step(s) off, scale max rel diff {scale_rel:.2e} (limit 2e-2)")
        print(f"[stats] {stats}", flush=True)
        return passed, f"{detail} {ref_detail} {stats}".strip()

    return compare


if __name__ == "__main__":
    import argparse
    import os
    import statistics

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_TEST_SEQ_LEN)
    parser.add_argument("--heads", type=int, default=H, help="value heads H")
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
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
        fn=build_kernel(t=args.seq_len, h=args.heads),
        specs=build_tensor_specs(t=args.seq_len, h=args.heads, weights=args.weights),
        golden_fn=golden_gdn_gated_rmsnorm,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-3,
        atol=1e-5,
        compare_fn={"y_q": compare_y_q()},
        compile_only=args.compile_only,
    )
    if args.rounds and result.bench is not None:
        samples = [s for s in result.bench.per_round("effective") if s > 0]
        if samples:
            print(f"[bench] gated_rmsnorm T={args.seq_len} H={args.heads} "
                  f"rounds={len(samples)} mean={statistics.fmean(samples):.1f} us "
                  f"median={statistics.median(samples):.1f} min={min(samples):.1f} "
                  f"max={max(samples):.1f}", flush=True)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
