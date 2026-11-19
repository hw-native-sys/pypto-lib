# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B's whole Gated DeltaNet block: hidden states in, hidden states out.

    x_q, x_scale = quant_x(x)                     per-token int8 of the hidden states
    qkv          = in_proj_qkv(x_q, x_scale)      5120 -> 10240, A8W8
    z            = in_proj_z(x_q, x_scale)        5120 -> 6144,  A8W8
    a, b         = in_proj_ab(x)                  5120 -> 48 each, bf16
    q, k, v      = short_conv(qkv)                depthwise causal conv, then silu
    q, k, beta, g = qk_norm_gate(q, k, a, b)      L2-norm, sigmoid, softplus
    o            = gdn_layer(q, k, v, g, beta)    the delta rule, six stages
    y_q, y_scale = gated_rmsnorm(o, z)            the gated norm, int8 epilogue
    out          = out_proj(y_q, y_scale)         6144 -> 5120, A8W8

Fourteen operators as one program. Everything between `x` and `out` is
allocated here and never leaves.

Three interfaces are shaped by the composition rather than by any one operator.
The convolution reads the K-1 tokens before each window, so `in_proj_qkv`
writes at `row_off` into the padded buffer the convolution wants, and the K-1
rows above are zeroed here; nothing copies `[T, C]` into `[T + K - 1, C]`. The
convolution writes q, k and v as three tensors rather than one, so each
consumer reads the rows of a contiguous buffer -- and v, which goes straight to
the delta rule, is written in fp16 there rather than cast by a later pass. And
`in_proj_ab` reads `x` in bf16 while everything else reads the quantised copy,
which is the one place the block reads the hidden states twice.

The dtype at each boundary is the one the standalone operators exchange, so
this is numerically the chain `gdn_layer.py` validates, with the same fp16
delta rule inside.
"""
import pypto.language as pl

import a8w8_linear
import gated_rmsnorm
import gdn_layer
import in_proj_ab
import in_proj_qkv
import in_proj_z
import out_proj
import qk_norm_gate
import quant_x
import short_conv
import solve_tril
from chunk_cumsum import CHUNK_TILE, DISPATCH_TOKENS
from config import GDN_QUANT, QWEN3_8_27B

# model shape
C = QWEN3_8_27B.hidden_size                 # the residual stream
CQKV = QWEN3_8_27B.qkv_width                # what in_proj_qkv writes
CV = QWEN3_8_27B.value_width                # what in_proj_z writes, out_proj reads
H = QWEN3_8_27B.linear_num_value_heads
HG = QWEN3_8_27B.linear_num_key_heads
D = QWEN3_8_27B.linear_value_head_dim
KCONV = QWEN3_8_27B.linear_conv_kernel_dim  # convolution taps
CHUNK = CHUNK_TILE

# validation
DEFAULT_TEST_SEQ_LEN = 512

# Per-ring heap override for sequences longer than one dispatch group.
BLOCK_RING_HEAP = 4 * 1024 * 1024 * 1024

# tiling
HALO_TILE = 1024        # channels per block of the zero fill above the projection

OUTPUTS = ("out",)      # bench.py reads it


def run_config(backend_type: str = "a2a3", device: int = 0, sim: bool = False, *, t: int = DEFAULT_TEST_SEQ_LEN) -> dict:
    """Configure the block's backend and ring heap for the token count."""
    config = dict(backend_type=backend_type, execution_mode=sim, device_id=device)
    if t > DISPATCH_TOKENS:
        config["ring_heap"] = BLOCK_RING_HEAP
    return config


def build_kernel(t: int = DEFAULT_TEST_SEQ_LEN, h: int = H, hg: int = HG, d: int = D, chunk: int = CHUNK,
                 inline: bool = False):
    """The block at one shape, composing fourteen operators as inlined callees."""
    if t <= 0 or t % DISPATCH_TOKENS:
        raise ValueError(f"t={t} must be a positive multiple of {DISPATCH_TOKENS}")

    c, cqkv, cv, k = C, CQKV, CV, KCONV
    key_width = hg * d

    op_quant = quant_x.build_kernel(t=t, inline=True)
    op_qkv = a8w8_linear.build_kernel(
        t, c, cqkv, name=in_proj_qkv.MODULE, inline=True, row_off=k - 1,
        m_tile=in_proj_qkv.M_TILE, n_tile=in_proj_qkv.N_TILE, k_tile=in_proj_qkv.K_TILE,
        m_group=in_proj_qkv.M_GROUP, n_group=in_proj_qkv.N_GROUP,
    )
    op_z = in_proj_z.build_kernel(t=t, inline=True)
    op_ab = in_proj_ab.build_kernel(t=t, inline=True)
    conv = dict(t=t, c=cqkv, k=k, inline=True)
    op_conv_q = short_conv.build_kernel(**conv, c_off=0, c_out=key_width,
                                        out_heads=hg, name="conv_q")
    op_conv_k = short_conv.build_kernel(**conv, c_off=key_width, c_out=key_width,
                                        out_heads=hg, name="conv_k")
    op_conv_v = short_conv.build_kernel(**conv, c_off=2 * key_width, c_out=h * d,
                                        out_heads=h, out_dtype=pl.FP16, name="conv_v")
    op_qkng = qk_norm_gate.build_kernel(t=t, h=h, hg=hg, d=d, inline=True)
    # The delta rule is one inline callee already, and it reads its shape off the
    # tensors it is handed rather than off a per-shape build.
    op_delta = gdn_layer.gdn_layer
    op_norm = gated_rmsnorm.build_kernel(t=t, h=h, d=d, inline=True)
    op_out = out_proj.build_kernel(t=t, inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def gdn_block(
        x: pl.Tensor[[t, c], pl.BF16],
        w_qkv: pl.Tensor[[cqkv, c], pl.INT8],
        w_qkv_scale: pl.Tensor[[cqkv], pl.FP32],
        w_z: pl.Tensor[[cv, c], pl.INT8],
        w_z_scale: pl.Tensor[[cv], pl.FP32],
        w_ab: pl.Tensor[[2 * h, c], pl.BF16],
        conv_w: pl.Tensor[[k, cqkv], pl.FP32],
        a_log: pl.Tensor[[h], pl.BF16],
        dt_bias: pl.Tensor[[h], pl.BF16],
        norm_w: pl.Tensor[[d], pl.BF16],
        w_out: pl.Tensor[[c, cv], pl.INT8],
        w_out_scale: pl.Tensor[[c], pl.FP32],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        out: pl.Out[pl.Tensor[[t, c], pl.BF16]],
    ):
        x_q = pl.create_tensor([t, c], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        op_quant(x, x_q, x_scale)

        # the convolution's first window reads the K-1 tokens before the sequence;
        # for prefill they are zero, and they are where a decode cache would sit
        qkv_pad = pl.create_tensor([t + k - 1, cqkv], dtype=pl.BF16)
        for blk in pl.spmd(cqkv // HALO_TILE, name_hint="conv_halo"):
            c0 = blk * HALO_TILE
            qkv_pad[0 : k - 1, c0 : c0 + HALO_TILE] = pl.full(
                [k - 1, HALO_TILE], dtype=pl.BF16, value=0.0)
        op_qkv(x_q, x_scale, w_qkv, w_qkv_scale, qkv_pad)

        z = pl.create_tensor([t, cv], dtype=pl.BF16)
        op_z(x_q, x_scale, w_z, w_z_scale, z)

        a_proj = pl.create_tensor([t, h], dtype=pl.BF16)
        b_proj = pl.create_tensor([t, h], dtype=pl.BF16)
        op_ab(x, w_ab, a_proj, b_proj)

        q_conv = pl.create_tensor([t, hg, d], dtype=pl.BF16)
        k_conv = pl.create_tensor([t, hg, d], dtype=pl.BF16)
        v_conv = pl.create_tensor([t, h, d], dtype=pl.FP16)
        op_conv_q(qkv_pad, conv_w, q_conv)
        op_conv_k(qkv_pad, conv_w, k_conv)
        op_conv_v(qkv_pad, conv_w, v_conv)

        q_n = pl.create_tensor([t, hg, d], dtype=pl.FP16)
        k_n = pl.create_tensor([t, hg, d], dtype=pl.FP16)
        beta = pl.create_tensor([h, t], dtype=pl.FP32)
        g = pl.create_tensor([t, h], dtype=pl.FP32)
        op_qkng(q_conv, k_conv, a_proj, b_proj, a_log, dt_bias, q_n, k_n, beta, g)

        o = pl.create_tensor([t, h, d], dtype=pl.FP16)
        op_delta(q_n, k_n, v_conv, g, beta, tril, mask_strict, eye,
                 m_diag, m_low, o)

        y_q = pl.create_tensor([t, cv], dtype=pl.INT8)
        y_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        op_norm(o, z, norm_w, y_q, y_scale)

        op_out(y_q, y_scale, w_out, w_out_scale, out)
        return out

    return gdn_block


def build_tensor_specs(t: int = DEFAULT_TEST_SEQ_LEN, h: int = H, hg: int = HG, d: int = D,
                       chunk: int = CHUNK, weights: str | None = None):
    """Every weight the block reads, plus the delta rule's three constants."""
    import torch
    from golden import TensorSpec

    import reference
    from weights import load_weights, quantized_weight, stacked_ab_weight

    def block_weights():
        return load_weights(QWEN3_8_27B, weights)

    w_qkv, w_qkv_scale = quantized_weight(QWEN3_8_27B, in_proj_qkv.MODULE, weights)
    w_z, w_z_scale = quantized_weight(QWEN3_8_27B, in_proj_z.MODULE, weights)
    w_out, w_out_scale = quantized_weight(QWEN3_8_27B, out_proj.MODULE, weights)

    def stacked_ab():
        return stacked_ab_weight(block_weights())

    def taps():
        return block_weights()["conv1d.weight"].reshape(CQKV, KCONV).t().contiguous().float()

    def param(name):
        return lambda: block_weights()[name].to(torch.bfloat16)

    def init_tril():
        return torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))

    def init_mask_strict():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows > cols).float()

    return [
        TensorSpec("x", [t, C], torch.bfloat16,
                   init_value=lambda: reference.make_block_inputs(t, QWEN3_8_27B)),
        TensorSpec("w_qkv", [CQKV, C], torch.int8, init_value=w_qkv),
        TensorSpec("w_qkv_scale", [CQKV], torch.float32, init_value=w_qkv_scale),
        TensorSpec("w_z", [CV, C], torch.int8, init_value=w_z),
        TensorSpec("w_z_scale", [CV], torch.float32, init_value=w_z_scale),
        TensorSpec("w_ab", [2 * h, C], torch.bfloat16, init_value=stacked_ab),
        TensorSpec("conv_w", [KCONV, CQKV], torch.float32, init_value=taps),
        TensorSpec("a_log", [h], torch.bfloat16, init_value=param("A_log")),
        TensorSpec("dt_bias", [h], torch.bfloat16, init_value=param("dt_bias")),
        TensorSpec("norm_w", [d], torch.bfloat16, init_value=param("norm.weight")),
        TensorSpec("w_out", [C, CV], torch.int8, init_value=w_out),
        TensorSpec("w_out_scale", [C], torch.float32, init_value=w_out_scale),
        TensorSpec("tril", [chunk, chunk], torch.float32, init_value=init_tril),
        TensorSpec("mask_strict", [chunk, chunk], torch.float32, init_value=init_mask_strict),
        TensorSpec("eye", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.eye_block(chunk)),
        TensorSpec("m_diag", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[0]),
        TensorSpec("m_low", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[1]),
        TensorSpec("out", [t, C], torch.bfloat16),
    ]


def _reference_weights(inputs):
    """Translate the block's weight tensors to the Torch reference layout."""
    w_a, w_b = inputs["w_ab"].chunk(2, dim=0)
    return {
        "in_proj_qkv.weight": inputs["w_qkv"],
        "in_proj_qkv.weight_scale": inputs["w_qkv_scale"],
        "in_proj_z.weight": inputs["w_z"],
        "in_proj_z.weight_scale": inputs["w_z_scale"],
        "in_proj_a.weight": w_a,
        "in_proj_b.weight": w_b,
        "conv1d.weight": inputs["conv_w"].t().unsqueeze(1),
        "A_log": inputs["a_log"],
        "dt_bias": inputs["dt_bias"],
        "norm.weight": inputs["norm_w"],
        "out_proj.weight": inputs["w_out"],
        "out_proj.weight_scale": inputs["w_out_scale"],
    }


def _reference_out(inputs, weights, *, quant: bool):
    import block_reference

    chunk = inputs["tril"].shape[0]
    return block_reference.block(inputs["x"], weights, QWEN3_8_27B, chunk, quant_x=quant, quant_y=quant)["out"]


def golden_gdn_block(tensors):
    """Fill the fake-quant golden from the block's input tensors."""
    from reference import host_torch_setup

    host_torch_setup()
    tensors["out"].copy_(_reference_out(tensors, _reference_weights(tensors), quant=True))


# Acceptance, both against the quantization floor rather than a fixed number.
# The block may sit this far above what W8A8 alone costs, in total and on its
# worst token row.
MAX_OVER_FLOOR = 1.05
MAX_ROW_OVER_FLOOR = 1.25
# ...and the distance to the fake-quant chain itself, as a fraction of that
# floor. The ratios above compare two distances to a third quantity, so they
# do not on their own constrain how far the kernel is from the answer it
# should produce: an output mirrored across the truth, `2 * truth - fq`, sits
# at exactly the floor distance from the truth and passes them at 1.000x while
# being twice the floor from `fq`.
MAX_VS_GOLDEN = 1.00
MAX_ROW_VS_GOLDEN = 1.50


def compare_out(weights: str | None):
    """Compare the harness golden and FP64 truth using the run's quantization floor."""
    import math

    import torch

    import reference
    from weights import load_weights

    def compare(actual, expected, *, inputs, **_kwargs):
        if actual.shape != expected.shape:
            return False, f"output shape {tuple(actual.shape)} != golden shape {tuple(expected.shape)}"

        reference.host_torch_setup()
        quantized_weights = _reference_weights(inputs)
        original_weights = load_weights(QWEN3_8_27B, weights)
        truth_weights = quantized_weights.copy()
        for name in GDN_QUANT.weights:
            weight = original_weights.get(name)
            if weight is None or not weight.is_floating_point():
                return False, f"quantization baseline requires the original floating-point weight {name}"
            q, scale = reference.quantize_rows(weight)
            scale_name = name.removesuffix(".weight") + ".weight_scale"
            if not torch.equal(q, quantized_weights[name]) or not torch.equal(scale, quantized_weights[scale_name]):
                return False, f"{name} does not match the run's quantized weight and scale; use the original --weights"
            truth_weights[name] = weight

        dev = actual.double()
        golden = expected.double()
        fq = _reference_out(inputs, quantized_weights, quant=True)
        truth = _reference_out(inputs, truth_weights, quant=False)

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())

        floor, dev_truth = rel(fq, truth), rel(dev, truth)
        row_floor, dev_row = worst_row(fq, truth), worst_row(dev, truth)
        if not (math.isfinite(floor) and floor > 0 and math.isfinite(row_floor) and row_floor > 0):
            return False, "quantization baseline requires finite, positive tensor and worst-row errors"
        over, row_over = dev_truth / floor, dev_row / row_floor
        dev_golden, dev_row_golden = rel(dev, golden), worst_row(dev, golden)
        near, row_near = dev_golden / floor, dev_row_golden / row_floor
        print(f"[stats] out: rel frob {dev_truth:.3e} vs the float64 truth against a "
              f"quantization floor of {floor:.3e} ({over:.3f}x); worst token row "
              f"{dev_row:.3e} against {row_floor:.3e} ({row_over:.3f}x); "
              f"{dev_golden:.3e} vs the fake-quant golden ({near:.3f} of the floor, "
              f"worst row {row_near:.3f})", flush=True)
        ok = (over <= MAX_OVER_FLOOR and row_over <= MAX_ROW_OVER_FLOOR
              and near <= MAX_VS_GOLDEN and row_near <= MAX_ROW_VS_GOLDEN)
        detail = (f"{over:.3f}x the quantization floor (<={MAX_OVER_FLOOR}), "
                  f"worst row {row_over:.3f}x (<={MAX_ROW_OVER_FLOOR}); "
                  f"vs the fake-quant golden {near:.3f} (<={MAX_VS_GOLDEN}), "
                  f"worst row {row_near:.3f} (<={MAX_ROW_VS_GOLDEN})")
        return ok, detail

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
    parser.add_argument("--weights", type=str, default=None,
                        help="the original floating-point layer from weights.py; default: random weights")
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the block over this many rounds (device only)")
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
        fn=build_kernel(t=args.seq_len),
        specs=build_tensor_specs(t=args.seq_len, weights=args.weights),
        golden_fn=golden_gdn_block,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=run_config(args.backend_type, args.device, args.execution_mode, t=args.seq_len),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"out": compare_out(args.weights)},
        compile_only=args.compile_only,
    )
    if args.rounds and result.bench is not None:
        samples = [s for s in result.bench.per_round("effective") if s > 0]
        if samples:
            print(f"[bench] gdn_block T={args.seq_len} rounds={len(samples)} "
                  f"mean={statistics.fmean(samples):.1f} us "
                  f"median={statistics.median(samples):.1f} min={min(samples):.1f}", flush=True)
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
