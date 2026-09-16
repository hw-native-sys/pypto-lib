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
this is numerically the chain `test_gdn_stages.py --chain` validates, with the
same fp16 delta rule inside.
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
from config import GDN_TILING, QWEN3_8_27B

# model shape
C = QWEN3_8_27B.hidden_size                 # the residual stream
CQKV = QWEN3_8_27B.qkv_width                # what in_proj_qkv writes
CV = QWEN3_8_27B.value_width                # what in_proj_z writes, out_proj reads
H = QWEN3_8_27B.linear_num_value_heads
HG = QWEN3_8_27B.linear_num_key_heads
D = QWEN3_8_27B.linear_value_head_dim
KCONV = QWEN3_8_27B.linear_conv_kernel_dim  # convolution taps
CHUNK = GDN_TILING.chunk

# case shape
T = 8192                # tokens (single sequence, B = 1)

# At T = 8192 the block holds about 1.3 GB of intermediates at once: 0.6 GB
# inside the delta rule, and the projections, the convolution's three outputs
# and the quantised hand-offs on top. `gdn_layer` needs 1 GiB for its 0.6 GB,
# so this follows the same ratio, rounded up: the runtime takes powers of two
# only. Under the default 256 MiB per ring the layer alone fails with
# orch_error_code=2 HEAP_RING_DEADLOCK.
BLOCK_RING_HEAP = 4 * 1024 * 1024 * 1024

# tiling
HALO_TILE = 1024        # channels per block of the zero fill above the projection

OUTPUTS = ("out",)      # bench.py reads it


def run_config(platform: str = "a2a3", device: int = 0) -> dict:
    """The config `golden.run` needs for this block. The ring heap is not optional."""
    return dict(platform=platform, device_id=device, ring_heap=BLOCK_RING_HEAP)


def build_kernel(t: int = T, h: int = H, hg: int = HG, d: int = D, chunk: int = CHUNK,
                 inline: bool = False):
    """The block at one shape, composing fourteen operators as inlined callees."""
    c, cqkv, cv, k = C, CQKV, CV, KCONV
    key_width = hg * d

    op_quant = quant_x.build_kernel(t=t, inline=True)
    op_qkv = in_proj_qkv.build_kernel(t=t, inline=True, row_off=k - 1)
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
    op_delta = gdn_layer.build_kernel(t=t, h=h, d=d, chunk=chunk, hg=hg, inline=True)
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
        neg_eye2: pl.Tensor[[2 * chunk, chunk], pl.FP16],
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
        op_delta(q_n, k_n, v_conv, g, beta, tril, mask_strict, neg_eye2, o)

        y_q = pl.create_tensor([t, cv], dtype=pl.INT8)
        y_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        op_norm(o, z, norm_w, y_q, y_scale)

        op_out(y_q, y_scale, w_out, w_out_scale, out)
        return out

    return gdn_block


def build_tensor_specs(t: int = T, h: int = H, hg: int = HG, d: int = D,
                       chunk: int = CHUNK, weights: str | None = None):
    """Every weight the block reads, plus the delta rule's three constants."""
    import torch
    from golden import TensorSpec

    import reference

    def block_weights():
        if weights is None:
            return reference.make_block_weights(QWEN3_8_27B)
        return torch.load(weights, weights_only=True)

    def quantized(module):
        q, scale = a8w8_linear.quantized_weight(QWEN3_8_27B, module, weights)
        return q, scale

    w_qkv, w_qkv_scale = quantized("in_proj_qkv")
    w_z, w_z_scale = quantized("in_proj_z")
    w_out, w_out_scale = quantized("out_proj")

    def stacked_ab():
        w = block_weights()
        return torch.cat([w["in_proj_a.weight"], w["in_proj_b.weight"]]).to(torch.bfloat16)

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
        TensorSpec("neg_eye2", [2 * chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.neg_eye_stack(chunk)),
        TensorSpec("out", [t, C], torch.bfloat16),
    ]


def golden_gdn_block(tensors):
    """The fake-quant chain: the block in float64 with the two activation quantisations."""
    tensors["out"].copy_(_chain(tensors["x"].shape[0], None, quant=True)["out"])


_CHAIN_CACHE: dict[tuple, dict] = {}


def _chain(t: int, weights: str | None, quant: bool, chunk: int = CHUNK):
    """The float64 block, cached. *quant* picks the fake-quant chain or the truth."""
    import torch

    import reference

    a8w8_linear.host_torch_setup()
    key = (t, weights, quant, chunk)
    if key not in _CHAIN_CACHE:
        if weights is None:
            w = reference.make_block_weights(QWEN3_8_27B)
        else:
            w = torch.load(weights, weights_only=True)
        if quant:
            w = reference.quantize_weights(w)
        _CHAIN_CACHE[key] = reference.block(
            reference.make_block_inputs(t, QWEN3_8_27B), w, QWEN3_8_27B, chunk,
            quant_x=quant, quant_y=quant)
    return _CHAIN_CACHE[key]


# Acceptance, both against the quantization floor rather than a fixed number.
# The block may sit this far above what W8A8 alone costs, in total and on its
# worst token row.
MAX_OVER_FLOOR = 1.05
MAX_ROW_OVER_FLOOR = 1.25


def compare_out(t: int, weights: str | None):
    """Score the block against the float64 truth, beside what quantization alone costs.

    **Not** an element-wise gate against the fake-quant chain, which is the
    obvious choice and the wrong one. `out_proj` reads int8, so a difference
    far below the int8 step still flips values that sit near a rounding
    boundary: the kernels' own dtypes move `y` by 2.6e-03, that flips 1.0% of
    `y_q` by one step, and those flips alone move `out` by 1.4e-02. An
    element-wise comparison after a quantization step measures which side of a
    boundary a value landed on, not whether the kernel is right.

    What is meaningful is how much the block adds to what W8A8 already costs,
    so both gates are ratios against the quantization floor and neither
    carries a tolerance of its own. The second one is per token row: a
    corrupted row barely moves the whole-tensor norm -- one dead row in 2048
    is 1.12x the floor -- while moving the worst row by more than 10x.
    """
    def compare(actual, expected, **_kwargs):
        dev = actual.double()
        fq = _chain(t, weights, quant=True)["out"]
        truth = _chain(t, weights, quant=False)["out"]

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())

        floor, dev_truth = rel(fq, truth), rel(dev, truth)
        row_floor, dev_row = worst_row(fq, truth), worst_row(dev, truth)
        over, row_over = dev_truth / floor, dev_row / row_floor
        print(f"[stats] out: rel frob {dev_truth:.3e} vs the float64 truth against a "
              f"quantization floor of {floor:.3e} ({over:.3f}x); worst token row "
              f"{dev_row:.3e} against {row_floor:.3e} ({row_over:.3f}x); "
              f"{rel(dev, fq):.3e} vs the fake-quant chain", flush=True)
        ok = over <= MAX_OVER_FLOOR and row_over <= MAX_ROW_OVER_FLOOR
        detail = (f"{over:.3f}x the quantization floor (<={MAX_OVER_FLOOR}), "
                  f"worst row {row_over:.3f}x (<={MAX_ROW_OVER_FLOOR})")
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
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
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
        config=run_config(args.platform, args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"out": compare_out(args.seq_len, args.weights)},
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
