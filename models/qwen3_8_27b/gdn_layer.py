# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B's Gated DeltaNet layer: the six operators as one forward.

    g_sum   = chunk_cumsum(g)                     chunk-local prefix sum of the gate
    A       = scaled_dot_kkt(k, beta, g_sum)      gated key-key matrix, strictly lower
    A_inv   = solve_tril(A)                       (I + A)^-1 per chunk
    W, U    = wy_fast(k, v, beta, A_inv, g_sum)   the WY representation
    S, Vnew = chunk_h(k, W, U, g_sum)             inter-chunk state recurrence
    O       = chunk_o(q, k, Vnew, S, g_sum)       chunk output

Everything between `g_sum` and `O` is internal: A, A_inv, W, U, the per-chunk
state snapshots and V_new are allocated here and never leave. Only q, k, v, the
gate and beta come in, and only O goes out.

This is the delta rule alone. The projections that produce q, k, v, beta and the
gate, the short convolution over them, the output gate and `out_proj` are the
surrounding block and are not built yet -- see the model's docs page.

The dtype at each boundary is the one the standalone stages exchange, so this
composition is numerically the pipeline `test_gdn_stages.py --chain` validates,
with one difference: solve_tril writes A_inv as FP16 directly rather than FP32
for the host to narrow, which is what its consumer reads anyway.
"""
import pypto.language as pl

from models.qwen3_8_27b import (
    chunk_cumsum,
    chunk_h,
    chunk_o,
    scaled_dot_kkt,
    solve_tril,
    wy_fast,
)
from models.qwen3_8_27b.config import GDN_TILING, QWEN3_8_27B

# model shape
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
CHUNK = GDN_TILING.chunk                    # chunk size in tokens, our tiling choice

# case shape
T = 8192                # tokens (single sequence, B = 1)

# A, A_inv, W, U, the state snapshots and V_new are all [T, H, *] and live at once:
# about 600 MiB at T = 8192, H = 48, against the runtime's default 256 MiB per ring.
# Without this the layer fails with orch_error_code=2 HEAP_RING_DEADLOCK. The same
# thing happens to models/deepseek_v4_pro/prefill_layer.py, at the same value.
LAYER_RING_HEAP = 1024 * 1024 * 1024


def run_config(platform: str = "a2a3", device: int = 0) -> dict:
    """The config `golden.run` needs for this layer. The ring heap is not optional."""
    return dict(platform=platform, device_id=device, ring_heap=LAYER_RING_HEAP)


def build_kernel(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK,
                 hg: int = HG):
    """The layer at one shape, composing the six operators as inlined callees."""
    nchunk = t // chunk
    shape = dict(t=t, h=h, d=d, chunk=chunk)

    op_cumsum = chunk_cumsum.build_kernel(**shape, inline=True)
    op_kkt = scaled_dot_kkt.build_kernel(**shape, hg=hg, inline=True)
    op_tril = solve_tril.build_kernel(**shape, inline=True, out_dtype=pl.FP16)
    op_wy = wy_fast.build_kernel(**shape, hg=hg, inline=True)
    op_h = chunk_h.build_kernel(**shape, hg=hg, inline=True)
    op_o = chunk_o.build_kernel(**shape, hg=hg, inline=True)

    @pl.jit
    def gdn_layer(
        q: pl.Tensor[[t, hg, d], pl.FP16],
        k: pl.Tensor[[t, hg, d], pl.FP16],
        v: pl.Tensor[[t, h, d], pl.FP16],
        g: pl.Tensor[[t, h], pl.FP32],
        beta: pl.Tensor[[h, t], pl.FP32],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        neg_eye: pl.Tensor[[chunk, chunk], pl.FP16],
        o_out: pl.Out[pl.Tensor[[t, h, d], pl.FP16]],
    ):
        # `tril` serves twice: chunk_cumsum contracts against it, and it is also
        # chunk_o's causal mask -- both want the inclusive lower triangle. Only
        # scaled_dot_kkt's is strict, so that one is a second constant.
        g_sum = pl.create_tensor([h, t], dtype=pl.FP32)
        op_cumsum(g, tril, g_sum)

        a = pl.create_tensor([t, h, chunk], dtype=pl.FP16)
        op_kkt(k, beta, g_sum, mask_strict, a)

        a_inv = pl.create_tensor([t, h, chunk], dtype=pl.FP16)
        op_tril(a, neg_eye, a_inv)

        w = pl.create_tensor([t, h, d], dtype=pl.FP16)
        u = pl.create_tensor([t, h, d], dtype=pl.FP16)
        op_wy(k, v, a_inv, beta, g_sum, w, u)

        state = pl.create_tensor([nchunk * h * d, d], dtype=pl.FP16)
        v_new = pl.create_tensor([t, h, d], dtype=pl.FP16)
        op_h(k, w, u, g_sum, state, v_new)

        op_o(q, k, v_new, state, g_sum, tril, o_out)
        return o_out

    return gdn_layer


gdn_layer = build_kernel()


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK,
                       hg: int = HG):
    import torch
    from golden import TensorSpec

    from models.qwen3_8_27b import reference

    def init_tril():
        return torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))

    def init_mask_strict():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows > cols).float()

    def draw(key, transform=None):
        return reference.lazy("chunk_cumsum", key, t, h, d, chunk, transform, hg=hg)

    return [
        TensorSpec("q", [t, hg, d], torch.float16, init_value=draw("q")),
        TensorSpec("k", [t, hg, d], torch.float16, init_value=draw("k")),
        TensorSpec("v", [t, h, d], torch.float16, init_value=draw("v")),
        TensorSpec("g", [t, h], torch.float32, init_value=draw("g")),
        TensorSpec("beta", [h, t], torch.float32,
                   init_value=draw("beta", reference.to_hT)),
        TensorSpec("tril", [chunk, chunk], torch.float32, init_value=init_tril),
        TensorSpec("mask_strict", [chunk, chunk], torch.float32,
                   init_value=init_mask_strict),
        TensorSpec("neg_eye", [chunk, chunk], torch.float16,
                   init_value=lambda: -torch.eye(chunk, dtype=torch.float16)),
        TensorSpec("o_out", [t, h, d], torch.float16),
    ]


def golden_gdn_layer(tensors):
    """The whole chain in float64, scored end to end rather than stage by stage."""
    from models.qwen3_8_27b import reference

    t, h, d = tensors["v"].shape
    chunk = tensors["tril"].shape[0]
    hg = tensors["q"].shape[1]
    tensors["o_out"].copy_(reference.compute("chunk_o", t, h, d, chunk, hg=hg)["o"])


def _stats_ok(actual, expected, **_kwargs):
    """megagdn-pto's criterion (tests/utils.py: NumericalAccuracy)."""
    from models.qwen3_8_27b import reference

    ok, detail = reference.stats_ok(actual, expected, chunk=CHUNK)
    print(f"[stats] {detail}", flush=True)
    return ok, detail


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=T)
    parser.add_argument("--heads", type=int, default=H, help="value heads H")
    parser.add_argument("--qk-heads", type=int, default=HG,
                        help="QK heads Hg; pass the same value as --heads for no "
                             "grouping")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    if args.heads % args.qk_heads:
        parser.error(f"H={args.heads} must be divisible by Hg={args.qk_heads}")
    shape = dict(t=args.seq_len, h=args.heads, hg=args.qk_heads)

    result = run(
        fn=build_kernel(**shape),
        specs=build_tensor_specs(**shape),
        golden_fn=golden_gdn_layer,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=run_config(args.platform, args.device),
        rtol=1e-2, atol=1e-5,
        compare_fn={"o_out": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
