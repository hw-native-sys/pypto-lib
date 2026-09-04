# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet wy_fast: the WY representation of the chunk update,

    A2[i, j] = A[i, j] * beta_j                 U = A2 @ V
    A1[i, j] = A[i, j] * beta_j * exp(g_j)      W = A1 @ K

Scaling A's columns rather than V's and K's rows gives the same result and lets
the cube read V and K straight from BSND. exp(g) underflows for late columns of a
chunk, and those columns are negligible in the sum, so the scaled A columns go to
zero as they do in the reference.
"""
import pypto.language as pl

# model config
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # value heads
D = 128                 # head dimension
CHUNK = 128             # chunk size in tokens


def build_kernel(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    """The stage kernel at one shape."""

    @pl.jit
    def gdn_wy_fast(
        k: pl.Tensor[[t, h, d], pl.FP16],
        v: pl.Tensor[[t, h, d], pl.FP16],
        a_in: pl.Tensor[[t, h, chunk], pl.FP16],
        beta: pl.Tensor[[h, t], pl.FP32],
        g_sum: pl.Tensor[[h, t], pl.FP32],
        w_out: pl.Out[pl.Tensor[[t, h, d], pl.FP16]],
        u_out: pl.Out[pl.Tensor[[t, h, d], pl.FP16]],
    ):
        k_flat = pl.reshape(k, [t, h * d])
        v_flat = pl.reshape(v, [t, h * d])
        a_flat = pl.reshape(a_in, [t, h * chunk])
        w_flat = pl.reshape(w_out, [t, h * d])
        u_flat = pl.reshape(u_out, [t, h * d])
        for c0 in pl.spmd(t // chunk, name_hint="wy_fast"):
            t0 = c0 * chunk
            for hh in pl.range(h):
                col = hh * chunk
                a_chunk = a_flat[t0 : t0 + chunk, col : col + chunk]
                beta_row = beta[hh : hh + 1, t0 : t0 + chunk]
                g_row = g_sum[hh : hh + 1, t0 : t0 + chunk]
                beta16 = pl.cast(beta_row, target_type=pl.FP16, mode="rint")
                gate = pl.mul(pl.exp(g_row), beta_row)
                gate16 = pl.cast(gate, target_type=pl.FP16, mode="rint")
                a2 = pl.col_expand_mul(a_chunk, beta16)
                a1 = pl.col_expand_mul(a_chunk, gate16)
                d0 = hh * d
                v_blk = v_flat[t0 : t0 + chunk, d0 : d0 + d]
                k_blk = k_flat[t0 : t0 + chunk, d0 : d0 + d]
                # FP16 operands promote to an FP16 result, accumulated FP32 in the cube
                u_flat[t0 : t0 + chunk, d0 : d0 + d] = pl.matmul(a2, v_blk)
                w_flat[t0 : t0 + chunk, d0 : d0 + d] = pl.matmul(a1, k_blk)
        return w_out, u_out

    return gdn_wy_fast


gdn_wy_fast = build_kernel()


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    from models.gdn import reference

    return [
        TensorSpec("k", [t, h, d], torch.float16, init_value=reference.lazy("wy_fast", "k", t, h, d, chunk)),
        TensorSpec("v", [t, h, d], torch.float16, init_value=reference.lazy("wy_fast", "v", t, h, d, chunk)),
        TensorSpec("a_in", [t, h, chunk], torch.float16,
                   init_value=reference.lazy("wy_fast", "a_inv16", t, h, d, chunk)),
        TensorSpec("beta", [h, t], torch.float32,
                   init_value=reference.lazy("wy_fast", "beta", t, h, d, chunk, reference.to_hT)),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=reference.lazy("wy_fast", "g_sum", t, h, d, chunk, reference.to_hT)),
        TensorSpec("w_out", [t, h, d], torch.float16, is_output=True),
        TensorSpec("u_out", [t, h, d], torch.float16, is_output=True),
    ]


def golden_gdn_wy_fast(tensors):
    from models.gdn import reference

    chunk = tensors["a_in"].shape[-1]
    w, u = reference.wy_fast(tensors["k"], tensors["v"], tensors["beta"].t(),
                             tensors["a_in"], tensors["g_sum"].t(), chunk)
    tensors["w_out"].copy_(w)
    tensors["u_out"].copy_(u)


def _stats_ok(actual, expected, **_kwargs):
    """megagdn-pto's criterion for this stage (tests/utils.py: NumericalAccuracy)."""
    from models.gdn import reference

    ok, detail = reference.stats_ok(actual, expected, chunk=CHUNK)
    print(f"[stats] {detail}", flush=True)
    return ok, detail


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=gdn_wy_fast,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_wy_fast,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"w_out": _stats_ok, "u_out": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
