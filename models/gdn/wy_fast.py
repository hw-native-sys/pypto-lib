# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet wy_fast — the WY representation of the chunk update.

    A2[i, j] = A[i, j] * beta_j                 U = A2 @ V
    A1[i, j] = A[i, j] * beta_j * exp(g_j)      W = A1 @ K

Stage 5 of the GDN pipeline, consuming the triangular inverse A from solve_tril.
Scaling A's columns rather than V's and K's rows is what the reference does and
gives the same result, and it lets the cube read V and K straight from BSND.

exp(g) underflows for late columns of a chunk -- a within-chunk cumulative gate
reaches around -100 -- and those columns are genuinely negligible in the sum, so
the scaled A columns go to zero as they do in the reference.
"""
import pypto.language as pl

# Model
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # value heads
D = 128                 # head dimension
CHUNK = 128             # chunk size in tokens

# Tiling
D_TILE = 128            # output columns per matmul; sizes the cube/vector crossing tile.
                        # The full head dim fits, so there is no inner block loop:
                        # 32 / 64 / 128 measure 267.1 / 173.4 / 145.6 us at T=8192.


@pl.jit
def gdn_wy_fast(
    k: pl.Tensor[[T, H, D], pl.FP16],
    v: pl.Tensor[[T, H, D], pl.FP16],
    a_in: pl.Tensor[[T, H, CHUNK], pl.FP16],
    beta: pl.Tensor[[H, T], pl.FP32],
    g_sum: pl.Tensor[[H, T], pl.FP32],
    w_out: pl.Out[pl.Tensor[[T, H, D], pl.FP16]],
    u_out: pl.Out[pl.Tensor[[T, H, D], pl.FP16]],
):
    k_flat = pl.reshape(k, [T, H * D])
    v_flat = pl.reshape(v, [T, H * D])
    a_flat = pl.reshape(a_in, [T, H * CHUNK])
    w_flat = pl.reshape(w_out, [T, H * D])
    u_flat = pl.reshape(u_out, [T, H * D])
    for t0 in pl.parallel(0, T, CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="wy_fast"):
            for h in pl.range(H):
                a_chunk = pl.slice(a_flat, [CHUNK, CHUNK], [t0, h * CHUNK])
                beta_row = pl.slice(beta, [1, CHUNK], [h, t0])
                g_row = pl.slice(g_sum, [1, CHUNK], [h, t0])
                beta16 = pl.cast(beta_row, target_type=pl.FP16, mode="rint")
                gate16 = pl.cast(pl.mul(pl.exp(g_row), beta_row),
                                 target_type=pl.FP16, mode="rint")
                a2 = pl.col_expand_mul(a_chunk, beta16)
                a1 = pl.col_expand_mul(a_chunk, gate16)
                for d in pl.unroll(D // D_TILE):
                    d0 = h * D + d * D_TILE
                    v_blk = pl.slice(v_flat, [CHUNK, D_TILE], [t0, d0])
                    k_blk = pl.slice(k_flat, [CHUNK, D_TILE], [t0, d0])
                    # FP16 operands promote to an FP16 result, accumulated FP32
                    # in the cube; no cast needed on the way out.
                    u_flat = pl.assemble(u_flat, pl.matmul(a2, v_blk), [t0, d0])
                    w_flat = pl.assemble(w_flat, pl.matmul(a1, k_blk), [t0, d0])
    return w_out, u_out


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    def init_k():
        return torch.nn.functional.normalize(
            torch.randn(t, h, d, dtype=torch.float16), dim=-1, p=2)

    def init_beta():
        return torch.rand(h, t, dtype=torch.float32)

    def init_g():
        torch.manual_seed(7)
        g = torch.nn.functional.logsigmoid(torch.randn(h, t, dtype=torch.float32))
        out = torch.zeros_like(g)
        for t0 in range(0, t, chunk):
            out[:, t0 : t0 + chunk] = g[:, t0 : t0 + chunk].cumsum(dim=1)
        return out

    def init_a():
        # A as solve_tril produces it: unit lower triangular, from the inverse of
        # a strictly-lower matrix of the scale kkt emits.
        torch.manual_seed(11)
        raw = (torch.randn(t // chunk, h, chunk, chunk) / chunk ** 0.5).tril(-1)
        eye = torch.eye(chunk).expand_as(raw)
        inv = torch.linalg.inv((eye + raw).double()).float()
        return inv.permute(0, 2, 1, 3).reshape(t, h, chunk).half()

    return [
        TensorSpec("k", [t, h, d], torch.float16, init_value=init_k),
        TensorSpec("v", [t, h, d], torch.float16, init_value=torch.randn),
        TensorSpec("a_in", [t, h, chunk], torch.float16, init_value=init_a),
        TensorSpec("beta", [h, t], torch.float32, init_value=init_beta),
        TensorSpec("g_sum", [h, t], torch.float32, init_value=init_g),
        TensorSpec("w_out", [t, h, d], torch.float16, is_output=True),
        TensorSpec("u_out", [t, h, d], torch.float16, is_output=True),
    ]


def _stats_ok(actual, expected, **_kwargs):
    """Upstream's acceptance criterion for this kernel (test_gdn_wy_fast.py).

    A2 and A1 are formed in FP16, as they are in the reference's workspace, so
    elementwise agreement with an FP32 golden is not attainable and upstream
    falls back to RMSE ratio and R2. Reproduced in torch: max diff 0.0016
    against a 1.0 hard fail, RMSE ratio 0.0004 against 0.05, R2 1.000000.
    """
    import numpy as np
    import torch

    diff = (actual.float() - expected.float()).abs()
    if diff.max().item() > 1.0:
        return False, f"max abs diff {diff.max().item():.4g} exceeds hard fail 1.0"
    if bool((diff <= 1e-5 + 1e-2 * expected.float().abs()).all()):
        return True, ""
    mean_abs = float(expected.float().abs().mean())
    rmse = float(torch.sqrt((diff.flatten() ** 2).mean()))
    ratio = rmse / max(mean_abs, 1e-15)
    ref = expected.float().flatten().numpy().astype(np.float64)
    pred = actual.float().flatten().numpy().astype(np.float64)
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    r2 = float("nan") if ss_tot < 1e-30 else 1.0 - float(np.sum((ref - pred) ** 2)) / ss_tot
    ok = ratio <= 0.05 and np.isfinite(r2) and r2 >= 0.99
    return ok, f"rmse/mean={ratio:.4g} (<=0.05), r2={r2:.6f} (>=0.99)"


def golden_gdn_wy_fast(tensors):
    import torch

    k = tensors["k"].float()
    v = tensors["v"].float()
    a = tensors["a_in"].float()
    beta = tensors["beta"].float()
    g = tensors["g_sum"].float()
    w = tensors["w_out"]
    u = tensors["u_out"]
    for t0 in range(0, k.shape[0], CHUNK):
        for h in range(k.shape[1]):
            ab = a[t0 : t0 + CHUNK, h, :]
            bc = beta[h, t0 : t0 + CHUNK]
            gc = g[h, t0 : t0 + CHUNK]
            u[t0 : t0 + CHUNK, h, :] = (
                ab @ (v[t0 : t0 + CHUNK, h, :] * bc[:, None])).half()
            w[t0 : t0 + CHUNK, h, :] = (ab @ (
                k[t0 : t0 + CHUNK, h, :] * bc[:, None] * torch.exp(gc)[:, None])).half()


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
