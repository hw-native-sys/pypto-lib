# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet scaled_dot_kkt — the gated intra-chunk key-key matrix.

    A[i, j] = (k_i . k_j) * exp(min(g_i - g_j, 0)) * beta_i    for j < i, else 0

Stage 3 of the GDN pipeline, consuming g_sum from chunk_cumsum. The decay is
formed as a difference of cumulative gates and clamped at zero, never as a
product of exp(g_i) and exp(-g_j): the latter overflows once the cumulative gate
grows large.

Chunks are independent and run one per core group; the strict-lower mask is a
constant, so it is loaded once per scope and reused across all heads.
"""
import pypto.language as pl

# Model
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # value heads
D = 128                 # head dimension
CHUNK = 128             # chunk size in tokens

# Tiling
COL_TILE = 64           # columns of A per matmul; sizes the cube/vector crossing tile.
                        # 128 does not fit: with the mask resident, a [CHUNK, 128] FP32
                        # crossing tile pushes the ring reserve past the Vec budget.


@pl.jit
def gdn_scaled_dot_kkt(
    k: pl.Tensor[[T, H, D], pl.FP16],
    beta: pl.Tensor[[H, T], pl.FP32],
    g_sum: pl.Tensor[[H, T], pl.FP32],
    mask: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    a_out: pl.Out[pl.Tensor[[T, H, CHUNK], pl.FP16]],
):
    # BSND [T, H, D] viewed as [T, H*D]: a per-head slice is then a strided 2D
    # window (row stride H*D, D columns), which is the shape the loader collapses.
    k_flat = pl.reshape(k, [T, H * D])
    a_flat = pl.reshape(a_out, [T, H * CHUNK])
    for t0 in pl.parallel(0, T, CHUNK):
        # An a2a3 core has two vector sub-cores, and an unsplit mixed region runs
        # its vector work on lane 0 only (lane 1 replays with valid_shape zeroed
        # just to keep the AIC<->AIV handshake symmetric). This stage is vector
        # dominated -- stripping the gating chain drops it from 203.5 to 78.8 us
        # -- so giving lane 1 the bottom half of the rows is worth 22.7%:
        # 203.5 -> 157.3 us at T=8192. No effect on wy_fast, which is
        # memory-bound, nor on chunk_cumsum, whose region is pure cube.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scaled_dot_kkt",
                   optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            # The mask is a constant: hold it for the whole scope rather than
            # re-reading a slice of it per head and per column block.
            msk = mask[:, :]
            for h in pl.range(H):
                kc = pl.slice(k_flat, [CHUNK, D], [t0, h * D])
                # Head-major keeps a head's chunk contiguous, so the same window
                # views as [1, CHUNK] for a row broadcast and [CHUNK, 1] for a
                # column one. A strided column slice of BSND does not build on
                # device, and an in-register transpose cannot be allocated.
                g_col = pl.reshape(pl.slice(g_sum, [1, CHUNK], [h, t0]), [CHUNK, 1])
                beta_col = pl.reshape(pl.slice(beta, [1, CHUNK], [h, t0]), [CHUNK, 1])
                for j in pl.unroll(CHUNK // COL_TILE):
                    j0 = j * COL_TILE
                    kj = pl.slice(k_flat, [COL_TILE, D], [t0 + j0, h * D])
                    scores = pl.matmul(kc, kj, b_trans=True)
                    g_row = pl.slice(g_sum, [1, COL_TILE], [h, t0 + j0])
                    diff = pl.full([CHUNK, COL_TILE], dtype=pl.FP32, value=0.0)
                    diff = pl.row_expand_add(diff, g_col)
                    diff = pl.col_expand_sub(diff, g_row)
                    decay = pl.exp(pl.minimum(diff, 0.0))
                    gated = pl.mul(scores, decay)
                    gated = pl.row_expand_mul(gated, beta_col)
                    masked = pl.mul(gated, pl.slice(msk, [CHUNK, COL_TILE], [0, j0]))
                    a_flat = pl.assemble(a_flat, pl.cast(masked, target_type=pl.FP16, mode="rint"),
                                         [t0, h * CHUNK + j0])
    return a_out


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    def init_mask():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows > cols).float()

    def init_beta():
        return torch.rand(h, t, dtype=torch.float32)

    def init_g():
        # g_sum as chunk_cumsum produces it: a per-chunk prefix sum of the
        # log-sigmoid gates, matching upstream's own generator.
        torch.manual_seed(7)
        g = torch.nn.functional.logsigmoid(torch.randn(h, t, dtype=torch.float32))
        out = torch.zeros_like(g)
        for t0 in range(0, t, chunk):
            out[:, t0 : t0 + chunk] = g[:, t0 : t0 + chunk].cumsum(dim=1)
        return out

    def init_k():
        # L2-normalised along the head dimension, as the model produces it and as
        # upstream's own test draws it (test_gdn_scaled_dot_kkt.py). NOT raw
        # randn: with unnormalised k, K K^T entries scale with D and `A` reaches
        # ~40, which makes (I + A)^-1 about 1e36 with a condition number of 1e20.
        # The kernel is indifferent, but every consumer of `A` is not -- solve_tril
        # chained onto that `A` inverts a matrix whose fp64 golden is already inf.
        torch.manual_seed(11)
        k = torch.randn(t, h, d, dtype=torch.float16)
        return torch.nn.functional.normalize(k, dim=-1, p=2)

    return [
        TensorSpec("k", [t, h, d], torch.float16, init_value=init_k),
        TensorSpec("beta", [h, t], torch.float32, init_value=init_beta),
        TensorSpec("g_sum", [h, t], torch.float32, init_value=init_g),
        TensorSpec("mask", [chunk, chunk], torch.float32, init_value=init_mask),
        TensorSpec("a_out", [t, h, chunk], torch.float16, is_output=True),
    ]


def golden_gdn_scaled_dot_kkt(tensors):
    import torch

    k = tensors["k"].float()
    beta = tensors["beta"].float()
    g = tensors["g_sum"].float()
    out = tensors["a_out"]
    out.zero_()
    rows = torch.arange(CHUNK)[:, None]
    cols = torch.arange(CHUNK)[None, :]
    causal = (rows > cols).float()
    for t0 in range(0, k.shape[0], CHUNK):
        for h in range(k.shape[1]):
            kc = k[t0 : t0 + CHUNK, h, :]
            gc = g[h, t0 : t0 + CHUNK]
            bc = beta[h, t0 : t0 + CHUNK]
            diff = gc[:, None] - gc[None, :]
            decay = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
            out[t0 : t0 + CHUNK, h, :] = ((kc @ kc.T) * decay * bc[:, None] * causal).half()


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
        fn=gdn_scaled_dot_kkt,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_scaled_dot_kkt,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-2,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
