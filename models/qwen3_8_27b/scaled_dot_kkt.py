# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet scaled_dot_kkt: the gated intra-chunk key-key matrix,
A[i, j] = (k_i . k_j) * exp(min(g_i - g_j, 0)) * beta_i for j < i, else 0."""
import pypto.language as pl

from models.qwen3_8_27b.config import GDN_TILING, QWEN3_8_27B

# model shape
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
CHUNK = GDN_TILING.chunk                    # chunk size in tokens, our tiling choice

# case shape
T = 8192                # tokens (single sequence, B = 1)

# tiling
COL_TILE = 128          # columns of A per matmul
SLOT_NUM = 1            # cross-core ring depth; the default depth cannot hold a
                        # [CHUNK, COL_TILE] FP32 crossing tile at COL_TILE = 128


def build_kernel(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK, hg: int = HG,
                 col_tile: int = COL_TILE, slot_num: int = SLOT_NUM,
                 inline: bool = False):
    """The stage kernel at one shape.

    `hg` is the number of QK heads, `h` the number of value heads; they differ
    under GQA. Pass `hg=h` for an ungrouped shape. `inline` makes the kernel a
    callee for `gdn_layer` rather than a program of its own.
    """
    grp = h // hg

    @(pl.jit.inline if inline else pl.jit)
    def gdn_scaled_dot_kkt(
        k: pl.Tensor[[t, hg, d], pl.FP16],
        beta: pl.Tensor[[h, t], pl.FP32],
        g_sum: pl.Tensor[[h, t], pl.FP32],
        mask: pl.Tensor[[chunk, chunk], pl.FP32],
        a_out: pl.Out[pl.Tensor[[t, h, chunk], pl.FP16]],
    ):
        # BSND [T, Hg, D] viewed as [T, Hg*D]: a per-head slice is a strided 2D window
        k_flat = pl.reshape(k, [t, hg * d])
        a_flat = pl.reshape(a_out, [t, h * chunk])
        for c0 in pl.spmd(t // chunk, name_hint="scaled_dot_kkt",
                          optimizations=[pl.cross_core_slot(slot_num=slot_num),
                                         pl.split(pl.SplitMode.UP_DOWN)]):
            t0 = c0 * chunk
            msk = mask[:, :]                       # constant, held for the whole scope
            for hh in pl.range(h):
                # GQA: value head hh reads key head hh // grp, the same mapping the
                # model reaches by repeat_interleave. grp == 1 leaves this an identity.
                d0 = (hh // grp) * d
                kc = k_flat[t0 : t0 + chunk, d0 : d0 + d]
                # Head-major keeps a head's chunk contiguous, so the same window views
                # as [1, CHUNK] for a row broadcast and [CHUNK, 1] for a column one. A
                # strided column slice of BSND does not build, and an in-register
                # transpose cannot be allocated.
                g_col = pl.reshape(g_sum[hh : hh + 1, t0 : t0 + chunk], [chunk, 1])
                beta_col = pl.reshape(beta[hh : hh + 1, t0 : t0 + chunk], [chunk, 1])
                for j0 in pl.unroll(0, chunk, col_tile):
                    kj = k_flat[t0 + j0 : t0 + j0 + col_tile, d0 : d0 + d]
                    scores = pl.matmul(kc, kj, b_trans=True)
                    g_row = g_sum[hh : hh + 1, t0 + j0 : t0 + j0 + col_tile]
                    diff = pl.full([chunk, col_tile], dtype=pl.FP32, value=0.0)
                    diff = pl.row_expand_add(diff, g_col)
                    diff = pl.col_expand_sub(diff, g_row)
                    decay = pl.exp(pl.minimum(diff, 0.0))
                    gated = pl.mul(scores, decay)
                    gated = pl.row_expand_mul(gated, beta_col)
                    masked = pl.mul(gated, msk[:, j0 : j0 + col_tile])
                    masked16 = pl.cast(masked, target_type=pl.FP16, mode="rint")
                    col = hh * chunk + j0
                    a_flat[t0 : t0 + chunk, col : col + col_tile] = masked16
        return a_out

    return gdn_scaled_dot_kkt


gdn_scaled_dot_kkt = build_kernel()


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK,
                       hg: int = HG):
    import torch
    from golden import TensorSpec

    from models.qwen3_8_27b import reference


    def init_mask():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows > cols).float()

    return [
        TensorSpec("k", [t, hg, d], torch.float16,
                   init_value=reference.lazy("scaled_dot_kkt", "k", t, h, d, chunk, hg=hg)),
        TensorSpec("beta", [h, t], torch.float32,
                   init_value=reference.lazy("scaled_dot_kkt", "beta", t, h, d, chunk, reference.to_hT, hg=hg)),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=reference.lazy("scaled_dot_kkt", "g_sum", t, h, d, chunk, reference.to_hT, hg=hg)),
        TensorSpec("mask", [chunk, chunk], torch.float32, init_value=init_mask),
        TensorSpec("a_out", [t, h, chunk], torch.float16),
    ]


def golden_gdn_scaled_dot_kkt(tensors):
    from models.qwen3_8_27b import reference

    chunk = tensors["mask"].shape[0]
    ref = reference.kkt(tensors["k"], tensors["beta"].t(), tensors["g_sum"].t(), chunk)
    tensors["a_out"].copy_(ref)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run(
        fn=gdn_scaled_dot_kkt,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_scaled_dot_kkt,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(
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
