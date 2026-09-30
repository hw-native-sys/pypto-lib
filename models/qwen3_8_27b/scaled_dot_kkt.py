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

from config import GDN_TILING, PREFILL_SEQ, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
CHUNK_TILE = GDN_TILING.chunk               # chunk size in tokens, our tiling choice
GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D
A_WIDTH = H * CHUNK_TILE
T = PREFILL_SEQ                             # tokens (single sequence, B = 1)

# tiling
COL_TILE = 128          # columns of A per matmul
SLOT_NUM = 1            # cross-core ring depth; the default cannot hold a
                        # [CHUNK_TILE, COL_TILE] FP32 crossing tile (pypto#2903)


def _gdn_scaled_dot_kkt(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    beta: pl.Tensor[[H, T_DYN], pl.FP32],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    mask: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    a_out: pl.Out[pl.Tensor[[T_DYN, H, CHUNK_TILE], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE; a block walks one whole chunk."""
    k.bind_dynamic(0, T_DYN)
    beta.bind_dynamic(1, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    a_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(k, 0)
    # BSND [T, Hg, D] viewed as [T, Hg*D]: a per-head slice is a strided 2D window
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    a_flat = pl.reshape(a_out, [t_dim, A_WIDTH])
    for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="scaled_dot_kkt",
                      optimizations=[pl.cross_core_slot(slot_num=SLOT_NUM),
                                     pl.split(pl.SplitMode.UP_DOWN)]):
        t0 = c0 * CHUNK_TILE
        msk = mask[:, :]                       # constant, held for the whole scope
        for hh in pl.range(H):
            # GQA: value head hh reads key head hh // grp, the same mapping the
            # model reaches by repeat_interleave. GRP == 1 leaves this an identity.
            d0 = (hh // GRP) * D
            kc = k_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            # Head-major keeps a head's chunk contiguous, so the same window views
            # as [1, CHUNK_TILE] for a row broadcast and [CHUNK_TILE, 1] for a column one. A
            # strided column slice of BSND does not build, and an in-register
            # transpose cannot be allocated.
            g_col = pl.reshape(g_sum[hh : hh + 1, t0 : t0 + CHUNK_TILE], [CHUNK_TILE, 1])
            beta_col = pl.reshape(beta[hh : hh + 1, t0 : t0 + CHUNK_TILE], [CHUNK_TILE, 1])
            for j0 in pl.unroll(0, CHUNK_TILE, COL_TILE):
                kj = k_flat[t0 + j0 : t0 + j0 + COL_TILE, d0 : d0 + D]
                scores = pl.matmul(kc, kj, b_trans=True)
                g_row = g_sum[hh : hh + 1, t0 + j0 : t0 + j0 + COL_TILE]
                diff = pl.full([CHUNK_TILE, COL_TILE], dtype=pl.FP32, value=0.0)
                diff = pl.row_expand_add(diff, g_col)
                diff = pl.col_expand_sub(diff, g_row)
                decay = pl.exp(pl.minimum(diff, 0.0))
                gated = pl.mul(scores, decay)
                gated = pl.row_expand_mul(gated, beta_col)
                masked = pl.mul(gated, msk[:, j0 : j0 + COL_TILE])
                masked16 = pl.cast(masked, target_type=pl.FP16, mode="rint")
                col = hh * CHUNK_TILE + j0
                a_flat[t0 : t0 + CHUNK_TILE, col : col + COL_TILE] = masked16
    return a_out


gdn_scaled_dot_kkt = pl.jit.inline(_gdn_scaled_dot_kkt)
gdn_scaled_dot_kkt_test = pl.jit(_gdn_scaled_dot_kkt)


_INPUTS: dict = {}


def _inputs(t: int, h: int, d: int, chunk: int, hg: int) -> dict:
    """The chain this stage consumes, run here rather than shared with the other stages.

    Memoised so the several specs that draw from it pay for it once.
    """

    import reference

    key = (t, h, d, chunk, hg)
    if key not in _INPUTS:
        st = reference.make_inputs(t, h, d, hg)
        st["g_sum"] = reference.cumsum(st["g"], chunk)
        _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK_TILE,
                       hg: int = HG):
    import torch
    from golden import TensorSpec

    import reference

    def draw(key, transform=None):
        def make():
            value = _inputs(t, h, d, chunk, hg)[key]
            return transform(value) if transform is not None else value

        return make



    def init_mask():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows > cols).float()

    return [
        TensorSpec("k", [t, hg, d], torch.float16,
                   init_value=draw("k")),
        TensorSpec("beta", [h, t], torch.float32,
                   init_value=draw("beta", reference.to_hT)),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=draw("g_sum", reference.to_hT)),
        TensorSpec("mask", [chunk, chunk], torch.float32, init_value=init_mask),
        TensorSpec("a_out", [t, h, chunk], torch.float16),
    ]


def golden_gdn_scaled_dot_kkt(tensors):
    import reference

    chunk = tensors["mask"].shape[0]
    ref = reference.kkt(tensors["k"], tensors["beta"].t(), tensors["g_sum"].t(), chunk)
    tensors["a_out"].copy_(ref)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=T)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()
    if args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")

    result = run(
        fn=gdn_scaled_dot_kkt_test,
        specs=build_tensor_specs(t=args.seq_len),
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
