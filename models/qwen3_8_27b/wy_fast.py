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
"""
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
VAL_WIDTH = H * D
A_WIDTH = H * CHUNK_TILE
T = PREFILL_SEQ                             # tokens (single sequence, B = 1)


def _gdn_wy_fast(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    v: pl.Tensor[[T_DYN, H, D], pl.FP16],
    a_in: pl.Tensor[[T_DYN, H, CHUNK_TILE], pl.FP16],
    beta: pl.Tensor[[H, T_DYN], pl.FP32],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    w_out: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
    u_out: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE; a block walks one whole chunk."""
    k.bind_dynamic(0, T_DYN)
    v.bind_dynamic(0, T_DYN)
    a_in.bind_dynamic(0, T_DYN)
    beta.bind_dynamic(1, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    w_out.bind_dynamic(0, T_DYN)
    u_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(k, 0)
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    v_flat = pl.reshape(v, [t_dim, VAL_WIDTH])
    a_flat = pl.reshape(a_in, [t_dim, A_WIDTH])
    w_flat = pl.reshape(w_out, [t_dim, VAL_WIDTH])
    u_flat = pl.reshape(u_out, [t_dim, VAL_WIDTH])
    for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="wy_fast"):
        t0 = c0 * CHUNK_TILE
        for hh in pl.range(H):
            col = hh * CHUNK_TILE
            a_chunk = a_flat[t0 : t0 + CHUNK_TILE, col : col + CHUNK_TILE]
            beta_row = beta[hh : hh + 1, t0 : t0 + CHUNK_TILE]
            g_row = g_sum[hh : hh + 1, t0 : t0 + CHUNK_TILE]
            beta16 = pl.cast(beta_row, target_type=pl.FP16, mode="rint")
            gate = pl.mul(pl.exp(g_row), beta_row)
            gate16 = pl.cast(gate, target_type=pl.FP16, mode="rint")
            a2 = pl.col_expand_mul(a_chunk, beta16)
            a1 = pl.col_expand_mul(a_chunk, gate16)
            d0 = hh * D
            # GQA: W reads key head hh // GRP, U reads value head hh.
            dg0 = (hh // GRP) * D
            v_blk = v_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            k_blk = k_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            # FP16 operands promote to an FP16 result, accumulated FP32 in the cube
            u_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D] = pl.matmul(a2, v_blk)
            w_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D] = pl.matmul(a1, k_blk)
    return w_out, u_out


gdn_wy_fast = pl.jit.inline(_gdn_wy_fast)
gdn_wy_fast_test = pl.jit(_gdn_wy_fast)


_INPUTS: dict = {}


def _inputs(t: int, h: int, d: int, chunk: int, hg: int) -> dict:
    """The chain this stage consumes, run here rather than shared with the other stages.

    Memoised so the several specs that draw from it pay for it once.
    """
    import torch

    import reference

    key = (t, h, d, chunk, hg)
    if key not in _INPUTS:
        st = reference.make_inputs(t, h, d, hg)
        st["g_sum"] = reference.cumsum(st["g"], chunk)
        a16 = reference.kkt(st["k"], st["beta"], st["g_sum"], chunk).to(torch.float16)
        st["a_inv16"] = reference.solve_tril(a16, chunk).to(torch.float16)
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



    return [
        TensorSpec("k", [t, hg, d], torch.float16,
                   init_value=draw("k")),
        TensorSpec("v", [t, h, d], torch.float16, init_value=draw("v")),
        TensorSpec("a_in", [t, h, chunk], torch.float16,
                   init_value=draw("a_inv16")),
        TensorSpec("beta", [h, t], torch.float32,
                   init_value=draw("beta", reference.to_hT)),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=draw("g_sum", reference.to_hT)),
        TensorSpec("w_out", [t, h, d], torch.float16),
        TensorSpec("u_out", [t, h, d], torch.float16),
    ]


def golden_gdn_wy_fast(tensors):
    import reference

    chunk = tensors["a_in"].shape[-1]
    w, u = reference.wy_fast(tensors["k"], tensors["v"], tensors["beta"].t(),
                             tensors["a_in"], tensors["g_sum"].t(), chunk)
    tensors["w_out"].copy_(w)
    tensors["u_out"].copy_(u)


def _stats_ok(actual, expected, **_kwargs):
    """Relative Frobenius norm against the float64 golden, with the peak reported."""
    import reference

    ok, detail = reference.stats_ok(actual, expected, chunk=CHUNK_TILE)
    print(f"[stats] {detail}", flush=True)
    return ok, detail


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
        fn=gdn_wy_fast_test,
        specs=build_tensor_specs(t=args.seq_len),
        golden_fn=golden_gdn_wy_fast,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(
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
