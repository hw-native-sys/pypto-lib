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
import torch

from chunk_cumsum import cumsum_ref
from config import QWEN3_8_27B
from reference import REF_DTYPE, frob_ok, make_inputs, to_hT
from scaled_dot_kkt import kkt_ref
from solve_tril import CHUNK_TILE, solve_tril_ref

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D
VAL_WIDTH = H * D

# tiling
A_WIDTH = H * CHUNK_TILE

# validation
# One chunk, the cross-chunk recurrence, then eight chunks. The token count must be a
# multiple of CHUNK_TILE.
DEFAULT_CASES = (128, 256, 1024)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len


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


def wy_fast_ref(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
            a_inv: torch.Tensor, g_sum: torch.Tensor,
            chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """S5: the WY representation. -> (W, U), both [T, H, D]."""
    t, hg, d = k.shape
    h = v.shape[1]
    grp = h // hg
    kf, vf, bf, af, gf = (x.to(REF_DTYPE) for x in (k, v, beta, a_inv, g_sum))
    w = torch.zeros(t, h, d, dtype=REF_DTYPE)
    u = torch.zeros(t, h, d, dtype=REF_DTYPE)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            ab = af[t0 : t0 + chunk, hh, :]
            bc = bf[t0 : t0 + chunk, hh, None]
            gc = gf[t0 : t0 + chunk, hh, None]
            u[t0 : t0 + chunk, hh, :] = ab @ (vf[t0 : t0 + chunk, hh, :] * bc)
            w[t0 : t0 + chunk, hh, :] = ab @ (
                kf[t0 : t0 + chunk, hh // grp, :] * bc * torch.exp(gc))
    return w, u


_INPUTS: dict = {}


def _inputs(t: int) -> dict:
    """The chain this stage consumes, memoised so the several specs pay for it once."""
    key = t
    if key not in _INPUTS:
        st = make_inputs(t, H, D, HG)
        st["g_sum"] = cumsum_ref(st["g"], CHUNK_TILE)
        a16 = kkt_ref(st["k"], st["beta"], st["g_sum"], CHUNK_TILE).to(torch.float16)
        st["a_inv16"] = solve_tril_ref(a16, CHUNK_TILE).to(torch.float16)
        _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ):
    from golden import TensorSpec

    def draw(key, transform=None):
        def make():
            value = _inputs(t)[key]
            return transform(value) if transform is not None else value

        return make

    return [
        TensorSpec("k", [t, HG, D], torch.float16,
                   init_value=draw("k")),
        TensorSpec("v", [t, H, D], torch.float16, init_value=draw("v")),
        TensorSpec("a_in", [t, H, CHUNK_TILE], torch.float16,
                   init_value=draw("a_inv16")),
        TensorSpec("beta", [H, t], torch.float32,
                   init_value=draw("beta", to_hT)),
        TensorSpec("g_sum", [H, t], torch.float32,
                   init_value=draw("g_sum", to_hT)),
        TensorSpec("w_out", [t, H, D], torch.float16),
        TensorSpec("u_out", [t, H, D], torch.float16),
    ]


def golden_gdn_wy_fast(tensors):
    chunk = tensors["a_in"].shape[-1]
    w, u = wy_fast_ref(tensors["k"], tensors["v"], tensors["beta"].t(), tensors["a_in"], tensors["g_sum"].t(), chunk)
    tensors["w_out"].copy_(w)
    tensors["u_out"].copy_(u)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        print(f"[case] wy_fast T={t}", flush=True)
        result = run(
            fn=gdn_wy_fast_test,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_gdn_wy_fast,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compare_fn={"w_out": frob_ok(), "u_out": frob_ok()},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
