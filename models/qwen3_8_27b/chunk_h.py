# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_h: the inter-chunk state recurrence. Per chunk, per head,
with S the state entering the chunk,

    snapshot = S
    V_new    = U - W @ S
    S        = exp(g_last) * S + K^T (V_new * exp(g_last - g))

Chunks carry state, so work is parallel over heads and sequential within one.
"""
import pypto.language as pl
import torch

from chunk_cumsum import cumsum_ref
from config import QWEN3_8_27B
from reference import REF_DTYPE, flat_state, frob_ok, make_inputs, to_hT
from scaled_dot_kkt import kkt_ref
from solve_tril import solve_tril_ref
from wy_fast import CHUNK_TILE, wy_fast_ref

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens
STATE_DYN = pl.dynamic("GDN_STATE_DYN")     # (T // CHUNK_TILE) * H * D, the snapshot rows

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D
VAL_WIDTH = H * D

# validation
BENCH_SEQ = 8192        # prefill tokens, this entry's default case


def _gdn_chunk_h(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    w: pl.Tensor[[T_DYN, H, D], pl.FP16],
    u: pl.Tensor[[T_DYN, H, D], pl.FP16],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    state: pl.Out[pl.Tensor[[STATE_DYN, D], pl.FP16]],
    v_new: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE, and `state` hold (T // CHUNK_TILE) * H * D rows."""
    k.bind_dynamic(0, T_DYN)
    w.bind_dynamic(0, T_DYN)
    u.bind_dynamic(0, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    state.bind_dynamic(0, STATE_DYN)
    v_new.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(k, 0)
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    w_flat = pl.reshape(w, [t_dim, VAL_WIDTH])
    u_flat = pl.reshape(u, [t_dim, VAL_WIDTH])
    v_flat = pl.reshape(v_new, [t_dim, VAL_WIDTH])
    for hh in pl.spmd(H, name_hint="chunk_h",
                      optimizations=[pl.cross_core_slot(slot_num=1),
                                     pl.split(pl.SplitMode.UP_DOWN)]):
        s0 = pl.full([D, D], dtype=pl.FP32, value=0.0)
        for c, (s_cur,) in pl.range(t_dim // CHUNK_TILE, init_values=(s0,)):
            t0 = c * CHUNK_TILE
            row = (c * H + hh) * D
            d0 = hh * D
            # GQA: K comes from key head hh // GRP; W, U and the state are per
            # value head.
            dg0 = (hh // GRP) * D

            s16 = pl.cast(s_cur, target_type=pl.FP16, mode="rint")
            state[row : row + D, 0:D] = s16                 # state ENTERING this chunk

            # coeff[i] = exp(g_last - g_i), decay = exp(g_last); both arguments <= 0
            g_last = pl.read(g_sum, [hh, t0 + CHUNK_TILE - 1])
            g_row = g_sum[hh : hh + 1, t0 : t0 + CHUNK_TILE]
            zero_row = pl.full([1, CHUNK_TILE], dtype=pl.FP32, value=0.0)
            neg_g = pl.sub(zero_row, g_row)
            # exp before the [1,C]->[C,1] reshape, not after (pypto#2947)
            coeff = pl.reshape(pl.exp(pl.add(neg_g, g_last)), [CHUNK_TILE, 1])
            zero_d = pl.full([1, D], dtype=pl.FP32, value=0.0)
            decay = pl.reshape(pl.exp(pl.add(zero_d, g_last)), [D, 1])

            wc = w_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            ws = pl.matmul(wc, s16, out_dtype=pl.FP32)
            uc = u_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            vc = pl.sub(pl.cast(uc, target_type=pl.FP32), ws)   # V_new = U - W @ S
            vc16 = pl.cast(vc, target_type=pl.FP16, mode="rint")

            # S = exp(g_last) * S + K^T @ (coeff * V_new). The decay goes on V, not
            # on K: K^T (V c) == (K c)^T V. Under pl.split a matmul whose transposed
            # operand came from the vector unit cannot be lowered (pypto#2902).
            kc = k_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            vs = pl.row_expand_mul(vc, coeff)
            vs16 = pl.cast(vs, target_type=pl.FP16, mode="rint")
            kv = pl.matmul(kc, vs16, a_trans=True, out_dtype=pl.FP32)
            s_next = pl.add(pl.row_expand_mul(s_cur, decay), kv)

            v_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D] = vc16
            s_end, = pl.yield_(s_next)
    return state, v_new


gdn_chunk_h = pl.jit.inline(_gdn_chunk_h)
gdn_chunk_h_test = pl.jit(_gdn_chunk_h)


def chunk_h_ref(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_sum: torch.Tensor,
            chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S6: the inter-chunk state recurrence.

    Returns the state snapshot ENTERING each chunk, [NCHUNK, H, D, D]; the
    residual-corrected values V_new, [T, H, D]; and the state LEAVING the last
    chunk, [H, D, D], which is what an inference cache carries forward.
    """
    t, hg, d = k.shape
    h = w.shape[1]
    grp = h // hg
    nc = t // chunk
    kf, wf, uf, gf = (x.to(REF_DTYPE) for x in (k, w, u, g_sum))
    state = torch.zeros(nc, h, d, d, dtype=REF_DTYPE)
    v_new = torch.zeros(t, h, d, dtype=REF_DTYPE)
    final_state = torch.zeros(h, d, d, dtype=REF_DTYPE)
    for hh in range(h):
        s = torch.zeros(d, d, dtype=REF_DTYPE)
        for ci in range(nc):
            t0 = ci * chunk
            gc = gf[t0 : t0 + chunk, hh]
            g_last = gc[-1]
            state[ci, hh] = s
            vc = uf[t0 : t0 + chunk, hh, :] - wf[t0 : t0 + chunk, hh, :] @ s
            v_new[t0 : t0 + chunk, hh, :] = vc
            kv = (kf[t0 : t0 + chunk, hh // grp, :].T
                  @ (vc * torch.exp(g_last - gc)[:, None]))
            s = torch.exp(g_last) * s + kv
        final_state[hh] = s
    return state, v_new, final_state


_INPUTS: dict = {}


def _inputs(t: int, h: int, d: int, chunk: int, hg: int) -> dict:
    """The chain this stage consumes, memoised so the several specs pay for it once."""
    key = (t, h, d, chunk, hg)
    if key not in _INPUTS:
        st = make_inputs(t, h, d, hg)
        st["g_sum"] = cumsum_ref(st["g"], chunk)
        a16 = kkt_ref(st["k"], st["beta"], st["g_sum"], chunk).to(torch.float16)
        a_inv16 = solve_tril_ref(a16, chunk).to(torch.float16)
        w, u = wy_fast_ref(st["k"], st["v"], st["beta"], a_inv16, st["g_sum"], chunk)
        st["w16"] = w.to(torch.float16)
        st["u16"] = u.to(torch.float16)
        _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ, h: int = H, d: int = D, chunk: int = CHUNK_TILE,
                       hg: int = HG):
    from golden import TensorSpec

    def draw(key, transform=None):
        def make():
            value = _inputs(t, h, d, chunk, hg)[key]
            return transform(value) if transform is not None else value

        return make

    nc = t // chunk

    return [
        TensorSpec("k", [t, hg, d], torch.float16,
                   init_value=draw("k")),
        TensorSpec("w", [t, h, d], torch.float16,
                   init_value=draw("w16")),
        TensorSpec("u", [t, h, d], torch.float16,
                   init_value=draw("u16")),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=draw("g_sum", to_hT)),
        TensorSpec("state", [nc * h * d, d], torch.float16),
        TensorSpec("v_new", [t, h, d], torch.float16),
    ]


def golden_gdn_chunk_h(tensors):
    t, _, d = tensors["k"].shape          # k has Hg heads under GQA
    h = tensors["w"].shape[1]             # W, U and the state are per value head
    chunk = t // (tensors["state"].shape[0] // (h * d))
    state, v_new, _ = chunk_h_ref(tensors["k"], tensors["w"], tensors["u"],
                                  tensors["g_sum"].t(), chunk)
    tensors["state"].copy_(flat_state(state))
    tensors["v_new"].copy_(v_new)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=BENCH_SEQ)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()
    if args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")

    result = run(
        fn=gdn_chunk_h_test,
        specs=build_tensor_specs(t=args.seq_len),
        golden_fn=golden_gdn_chunk_h,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        compare_fn={"state": frob_ok("state"), "v_new": frob_ok("v_new")},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
