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

from config import GDN_TILING, PREFILL_SEQ, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens
STATE_DYN = pl.dynamic("GDN_STATE_DYN")     # (T // CHUNK_TILE) * H * D, the snapshot rows

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
CHUNK_TILE = GDN_TILING.chunk               # chunk size in tokens, our tiling choice
GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D
VAL_WIDTH = H * D
T = PREFILL_SEQ                             # tokens (single sequence, B = 1)


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
    # Without pl.split the kernel needs 197632 B of a 188416 B vector buffer.
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
        a_inv16 = reference.solve_tril(a16, chunk).to(torch.float16)
        w, u = reference.wy_fast(st["k"], st["v"], st["beta"], a_inv16, st["g_sum"], chunk)
        st["w16"] = w.to(torch.float16)
        st["u16"] = u.to(torch.float16)
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

    nc = t // chunk

    return [
        TensorSpec("k", [t, hg, d], torch.float16,
                   init_value=draw("k")),
        TensorSpec("w", [t, h, d], torch.float16,
                   init_value=draw("w16")),
        TensorSpec("u", [t, h, d], torch.float16,
                   init_value=draw("u16")),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=draw("g_sum", reference.to_hT)),
        TensorSpec("state", [nc * h * d, d], torch.float16),
        TensorSpec("v_new", [t, h, d], torch.float16),
    ]


def golden_gdn_chunk_h(tensors):
    import reference

    t, _, d = tensors["k"].shape          # k has Hg heads under GQA
    h = tensors["w"].shape[1]             # W, U and the state are per value head
    chunk = t // (tensors["state"].shape[0] // (h * d))
    state, v_new, _ = reference.chunk_h(tensors["k"], tensors["w"], tensors["u"],
                                        tensors["g_sum"].t(), chunk)
    tensors["state"].copy_(reference.flat_state(state))
    tensors["v_new"].copy_(v_new)


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
        fn=gdn_chunk_h_test,
        specs=build_tensor_specs(t=args.seq_len),
        golden_fn=golden_gdn_chunk_h,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"state": _stats_ok, "v_new": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
