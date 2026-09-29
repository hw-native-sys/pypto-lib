# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_o: the chunk output,

    inter = exp(g_i) * (Q @ S)
    intra = (Q @ K^T * exp(min(g_i - g_j, 0)) * causal) @ V_new
    O     = inter + intra

The causal mask includes the diagonal, unlike scaled_dot_kkt's strictly-lower one.
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
T = PREFILL_SEQ                             # tokens (single sequence, B = 1)


GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D
VAL_WIDTH = H * D


def _gdn_chunk_o(
    q: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    v: pl.Tensor[[T_DYN, H, D], pl.FP16],
    state: pl.Tensor[[STATE_DYN, D], pl.FP16],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    mask: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    o_out: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE, and `state` hold (T // CHUNK_TILE) * H * D rows."""
    q.bind_dynamic(0, T_DYN)
    k.bind_dynamic(0, T_DYN)
    v.bind_dynamic(0, T_DYN)
    state.bind_dynamic(0, STATE_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    o_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(q, 0)
    q_flat = pl.reshape(q, [t_dim, KEY_WIDTH])
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    v_flat = pl.reshape(v, [t_dim, VAL_WIDTH])
    o_flat = pl.reshape(o_out, [t_dim, VAL_WIDTH])
    for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="chunk_o",
                      optimizations=[pl.cross_core_slot(slot_num=1),
                                     pl.split(pl.SplitMode.UP_DOWN)]):
        t0 = c0 * CHUNK_TILE
        for hh in pl.range(H):
            d0 = hh * D
            # GQA: Q and K come from key head hh // GRP; V, O and the state
            # snapshot are per value head.
            dg0 = (hh // GRP) * D
            qc = q_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            g_row = g_sum[hh : hh + 1, t0 : t0 + CHUNK_TILE]
            g_col = pl.reshape(g_row, [CHUNK_TILE, 1])
            # exp before the [1,C]->[C,1] reshape, not after (pypto#2947)
            eg = pl.reshape(pl.exp(g_row), [CHUNK_TILE, 1])

            row = c0 * VAL_WIDTH + d0
            s_blk = state[row : row + D, 0 : D]
            inter = pl.row_expand_mul(pl.matmul(qc, s_blk), eg)

            # A matmul result, not pl.create_tensor: the accumulator must be a tile
            # private to this core group, not a tensor every chunk in flight shares.
            kc = k_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            qk = pl.matmul(qc, kc, b_trans=True)
            diff = pl.full([CHUNK_TILE, CHUNK_TILE], dtype=pl.FP32, value=0.0)
            diff = pl.row_expand_add(diff, g_col)
            diff = pl.col_expand_sub(diff, g_row)
            decay = pl.exp(pl.minimum(diff, 0.0))
            gate = pl.mul(decay, mask[:, :])
            # a matmul's result dtype follows its CONSUMER, not its operands
            gated = pl.cast(pl.mul(qk, gate), target_type=pl.FP16, mode="rint")
            vc = v_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            intra = pl.matmul(gated, vc, out_dtype=pl.FP32)

            o_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D] = pl.cast(pl.add(inter, intra),
                                                           target_type=pl.FP16, mode="rint")
    return o_out


gdn_chunk_o = pl.jit.inline(_gdn_chunk_o)
gdn_chunk_o_test = pl.jit(_gdn_chunk_o)



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
        state, v_new, _ = reference.chunk_h(st["k"], w.to(torch.float16), u.to(torch.float16),
        st["g_sum"], chunk)
        st["state"] = state
        st["v_new16"] = v_new.to(torch.float16)
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

    def init_mask():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows >= cols).float()      # inclusive diagonal

    return [
        TensorSpec("q", [t, hg, d], torch.float16,
                   init_value=draw("q")),
        TensorSpec("k", [t, hg, d], torch.float16,
                   init_value=draw("k")),
        TensorSpec("v", [t, h, d], torch.float16,
                   init_value=draw("v_new16")),
        TensorSpec("state", [nc * h * d, d], torch.float16,
                   init_value=draw("state", reference.flat_state)),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=draw("g_sum", reference.to_hT)),
        TensorSpec("mask", [chunk, chunk], torch.float32, init_value=init_mask),
        TensorSpec("o_out", [t, h, d], torch.float16),
    ]


def golden_gdn_chunk_o(tensors):
    import reference

    t, _, d = tensors["q"].shape          # q and k have Hg heads under GQA
    h = tensors["v"].shape[1]             # V, O and the state are per value head
    chunk = tensors["mask"].shape[0]
    state = tensors["state"].reshape(t // chunk, h, d, d)
    o = reference.chunk_o(tensors["q"], tensors["k"], tensors["v"], state, tensors["g_sum"].t(), chunk)
    tensors["o_out"].copy_(o)


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
        fn=gdn_chunk_o_test,
        specs=build_tensor_specs(t=args.seq_len),
        golden_fn=golden_gdn_chunk_o,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"o_out": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
