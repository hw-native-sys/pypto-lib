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
import torch

from chunk_cumsum import cumsum_ref
from chunk_h import CHUNK_TILE, chunk_h_ref
from config import QWEN3_8_27B
from reference import REF_DTYPE, flat_state, frob_ok, make_inputs, to_hT
from scaled_dot_kkt import kkt_ref
from solve_tril import solve_tril_ref
from wy_fast import wy_fast_ref

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
# One chunk, the cross-chunk recurrence, then eight chunks. The token count must be a
# multiple of CHUNK_TILE.
DEFAULT_CASES = (128, 256, 1024)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len


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



def chunk_o_ref(q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor,
            state: torch.Tensor, g_sum: torch.Tensor, chunk: int) -> torch.Tensor:
    """S7: the chunk output, inter-chunk plus intra-chunk. -> [T, H, D].

    The causal mask includes the diagonal here, unlike :func:`scaled_dot_kkt.kkt_ref`'s.
    """
    t, hg, d = q.shape
    h = v_new.shape[1]
    grp = h // hg
    qf, kf, vf, sf, gf = (x.to(REF_DTYPE) for x in (q, k, v_new, state, g_sum))
    out = torch.zeros(t, h, d, dtype=REF_DTYPE)
    rows = torch.arange(chunk)[:, None]
    cols = torch.arange(chunk)[None, :]
    causal = (rows >= cols).to(REF_DTYPE)
    zero = torch.zeros(chunk, chunk, dtype=REF_DTYPE)
    for ci in range(t // chunk):
        t0 = ci * chunk
        for hh in range(h):
            qc = qf[t0 : t0 + chunk, hh // grp, :]
            kc = kf[t0 : t0 + chunk, hh // grp, :]
            gc = gf[t0 : t0 + chunk, hh]
            inter = (qc @ sf[ci, hh]) * torch.exp(gc)[:, None]
            gate = torch.exp(torch.minimum(gc[:, None] - gc[None, :], zero))
            intra = ((qc @ kc.T) * gate * causal) @ vf[t0 : t0 + chunk, hh, :]
            out[t0 : t0 + chunk, hh, :] = inter + intra
    return out


_INPUTS: dict = {}


def _inputs(t: int) -> dict:
    """The chain this stage consumes, memoised so the several specs pay for it once."""
    key = t
    if key not in _INPUTS:
        st = make_inputs(t, H, D, HG)
        st["g_sum"] = cumsum_ref(st["g"], CHUNK_TILE)
        a16 = kkt_ref(st["k"], st["beta"], st["g_sum"], CHUNK_TILE).to(torch.float16)
        a_inv16 = solve_tril_ref(a16, CHUNK_TILE).to(torch.float16)
        w, u = wy_fast_ref(st["k"], st["v"], st["beta"], a_inv16, st["g_sum"], CHUNK_TILE)
        w16 = w.to(torch.float16)
        u16 = u.to(torch.float16)
        state, v_new, _ = chunk_h_ref(st["k"], w16, u16, st["g_sum"], CHUNK_TILE)
        st["state"] = state
        st["v_new16"] = v_new.to(torch.float16)
        _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ):
    from golden import TensorSpec

    def draw(key, transform=None):
        def make():
            value = _inputs(t)[key]
            return transform(value) if transform is not None else value

        return make

    nc = t // CHUNK_TILE

    def init_mask():
        rows = torch.arange(CHUNK_TILE)[:, None]
        cols = torch.arange(CHUNK_TILE)[None, :]
        return (rows >= cols).float()      # inclusive diagonal

    return [
        TensorSpec("q", [t, HG, D], torch.float16,
                   init_value=draw("q")),
        TensorSpec("k", [t, HG, D], torch.float16,
                   init_value=draw("k")),
        TensorSpec("v", [t, H, D], torch.float16,
                   init_value=draw("v_new16")),
        TensorSpec("state", [nc * H * D, D], torch.float16,
                   init_value=draw("state", flat_state)),
        TensorSpec("g_sum", [H, t], torch.float32,
                   init_value=draw("g_sum", to_hT)),
        TensorSpec("mask", [CHUNK_TILE, CHUNK_TILE], torch.float32, init_value=init_mask),
        TensorSpec("o_out", [t, H, D], torch.float16),
    ]


def golden_gdn_chunk_o(tensors):
    t, _, d = tensors["q"].shape          # q and k have Hg heads under GQA
    h = tensors["v"].shape[1]             # V, O and the state are per value head
    chunk = tensors["mask"].shape[0]
    state = tensors["state"].reshape(t // chunk, h, d, d)
    o = chunk_o_ref(tensors["q"], tensors["k"], tensors["v"], state, tensors["g_sum"].t(), chunk)
    tensors["o_out"].copy_(o)


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
        print(f"[case] chunk_o T={t}", flush=True)
        result = run(
            fn=gdn_chunk_o_test,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_gdn_chunk_o,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compare_fn={"o_out": frob_ok()},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
