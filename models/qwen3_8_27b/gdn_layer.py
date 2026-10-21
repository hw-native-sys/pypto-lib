# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B's Gated DeltaNet layer: the six operators as one forward.

    g_sum   = chunk_cumsum(g)                     chunk-local prefix sum of the gate
    A       = scaled_dot_kkt(k, beta, g_sum)      gated key-key matrix, strictly lower
    A_inv   = solve_tril(A)                       (I + A)^-1 per chunk
    W, U    = wy_fast(k, v, beta, A_inv, g_sum)   the WY representation
    S, Vnew = chunk_h(k, W, U, g_sum)             inter-chunk state recurrence
    O       = chunk_o(q, k, Vnew, S, g_sum)       chunk output

Only q, k, v, the gate and beta come in, and only O goes out.
"""
import pypto.language as pl
import torch

from chunk_cumsum import CHUNK_TILE, DISPATCH_TOKENS, cumsum_ref, gdn_chunk_cumsum
from chunk_h import chunk_h_ref, gdn_chunk_h
from chunk_o import chunk_o_ref, gdn_chunk_o
from config import QWEN3_8_27B
from reference import frob_ok, make_inputs, to_hT
from scaled_dot_kkt import gdn_scaled_dot_kkt, kkt_ref
import solve_tril
from solve_tril import gdn_solve_tril, solve_tril_ref
from wy_fast import gdn_wy_fast, wy_fast_ref

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
VAL_WIDTH = H * D

# validation
# The minimum dispatch, then two of them. One chunk is below DISPATCH_TOKENS, so this
# entry has no single-chunk case.
DEFAULT_CASES = (512, 1024)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# Bytes per ring, one entry per scope depth, 0 leaving a ring at its default. The stage
# scopes allocate on ring 1, the only ring the default cannot hold at BENCH_SEQ; a scalar
# here would reserve the same bytes on all four. The floor grows with the token count, so a
# longer --seq-len needs a larger entry.
LAYER_RING_HEAP = [0, 1024 * 1024 * 1024, 0, 0]


def run_config(backend_type: str = "a2a3", sim: bool = False, device: int = 0, swimlane: int = 0) -> dict:
    """The config `golden.run` needs for this layer, the ring heap included."""
    return dict(
        backend_type=backend_type, execution_mode=sim, device_id=device,
        ring_heap=LAYER_RING_HEAP, enable_chip_swimlane=swimlane,
    )


def _gdn_layer(
    q: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    v: pl.Tensor[[T_DYN, H, D], pl.FP16],
    g: pl.Tensor[[T_DYN, H], pl.FP32],
    beta: pl.Tensor[[H, T_DYN], pl.FP32],
    tril: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    mask_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    eye: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    m_diag: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    m_low: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    o_out: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
):
    """The token count must be a multiple of chunk_cumsum's DISPATCH_TOKENS, the coarsest tile here."""
    q.bind_dynamic(0, T_DYN)
    k.bind_dynamic(0, T_DYN)
    v.bind_dynamic(0, T_DYN)
    g.bind_dynamic(0, T_DYN)
    beta.bind_dynamic(1, T_DYN)
    o_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(q, 0)
    nchunk = t_dim // CHUNK_TILE
    # `tril` serves twice: chunk_cumsum contracts against it, and it is also
    # chunk_o's causal mask -- both want the inclusive lower triangle. Only
    # scaled_dot_kkt's is strict, so that one is a second constant.
    g_sum = pl.create_tensor([H, t_dim], dtype=pl.FP32)
    gdn_chunk_cumsum(g, tril, g_sum)

    a = pl.create_tensor([t_dim, H, CHUNK_TILE], dtype=pl.FP16)
    gdn_scaled_dot_kkt(k, beta, g_sum, mask_strict, a)

    a_inv = pl.create_tensor([t_dim, H, CHUNK_TILE], dtype=pl.FP16)
    gdn_solve_tril(a, eye, m_diag, m_low, a_inv)

    w = pl.create_tensor([t_dim, H, D], dtype=pl.FP16)
    u = pl.create_tensor([t_dim, H, D], dtype=pl.FP16)
    gdn_wy_fast(k, v, a_inv, beta, g_sum, w, u)

    state_rows = nchunk * VAL_WIDTH
    state = pl.create_tensor([state_rows, D], dtype=pl.FP16)
    v_new = pl.create_tensor([t_dim, H, D], dtype=pl.FP16)
    gdn_chunk_h(k, w, u, g_sum, state, v_new)

    gdn_chunk_o(q, k, v_new, state, g_sum, tril, o_out)
    return o_out


gdn_layer = pl.jit.inline(_gdn_layer)
gdn_layer_test = pl.jit(_gdn_layer)



def delta_rule_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               beta: torch.Tensor, g: torch.Tensor, chunk: int) -> dict[str, torch.Tensor]:
    """All six stages on the given inputs; every intermediate, `o` the output.

    Values crossing a stage boundary are narrowed to the dtype the kernels
    exchange -- FP16 for A, A_inv, W, U, V_new and the state snapshots -- so a
    stage reads what the preceding kernel would have handed it.
    """
    st = dict(q=q, k=k, v=v, beta=beta, g=g)
    st["g_sum"] = cumsum_ref(g, chunk)
    st["a16"] = kkt_ref(k, beta, st["g_sum"], chunk).to(torch.float16)
    st["a_inv16"] = solve_tril_ref(st["a16"], chunk).to(torch.float16)
    w, u = wy_fast_ref(k, v, beta, st["a_inv16"], st["g_sum"], chunk)
    st["w16"] = w.to(torch.float16)
    st["u16"] = u.to(torch.float16)
    state, v_new, st["final_state"] = chunk_h_ref(k, st["w16"], st["u16"], st["g_sum"], chunk)
    st["state16"] = state.to(torch.float16)
    st["v_new16"] = v_new.to(torch.float16)
    st["o"] = chunk_o_ref(q, k, st["v_new16"], st["state16"], st["g_sum"], chunk)
    return st


_INPUTS: dict = {}


def _inputs(t: int, h: int, d: int, hg: int) -> dict:
    """The drawn q, k, v, beta and g, memoised so the five specs pay for them once."""
    key = (t, h, d, hg)
    if key not in _INPUTS:
        _INPUTS[key] = make_inputs(t, h, d, hg)
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ, h: int = H, d: int = D, chunk: int = CHUNK_TILE, hg: int = HG):
    from golden import TensorSpec

    def init_tril():
        return torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))

    def init_mask_strict():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows > cols).float()

    def draw(key, transform=None):
        def make():
            value = _inputs(t, h, d, hg)[key]
            return transform(value) if transform is not None else value

        return make

    return [
        TensorSpec("q", [t, hg, d], torch.float16, init_value=draw("q")),
        TensorSpec("k", [t, hg, d], torch.float16, init_value=draw("k")),
        TensorSpec("v", [t, h, d], torch.float16, init_value=draw("v")),
        TensorSpec("g", [t, h], torch.float32, init_value=draw("g")),
        TensorSpec("beta", [h, t], torch.float32,
                   init_value=draw("beta", to_hT)),
        TensorSpec("tril", [chunk, chunk], torch.float32, init_value=init_tril),
        TensorSpec("mask_strict", [chunk, chunk], torch.float32,
                   init_value=init_mask_strict),
        TensorSpec("eye", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.eye_block(chunk)),
        TensorSpec("m_diag", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[0]),
        TensorSpec("m_low", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[1]),
        TensorSpec("o_out", [t, h, d], torch.float16),
    ]


def golden_gdn_layer(tensors):
    """The whole chain in float64 on the tensors the kernel was given, scored end to end."""
    chunk = tensors["tril"].shape[0]
    st = delta_rule_ref(tensors["q"].double(), tensors["k"].double(),
                        tensors["v"].double(), tensors["beta"].t().double(),
                        tensors["g"].double(), chunk)
    tensors["o_out"].copy_(st["o"])


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % DISPATCH_TOKENS:
        parser.error(f"--seq-len must be a multiple of DISPATCH_TOKENS={DISPATCH_TOKENS}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        print(f"[case] gdn_layer T={t}", flush=True)
        result = run(
            fn=gdn_layer_test,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_gdn_layer,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=run_config(args.backend_type, args.execution_mode, args.device, args.enable_chip_swimlane),
            compare_fn={"o_out": frob_ok("o_out")},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
