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

The layer reads and updates a caller-owned recurrent state row and returns O.
"""
import pypto.language as pl
import torch

from chunk_cumsum import CHUNK_TILE, DISPATCH_TOKENS, cumsum_ref, gdn_chunk_cumsum
from chunk_h import (FSTATE_DYN, chunk_h_ref, compare_final_state,
                     compare_packed_final_state, gdn_chunk_h_final, gdn_chunk_h_final_packed)
from chunk_o_pipe import gdn_chunk_o_pipe
from config import QWEN3_8_27B
from delta_rule_reference import chunk_o_ref
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
    row_base: pl.Tensor[[T_DYN], pl.INT32],
    state_row: pl.Scalar[pl.INDEX],
    final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, D, D], pl.FP32]],
    o_out: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
):
    """The token count must be a multiple of chunk_cumsum's DISPATCH_TOKENS, the coarsest tile here."""
    q.bind_dynamic(0, T_DYN)
    k.bind_dynamic(0, T_DYN)
    v.bind_dynamic(0, T_DYN)
    g.bind_dynamic(0, T_DYN)
    beta.bind_dynamic(1, T_DYN)
    row_base.bind_dynamic(0, T_DYN)
    final_state.bind_dynamic(0, FSTATE_DYN)
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
    gdn_chunk_h_final(k, w, u, g_sum, row_base, state, v_new, state_row, final_state)

    gdn_chunk_o_pipe(q, k, v_new, state, g_sum, tril, o_out)
    return o_out, final_state


gdn_layer = pl.jit.inline(_gdn_layer)
gdn_layer_test = pl.jit(_gdn_layer)


def build_packed_kernel(rows: int, inline: bool = False):
    """The delta rule with independent resumed state for each packed row."""
    if rows < 1:
        raise ValueError("packed GDN layer needs at least one row")

    @(pl.jit.inline if inline else pl.jit)
    def gdn_layer_packed(
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
        state_rows: pl.Tensor[[rows], pl.INT32],
        row_ends: pl.Tensor[[rows], pl.INT32],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, D, D], pl.FP32]],
        o_out: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
    ):
        q.bind_dynamic(0, T_DYN)
        k.bind_dynamic(0, T_DYN)
        v.bind_dynamic(0, T_DYN)
        g.bind_dynamic(0, T_DYN)
        beta.bind_dynamic(1, T_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        o_out.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(q, 0)
        nchunk = t_dim // CHUNK_TILE
        g_sum = pl.create_tensor([H, t_dim], dtype=pl.FP32)
        gdn_chunk_cumsum(g, tril, g_sum)
        a = pl.create_tensor([t_dim, H, CHUNK_TILE], dtype=pl.FP16)
        gdn_scaled_dot_kkt(k, beta, g_sum, mask_strict, a)
        a_inv = pl.create_tensor([t_dim, H, CHUNK_TILE], dtype=pl.FP16)
        gdn_solve_tril(a, eye, m_diag, m_low, a_inv)
        w = pl.create_tensor([t_dim, H, D], dtype=pl.FP16)
        u = pl.create_tensor([t_dim, H, D], dtype=pl.FP16)
        gdn_wy_fast(k, v, a_inv, beta, g_sum, w, u)
        state = pl.create_tensor([nchunk * VAL_WIDTH, D], dtype=pl.FP16)
        v_new = pl.create_tensor([t_dim, H, D], dtype=pl.FP16)
        gdn_chunk_h_final_packed(k, w, u, g_sum, state_rows, row_ends,
                                 state, v_new, final_state)
        gdn_chunk_o_pipe(q, k, v_new, state, g_sum, tril, o_out)
        return o_out, final_state

    return gdn_layer_packed



def delta_rule_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               beta: torch.Tensor, g: torch.Tensor, chunk: int,
               initial_state: torch.Tensor | None = None,
               row_base: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
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
    state, v_new, st["final_state"] = chunk_h_ref(
        k, st["w16"], st["u16"], st["g_sum"], chunk, initial_state, row_base)
    st["state16"] = state.to(torch.float16)
    st["v_new16"] = v_new.to(torch.float16)
    st["o"] = chunk_o_ref(q, k, st["v_new16"], st["state16"], st["g_sum"], chunk)
    return st


_INPUTS: dict = {}


def _inputs(t: int) -> dict:
    """The drawn q, k, v, beta and g, memoised so the five specs pay for them once."""
    key = t
    if key not in _INPUTS:
        _INPUTS[key] = make_inputs(t, H, D, HG)
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ, resume: bool = False):
    from golden import ScalarSpec, TensorSpec

    def init_tril():
        return torch.tril(torch.ones(CHUNK_TILE, CHUNK_TILE, dtype=torch.float32))

    def init_mask_strict():
        rows = torch.arange(CHUNK_TILE)[:, None]
        cols = torch.arange(CHUNK_TILE)[None, :]
        return (rows > cols).float()

    def draw(key, transform=None):
        def make():
            value = _inputs(t)[key]
            return transform(value) if transform is not None else value

        return make

    def pool():
        rows = torch.full((4, H, D, D), -1.0, dtype=torch.float32)
        if resume:
            rows[2] = torch.randn(H, D, D, generator=torch.Generator().manual_seed(91)) * 0.02
        else:
            rows[2] = 0.0
        return rows

    return [
        TensorSpec("q", [t, HG, D], torch.float16, init_value=draw("q")),
        TensorSpec("k", [t, HG, D], torch.float16, init_value=draw("k")),
        TensorSpec("v", [t, H, D], torch.float16, init_value=draw("v")),
        TensorSpec("g", [t, H], torch.float32, init_value=draw("g")),
        TensorSpec("beta", [H, t], torch.float32,
                   init_value=draw("beta", to_hT)),
        TensorSpec("tril", [CHUNK_TILE, CHUNK_TILE], torch.float32, init_value=init_tril),
        TensorSpec("mask_strict", [CHUNK_TILE, CHUNK_TILE], torch.float32,
                   init_value=init_mask_strict),
        TensorSpec("eye", [CHUNK_TILE, CHUNK_TILE], torch.float16,
                   init_value=lambda: solve_tril.eye_block(CHUNK_TILE)),
        TensorSpec("m_diag", [CHUNK_TILE, CHUNK_TILE], torch.float16,
                   init_value=lambda: solve_tril.blk_masks()[0]),
        TensorSpec("m_low", [CHUNK_TILE, CHUNK_TILE], torch.float16,
                   init_value=lambda: solve_tril.blk_masks()[1]),
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: torch.zeros(t, dtype=torch.int32)),
        ScalarSpec("state_row", torch.int64, 2, compile_runtime=True),
        TensorSpec("final_state", [4, H, D, D], torch.float32, init_value=pool),
        TensorSpec("o_out", [t, H, D], torch.float16),
    ]


def golden_gdn_layer(tensors):
    """The whole chain in float64 on the tensors the kernel was given, scored end to end."""
    chunk = tensors["tril"].shape[0]
    row = int(tensors["state_row"])
    initial = tensors["final_state"][row].transpose(-2, -1).contiguous()
    st = delta_rule_ref(tensors["q"].double(), tensors["k"].double(),
                        tensors["v"].double(), tensors["beta"].t().double(),
                        tensors["g"].double(), chunk, initial, tensors["row_base"])
    tensors["o_out"].copy_(st["o"])
    tensors["final_state"][row].copy_(st["final_state"].transpose(-2, -1))


def build_packed_tensor_specs(t: int = DISPATCH_TOKENS, starts: tuple[int, ...] = (0, 256),
                              state_rows: tuple[int, ...] = (1, 3),
                              pool_rows: int = 5, resume: bool = False):
    from golden import TensorSpec

    if t <= 0 or t % DISPATCH_TOKENS:
        raise ValueError(f"packed GDN layer needs a positive multiple of {DISPATCH_TOKENS} tokens")
    if (not starts or len(starts) != len(state_rows) or starts[0] != 0
            or any(start % 256 for start in starts)
            or any(b - a < 256 for a, b in zip(starts, starts[1:] + (t,)))
            or len(set(state_rows)) != len(state_rows)
            or any(not 0 <= row < pool_rows for row in state_rows)):
        raise ValueError("packed GDN layer needs disjoint complete rows and cache owners")
    ends = starts[1:] + (t,)

    def pool():
        value = torch.full((pool_rows, H, D, D), -1.0, dtype=torch.float32)
        for owner, row in enumerate(state_rows):
            gen = torch.Generator().manual_seed(91 + owner)
            value[row] = (torch.randn(H, D, D, generator=gen) * 0.02 if resume else 0.0)
        return value

    base = build_tensor_specs(t=t, resume=resume)
    return base[:10] + [
        TensorSpec("state_rows", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(state_rows, dtype=torch.int32)),
        TensorSpec("row_ends", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(ends, dtype=torch.int32)),
        TensorSpec("final_state", [pool_rows, H, D, D], torch.float32, init_value=pool),
    ] + base[13:]


def golden_gdn_layer_packed(tensors):
    ends = tensors["row_ends"].tolist()
    starts = [0] + ends[:-1]
    for owner, (start, end) in enumerate(zip(starts, ends)):
        row = int(tensors["state_rows"][owner])
        initial = tensors["final_state"][row].transpose(-2, -1).contiguous()
        st = delta_rule_ref(tensors["q"][start:end].double(),
                            tensors["k"][start:end].double(),
                            tensors["v"][start:end].double(),
                            tensors["beta"][:, start:end].t().double(),
                            tensors["g"][start:end].double(), CHUNK_TILE, initial)
        tensors["o_out"][start:end].copy_(st["o"])
        tensors["final_state"][row].copy_(st["final_state"].transpose(-2, -1))


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--packed", action="store_true", default=False)
    parser.add_argument("--row-starts", type=str, default="0,256")
    parser.add_argument("--compile-only", action="store_true", default=False)
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
        if args.packed:
            starts = tuple(int(value) for value in args.row_starts.split(","))
            state_rows = tuple(2 * owner + 1 for owner in range(len(starts)))
            print(f"[case] gdn_layer packed T={t} starts={starts} resume={args.resume}",
                  flush=True)
            result = run(
                fn=build_packed_kernel(rows=len(starts)),
                specs=build_packed_tensor_specs(t=t, starts=starts, state_rows=state_rows,
                                                pool_rows=2 * len(starts) + 1,
                                                resume=args.resume),
                golden_fn=golden_gdn_layer_packed,
                golden_data=args.golden_data,
                runtime_dir=args.runtime_dir,
                save_data=args.save_data,
                config=run_config(args.backend_type, args.execution_mode, args.device,
                                  args.enable_chip_swimlane),
                compare_fn={"o_out": frob_ok(),
                            "final_state": compare_packed_final_state(state_rows)},
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(f"FAILED at packed GDN layer T={t}")
            continue
        print(f"[case] gdn_layer T={t}", flush=True)
        result = run(
            fn=gdn_layer_test,
            specs=build_tensor_specs(t=t, resume=args.resume),
            golden_fn=golden_gdn_layer,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=run_config(args.backend_type, args.execution_mode, args.device, args.enable_chip_swimlane),
            compare_fn={"o_out": frob_ok(), "final_state": compare_final_state(2)},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
