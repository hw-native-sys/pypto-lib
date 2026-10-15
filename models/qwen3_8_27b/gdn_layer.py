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

from chunk_cumsum import DISPATCH_TOKENS, gdn_chunk_cumsum
from chunk_h import gdn_chunk_h
from chunk_o import gdn_chunk_o
from config import GDN_TILING, PREFILL_SEQ, QWEN3_8_27B
from scaled_dot_kkt import gdn_scaled_dot_kkt
import solve_tril
from solve_tril import gdn_solve_tril
from wy_fast import gdn_wy_fast

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
VAL_WIDTH = H * D
CHUNK_TILE = GDN_TILING.chunk               # chunk size in tokens, our tiling choice
T = PREFILL_SEQ                             # tokens (single sequence, B = 1)

# A, A_inv, W, U, the state snapshots and V_new are all [T, H, *] and live at once:
# about 600 MiB at T = 8192, H = 48, against the runtime's default 256 MiB per ring.
# Without this the layer fails with orch_error_code=2 HEAP_RING_DEADLOCK. The same
# thing happens to models/deepseek_v4_pro/prefill_layer.py, at the same value.
LAYER_RING_HEAP = 1024 * 1024 * 1024


def run_config(platform: str = "a2a3", device: int = 0, swimlane: int = 0) -> dict:
    """The config `golden.run` needs for this layer. The ring heap is not optional."""
    return dict(platform=platform, device_id=device, ring_heap=LAYER_RING_HEAP,
                enable_chip_swimlane=swimlane)


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



_INPUTS: dict = {}


def _inputs(t: int, h: int, d: int, hg: int) -> dict:
    """The drawn q, k, v, beta and g, memoised so the five specs pay for them once."""
    import reference

    key = (t, h, d, hg)
    if key not in _INPUTS:
        _INPUTS[key] = reference.make_inputs(t, h, d, hg)
    return _INPUTS[key]


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK_TILE,
                       hg: int = HG):
    import torch
    from golden import TensorSpec

    import reference

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
                   init_value=draw("beta", reference.to_hT)),
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
    import reference

    chunk = tensors["tril"].shape[0]
    st = reference.delta_rule(tensors["q"].double(), tensors["k"].double(),
                              tensors["v"].double(), tensors["beta"].t().double(),
                              tensors["g"].double(), chunk)
    tensors["o_out"].copy_(st["o"])


def _stats_ok(actual, expected, **_kwargs):
    """Relative Frobenius norm against the float64 golden, with the peak reported."""
    import reference

    ok, detail = reference.stats_ok(actual, expected, chunk=CHUNK_TILE)
    print(f"[stats] {detail}", flush=True)
    return ok, detail


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=T)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=range(5))
    args = parser.parse_args()
    if args.seq_len % DISPATCH_TOKENS:
        parser.error(f"--seq-len must be a multiple of {DISPATCH_TOKENS}")

    result = run(
        fn=gdn_layer_test,
        specs=build_tensor_specs(t=args.seq_len),
        golden_fn=golden_gdn_layer,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=run_config(args.platform, args.device, args.enable_chip_swimlane),
        rtol=1e-2, atol=1e-5,
        compare_fn={"o_out": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
