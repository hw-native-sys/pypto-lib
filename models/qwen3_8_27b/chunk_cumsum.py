# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_cumsum: chunk-local prefix sum of the gate logits,
g_sum[t, h] = sum over i <= t within the chunk of g[i, h]."""
import pypto.language as pl
import torch

from config import QWEN3_8_27B
from reference import REF_DTYPE, make_inputs, to_hT

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # gate heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension; unused here, drawn inputs match the pipeline

# tiling
CHUNK_TILE = 128        # chunk length the delta rule is blocked over
GROUP_TILE = 4          # chunks per dispatch, sharing one tril load
DISPATCH_TOKENS = CHUNK_TILE * GROUP_TILE

# validation
# The minimum dispatch, then two of them. One chunk is below DISPATCH_TOKENS, so this
# entry has no single-chunk case.
DEFAULT_CASES = (512, 1024)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len


def _gdn_chunk_cumsum(
    g: pl.Tensor[[T_DYN, H], pl.FP32],
    tril: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    g_sum: pl.Out[pl.Tensor[[H, T_DYN], pl.FP32]],
):
    """The token count must be a multiple of DISPATCH_TOKENS; `pl.unroll` needs a static group."""
    g.bind_dynamic(0, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    t_dim = pl.tensor.dim(g, 0)
    for c0 in pl.spmd(t_dim // DISPATCH_TOKENS, name_hint="chunk_cumsum"):
        t0 = c0 * DISPATCH_TOKENS
        tl = tril[:, :]                        # constant, held for the whole scope
        for c in pl.unroll(GROUP_TILE):
            s0 = t0 + c * CHUNK_TILE
            # [H, CHUNK_TILE] = (tril @ g_chunk)^T; the cube absorbs both transposes
            g_chunk = pl.matmul(g[s0 : s0 + CHUNK_TILE, :], tl, a_trans=True, b_trans=True)
            g_sum[:, s0 : s0 + CHUNK_TILE] = g_chunk
    return g_sum


gdn_chunk_cumsum = pl.jit.inline(_gdn_chunk_cumsum)
gdn_chunk_cumsum_test = pl.jit(_gdn_chunk_cumsum)


def cumsum_ref(g: torch.Tensor, chunk: int) -> torch.Tensor:
    """S1: chunk-local prefix sum of the gate logits. [T, H] -> [T, H]."""
    out = torch.zeros_like(g, dtype=REF_DTYPE)
    gf = g.to(REF_DTYPE)
    for t0 in range(0, g.shape[0], chunk):
        out[t0 : t0 + chunk] = gf[t0 : t0 + chunk].cumsum(dim=0)
    return out


_INPUTS: dict = {}


def _inputs(t: int) -> dict:
    """The chain this stage consumes, memoised so the several specs pay for it once."""
    key = t
    if key not in _INPUTS:
        _INPUTS[key] = make_inputs(t, H, D, HG)
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ):
    from golden import TensorSpec

    def draw(key, transform=None):
        def make():
            value = _inputs(t)[key]
            return transform(value) if transform is not None else value

        return make


    def init_tril():
        return torch.tril(torch.ones(CHUNK_TILE, CHUNK_TILE, dtype=torch.float32))

    return [
        TensorSpec("g", [t, H], torch.float32,
                   init_value=draw("g")),
        TensorSpec("tril", [CHUNK_TILE, CHUNK_TILE], torch.float32, init_value=init_tril),
        TensorSpec("g_sum", [H, t], torch.float32),
    ]


def golden_gdn_chunk_cumsum(tensors):
    g = tensors["g"]
    out = tensors["g_sum"]
    chunk = tensors["tril"].shape[0]
    out.copy_(to_hT(cumsum_ref(g, chunk)))


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
    if args.seq_len and args.seq_len % DISPATCH_TOKENS:
        parser.error(f"--seq-len must be a multiple of DISPATCH_TOKENS={DISPATCH_TOKENS}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        print(f"[case] chunk_cumsum T={t}", flush=True)
        result = run(
            fn=gdn_chunk_cumsum_test,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_gdn_chunk_cumsum,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-4,
            atol=1e-5,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
