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

from config import GDN_TILING, PREFILL_SEQ, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # gate heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension; unused here, drawn inputs match the pipeline
CHUNK_TILE = GDN_TILING.chunk               # chunk size in tokens, our tiling choice
T = PREFILL_SEQ                             # tokens (single sequence, B = 1)

# tiling
GROUP_TILE = 4          # chunks per dispatch, sharing one tril load
DISPATCH_TOKENS = CHUNK_TILE * GROUP_TILE


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


_INPUTS: dict = {}


def _inputs(t: int, h: int, d: int, chunk: int, hg: int) -> dict:
    """The chain this stage consumes, run here rather than shared with the other stages.

    Memoised so the several specs that draw from it pay for it once.
    """

    import reference

    key = (t, h, d, chunk, hg)
    if key not in _INPUTS:
        st = reference.make_inputs(t, h, d, hg)
        _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK_TILE,
                       hg: int = HG):
    # hg only picks which reference chain to draw from; this stage reads no q or k.
    import torch
    from golden import TensorSpec


    def draw(key, transform=None):
        def make():
            value = _inputs(t, h, d, chunk, hg)[key]
            return transform(value) if transform is not None else value

        return make


    def init_tril():
        return torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))

    return [
        TensorSpec("g", [t, h], torch.float32,
                   init_value=draw("g")),
        TensorSpec("tril", [chunk, chunk], torch.float32, init_value=init_tril),
        TensorSpec("g_sum", [h, t], torch.float32),
    ]


def golden_gdn_chunk_cumsum(tensors):
    import reference

    g = tensors["g"]
    out = tensors["g_sum"]
    chunk = tensors["tril"].shape[0]
    out.copy_(reference.to_hT(reference.cumsum(g, chunk)))


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
    if args.seq_len % DISPATCH_TOKENS:
        parser.error(f"--seq-len must be a multiple of DISPATCH_TOKENS={DISPATCH_TOKENS}")

    result = run(
        fn=gdn_chunk_cumsum_test,
        specs=build_tensor_specs(t=args.seq_len),
        golden_fn=golden_gdn_chunk_cumsum,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-4,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
