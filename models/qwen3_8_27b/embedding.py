# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B token embedding: out[t, :] = embed_weight[input_ids[t], :]."""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
HIDDEN = QWEN3_8_27B.hidden_size
VOCAB = QWEN3_8_27B.vocab_size
# validation
DEFAULT_CASES = (64, 128)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
# The whole row in one piece. A table row is 10240 bytes and the gather is
# scattered across 2.54 GiB, so splitting it costs: at 512 the same work
# measures 755.5 us against 122.3, ten 1024-byte reads per token rather than
# one. Narrower pieces only help if a row stops fitting.
K_TILE = HIDDEN
SPMD_BLOCKS = 48        # one per vector core

# host only: rows per band of the input draw
INIT_ROW_BAND = 16384


K_BLOCKS = HIDDEN // K_TILE             # column tiles of one embedding row


def _qwen3_embedding(
    input_ids: pl.Tensor[[T_DYN], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[T_DYN, HIDDEN], pl.BF16]],
):
    """One work item is one column tile of one token's row."""
    input_ids.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(input_ids, 0)
    work_items = t_dim * K_BLOCKS
    for core in pl.spmd(SPMD_BLOCKS, name_hint="embedding"):
        for work in pl.range(core, work_items, SPMD_BLOCKS):
            ti = work // K_BLOCKS
            k0 = (work - ti * K_BLOCKS) * K_TILE
            token_id = pl.tensor.read(input_ids, [ti])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            row_piece = pl.slice(embed_weight, [1, K_TILE], [token_row, k0])
            out[ti : ti + 1, k0 : k0 + K_TILE] = row_piece
    return out


qwen3_embedding = pl.jit.inline(_qwen3_embedding)
qwen3_embedding_test = pl.jit(_qwen3_embedding)


def build_l3_kernel(batch: int):
    """Build a HOST entry for a fixed token-row extent."""
    if batch < 1:
        raise ValueError("embedding requires at least one token")
    kernel = qwen3_embedding_test

    @pl.jit.host
    def l3_embedding(
        input_ids: pl.Tensor[[batch], pl.INT32],
        embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
        out: pl.Out[pl.Tensor[[batch, HIDDEN], pl.BF16]],
    ):
        return kernel(input_ids, embed_weight, out)

    return l3_embedding



def _banded_normal(rows: int, cols: int, seed: int, scale: float = 1.0):
    """A BF16 `[rows, cols]` normal draw built one band at a time.

    `torch.randn(248320, 5120)` is 5.09 GB of FP32 before it is narrowed, and
    simulator CI builds these at the full vocabulary. Banding keeps the peak at
    the BF16 result plus one band.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(rows, cols, dtype=torch.bfloat16)
    for r0 in range(0, rows, INIT_ROW_BAND):
        rows_here = min(INIT_ROW_BAND, rows - r0)
        band = torch.randn(rows_here, cols, generator=gen)
        if scale != 1.0:
            band *= scale
        out[r0 : r0 + rows_here] = band.to(torch.bfloat16)
    return out

def build_tensor_specs(t: int = BENCH_SEQ, hidden: int = HIDDEN, vocab: int = VOCAB):
    import torch
    from golden import TensorSpec

    def init_ids():
        gen = torch.Generator().manual_seed(42)
        return torch.randint(0, vocab, (t,), generator=gen, dtype=torch.int32)

    def init_table():
        return _banded_normal(vocab, hidden, seed=43)

    return [
        TensorSpec("input_ids", [t], torch.int32, init_value=init_ids),
        TensorSpec("embed_weight", [vocab, hidden], torch.bfloat16, init_value=init_table),
        TensorSpec("out", [t, hidden], torch.bfloat16),
    ]


def golden_qwen3_embedding(tensors):
    # The reference for a gather is the gather: torch indexing is exact here,
    # every output element being a copy of a table element.
    ids = tensors["input_ids"].long()
    tensors["out"].copy_(tensors["embed_weight"][ids])


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--vocab", type=int, default=VOCAB)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        print(f"[case] embedding T={t}", flush=True)
        result = run(
            fn=qwen3_embedding_test,
            specs=build_tensor_specs(t=t, vocab=args.vocab),
            golden_fn=golden_qwen3_embedding,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=0,
            atol=0,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={t}")
