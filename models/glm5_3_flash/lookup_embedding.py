# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Look up one vocabulary shard and replicate BF16 embeddings into mHC streams."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, HC_MULT, LOCAL_VOCAB, SUPPORTED_TP_SIZES, TP_SIZE, T_DYN


# tiling
D_TILE = 1024


@pl.jit.inline
def lookup_embedding(
    token_ids: pl.Tensor[[T_DYN], pl.INT32],
    table: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
    vocab_start: pl.Scalar[pl.INT32],
    hidden_streams: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    """Write local-shard rows or zeros; the caller reduces across vocabulary shards."""
    token_count = pl.tensor.dim(token_ids, 0)
    streams_flat = pl.reshape(hidden_streams, [token_count * HC_MULT, D])
    vocab_stop = vocab_start + LOCAL_VOCAB
    for block in pl.spmd(48, name_hint="lookup_embedding"):
        for work in pl.range(block, token_count * (D // D_TILE), 48):
            token = work // (D // D_TILE)
            hidden_offset = (work % (D // D_TILE)) * D_TILE
            token_id = pl.read(token_ids, [token])
            chunk = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            if token_id >= vocab_start:
                if token_id < vocab_stop:
                    local_id = pl.cast(token_id - vocab_start, pl.INDEX)
                    chunk = table[local_id : local_id + 1, hidden_offset : hidden_offset + D_TILE]
            for stream in pl.range(HC_MULT):
                row = token * HC_MULT + stream
                streams_flat[row : row + 1, hidden_offset : hidden_offset + D_TILE] = chunk
    return hidden_streams


@pl.jit
def lookup_embedding_test(
    token_ids: pl.Tensor[[T_DYN], pl.INT32],
    table: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
    vocab_start: pl.Scalar[pl.INT32],
    hidden_streams: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    token_ids.bind_dynamic(0, T_DYN)
    hidden_streams.bind_dynamic(0, T_DYN)
    return lookup_embedding(token_ids, table, vocab_start, hidden_streams)


def golden_lookup_embedding(
    token_ids: torch.Tensor,
    table: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    """Return replicated local-shard rows with exact zeros for nonlocal IDs."""
    local_ids = token_ids.to(torch.int64) - int(vocab_start)
    local = (local_ids >= 0) & (local_ids < table.shape[0])
    output = torch.zeros(token_ids.numel(), HC_MULT, D, dtype=table.dtype, device=table.device)
    rows = table.index_select(0, local_ids[local])
    output[local] = rows.unsqueeze(1).expand(-1, HC_MULT, -1)
    return output


def golden_lookup_embedding_test(tensors):
    inputs = tensors["token_ids"], tensors["table"], int(tensors["vocab_start"])
    tensors["hidden_streams"][:] = golden_lookup_embedding(*inputs)


def build_tensor_specs(token_ids, vocab_start):
    from golden import ScalarSpec, TensorSpec

    ids = torch.tensor(token_ids, dtype=torch.int32)
    generator = torch.Generator().manual_seed(53)
    table = torch.randn(LOCAL_VOCAB, D, dtype=torch.bfloat16, generator=generator)
    return [
        TensorSpec("token_ids", [ids.numel()], torch.int32, init_value=ids),
        TensorSpec("table", [LOCAL_VOCAB, D], torch.bfloat16, init_value=table),
        ScalarSpec("vocab_start", torch.int32, vocab_start),
        TensorSpec("hidden_streams", [ids.numel(), HC_MULT, D], torch.bfloat16),
    ]


def validation_cases():
    """Cover first, middle, last and absent vocabulary owners at one and 17 rows."""
    middle = (TP_SIZE // 2) * LOCAL_VOCAB
    last = (TP_SIZE - 1) * LOCAL_VOCAB
    middle_stop = middle + LOCAL_VOCAB
    last_stop = last + LOCAL_VOCAB
    mixed = [middle, middle_stop - 1, middle - 1, middle_stop, middle + 1, middle, -1]
    last_ids = [last, last_stop - 1, last - 1, last_stop, last + 1, last_stop - 1, -1]
    outside = [middle - 1, middle_stop, -1, -(2**31), 2**31 - 1]
    return [
        ("decode_first", [0], 0),
        ("prefill_middle", (mixed * 3)[:17], middle),
        ("decode_nonlocal", [-(2**31)], last),
        ("prefill_last", (last_ids * 3)[:17], last),
        ("prefill_nonlocal", (outside * 4)[:17], middle),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description="Validate vocabulary-sharded embedding lookup.")
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=SUPPORTED_TP_SIZES)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    for name, token_ids, vocab_start in validation_cases():
        print(f"--- lookup_embedding {name}: T={len(token_ids)}, vocab_start={vocab_start} ---")
        result = run(
            fn=lookup_embedding_test,
            specs=build_tensor_specs(token_ids, vocab_start),
            golden_fn=golden_lookup_embedding_test,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=0.0,
            atol=0.0,
            compile_only=args.compile_only,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)


__all__ = [
    "golden_lookup_embedding",
    "lookup_embedding",
]
