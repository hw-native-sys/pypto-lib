# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Embedding lookup for the text backbone (weights loaded pre-rotated by the export)."""

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN


# Dynamic shape variable.

# model config
D = M.hidden_size


@pl.jit.inline(auto_scope=False)
def embedding_lookup(
    input_ids: pl.Tensor,
    embed_weight: pl.Tensor,
    output: pl.Tensor,
    width: pl.constexpr,
):
    """Copy one BF16 embedding row per input token."""
    t_dim = pl.tensor.dim(input_ids, 0)
    for token in pl.spmd(t_dim, name_hint="embedding_lookup"):
        row = pl.cast(pl.read(input_ids, [token]), pl.INDEX)
        output[token : token + 1, :] = pl.slice(embed_weight, [1, width], [row, 0])
    return output


@pl.jit
def embedding_test(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[1024, 5120], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, 5120], pl.BF16]],
):
    input_ids.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    embedding_lookup(input_ids, embed_weight, output, width=5120)
    return output


def golden_embedding(tensors):
    """Torch reference: row gather from the embedding table."""
    tensors["output"][:] = tensors["embed_weight"][tensors["input_ids"].long()]


def build_tensor_specs(tokens: int = 16):
    from golden import TensorSpec

    return [
        TensorSpec("input_ids", [tokens], torch.int64, init_value=lambda: torch.randint(0, 1024, (tokens,))),
        TensorSpec("embed_weight", [1024, 5120], torch.bfloat16, init_value=torch.randn),
        TensorSpec("output", [tokens, 5120], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    result = run(
        fn=embedding_test,
        specs=build_tensor_specs(args.tokens),
        golden_fn=golden_embedding,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
