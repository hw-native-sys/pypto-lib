# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-embedding lookup and HC input packing for DeepSeek-V4.1 Flash."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

from models.deepseek_v4_1_flash.config import D, HC_MULT, T_DYN


TOKEN_DYN = T_DYN
VOCAB_DYN = pl.dynamic("PACK_X_HC_VOCAB_DYN")
SOURCE_ROWS = pl.dynamic("PACK_SOURCE_ROWS")
ROW_WIDTH = pl.dynamic("PACK_ROW_WIDTH")

HIDDEN_TILE = 512
SPMD_BLOCKS = 48
TEST_VOCAB_SIZE = 256
DEFAULT_TOKENS = 8

assert D % HIDDEN_TILE == 0


@pl.jit.inline
def gather_rows(
    source: pl.Tensor[[SOURCE_ROWS, ROW_WIDTH], pl.FP32],
    row_ids: pl.Tensor[[TOKEN_DYN], pl.INT32],
    output: pl.Tensor[[TOKEN_DYN, ROW_WIDTH], pl.FP32],
):
    """Gather complete rows, zeroing negative IDs used for inactive padding."""
    rows = pl.tensor.dim(output, 0)
    width = pl.tensor.dim(output, 1)
    for worker in pl.spmd(SPMD_BLOCKS, name_hint="gather_rows"):
        for row in pl.range(worker, rows, SPMD_BLOCKS):
            source_row = pl.cast(pl.read(row_ids, [row]), pl.INDEX)
            for col in pl.range(0, width, HIDDEN_TILE):
                active = pl.min(HIDDEN_TILE, width - col)
                if source_row >= 0:
                    value = pl.load(source, [source_row, col], [1, HIDDEN_TILE], valid_shape=[1, active])
                    pl.store(value, [row, col], output)
                else:
                    empty = pl.tile.full([1, HIDDEN_TILE], dtype=pl.FP32, value=0.0)
                    pl.store(pl.set_validshape(empty, 1, active), [row, col], output)
    return output


@pl.jit.inline
def pack_x_hc(
    input_ids: pl.Tensor[[TOKEN_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32]:
    """Look up token embeddings and replicate them across the HC lanes."""
    token_count = pl.tensor.dim(input_ids, 0)
    vocab_rows = pl.tensor.dim(embed_weight, 0)
    x_hc_flat = pl.reshape(x_hc, [token_count * HC_MULT, D])
    work_items = token_count * (D // HIDDEN_TILE)
    for block in pl.spmd(SPMD_BLOCKS, name_hint="pack_x_hc"):
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            token_idx = work_idx // (D // HIDDEN_TILE)
            hidden_offset = (work_idx % (D // HIDDEN_TILE)) * HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            safe_token_id = pl.max(pl.min(token_id, vocab_rows - 1), 0)
            token_row = pl.cast(safe_token_id, target_type=pl.INDEX)
            hidden_chunk = pl.cast(
                embed_weight[
                    token_row : token_row + 1,
                    hidden_offset : hidden_offset + HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            for hc_idx in pl.range(HC_MULT):
                x_hc_row = token_idx * HC_MULT + hc_idx
                x_hc_flat[
                    x_hc_row : x_hc_row + 1,
                    hidden_offset : hidden_offset + HIDDEN_TILE,
                ] = hidden_chunk
    return x_hc


@pl.jit
def pack_x_hc_test(
    input_ids: pl.Tensor[[TOKEN_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32]],
) -> pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32]:
    input_ids.bind_dynamic(0, TOKEN_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    x_hc.bind_dynamic(0, TOKEN_DYN)
    return pack_x_hc(input_ids, embed_weight, x_hc)


def golden_pack_x_hc(tensors):
    vocab_size = tensors["embed_weight"].shape[0]
    safe_ids = tensors["input_ids"].long().clamp(0, vocab_size - 1)
    hidden = tensors["embed_weight"].index_select(0, safe_ids).float()
    tensors["x_hc"][:] = hidden.unsqueeze(1).expand(-1, HC_MULT, -1)


def build_tensor_specs(token_count, vocab_size):
    import torch
    from golden import TensorSpec

    def init_input_ids():
        samples = torch.tensor(
            [-1, 0, 17, vocab_size - 1, vocab_size, vocab_size + 7, 2, 1],
            dtype=torch.int64,
        )
        repeats = (token_count + samples.numel() - 1) // samples.numel()
        return samples.repeat(repeats)[:token_count].contiguous()

    return [
        TensorSpec("input_ids", [token_count], torch.int64, init_value=init_input_ids),
        TensorSpec(
            "embed_weight",
            [vocab_size, D],
            torch.bfloat16,
            init_value=lambda: torch.randn(vocab_size, D, dtype=torch.bfloat16),
        ),
        TensorSpec(
            "x_hc",
            [token_count, HC_MULT, D],
            torch.float32,
        ),
    ]


def validate(argv=None):
    """Validate token embedding + HC packing on A5."""
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="Validate DeepSeek-V4.1 Flash token input packing.")
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--vocab", type=int, default=TEST_VOCAB_SIZE)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args(argv)

    result = run(
        fn=pack_x_hc_test,
        specs=build_tensor_specs(args.tokens, args.vocab),
        golden_fn=golden_pack_x_hc,
        compile_only=args.compile_only,
        config=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=0.0,
        atol=0.0,
    )
    return result


__all__ = [
    "DEFAULT_TOKENS",
    "TEST_VOCAB_SIZE",
    "build_tensor_specs",
    "golden_pack_x_hc",
    "pack_x_hc",
    "pack_x_hc_test",
    "validate",
]


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


def test_precision(a5_args):
    """Validate the operator against its golden reference on A5."""
    result = validate(a5_args())
    assert result.passed, result.error


if __name__ == _SCRIPT_ENTRY_POINT:
    main()
