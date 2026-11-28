# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-embedding lookup, HC input packing and cache lengths for DeepSeek-V4.1 Flash."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import pypto.language as pl

from .config import FLASH as M
from .dsl_types import T_DYN


# model config
D = M.hidden_size
HC_MULT = M.hc_mult


TOKEN_DYN = T_DYN
VOCAB_DYN = pl.dynamic("PACK_X_HC_VOCAB_DYN")
SOURCE_ROWS = pl.dynamic("PACK_SOURCE_ROWS")
ROW_WIDTH = pl.dynamic("PACK_ROW_WIDTH")

HIDDEN_TILE = 512
LENGTH_TILE = 128
SPMD_BLOCKS = 48
TEST_VOCAB_SIZE = 256
DEFAULT_TOKENS = 8


@pl.jit.inline(auto_scope=False)
def prepare_compressed_lens(
    position_ids: pl.Tensor[[TOKEN_DYN], pl.INT32],
    lengths: pl.Tensor[[TOKEN_DYN], pl.INT32],
    compression_shift: pl.constexpr,
):
    """Derive visible C1A (shift 0) or C2A (shift 1) lengths; position -1 gives zero."""
    rows = pl.tensor.dim(position_ids, 0)
    positions_row = pl.reshape(position_ids, [1, rows])
    lengths_row = pl.reshape(lengths, [1, rows])
    for row in pl.parallel(0, rows, LENGTH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prepare_compressed_lens"):
            active = pl.min(LENGTH_TILE, rows - row)
            positions = pl.load(positions_row, [0, row], [1, LENGTH_TILE], valid_shape=[1, active])
            visible = pl.add(positions, 1)
            compressed = pl.shrs(visible, compression_shift)
            pl.store(compressed, [0, row], lengths_row)
    return lengths


@pl.jit.inline
def gather_rows(
    source: pl.Tensor[[SOURCE_ROWS, ROW_WIDTH], pl.FP32],
    row_ids: pl.Tensor[[TOKEN_DYN], pl.INT32],
    output: pl.Tensor[[TOKEN_DYN, ROW_WIDTH], pl.FP32],
    source_ready: pl.Scalar[pl.TASK_ID],
    column_tile: pl.constexpr,
):
    """Gather rows after source_ready; negative IDs produce zero padding."""
    rows = pl.tensor.dim(output, 0)
    width = pl.tensor.dim(output, 1)
    workers = pl.max(1, pl.min(pl.min(rows, SPMD_BLOCKS), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="gather_rows", deps=[source_ready]):
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, rows, workers):
            source_row = pl.cast(pl.read(row_ids, [row]), pl.INDEX)
            for col in pl.range(0, width, column_tile):
                active = pl.min(column_tile, width - col)
                if source_row >= 0:
                    value = pl.load(source, [source_row, col], [1, column_tile], valid_shape=[1, active])
                    pl.store(value, [row, col], output)
                else:
                    empty = pl.tile.full([1, column_tile], dtype=pl.FP32, value=0.0)
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
    workers = pl.max(1, pl.min(pl.min(work_items, SPMD_BLOCKS), pl.system.available_aiv_count()))
    for block in pl.spmd(workers, name_hint="pack_x_hc"):
        for work_idx in pl.range(block, work_items, workers):
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
def l2_input_pack(
    input_ids: pl.Tensor[[TOKEN_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32]],
) -> pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32]:
    input_ids.bind_dynamic(0, TOKEN_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    x_hc.bind_dynamic(0, TOKEN_DYN)
    return pack_x_hc(input_ids, embed_weight, x_hc)


@pl.jit
def _validate_row_metadata(
    source: pl.Tensor[[16, 32], pl.FP32],
    ids: pl.Tensor[[16], pl.INT32],
    positions: pl.Tensor[[16], pl.INT32],
    output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    c1a: pl.Out[pl.Tensor[[16], pl.INT32]],
    c2a: pl.Out[pl.Tensor[[16], pl.INT32]],
):
    ready = pl.system.task_dummy(deps=[])
    gather_rows(source, ids, output, ready, 32)
    prepare_compressed_lens(positions, c1a, 0)
    prepare_compressed_lens(positions, c2a, 1)
    return output, c1a, c2a


def validate_row_metadata(config, compile_only):
    import torch
    from golden import TensorSpec, run

    ids = torch.tensor([3, 2, 2, -1, *range(4, 16)], dtype=torch.int32)
    positions = torch.tensor([-1, 0, 1, 2, *range(122, 134)], dtype=torch.int32)
    source = torch.arange(512, dtype=torch.float32).reshape(16, 32)

    def golden(tensors):
        values = source[ids.clamp_min(0).long()]
        values[ids < 0] = 0
        tensors["output"].copy_(values)
        tensors["c1a"].copy_(positions + 1)
        tensors["c2a"].copy_((positions + 1) // 2)

    specs = [TensorSpec(name, list(t.shape), t.dtype, init_value=t)
             for name, t in (("source", source), ("ids", ids), ("positions", positions))]
    specs.extend((TensorSpec("output", [16, 32], torch.float32),
                  TensorSpec("c1a", [16], torch.int32), TensorSpec("c2a", [16], torch.int32)))
    result = run(fn=_validate_row_metadata, specs=specs, golden_fn=golden, config=config,
                 compile_only=compile_only, rtol=0, atol=0)
    if not result.passed:
        raise SystemExit(result.error or 1)


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


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="Validate DeepSeek-V4.1 Flash token input packing.")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--vocab", type=int, default=TEST_VOCAB_SIZE)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=l2_input_pack,
        specs=build_tensor_specs(args.tokens, args.vocab),
        golden_fn=golden_pack_x_hc,
        compile_only=args.compile_only,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
        ),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
    validate_row_metadata(dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                          args.compile_only)
