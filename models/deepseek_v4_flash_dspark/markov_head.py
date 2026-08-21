# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 DSpark Markov embedding and full-vocabulary logits projection."""

import pypto.language as pl

from config import DECODE_BATCH, TP


# Dynamic shape variables.
T_DYN = pl.dynamic("MARKOV_HEAD_T_DYN")
VOCAB_DYN = pl.dynamic("MARKOV_HEAD_VOCAB_DYN")

# model config
MARKOV_RANK = 256

# tiling
T_TILE = 16
VOCAB_TILE = 128
SPMD_BLOCKS = 48


@pl.jit.inline
def markov_head(
    token_ids: pl.Tensor[[T_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    logits_bias: pl.Tensor[[T_DYN, VOCAB_DYN], pl.FP32],
    markov_embed: pl.Tensor[[T_DYN, MARKOV_RANK], pl.BF16],
):
    t_dim = pl.tensor.dim(token_ids, 0)
    t_linear = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE

    for token_idx in pl.spmd(t_dim, name_hint="markov_embedding", allow_early_resolve=True):
        token_id = pl.read(token_ids, [token_idx])
        token_row = pl.cast(token_id, target_type=pl.INDEX)
        markov_embed[token_idx : token_idx + 1, 0:MARKOV_RANK] = markov_w1[
            token_row : token_row + 1, 0:MARKOV_RANK
        ]

    vocab_dim = pl.tensor.dim(markov_w2, 0)
    work_items = (t_linear // T_TILE) * (vocab_dim // VOCAB_TILE)
    for block in pl.spmd(SPMD_BLOCKS, name_hint="markov_logits"):
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            t0 = (work_idx // (vocab_dim // VOCAB_TILE)) * T_TILE
            vocab0 = (work_idx % (vocab_dim // VOCAB_TILE)) * VOCAB_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            embed_bf16 = pl.slice(
                markov_embed, [T_TILE, MARKOV_RANK], [t0, 0], valid_shape=[valid_rows, MARKOV_RANK]
            )
            weight_bf16 = markov_w2[vocab0 : vocab0 + VOCAB_TILE, 0:MARKOV_RANK]
            logits_tile = pl.matmul(embed_bf16, weight_bf16, b_trans=True, out_dtype=pl.FP32)
            logits_valid = pl.set_validshape(logits_tile, valid_rows, VOCAB_TILE)
            logits_bias[t0 : t0 + T_TILE, vocab0 : vocab0 + VOCAB_TILE] = logits_valid

    return logits_bias, markov_embed


@pl.jit
def markov_head_test(
    token_ids: pl.Tensor[[T_DYN], pl.INT64],
    markov_w1: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    markov_w2: pl.Tensor[[VOCAB_DYN, MARKOV_RANK], pl.BF16],
    logits_bias: pl.Out[pl.Tensor[[T_DYN, VOCAB_DYN], pl.FP32]],
    markov_embed: pl.Out[pl.Tensor[[T_DYN, MARKOV_RANK], pl.BF16]],
):
    token_ids.bind_dynamic(0, T_DYN)
    markov_w1.bind_dynamic(0, VOCAB_DYN)
    markov_w2.bind_dynamic(0, VOCAB_DYN)
    logits_bias.bind_dynamic(0, T_DYN)
    logits_bias.bind_dynamic(1, VOCAB_DYN)
    markov_embed.bind_dynamic(0, T_DYN)
    return markov_head(token_ids, markov_w1, markov_w2, logits_bias, markov_embed)


def golden_markov_head(tensors):
    import torch

    markov_embed = tensors["markov_w1"].index_select(0, tensors["token_ids"].long())
    logits_bias = torch.matmul(markov_embed.float(), tensors["markov_w2"].float().t())
    tensors["markov_embed"][:] = markov_embed
    tensors["logits_bias"][:] = logits_bias


def build_tensor_specs(token_count, vocab_size):
    import torch
    from golden import TensorSpec

    def init_token_ids():
        sample_ids = torch.tensor([0, 1, 17, vocab_size - 1], dtype=torch.int64)
        repeats = (token_count + sample_ids.numel() - 1) // sample_ids.numel()
        return sample_ids.repeat(repeats)[:token_count].contiguous()

    def init_markov_w1():
        return torch.randn(vocab_size, MARKOV_RANK, dtype=torch.bfloat16) / MARKOV_RANK ** 0.5

    def init_markov_w2():
        return torch.randn(vocab_size, MARKOV_RANK, dtype=torch.bfloat16) / MARKOV_RANK ** 0.5

    return [
        TensorSpec("token_ids", [token_count], torch.int64, init_value=init_token_ids),
        TensorSpec("markov_w1", [vocab_size, MARKOV_RANK], torch.bfloat16, init_value=init_markov_w1),
        TensorSpec("markov_w2", [vocab_size, MARKOV_RANK], torch.bfloat16, init_value=init_markov_w2),
        TensorSpec("logits_bias", [token_count, vocab_size], torch.float32, is_output=True),
        TensorSpec("markov_embed", [token_count, MARKOV_RANK], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    TEST_VOCAB_SIZE = 4096

    parser = argparse.ArgumentParser(description="DeepSeek-V4 DSpark Markov-head validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--token-count", type=int, default=DECODE_BATCH // TP)
    parser.add_argument("--vocab-size", type=int, default=TEST_VOCAB_SIZE)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=markov_head_test,
        specs=build_tensor_specs(args.token_count, args.vocab_size),
        golden_fn=golden_markov_head,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=2e-3,
        atol=2e-3,
        compare_fn={
            "logits_bias": ratio_allclose(atol=2e-3, rtol=2e-3, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
