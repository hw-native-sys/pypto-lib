# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Flash token embedding lookup for packed prefill and decode IDs.

Emits both the plain hidden row and the Hyper-Connections view: every block
entry point consumes ``[T, HC_MULT, D]`` FP32, and layer 0 feeds it HC_MULT
identical copies of the embedding row.
"""

import pypto.language as pl

from config import FLASH as M, DECODE_TOKENS, PREFILL_TOKENS


# Dynamic shape variables.
T_DYN = pl.dynamic("LOOKUP_EMBEDDING_T_DYN")
VOCAB_DYN = pl.dynamic("LOOKUP_EMBEDDING_VOCAB_DYN")

# model config
D = M.hidden_size
HC_MULT = M.hc_mult

# tiling
HIDDEN_TILE = 512
SPMD_BLOCKS = 48


@pl.jit.inline
def lookup_embedding(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    token_count = pl.tensor.dim(input_ids, 0)
    x_hc_flat = pl.reshape(x_hc, [token_count * HC_MULT, D])
    work_items = token_count * (D // HIDDEN_TILE)
    for block in pl.spmd(SPMD_BLOCKS, name_hint="lookup_embedding"):
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            token_idx = work_idx // (D // HIDDEN_TILE)
            hidden_block = work_idx % (D // HIDDEN_TILE)
            hidden_offset = hidden_block * HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            hidden_chunk = embed_weight[token_row : token_row + 1, hidden_offset : hidden_offset + HIDDEN_TILE]
            hidden_states[token_idx : token_idx + 1, hidden_offset : hidden_offset + HIDDEN_TILE] = hidden_chunk
            hc_chunk = pl.cast(hidden_chunk, target_type=pl.FP32, mode="none")
            for hc_idx in pl.range(HC_MULT):
                hc_row = token_idx * HC_MULT + hc_idx
                x_hc_flat[hc_row : hc_row + 1, hidden_offset : hidden_offset + HIDDEN_TILE] = hc_chunk

    return hidden_states, x_hc


@pl.jit
def lookup_embedding_test(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    hidden_states: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_hc: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    input_ids.bind_dynamic(0, T_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    hidden_states.bind_dynamic(0, T_DYN)
    x_hc.bind_dynamic(0, T_DYN)

    return lookup_embedding(input_ids, embed_weight, hidden_states, x_hc)


def golden_lookup_embedding_test(tensors):
    embed = tensors["embed_weight"].index_select(0, tensors["input_ids"].long())
    tensors["hidden_states"][:] = embed
    tensors["x_hc"][:] = embed.float().unsqueeze(1).repeat(1, HC_MULT, 1)


def build_tensor_specs(token_count, vocab_size):
    import torch
    from golden import TensorSpec

    def init_input_ids():
        sample_ids = torch.tensor([0, 1, 17, vocab_size - 1, 17, 2, vocab_size // 2, 1], dtype=torch.int64)
        repeats = (token_count + sample_ids.numel() - 1) // sample_ids.numel()
        return sample_ids.repeat(repeats)[:token_count].contiguous()

    def init_embed_weight():
        return torch.randn(vocab_size, D, dtype=torch.bfloat16)

    return [
        TensorSpec("input_ids", [token_count], torch.int64, init_value=init_input_ids),
        TensorSpec("embed_weight", [vocab_size, D], torch.bfloat16, init_value=init_embed_weight),
        TensorSpec("hidden_states", [token_count, D], torch.bfloat16, is_output=True),
        TensorSpec("x_hc", [token_count, HC_MULT, D], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    MODES = {"decode": DECODE_TOKENS, "prefill": PREFILL_TOKENS}
    TEST_VOCAB_SIZE = 256

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash embedding lookup validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    args = parser.parse_args()

    modes_to_run = list(MODES) if args.mode == "all" else [args.mode]
    for mode_name in modes_to_run:
        token_count = MODES[mode_name]
        print(f"--- lookup_embedding_test {mode_name}: T={token_count} ---")
        result = run_jit(
            fn=lookup_embedding_test,
            specs=build_tensor_specs(token_count, TEST_VOCAB_SIZE),
            golden_fn=golden_lookup_embedding_test,
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
            ),
            rtol=0.0,
            atol=0.0,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
