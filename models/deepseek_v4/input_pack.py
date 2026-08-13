# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-embedding lookup and HC input packing for DeepSeek-V4 forwards."""

import pypto.language as pl

from config import ACTIVE as M, DECODE_TOKENS, PREFILL_TOKENS


TOKEN_DYN = pl.dynamic("PACK_X_HC_TOKEN_DYN")
VOCAB_DYN = pl.dynamic("PACK_X_HC_VOCAB_DYN")

D = M.hidden_size
HC_MULT = M.hc_mult

HIDDEN_TILE = 512
SPMD_BLOCKS = 48

assert D % HIDDEN_TILE == 0


@pl.jit.inline
def pack_x_hc(
    input_ids: pl.Tensor[[TOKEN_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[TOKEN_DYN, HC_MULT, D], pl.FP32]:
    """Look up token embeddings and replicate them across the HC lanes."""
    token_count = pl.tensor.dim(input_ids, 0)
    x_hc_flat = pl.reshape(x_hc, [token_count * HC_MULT, D])
    work_items = token_count * (D // HIDDEN_TILE)
    for block in pl.spmd(SPMD_BLOCKS, name_hint="pack_x_hc"):
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            token_idx = work_idx // (D // HIDDEN_TILE)
            hidden_offset = (work_idx % (D // HIDDEN_TILE)) * HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
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
    hidden = tensors["embed_weight"].index_select(0, tensors["input_ids"].long()).float()
    tensors["x_hc"][:] = hidden.unsqueeze(1).expand(-1, HC_MULT, -1)


def build_tensor_specs(token_count, vocab_size):
    import torch
    from golden import TensorSpec

    def init_input_ids():
        samples = torch.tensor(
            [0, 1, 17, vocab_size - 1, 17, 2, vocab_size // 2, 1],
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
            is_output=True,
        ),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    modes = {"decode": DECODE_TOKENS, "prefill": PREFILL_TOKENS}
    test_vocab_size = 256

    parser = argparse.ArgumentParser(description="Validate DeepSeek-V4 token input packing.")
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a5",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(modes) if args.mode == "all" else [args.mode]
    for mode_name in modes_to_run:
        token_count = modes[mode_name]
        print(f"--- pack_x_hc_test {mode_name}: T={token_count} ---")
        result = run_jit(
            fn=pack_x_hc_test,
            specs=build_tensor_specs(token_count, test_vocab_size),
            golden_fn=golden_pack_x_hc,
            compile_only=args.compile_only,
            runtime_cfg=dict(platform=args.platform, device_id=args.device),
            rtol=0.0,
            atol=0.0,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
