# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 LM head projection for one TP shard.

The input hidden states are expected to have already passed the final RMSNorm,
matching cann-recipes' ``DeepseekV3Model.forward`` + ``forward_lm_head`` split.
This kernel computes only the local vocabulary shard for TP=16 and intentionally
does not perform the cross-rank gather/all-to-all needed to form full logits.
"""

import pypto.language as pl

from config import DECODE_TOKENS, FLASH as M, LM_HEAD_TP_SIZE


T = DECODE_TOKENS
D = M.hidden_size
VOCAB = M.vocab_size

VOCAB_PER_TP = VOCAB // LM_HEAD_TP_SIZE
LM_HEAD_K_CHUNK = 128
VOCAB_CHUNK = 80
T_TILE = 16

assert VOCAB % LM_HEAD_TP_SIZE == 0
assert VOCAB_PER_TP % VOCAB_CHUNK == 0
assert D % LM_HEAD_K_CHUNK == 0
assert T % T_TILE == 0

VOCAB_BLOCKS = VOCAB_PER_TP // VOCAB_CHUNK
K_BLOCKS = D // LM_HEAD_K_CHUNK


@pl.jit.inline
def lm_head(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits_shard: pl.Tensor[[T, VOCAB_PER_TP], pl.FP32],
) -> pl.Tensor[[T, VOCAB_PER_TP], pl.FP32]:
    # Communication contract for the full TP=16 path:
    # 1. Each rank owns one contiguous vocab shard of lm_head_weight:
    #    [VOCAB_PER_TP, D], where global vocab range is
    #    [tp_rank * VOCAB_PER_TP, (tp_rank + 1) * VOCAB_PER_TP).
    # 2. This kernel computes only local logits_shard [T, VOCAB_PER_TP].
    # 3. The caller/orchestrator must gather shards across the lm_head TP group
    #    to form full logits [T, VOCAB] before sampling. For a pure TP layout
    #    this is an all-gather over the vocab axis.
    # 4. If attention/data parallel groups also split the token dimension, match
    #    cann-recipes' forward_lm_head behavior: gather hidden states before the
    #    matmul when needed, then use all-to-all/all-gather so every sampling
    #    owner receives the full-vocab logits for its local tokens.
    for t0 in pl.parallel(0, T, T_TILE):
        for ob in pl.parallel(VOCAB_BLOCKS):
            o0 = ob * VOCAB_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head"):
                hidden_chunk = pl.slice(hidden_states, [T_TILE, LM_HEAD_K_CHUNK], [t0, 0])
                weight_chunk = pl.slice(lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [o0, 0])
                acc = pl.matmul(hidden_chunk, weight_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, K_BLOCKS):
                    k0 = kb * LM_HEAD_K_CHUNK
                    hidden_chunk = pl.slice(hidden_states, [T_TILE, LM_HEAD_K_CHUNK], [t0, k0])
                    weight_chunk = pl.slice(lm_head_weight, [VOCAB_CHUNK, LM_HEAD_K_CHUNK], [o0, k0])
                    acc = pl.matmul_acc(acc, hidden_chunk, weight_chunk, b_trans=True)
                logits_shard = pl.assemble(logits_shard, acc, [t0, o0])

    return logits_shard


@pl.jit
def lm_head_test(
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logits_shard: pl.Out[pl.Tensor[[T, VOCAB_PER_TP], pl.FP32]],
) -> pl.Tensor[[T, VOCAB_PER_TP], pl.FP32]:
    logits_shard = lm_head(hidden_states, lm_head_weight, logits_shard)
    return logits_shard


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    weight = tensors["lm_head_weight"].float()
    logits = torch.matmul(hidden, weight.t())
    tensors["logits_shard"][:] = logits


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_hidden_states():
        return torch.randn(T, D) * 0.1

    def init_lm_head_weight():
        return (torch.randn(VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)

    return [
        TensorSpec("hidden_states", [T, D], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("lm_head_weight", [VOCAB_PER_TP, D], torch.bfloat16, init_value=init_lm_head_weight),
        TensorSpec("logits_shard", [T, VOCAB_PER_TP], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=lm_head_test,
        specs=build_tensor_specs(),
        golden_fn=golden_lm_head,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
