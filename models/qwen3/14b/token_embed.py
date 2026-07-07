# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Embedding lookup for sampled Qwen3-14B token ids."""

from __future__ import annotations

import pypto.language as pl

from config import BATCH, HIDDEN, VOCAB
from greedy_sample import SAMPLED_IDS_PAD


HIDDEN_CHUNK = 256

assert HIDDEN % HIDDEN_CHUNK == 0


@pl.jit
def token_embed_fwd(
    sampled_ids: pl.Tensor[[BATCH, SAMPLED_IDS_PAD], pl.INT32],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    next_hidden: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
):
    """Gather embedding rows for sampled token ids."""
    for b in pl.parallel(0, BATCH, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_embed"):
            token_id = pl.read(sampled_ids, [b, 0])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            for k0 in pl.range(0, HIDDEN, HIDDEN_CHUNK):
                hidden_chunk = pl.slice(embed_weight, [1, HIDDEN_CHUNK], [token_row, k0])
                next_hidden = pl.assemble(next_hidden, hidden_chunk, [b, k0])

    return next_hidden


def build_tensor_specs():
    import torch

    from golden import TensorSpec

    def ids():
        base = torch.arange(BATCH, dtype=torch.int32).view(BATCH, 1) % VOCAB
        out = torch.zeros(BATCH, SAMPLED_IDS_PAD, dtype=torch.int32)
        out[:, :1] = base
        return out

    return [
        TensorSpec("sampled_ids", [BATCH, SAMPLED_IDS_PAD], torch.int32, init_value=ids),
        TensorSpec("embed_weight", [VOCAB, HIDDEN], torch.bfloat16, init_value=torch.randn),
        TensorSpec("next_hidden", [BATCH, HIDDEN], torch.bfloat16, is_output=True),
    ]


def golden_token_embed(tensors):
    sampled_ids = tensors["sampled_ids"][:, :1].long().view(-1)
    tensors["next_hidden"][:] = tensors["embed_weight"].index_select(0, sampled_ids).to(tensors["next_hidden"].dtype)


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=token_embed_fwd,
        specs=build_tensor_specs(),
        golden_fn=golden_token_embed,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
