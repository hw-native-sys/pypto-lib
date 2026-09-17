# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Chunked Kimi-delta-attention for prefill — the largest single assignment here.

``chunk_kimi_delta_attention`` expands the per-token recurrence into a chunked form:
a decay cumulative sum inside each chunk, a ``K K^T`` Gram matrix, the inverse of a
unit lower-triangular matrix (the UT transform), the ``W`` / ``U`` recomputation,
an inter-chunk scan over the ``[KDA_DIM, KDA_DIM]`` state, and the intra-chunk
output. q and k are L2-normalised in FP32 first and q is scaled by
``KDA_DIM ** -0.5``.

Two pypto facts shape the assignment:

* there is **no scan or cumsum tile op**, so the decay cumulative sum and the
  inter-chunk state scan must both be written as explicit ``pl.range`` loops;
* the recurrent state is FP32 ``[LOCAL_KDA_H, KDA_DIM, KDA_DIM]`` = 256 KB per
  request per layer at TP16, which does not fit UB (192 KB, ~184 KB usable), so the
  state has to be tiled over heads.

**Start from the existing pypto implementation, not from scratch.** cann-recipes-infer
carries a complete chunked delta-rule forward in pypto at
``integration/vllm/ling-3.0-flash/npu_patch/.../ops/pypto/kda/chunk_kda_impl.py``, developed and
tuned on Ascend910B3 — the same a2a3 generation — with a stage-by-stage design
write-up in ``docs/integration/ling-3.0-flash/bailing_v3_pypto_operator_guide.md``.
Its quoted shape, ``T=4K, H=4, K=V=128``, is literally our per-rank TP16 shape.

It is **not** code to copy: it targets a different pypto frontend
(``pypto.frontend.jit`` / ``pypto.Tensor`` / ``pypto_impl``, with automatic tiling),
and the pypto this repo pins exports none of those. The port is from implicit tiling
to the explicit ``pl`` tile DSL.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import B_DYN, KDA_DIM, LOCAL_KDA_H, T_DYN


CHUNK = 64  # chunk length of the delta-rule recurrence; retune on a2a3


def golden_prefill_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("chunked KDA golden is assigned with the kernel")


@pl.jit.inline
def prefill_kda(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, LOCAL_KDA_H], pl.FP32],
    recurrent_state: pl.Tensor[[B_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[B_DYN + 1], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    raise NotImplementedError("chunked KDA kernel body is assigned independently")


__all__ = [
    "CHUNK",
    "golden_prefill_kda",
    "prefill_kda",
]
