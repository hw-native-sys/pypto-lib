# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The single-token KDA recurrence used by decode and by the MTP draft step.

Per head, with the state ``S`` in FP32 ``[KDA_DIM, KDA_DIM]``:

    S      <- S * exp(g)
    kv_mem <- sum_k S[k, :] * key[k]
    delta  <- (value - kv_mem) * beta
    S      <- S + key[:, None] * delta[None, :]
    out    <- sum_k S[k, :] * query[k]

q and k are L2-normalised in FP32 and q carries the ``KDA_DIM ** -0.5`` scale, both
inside the kernel, matching ``use_qk_l2norm_in_kernel=True``.

This is the decode critical path: 34 layers x 4 heads per rank, each a pair of
128x128 rank-1 updates, and the state is read-modify-written in place, so it is the
one KDA kernel that cannot be batched across layers.

**Start from the existing pypto implementation, not from scratch.** cann-recipes-infer
carries a complete single-step recurrent KDA in pypto at
``integration/vllm/ling-3.0-flash/npu_patch/.../ops/pypto/kda/fused_recurrent_kda_impl.py``, developed and
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


def golden_decode_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("recurrent KDA golden is assigned with the kernel")


@pl.jit.inline
def decode_kda(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, LOCAL_KDA_H], pl.FP32],
    recurrent_state: pl.Tensor[[B_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    raise NotImplementedError("recurrent KDA kernel body is assigned independently")


__all__ = [
    "decode_kda",
    "golden_decode_kda",
]
