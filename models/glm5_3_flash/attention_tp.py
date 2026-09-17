# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The shared TP16 reduction over row-parallel partial sums.

Four kernels finish a layer by producing an FP32 partial of the full hidden width
rather than a finished activation: ``kda_output``, ``mla_epilog_prefill``,
``mla_epilog_decode`` and ``dense_mlp``. All four are row-parallel — they contract
over a head or intermediate axis that TP split — so each rank holds a summand, and
the layer is not complete until the 16 summands are added.

This file owns that addition, once, for all of them. Without a single owner the
three producers each declare an FP32 ``[T_DYN, D]`` output that nothing consumes,
which is how a scaffold ends up with a hole that only shows up at integration.

Two facts shape the implementation:

* No model in this repo uses a one-call collective for this.
  ``models/deepseek_v4_1_flash/attention_tp.py`` hand-rolls the reduction out of
  ``pld.DistributedTensor`` windows plus ``pld.system.wait`` / ``notify`` /
  ``remote_store`` under ``@pl.jit.inline(auto_scope=False)``. Start there.
* Nothing in ``models/`` has ever been exercised above ``# ci: devices=4``, and that
  marker appears on exactly two files. TP16 correctness therefore cannot be a CI
  case discovered by the daily sweep; it needs a dedicated job that borrows 16 dies.
  Bring the kernel up at TP2 and TP4 first — ``config.py`` accepts both — and treat
  the 16-rank run as a separate milestone.

The MoE all-to-all is **not** here: dispatch and combine move token payloads with
their own routing metadata and belong with the expert kernels.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.glm5_3_flash.config import D, EP_SIZE, T_DYN, TP_SIZE


def golden_tp_all_reduce(partials: torch.Tensor) -> torch.Tensor:
    """Sum one ``[TP_SIZE, T, D]`` stack of per-rank partials into ``[T, D]``.

    The golden takes the stack because a single-process reference cannot observe a
    collective; the kernel takes one rank's slice and the window.
    """
    if partials.shape[0] != TP_SIZE:
        raise ValueError(f"expected {TP_SIZE} partials, got {partials.shape[0]}")
    return partials.float().sum(dim=0).to(torch.bfloat16)


@pl.jit.inline(auto_scope=False)
def tp_all_reduce(
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    exchange: pld.DistributedTensor[[EP_SIZE, T_DYN, D], pl.FP32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    raise NotImplementedError("TP all-reduce kernel body is assigned independently")


__all__ = [
    "golden_tp_all_reduce",
    "tp_all_reduce",
]
