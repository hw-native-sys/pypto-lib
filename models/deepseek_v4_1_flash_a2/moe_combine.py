# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Restore dispatch order and reduce FP32 weighted routes before the shared sum.

``expanded_row_idx`` maps each original token/top-k route to its dispatch row.
Each token owns one output row, avoiding cross-core writes to the same row.
"""

import os

import pypto.language as pl

T = int(os.environ.get("A2_MOE_COMBINE_T", "17"))
TOPK = 6
D = 5120


@pl.jit(auto_scope=False)
def moe_combine_test(
    dispatch_y: pl.Tensor[[T * TOPK, D], pl.FP32],
    expanded_row_idx: pl.Tensor[[T * TOPK], pl.INT32],
    shared: pl.Tensor[[T, D], pl.BF16],
    output: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    with pl.spmd(T, name_hint="moe_token_combine"):
        token = pl.tile.get_block_idx()
        acc = pl.full([1, D], dtype=pl.FP32, value=0.0)
        for rank in pl.range(TOPK):
            route = token * TOPK + rank
            dispatch_index = pl.read(expanded_row_idx, [route])
            dispatch_row = pl.cast(dispatch_index, pl.INDEX)
            acc = pl.add(acc, dispatch_y[dispatch_row : dispatch_row + 1, :])
        routed_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
        routed_fp32 = pl.cast(routed_bf16, target_type=pl.FP32)
        shared_fp32 = pl.cast(shared[token : token + 1, :], target_type=pl.FP32)
        total = pl.add(routed_fp32, shared_fp32)
        output[token : token + 1, :] = pl.cast(total, target_type=pl.BF16, mode="rint")
    return output
