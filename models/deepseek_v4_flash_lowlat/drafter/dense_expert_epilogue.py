# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Use the Drafter's original per-expert rounding on each TP8 output stripe."""

import pypto.language as pl


B1_QUERY = 5
ROUTED = 6
ROWS = 40
SCALE_COLS = 8
COLS = 512
BAND = 128


@pl.jit.inline(auto_scope=False)
def round_expert_stripe(
    w2: pl.Tensor[[ROWS, COLS], pl.INT32],
    activation_scale: pl.Tensor[[ROWS, SCALE_COLS], pl.FP32],
    channel_scale: pl.Tensor[[ROWS, COLS], pl.FP32],
    route_weight: pl.Tensor[[ROWS, SCALE_COLS], pl.FP32],
    live: pl.Scalar[pl.INT32],
    output: pl.Out[pl.Tensor[[B1_QUERY, COLS], pl.BF16]],
    producer: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Round each expert before combining shared then the six original routes.

    Slots 0..29 are query-major, with stable route order. Slots 30..34
    contain shared results. Rows 35..39 are allocation padding.
    """
    with pl.spmd(B1_QUERY * (COLS // BAND), name_hint="draft_tp8_w2_round_combine",
                 deps=[producer]) as done:
        block = pl.tile.get_block_idx()
        token = block // (COLS // BAND)
        col = block % (COLS // BAND) * BAND
        if token < live:
            shared = B1_QUERY * ROUTED + token
            shared_scale = pl.read(activation_scale,[shared,0])
            shared_i32 = pl.load(w2,[shared,col],[1,BAND])
            shared_w2_scale = pl.load(channel_scale,[shared,col],[1,BAND])
            shared_acc = pl.cast(shared_i32, target_type=pl.FP32, mode="none")
            shared_out = pl.mul(pl.mul(shared_acc,shared_scale),shared_w2_scale)
            combined = pl.cast(pl.cast(shared_out,target_type=pl.BF16,mode="rint"),
                               target_type=pl.FP32,mode="none")
            for j in pl.range(ROUTED):
                route = token * ROUTED + j
                route_scale = pl.read(activation_scale,[route,0]) * pl.read(route_weight,[route,0])
                route_i32 = pl.load(w2,[route,col],[1,BAND])
                route_w2_scale = pl.load(channel_scale,[route,col],[1,BAND])
                route_acc = pl.cast(route_i32,target_type=pl.FP32,mode="none")
                route_out = pl.mul(pl.mul(route_acc,route_scale),route_w2_scale)
                combined = pl.add(combined,pl.cast(pl.cast(route_out,target_type=pl.BF16,mode="rint"),
                                                   target_type=pl.FP32,mode="none"))
            pl.store(pl.cast(combined,target_type=pl.BF16,mode="rint"),[token,col],output)
        else:
            pl.store(pl.tile.full([1,BAND],dtype=pl.BF16,value=0.0),[token,col],output)
    return done
