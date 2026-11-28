# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Drafter output projection transport preserving the original TP4 round points."""

import pypto.language as pl
import pypto.language.distributed as pld


GROUPS = 8
PAIR_COUNT = GROUPS // 2
LIVE_ROWS = 5
D = 4096
BAND = 1024
LANE_ROWS = GROUPS * LIVE_ROWS


@pl.jit.inline(auto_scope=False)
def gather_and_round_pairs(
    projected: pl.Tensor[[LIVE_ROWS,D],pl.FP32],
    common_weight_scale: pl.Tensor[[D],pl.FP32],
    combined: pl.Out[pl.Tensor[[LIVE_ROWS,D],pl.BF16]],
    window: pld.DistributedTensor[[2*LANE_ROWS,D],pl.FP32],
    signals: pld.DistributedTensor[[GROUPS,32],pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    producer: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Transfer unrounded output groups; round each original pair before folding.

    All eight inputs refer to the same five real B=1 draft tokens, in the
    original query order. The caller must keep the shared scale and previous
    lane's real retirement dependency. Epochs are positive and monotone here.
    """
    lane = pl.cast((epoch % 2) * LANE_ROWS,pl.INDEX)
    with pl.spmd(GROUPS,name_hint="draft_tp8_publish_o_group",deps=[producer]) as sent:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(dst=window,peer=peer,src=projected,
                       dst_offsets=[lane+rank*LIVE_ROWS,0],src_offsets=[0,0],
                       shape=[LIVE_ROWS,D],chunk_rows=1,chunk_cols=BAND,pipeline=True)
        pld.system.notify(target=signals,peer=peer,offsets=[rank,0],
                          value=epoch,op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="draft_tp8_o_groups_ready",deps=[sent]) as ready:
        for group in pl.unroll(GROUPS):
            pld.system.defer_wait(signal=signals,offsets=[group,0],expected=epoch,cmp=pld.WaitCmp.Ge)
    with pl.spmd(LIVE_ROWS*(D//BAND),name_hint="draft_tp8_reduce_original_o_pairs",deps=[ready]) as done:
        block = pl.tile.get_block_idx()
        token = block // (D//BAND)
        col = block % (D//BAND) * BAND
        weight_scale = pl.load(common_weight_scale,[col],[BAND])
        scale = pl.reshape(weight_scale,[1,BAND])
        a = pl.load(window,[lane+token,col],[1,BAND])
        b = pl.load(window,[lane+LIVE_ROWS+token,col],[1,BAND])
        zero = pl.tile.full([1,BAND],dtype=pl.FP32,value=0.0)
        p0 = pl.mul(pl.add(pl.add(zero,a),b),scale)
        accum = pl.cast(pl.cast(p0,target_type=pl.BF16,mode="rint"),target_type=pl.FP32,mode="none")
        for pair in pl.range(1,PAIR_COUNT):
            first = lane + 2*pair*LIVE_ROWS + token
            left = pl.load(window,[first,col],[1,BAND])
            right = pl.load(window,[first+LIVE_ROWS,col],[1,BAND])
            pair_fp32 = pl.mul(pl.add(pl.add(zero,left),right),scale)
            pair_bf16 = pl.cast(pair_fp32,target_type=pl.BF16,mode="rint")
            accum = pl.add(accum,pl.cast(pair_bf16,target_type=pl.FP32,mode="none"))
        pl.store(pl.cast(accum,target_type=pl.BF16,mode="rint"),[token,col],combined)
    return done
