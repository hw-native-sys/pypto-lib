# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact collectives for splitting a Drafter expert's intermediate channels."""

import pypto.language as pl
import pypto.language.distributed as pld


TP = 8
ROW_PAD = 40
LOCAL_INTERMEDIATE = 256
FULL_HIDDEN = 4096
COLS = FULL_HIDDEN // TP
ROW_TILE = 8
D_TILE = 128
LANE_ROWS = TP * ROW_PAD
SCALE_COLS = 8
INT8_AMAX_EPS = 1e-4
INT8_SCALE_MAX = 127.0


@pl.jit.inline(auto_scope=False)
def globally_scaled_rows(
    middle: pl.Tensor[[ROW_PAD, LOCAL_INTERMEDIATE], pl.FP32],
    local: pl.Tensor[[ROW_PAD, SCALE_COLS], pl.FP32],
    quant_input: pl.Out[pl.Tensor[[ROW_PAD, LOCAL_INTERMEDIATE], pl.INT8]],
    scale_columns: pl.Out[pl.Tensor[[ROW_PAD, SCALE_COLS], pl.FP32]],
    maxima: pld.DistributedTensor[[2*LANE_ROWS, SCALE_COLS], pl.FP32],
    arrived: pld.DistributedTensor[[TP, 32], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    producer: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Use one shared activation amax per original expert/token route row."""
    lane=pl.cast(((epoch-1)%2)*LANE_ROWS,pl.INDEX)
    with pl.spmd(TP,name_hint="draft_tp8_publish_expert_amax",deps=[producer]) as pushed:
        peer=pl.tile.get_block_idx()
        pld.tensor.put(dst=maxima,peer=peer,src=local,dst_offsets=[lane+rank*ROW_PAD,0],
                       src_offsets=[0,0],shape=[ROW_PAD,SCALE_COLS])
        pld.system.notify(target=arrived,peer=peer,offsets=[rank,0],value=epoch,op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="draft_tp8_amax_ready",deps=[pushed]) as ready:
        for owner in pl.unroll(TP):
            pld.system.defer_wait(signal=arrived,offsets=[owner,0],expected=epoch,cmp=pld.WaitCmp.Ge)
    with pl.spmd(ROW_PAD//ROW_TILE,name_hint="draft_tp8_quant_global_scale",deps=[ready]) as done:
        row0=pl.tile.get_block_idx()*ROW_TILE
        full=pl.load(maxima,[lane+row0,0],[ROW_TILE,SCALE_COLS])
        for owner in pl.range(1,TP):
            other=pl.load(maxima,[lane+owner*ROW_PAD+row0,0],[ROW_TILE,SCALE_COLS])
            full=pl.maximum(full,other)
        row_amax=pl.tile.full([1,ROW_TILE],dtype=pl.FP32,value=0.0)
        for row in pl.range(ROW_TILE):
            pl.tile.write(row_amax,[0,row],pl.tile.read(full,[row,0]))
        floor=pl.tile.full([1,ROW_TILE],dtype=pl.FP32,value=INT8_AMAX_EPS)
        absolute_max=pl.maximum(row_amax,floor)
        scale_max=pl.tile.full([1,ROW_TILE],dtype=pl.FP32,value=INT8_SCALE_MAX)
        quant_scale=pl.div(scale_max,absolute_max)
        dequant=pl.reshape(pl.recip(quant_scale),[ROW_TILE,1])
        zeros=pl.tile.full([ROW_TILE,SCALE_COLS],dtype=pl.FP32,value=0.0)
        pl.store(pl.row_expand(zeros,dequant),[row0,0],scale_columns)
        scaled=pl.row_expand_mul(pl.load(middle,[row0,0],[ROW_TILE,LOCAL_INTERMEDIATE]),
                                 pl.reshape(quant_scale,[ROW_TILE,1]))
        q32=pl.cast(scaled,target_type=pl.INT32,mode="rint")
        q16=pl.cast(q32,target_type=pl.FP16,mode="round")
        q8=pl.cast(q16,target_type=pl.INT8,mode="trunc")
        pl.store(q8,[row0,0],quant_input)
    return done
