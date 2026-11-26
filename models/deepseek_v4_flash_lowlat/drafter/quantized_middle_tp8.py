# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Transfer quantized Drafter activations instead of large INT32 W2 partials."""

import pypto.language as pl
import pypto.language.distributed as pld


TP=8
ROWS=40
QCOL=256
INTER=TP*QCOL
STRIPE=512
TILE=128
DYNAMIC_EXPERTS=pl.dynamic("DRAFT_TP8_EXPERT_COUNT")
ROW_TILE=8
LANE_ROWS=TP*ROWS
CUBE_BLOCKS=24


@pl.jit.inline(auto_scope=False)
def gather_quantized_middle(
    quant: pl.Tensor[[ROWS,QCOL],pl.INT8],
    gathered: pl.Out[pl.Tensor[[ROWS,INTER],pl.INT8]],
    window: pld.DistributedTensor[[2*LANE_ROWS,QCOL],pl.INT8],
    signals: pld.DistributedTensor[[TP,32],pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    producer: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """All experts retain the one global scale used by the original activation."""
    lane=pl.cast((epoch % 2)*LANE_ROWS,pl.INDEX)
    with pl.spmd(TP,name_hint="draft_tp8_publish_int8_middle",deps=[producer]) as sent:
        peer=pl.tile.get_block_idx()
        pld.tensor.put(dst=window,peer=peer,src=quant,dst_offsets=[lane+rank*ROWS,0],
                       src_offsets=[0,0],shape=[ROWS,QCOL],chunk_rows=ROWS,chunk_cols=QCOL)
        pld.system.notify(target=signals,peer=peer,offsets=[rank,0],value=epoch,op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="draft_tp8_int8_middle_ready",deps=[sent]) as ready:
        for owner in pl.unroll(TP):
            pld.system.defer_wait(signal=signals,offsets=[owner,0],expected=epoch,cmp=pld.WaitCmp.Ge)
    with pl.spmd(TP*(ROWS//ROW_TILE),name_hint="draft_tp8_unpack_int8_middle",deps=[ready]) as done:
        block=pl.tile.get_block_idx()
        owner=block//(ROWS//ROW_TILE)
        row=block%(ROWS//ROW_TILE)*ROW_TILE
        data=pl.load(window,[lane+owner*ROWS+row,0],[ROW_TILE,QCOL])
        pl.store(data,[row,owner*QCOL],gathered)
    return done


@pl.jit.inline(auto_scope=False)
def project_w2_output_columns(
    gathered: pl.Tensor[[ROWS,INTER],pl.INT8],
    route_slot: pl.Tensor[[ROWS,8],pl.INT32],
    weights: pl.Tensor[[DYNAMIC_EXPERTS,STRIPE,INTER],pl.INT8,pl.NZ],
    outputs: pl.Out[pl.Tensor[[ROWS,STRIPE],pl.INT32]],
    producer: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """One output-column shard, full K. A negative route slot is padding."""
    with pl.spmd(CUBE_BLOCKS,name_hint="draft_tp8_w2_owned_columns",deps=[producer]) as done:
        pl.set_cache_policy(weights,pl.CachePolicy.BYPASS)
        block=pl.tile.get_block_idx()
        for item in pl.range(block,ROWS*(STRIPE//TILE),CUBE_BLOCKS):
            row=item//(STRIPE//TILE)
            col=item%(STRIPE//TILE)*TILE
            expert=pl.read(route_slot,[row,0])
            if expert>=0:
                result=pl.create_tensor([16,TILE],dtype=pl.INT32)
                for k0 in pl.pipeline(0,INTER,QCOL,stage=2):
                    part=pl.slice(gathered,[16,QCOL],[row,k0],valid_shape=[1,QCOL])
                    w=weights[expert,col:col+TILE,k0:k0+QCOL]
                    result=pl.matmul_acc(result,part,w,b_trans=True,init_cond=k0==0)
                outputs[row:row+1,col:col+TILE]=pl.set_validshape(result,1,TILE)
            else:
                outputs[row:row+1,col:col+TILE]=pl.full([1,TILE],dtype=pl.INT32,value=0)
    return done
