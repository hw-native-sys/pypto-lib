# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Materialize admitted Drafter routes and collect final BF16 output stripes."""

import pypto.language as pl
import pypto.language.distributed as pld


TP=8
D=4096
ROWS=40
QUERIES=5
FANOUT=6
SHARED_EID=256
STRIPE=D//TP
GATE_ROWS=32
QUANT_COLS=8


@pl.jit.inline(auto_scope=False)
def pack_dense_route_rows(
    q8: pl.Tensor[[GATE_ROWS,D],pl.INT8],
    q_scale: pl.Tensor[[GATE_ROWS,1],pl.FP32],
    indices: pl.Tensor[[GATE_ROWS,FANOUT],pl.INT32],
    router_weights: pl.Tensor[[GATE_ROWS,FANOUT],pl.FP32],
    expert_output_scales: pl.Tensor[[SHARED_EID+1,STRIPE],pl.FP32],
    packed_q: pl.Out[pl.Tensor[[ROWS,D],pl.INT8]],
    scales: pl.Out[pl.Tensor[[ROWS,QUANT_COLS],pl.FP32]],
    route_weights: pl.Out[pl.Tensor[[ROWS,QUANT_COLS],pl.FP32]],
    slots: pl.Out[pl.Tensor[[ROWS,8],pl.INT32]],
    w2_scales: pl.Out[pl.Tensor[[ROWS,STRIPE],pl.FP32]],
    live: pl.Scalar[pl.INT32],
    gate_done: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """One task consumes the legacy stable route order and owns non-overlapping rows."""
    # Each instance owns a 64 B sector in each scalar metadata buffer.
    with pl.spmd(ROWS//2,name_hint="draft_tp8_pack_gate_rows",deps=[gate_done]) as retired:
        base=pl.tile.get_block_idx()*2
        for part in pl.unroll(2):
            row=base+part
            token=row//FANOUT
            k=row%FANOUT
            exp=pl.cast(-1,pl.INT32)
            x_scale=0.0
            route_weight=0.0
            if row>=QUERIES*FANOUT:
                token=row-QUERIES*FANOUT
                if token<live:
                    exp=pl.cast(SHARED_EID,pl.INT32)
                    x_scale=pl.read(q_scale,[token,0])
                    route_weight=1.0
            else:
                if token<live:
                    exp=pl.read(indices,[token,k])
                    x_scale=pl.read(q_scale,[token,0])
                    route_weight=pl.read(router_weights,[token,k])
            pl.write(slots,[row,0],exp)
            pl.write(scales,[row,0],x_scale)
            pl.write(route_weights,[row,0],route_weight)
            if exp>=0:
                for col in pl.range(0,D,1024):
                    values=pl.load(q8,[token,col],[1,1024])
                    pl.store(values,[row,col],packed_q)
                scale=pl.load(expert_output_scales,[exp,0],[1,STRIPE])
                pl.store(scale,[row,0],w2_scales)
            else:
                zeros=pl.tile.full([1,STRIPE],dtype=pl.FP32,value=0.0)
                pl.store(zeros,[row,0],w2_scales)
    return retired


@pl.jit.inline(auto_scope=False)
def replicate_drafter_output(
    stripe: pl.Tensor[[QUERIES,STRIPE],pl.BF16],
    result: pl.Out[pl.Tensor[[QUERIES,D],pl.BF16]],
    slots: pld.DistributedTensor[[2*TP*QUERIES,STRIPE],pl.BF16],
    signals: pld.DistributedTensor[[TP,32],pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Exchange only the already combined output, with no per-expert outputs."""
    lane=pl.cast((epoch % 2)*TP*QUERIES,pl.INDEX)
    with pl.spmd(TP,name_hint="draft_tp8_publish_combined_stripe",deps=[source_ready]) as sent:
        peer=pl.tile.get_block_idx()
        pld.tensor.put(dst=slots,peer=peer,src=stripe,dst_offsets=[lane+rank*QUERIES,0],
                       src_offsets=[0,0],shape=[QUERIES,STRIPE],chunk_rows=QUERIES,chunk_cols=STRIPE)
        pld.system.notify(target=signals,peer=peer,offsets=[rank,0],value=epoch,op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="draft_tp8_combined_stripes_ready",deps=[sent]) as ready:
        for owner in pl.unroll(TP):
            pld.system.defer_wait(signal=signals,offsets=[owner,0],expected=epoch,cmp=pld.WaitCmp.Ge)
    with pl.spmd(TP,name_hint="draft_tp8_materialize_combined_hidden",deps=[ready]) as done:
        owner=pl.tile.get_block_idx()
        payload=pl.load(slots,[lane+owner*QUERIES,0],[QUERIES,STRIPE])
        pl.store(payload,[0,owner*STRIPE],result)
    return done
