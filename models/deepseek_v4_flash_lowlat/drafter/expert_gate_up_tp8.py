# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The dense W1/W3 channel shard that feeds the original global activation scale."""

import pypto.language as pl


ROWS=40
D=4096
INTER=256
INTER_TILE=128
K_TILE=512
M_TILE=16
CUBE_BLOCKS=24
EXPERTS=pl.dynamic("DRAFT_TP8_GATE_WEIGHT_BANK_COUNT")


@pl.jit.inline(auto_scope=False)
def project_expert_gate_up(
    activations: pl.Tensor[[ROWS,D],pl.INT8],
    slots: pl.Tensor[[ROWS,8],pl.INT32],
    w1: pl.Tensor[[EXPERTS,INTER,D],pl.INT8,pl.NZ],
    w3: pl.Tensor[[EXPERTS,INTER,D],pl.INT8,pl.NZ],
    gate: pl.Out[pl.Tensor[[ROWS,INTER],pl.INT32]],
    up: pl.Out[pl.Tensor[[ROWS,INTER],pl.INT32]],
    producer: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Retain every admitted route, including repeated expert IDs."""
    with pl.spmd(CUBE_BLOCKS,name_hint="draft_tp8_gate_up_channel_shards",deps=[producer]) as ready:
        block=pl.tile.get_block_idx()
        pl.set_cache_policy(w1,pl.CachePolicy.BYPASS)
        pl.set_cache_policy(w3,pl.CachePolicy.BYPASS)
        for work in pl.range(block,ROWS*(INTER//INTER_TILE),CUBE_BLOCKS):
            row=work//(INTER//INTER_TILE)
            col=work%(INTER//INTER_TILE)*INTER_TILE
            expert=pl.read(slots,[row,0])
            if expert>=0:
                gate_dot=pl.create_tensor([M_TILE,INTER_TILE],dtype=pl.INT32)
                up_dot=pl.create_tensor([M_TILE,INTER_TILE],dtype=pl.INT32)
                for k in pl.pipeline(0,D,K_TILE,stage=2):
                    x=pl.slice(activations,[M_TILE,K_TILE],[row,k],valid_shape=[1,K_TILE])
                    a=w1[expert,col:col+INTER_TILE,k:k+K_TILE]
                    b=w3[expert,col:col+INTER_TILE,k:k+K_TILE]
                    gate_dot=pl.matmul_acc(gate_dot,x,a,b_trans=True,init_cond=k==0)
                    up_dot=pl.matmul_acc(up_dot,x,b,b_trans=True,init_cond=k==0)
                gate[row:row+1,col:col+INTER_TILE]=pl.set_validshape(gate_dot,1,INTER_TILE)
                up[row:row+1,col:col+INTER_TILE]=pl.set_validshape(up_dot,1,INTER_TILE)
            else:
                zeros=pl.full([1,INTER_TILE],dtype=pl.INT32,value=0)
                gate[row:row+1,col:col+INTER_TILE]=zeros
                up[row:row+1,col:col+INTER_TILE]=zeros
    return ready


@pl.jit.inline(auto_scope=False)
def activate_expert_gate_up(
    gate: pl.Tensor[[ROWS,INTER],pl.INT32],
    up: pl.Tensor[[ROWS,INTER],pl.INT32],
    x_scale: pl.Tensor[[ROWS,8],pl.FP32],
    slots: pl.Tensor[[ROWS,8],pl.INT32],
    s1: pl.Tensor[[EXPERTS,INTER],pl.FP32],
    s3: pl.Tensor[[EXPERTS,INTER],pl.FP32],
    result: pl.Out[pl.Tensor[[ROWS,INTER],pl.FP32]],
    maxima: pl.Out[pl.Tensor[[ROWS,8],pl.FP32]],
    computed: pl.Scalar[pl.TASK_ID],
    swiglu_limit: pl.constexpr,
) -> pl.Scalar[pl.TASK_ID]:
    """Apply the unchanged Drafter scaling, clipping and SwiGLU order."""
    # Two adjacent maxima rows share a cache line and a single writer.
    with pl.spmd(ROWS // 2,name_hint="draft_tp8_swiglu_and_amax",deps=[computed]) as ready:
        row0 = pl.tile.get_block_idx() * 2
        payload = pl.tile.full([2,8],dtype=pl.FP32,value=0.0)
        max_input = pl.create_tile([8,INTER],dtype=pl.FP32)
        scratch = pl.create_tile([8,INTER],dtype=pl.FP32)
        for local_row in pl.range(2):
            row = row0 + local_row
            expert=pl.read(slots,[row,0])
            if expert>=0:
                scale=pl.read(x_scale,[row,0])
                g=pl.cast(pl.load(gate,[row,0],[1,INTER]),target_type=pl.FP32,mode="none")
                u=pl.cast(pl.load(up,[row,0],[1,INTER]),target_type=pl.FP32,mode="none")
                g=pl.mul(pl.mul(g,scale),pl.load(s1,[expert,0],[1,INTER]))
                u=pl.mul(pl.mul(u,scale),pl.load(s3,[expert,0],[1,INTER]))
                if swiglu_limit>0.0:
                    g=pl.minimum(g,swiglu_limit)
                    u=pl.maximum(pl.minimum(u,swiglu_limit),-swiglu_limit)
                sigmoid=pl.recip(pl.add(pl.exp(pl.neg(g)),1.0))
                activation = pl.mul(pl.mul(g,sigmoid),u)
                pl.store(activation,[row,0],result)
                max_input[0:1,:] = pl.abs(activation)
                maximum = pl.row_max(pl.set_validshape(max_input,1,INTER),scratch)
                pl.tile.write(payload,[local_row,0],pl.tile.read(maximum,[0,0]))
            else:
                pl.store(pl.tile.full([1,INTER],dtype=pl.FP32,value=0.0),[row,0],result)
        pl.store(payload,[row0,0],maxima)
    return ready
