# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Project one of the eight Drafter O groups before the original pair rounding."""

import pypto.language as pl


T=5
M_TILE=16
D=4096
O_LORA=1024
A_N=128
K=256
B_N=256
Q_ROWS=8


@pl.jit.inline(auto_scope=False)
def local_o_group(
    packed_heads: pl.Tensor[[T,D],pl.BF16],
    wo_a: pl.Tensor[[O_LORA,D],pl.BF16,pl.NZ],
    wo_b: pl.Tensor[[D,O_LORA],pl.INT8,pl.NZ],
    output: pl.Out[pl.Tensor[[T,D],pl.FP32]],
    input_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Produce FP32 scaled INT32 products before the common channel multiplier."""
    activation=pl.create_tensor([M_TILE,O_LORA],dtype=pl.FP32)
    with pl.spmd(O_LORA//A_N,name_hint="draft_tp8_o_group_a",deps=[input_ready]) as a_done:
        first_col=pl.tile.get_block_idx()*A_N
        pl.set_cache_policy(wo_a,pl.CachePolicy.BYPASS)
        first_dot=pl.create_tensor([M_TILE,A_N],dtype=pl.FP32)
        for first_k0 in pl.pipeline(0,D,K,stage=2):
            first_x=pl.slice(packed_heads,[M_TILE,K],[0,first_k0],valid_shape=[T,K])
            first_w=wo_a[first_col:first_col+A_N,first_k0:first_k0+K]
            first_dot=pl.matmul_acc(first_dot,first_x,first_w,b_trans=True,init_cond=first_k0==0)
        activation[0:M_TILE,first_col:first_col+A_N]=first_dot
    q8=pl.create_tensor([M_TILE,O_LORA],dtype=pl.INT8)
    scale=pl.create_tensor([1,Q_ROWS],dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="draft_tp8_o_group_quant",deps=[a_done]) as q_done:
        quant_x=pl.load(activation,[0,0],[Q_ROWS,O_LORA],valid_shape=[T,O_LORA])
        masked=pl.fillpad(quant_x,pad_value=pl.PadValue.zero)
        maximum_tmp=pl.create_tile([Q_ROWS,O_LORA],dtype=pl.FP32)
        absmax=pl.reshape(pl.row_max(pl.abs(masked),maximum_tmp),[1,Q_ROWS])
        safe=pl.maximum(pl.tile.full([1,Q_ROWS],dtype=pl.FP32,value=1e-4),absmax)
        qscale=pl.div(pl.tile.full([1,Q_ROWS],dtype=pl.FP32,value=127.0),safe)
        pl.store(pl.recip(qscale),[0,0],scale)
        y=pl.row_expand_mul(masked,pl.reshape(qscale,[Q_ROWS,1]))
        y32=pl.cast(y,target_type=pl.INT32,mode="rint")
        y16=pl.cast(y32,target_type=pl.FP16,mode="round")
        y8=pl.cast(y16,target_type=pl.INT8,mode="trunc")
        pl.store(pl.set_validshape(y8,T,O_LORA),[0,0],q8)

    partial=pl.create_tensor([M_TILE,D],dtype=pl.INT32)
    with pl.spmd(D//B_N,name_hint="draft_tp8_o_group_b",deps=[q_done]) as b_done:
        second_col=pl.tile.get_block_idx()*B_N
        pl.set_cache_policy(wo_b,pl.CachePolicy.BYPASS)
        second_dot=pl.create_tensor([M_TILE,B_N],dtype=pl.INT32)
        for second_k0 in pl.pipeline(0,O_LORA,K,stage=2):
            second_x=pl.slice(q8,[M_TILE,K],[0,second_k0],valid_shape=[T,K])
            second_w=wo_b[second_col:second_col+B_N,second_k0:second_k0+K]
            second_dot=pl.matmul_acc(second_dot,second_x,second_w,b_trans=True,init_cond=second_k0==0)
        partial[0:M_TILE,second_col:second_col+B_N]=second_dot

    with pl.spmd(D//512,name_hint="draft_tp8_o_group_dequant",deps=[b_done]) as done:
        dequant_col=pl.tile.get_block_idx()*512
        act=pl.reshape(pl.load(scale,[0,0],[1,Q_ROWS]),[Q_ROWS,1])
        inner=pl.load(partial,[0,dequant_col],[Q_ROWS,512])
        result=pl.row_expand_mul(pl.cast(inner,target_type=pl.FP32,mode="none"),act)
        pl.store(pl.set_validshape(result,T,512),[0,dequant_col],output)
    return done
