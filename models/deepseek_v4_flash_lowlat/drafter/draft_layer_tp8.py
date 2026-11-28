# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One B=1 Drafter layer with real local heads and dense TP8 experts."""

import pypto.language as pl
import pypto.language.distributed as pld

from .attention_local_tp8 import attention_local_tp8
from .config import FLASH
from .dense_ffn_core_tp8 import dense_ffn_from_gate
from .hc_pre import draft_hc_mix_attn_norm, hc_pre_gates
from .local_oproj_tp8 import local_o_group
from .moe import gate_tp8
from .oproj_pairs_tp8 import gather_and_round_pairs


D=4096
HC=4
T=32
LIVE=5
LIMIT=FLASH.swiglu_limit


TP8_POST_TILE = 4


@pl.jit.inline
def hc_post_tp8_live(
    update: pl.Tensor[[LIVE,D],pl.BF16],
    residual: pl.Tensor[[T,HC,D],pl.FP32],
    post: pl.Tensor[[T,HC],pl.FP32],
    comb: pl.Tensor[[T,HC*HC],pl.FP32],
    result: pl.Out[pl.Tensor[[T,HC,D],pl.FP32]],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Use the five live rows and retire the zero tail in the same AIV grid."""
    residual_flat=pl.reshape(residual,[T,HC*D])
    output=pl.reshape(result,[T,HC*D])
    with pl.spmd((T//TP8_POST_TILE)*HC,name_hint="draft_tp8_hc_post_live",deps=[ready],
                 allow_early_resolve=True):
        block=pl.tile.get_block_idx()
        t0=block//HC*TP8_POST_TILE
        channel=block%HC
        dst=channel*D
        zero=pl.full([1,D],dtype=pl.FP32,value=0.0)
        for token in pl.pipeline(t0,t0+TP8_POST_TILE,stage=2):
            if token<LIVE:
                post_w=pl.read(post,[token,channel])
                row=pl.cast(update[token:token+1,:],target_type=pl.FP32)
                mixed=pl.mul(row,post_w)
                for source in pl.pipeline(HC,stage=4):
                    coeff=pl.read(comb,[token,source*HC+channel])
                    channel_start=source*D
                    old=residual_flat[token:token+1,channel_start:channel_start+D]
                    mixed=pl.add(mixed,pl.mul(old,coeff))
                output[token:token+1,dst:dst+D]=mixed
            else:
                output[token:token+1,dst:dst+D]=zero
    return result


@pl.jit.inline(auto_scope=False)
def layer_tp8(
    x_hc: pl.Tensor[[T,HC,D],pl.FP32],
    query_token_ids: pl.Tensor[[T],pl.INT64],
    attention_hc_fn: pl.Tensor[[24,16384],pl.FP32],
    attention_hc_scale: pl.Tensor[[3],pl.FP32],
    attention_hc_base: pl.Tensor[[24],pl.FP32],
    attention_norm: pl.Tensor[[D],pl.BF16],
    wq_a: pl.Tensor[[D,1024],pl.BF16,pl.NZ],
    wq_b: pl.Tensor[[1024,4096],pl.INT8,pl.NZ],
    wq_b_scale: pl.Tensor[[4096],pl.FP32],
    wkv: pl.Tensor[[D,512],pl.BF16],
    gamma_cq: pl.Tensor[[1024],pl.BF16],
    gamma_ckv: pl.Tensor[[512],pl.BF16],
    cos: pl.Tensor[[LIVE,64],pl.BF16],
    sin: pl.Tensor[[LIVE,64],pl.BF16],
    positions: pl.Tensor[[LIVE],pl.INT32],
    query_slots: pl.Tensor[[LIVE],pl.INT64],
    cache: pl.Tensor[[512,32,1,512],pl.BF16],
    swa_indices: pl.Tensor[[1,192],pl.INT32],
    swa_lens: pl.Tensor[[1],pl.INT32],
    sink: pl.Tensor[[8],pl.FP32],
    wo_a: pl.Tensor[[1024,D],pl.BF16,pl.NZ],
    wo_b: pl.Tensor[[D,1024],pl.INT8,pl.NZ],
    wo_scale: pl.Tensor[[D],pl.FP32],
    ffn_hc_fn: pl.Tensor[[24,16384],pl.FP32],
    ffn_hc_scale: pl.Tensor[[3],pl.FP32],
    ffn_hc_base: pl.Tensor[[24],pl.FP32],
    ffn_norm: pl.Tensor[[D],pl.BF16],
    gate_w: pl.Tensor[[256,D],pl.FP32],
    gate_bias: pl.Tensor[[256],pl.FP32],
    hash_routes: pl.Tensor[[129280,6],pl.INT32],
    w1: pl.Tensor[[257,256,D],pl.INT8,pl.NZ],
    s1: pl.Tensor[[257,256],pl.FP32],
    w3: pl.Tensor[[257,256,D],pl.INT8,pl.NZ],
    s3: pl.Tensor[[257,256],pl.FP32],
    w2: pl.Tensor[[257,512,2048],pl.INT8,pl.NZ],
    s2: pl.Tensor[[257,512],pl.FP32],
    result: pl.Out[pl.Tensor[[T,HC,D],pl.FP32]],
    o_window: pld.DistributedTensor[[80,D],pl.FP32],
    o_flags: pld.DistributedTensor[[8,32],pl.INT32],
    max_window: pld.DistributedTensor[[640,8],pl.FP32],
    max_flags: pld.DistributedTensor[[8,32],pl.INT32],
    q8_window: pld.DistributedTensor[[640,256],pl.INT8],
    q8_flags: pld.DistributedTensor[[8,32],pl.INT32],
    output_window: pld.DistributedTensor[[80,512],pl.BF16],
    output_flags: pld.DistributedTensor[[8,32],pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    prior: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor[[T,HC,D],pl.FP32],pl.Scalar[pl.TASK_ID]]:
    """Consume real owner-zero metadata for all TP ranks and preserve five rows."""
    attn_post=pl.create_tensor([T,HC],dtype=pl.FP32)
    attn_comb=pl.create_tensor([T,HC*HC],dtype=pl.FP32)
    attn_pre=pl.create_tensor([T,8],dtype=pl.FP32)
    hc_pre_gates(x_hc,attention_hc_fn,attention_hc_scale,attention_hc_base,
                 attn_pre,attn_post,attn_comb,False)
    norm=pl.create_tensor([LIVE,D],dtype=pl.BF16)
    attn_flat: pl.Tensor[[T,HC*D],pl.FP32] = pl.reshape(x_hc,[T,HC*D])
    draft_hc_mix_attn_norm(attn_flat,attn_pre,attention_norm,norm)
    heads=pl.create_tensor([256,512],dtype=pl.BF16)
    attention_local_tp8(norm,wq_a,wq_b,wq_b_scale,wkv,gamma_cq,gamma_ckv,
                       cos,sin,cache,query_slots,swa_indices,swa_lens,sink,heads)
    real_heads=pl.slice(heads,[LIVE*8,512],[0,0])
    group=pl.reshape(real_heads,[LIVE,D])
    partial=pl.create_tensor([LIVE,D],dtype=pl.FP32)
    local_ready=local_o_group(group,wo_a,wo_b,partial,prior)
    attention=pl.create_tensor([LIVE,D],dtype=pl.BF16)
    attn_done=gather_and_round_pairs(partial,wo_scale,attention,o_window,o_flags,rank,epoch,local_ready)

    new_hc=pl.create_tensor([T,HC,D],dtype=pl.FP32)
    live=pl.const(5,pl.INT32)
    hc_post_tp8_live(attention,x_hc,attn_post,attn_comb,new_hc,attn_done)

    ffn_pre=pl.create_tensor([T,8],dtype=pl.FP32)
    ffn_post=pl.create_tensor([T,HC],dtype=pl.FP32)
    ffn_comb=pl.create_tensor([T,HC*HC],dtype=pl.FP32)
    hc_pre_gates(new_hc,ffn_hc_fn,ffn_hc_scale,ffn_hc_base,ffn_pre,ffn_post,ffn_comb,False)
    act=pl.create_tensor([T,D],dtype=pl.INT8)
    scale=pl.create_tensor([T,1],dtype=pl.FP32)
    route_ids=pl.create_tensor([T,6],dtype=pl.INT32)
    route_weight=pl.create_tensor([T,6],dtype=pl.FP32)
    unused_mixed=pl.create_tensor([T,D],dtype=pl.BF16)
    ffn_flat: pl.Tensor[[T,HC*D],pl.FP32] = pl.reshape(new_hc,[T,HC*D])
    gate_tp8(unused_mixed,ffn_norm,gate_w,gate_bias,layer_id,live,hash_routes,query_token_ids,
         act,scale,route_ids,route_weight,ffn_flat,ffn_pre,True)

    ffn=pl.create_tensor([LIVE,D],dtype=pl.BF16)
    expert_done=dense_ffn_from_gate(act,scale,route_ids,route_weight,w1,s1,w3,s3,w2,s2,ffn,
                      max_window,max_flags,q8_window,q8_flags,output_window,output_flags,rank,live,epoch,attn_done,LIMIT)
    hc_post_tp8_live(ffn,new_hc,ffn_post,ffn_comb,result,expert_done)
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="draft_tp8_layer_retire") as retired:
        _retired=pl.read(result,[0,0,0])
    return result,retired
