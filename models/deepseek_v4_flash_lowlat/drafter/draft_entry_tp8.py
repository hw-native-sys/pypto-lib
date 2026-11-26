# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Three local-head and dense-expert Drafter layers on the native eight ranks."""

import pypto.language as pl
import pypto.language.distributed as pld

from .draft_layer_tp8 import layer_tp8,D,HC,T,LIVE
from .initial_context_tp8 import prepare_context_tp8
from .hc_pre import hc_head


@pl.jit.inline(auto_scope=False)
def draft_tp8_b1(
    history: pl.Tensor[[128,12288],pl.BF16],
    main_proj: pl.Tensor[[4096,12288],pl.BF16],
    main_norm: pl.Tensor[[D],pl.BF16],
    num_sampled: pl.Tensor[[1],pl.INT32],
    last_sampled: pl.Tensor[[1],pl.INT64],
    next_prefill_tokens: pl.Tensor[[1],pl.INT64],
    embedding: pl.Tensor[[129280,D],pl.BF16],
    anchor_pos: pl.Tensor[[1],pl.INT32],
    context_pos: pl.Tensor[[128],pl.INT32],
    context_cos: pl.Tensor[[128,64],pl.BF16],
    context_sin: pl.Tensor[[128,64],pl.BF16],
    context_slots: pl.Tensor[[3,512],pl.INT64],
    query_cos: pl.Tensor[[LIVE,64],pl.BF16],
    query_sin: pl.Tensor[[LIVE,64],pl.BF16],
    query_pos: pl.Tensor[[LIVE],pl.INT32],
    query_slots: pl.Tensor[[3,LIVE],pl.INT64],
    kv_caches: pl.InOut[pl.Tensor[[3,512,32,1,512],pl.BF16]],
    swa_indices: pl.Tensor[[3,1,192],pl.INT32],
    swa_lens: pl.Tensor[[3,1],pl.INT32],
    head_fn: pl.Tensor[[4,16384],pl.FP32],
    head_scale: pl.Tensor[[1],pl.FP32],
    head_base: pl.Tensor[[4],pl.FP32],
    head: pl.Out[pl.Tensor[[1,LIVE,D],pl.BF16]],
    attention_hc_fn: pl.Tensor[[3, 24, 16384], pl.FP32],
    attention_hc_scale: pl.Tensor[[3, 3], pl.FP32],
    attention_hc_base: pl.Tensor[[3, 24], pl.FP32],
    attention_norm: pl.Tensor[[3, D], pl.BF16],
    wq_a: pl.Tensor[[3, D, 1024], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[3, 1024, 4096], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[3, 4096], pl.FP32],
    wkv: pl.Tensor[[3, D, 512], pl.BF16],
    gamma_cq: pl.Tensor[[3, 1024], pl.BF16],
    gamma_ckv: pl.Tensor[[3, 512], pl.BF16],
    sink: pl.Tensor[[3, 8], pl.FP32],
    wo_a: pl.Tensor[[3, 1024, D], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[3, D, 1024], pl.INT8, pl.NZ],
    wo_scale: pl.Tensor[[3, D], pl.FP32],
    ffn_hc_fn: pl.Tensor[[3, 24, 16384], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3, 3], pl.FP32],
    ffn_hc_base: pl.Tensor[[3, 24], pl.FP32],
    ffn_norm: pl.Tensor[[3, D], pl.BF16],
    gate_w: pl.Tensor[[3, 256, D], pl.FP32],
    gate_bias: pl.Tensor[[3, 256], pl.FP32],
    hash_routes: pl.Tensor[[3, 129280, 6], pl.INT32],
    w1: pl.Tensor[[3, 257, 256, D], pl.INT8, pl.NZ],
    s1: pl.Tensor[[3, 257, 256], pl.FP32],
    w3: pl.Tensor[[3, 257, 256, D], pl.INT8, pl.NZ],
    s3: pl.Tensor[[3, 257, 256], pl.FP32],
    w2: pl.Tensor[[3, 257, 512, 2048], pl.INT8, pl.NZ],
    s2: pl.Tensor[[3, 257, 512], pl.FP32],
    o_window: pld.DistributedTensor[[80, D], pl.FP32],
    o_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    max_window: pld.DistributedTensor[[640, 8], pl.FP32],
    max_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    q8_window: pld.DistributedTensor[[640, 256], pl.INT8],
    q8_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    output_window: pld.DistributedTensor[[80, 512], pl.BF16],
    output_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    initial_hidden: pl.Out[pl.Tensor[[T,HC,D],pl.FP32]],
    intermediate_hidden: pl.Out[pl.Tensor[[3,T,HC,D],pl.FP32]],
    rank: pl.Scalar[pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor[[1,LIVE,D],pl.BF16],pl.Scalar[pl.TASK_ID]]:
    layer_in=initial_hidden
    ids=pl.create_tensor([T],dtype=pl.INT64)
    prepare_context_tp8(history,main_proj,main_norm,num_sampled,last_sampled,next_prefill_tokens,embedding,
        anchor_pos,context_pos,context_cos,context_sin,context_slots,wkv,gamma_ckv,kv_caches,layer_in,ids,prior_cycle_gate)
    prev=prior_cycle_gate
    for index in pl.unroll(3):
        layer_out=intermediate_hidden[index]
        layer_in,prev=layer_tp8(
            layer_in,
            ids,
            attention_hc_fn[index],
            attention_hc_scale[index],
            attention_hc_base[index],
            attention_norm[index],
            wq_a[index],
            wq_b[index],
            wq_b_scale[index],
            wkv[index],
            gamma_cq[index],
            gamma_ckv[index],
            query_cos,
            query_sin,
            query_pos,
            query_slots[index],
            kv_caches[index],
            swa_indices[index],
            swa_lens[index],
            sink[index],
            wo_a[index],
            wo_b[index],
            wo_scale[index],
            ffn_hc_fn[index],
            ffn_hc_scale[index],
            ffn_hc_base[index],
            ffn_norm[index],
            gate_w[index],
            gate_bias[index],
            hash_routes[index],
            w1[index],
            s1[index],
            w3[index],
            s3[index],
            w2[index],
            s2[index],
            layer_out,
            o_window,
            o_flags,
            max_window,
            max_flags,
            q8_window,
            q8_flags,
            output_window,
            output_flags,
            pl.cast(40+index,pl.INT32),
            rank,
            epoch_base+pl.cast(index+1,pl.INT32),
            prev,
        )
    padded_head=pl.create_tensor([T,D],dtype=pl.BF16)
    hc_head(layer_in,head_fn,head_scale,head_base,padded_head)
    head_flat=pl.reshape(head,[LIVE,D])
    with pl.spmd(LIVE,name_hint="draft_tp8_real_head_rows",deps=[prev]) as ready:
        row=pl.tile.get_block_idx()
        head_flat[row:row+1,:]=padded_head[row:row+1,:]
    return head,ready
