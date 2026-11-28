# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compose the B=1 eight-rank Drafter FFN after the existing admission gate."""

import pypto.language as pl
import pypto.language.distributed as pld

from .dense_expert_epilogue import round_expert_stripe
from .expert_gate_up_tp8 import activate_expert_gate_up,project_expert_gate_up
from .global_scale_tp8 import globally_scaled_rows
from .quantized_middle_tp8 import gather_quantized_middle,project_w2_output_columns
from .route_rows_tp8 import pack_dense_route_rows,replicate_drafter_output


@pl.jit.inline(auto_scope=False)
def dense_ffn_from_gate(
    q8: pl.Tensor[[32,4096],pl.INT8],
    q_scale: pl.Tensor[[32,1],pl.FP32],
    indices: pl.Tensor[[32,6],pl.INT32],
    router_weight: pl.Tensor[[32,6],pl.FP32],
    w1: pl.Tensor[[257,256,4096],pl.INT8,pl.NZ],
    s1: pl.Tensor[[257,256],pl.FP32],
    w3: pl.Tensor[[257,256,4096],pl.INT8,pl.NZ],
    s3: pl.Tensor[[257,256],pl.FP32],
    w2: pl.Tensor[[257,512,2048],pl.INT8,pl.NZ],
    s2: pl.Tensor[[257,512],pl.FP32],
    output: pl.Out[pl.Tensor[[5,4096],pl.BF16]],
    max_window: pld.DistributedTensor[[640,8],pl.FP32],
    max_flags: pld.DistributedTensor[[8,32],pl.INT32],
    q8_window: pld.DistributedTensor[[640,256],pl.INT8],
    q8_flags: pld.DistributedTensor[[8,32],pl.INT32],
    output_window: pld.DistributedTensor[[80,512],pl.BF16],
    output_flags: pld.DistributedTensor[[8,32],pl.INT32],
    rank: pl.Scalar[pl.INT32],
    live_tokens: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    swiglu_limit: pl.constexpr,
) -> pl.Scalar[pl.TASK_ID]:
    """Use the same six routed results and shared result for each real row."""
    packed=pl.create_tensor([40,4096],dtype=pl.INT8)
    sx=pl.create_tensor([40,8],dtype=pl.FP32)
    route=pl.create_tensor([40,8],dtype=pl.FP32)
    slots=pl.create_tensor([40,8],dtype=pl.INT32)
    channel=pl.create_tensor([40,512],dtype=pl.FP32)
    packed_ready=pack_dense_route_rows(q8,q_scale,indices,router_weight,s2,
                                     packed,sx,route,slots,channel,live_tokens,ready)

    dots_gate=pl.create_tensor([40,256],dtype=pl.INT32)
    dots_up=pl.create_tensor([40,256],dtype=pl.INT32)
    activation=pl.create_tensor([40,256],dtype=pl.FP32)
    dots=project_expert_gate_up(packed,slots,w1,w3,dots_gate,dots_up,packed_ready)
    max_local=pl.create_tensor([40,8],dtype=pl.FP32)
    activated=activate_expert_gate_up(dots_gate,dots_up,sx,slots,s1,s3,activation,max_local,dots,swiglu_limit)

    q_local=pl.create_tensor([40,256],dtype=pl.INT8)
    dq=pl.create_tensor([40,8],dtype=pl.FP32)
    quantized=globally_scaled_rows(activation,max_local,q_local,dq,max_window,max_flags,rank,epoch,activated)
    all_channels=pl.create_tensor([40,2048],dtype=pl.INT8)
    gathered=gather_quantized_middle(q_local,all_channels,q8_window,q8_flags,rank,epoch,quantized)

    w2_partial=pl.create_tensor([40,512],dtype=pl.INT32)
    computed=project_w2_output_columns(all_channels,slots,w2,w2_partial,gathered)
    out_local=pl.create_tensor([5,512],dtype=pl.BF16)
    rounded=round_expert_stripe(w2_partial,dq,channel,route,live_tokens,out_local,computed)
    return replicate_drafter_output(out_local,output,output_window,output_flags,rank,epoch,rounded)
