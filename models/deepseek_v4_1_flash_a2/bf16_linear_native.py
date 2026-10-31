# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generic BF16 NPU linear custom-op; no CPU implementation or FP32 widening."""

NAMESPACE='pypto_dsv41_bf16_linear_v1'
SCHEMA='linear(Tensor source, Tensor weight_nk) -> Tensor'

def register(torch):
    from torch.utils._python_dispatch import TorchDispatchMode
    trace=[]
    def tensors(value):
        if isinstance(value,torch.Tensor):return [value]
        if isinstance(value,(list,tuple)):return [x for v in value for x in tensors(v)]
        if isinstance(value,dict):return [x for v in value.values() for x in tensors(v)]
        return []
    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            values=tensors((args,kwargs or {}))
            if any(v.device.type!='npu' or v.dtype!=torch.bfloat16 for v in values):
                raise ValueError('BF16 linear dispatch reached a different device/dtype')
            result=func(*args,**(kwargs or {}))
            if any(v.device.type!='npu' or v.dtype!=torch.bfloat16 for v in tensors(result)):
                raise ValueError('BF16 linear dispatch returned a different device/dtype')
            trace.append(dict(op=str(func),schema=str(func._schema),
                inputs=[dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype),device=str(v.device)) for v in values]))
            return result
    def linear(source,weight_nk):
        if (source.device.type!='npu' or weight_nk.device.type!='npu' or
            source.dtype!=torch.bfloat16 or weight_nk.dtype!=torch.bfloat16 or
            source.ndim!=2 or weight_nk.ndim!=2 or source.shape[1]!=weight_nk.shape[1]):
            raise ValueError('Generic NPU BF16 M,K / N,K operands required')
        with Observe():
            return torch.ops.aten.linear.default(source,weight_nk,None)
    definition=torch.library.Library(NAMESPACE,'DEF')
    definition.define(SCHEMA)
    implementation=torch.library.Library(NAMESPACE,'IMPL','PrivateUse1')
    implementation.impl('linear',linear)
    return getattr(torch.ops,NAMESPACE).linear.default,(definition,implementation),trace
