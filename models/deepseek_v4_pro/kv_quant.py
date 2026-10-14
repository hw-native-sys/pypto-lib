# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Group-64 FP8 quantize/dequantize for the BF16 non-RoPE KV cache."""

def reference_kv_quant_fp8(x):
    """Independent torch expression of the official group-64 cache epilog."""
    import torch
    values = x.to(torch.bfloat16).float().reshape(*x.shape[:-1], -1, 64)
    maximum = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2((maximum / 448.0).double()))).float()
    payload = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (payload.float() * scale).to(torch.bfloat16).float().reshape(x.shape)
