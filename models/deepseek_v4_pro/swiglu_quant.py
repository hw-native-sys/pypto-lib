# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CANN SwiGLU group-32 MXFP8 quantization with upward-rounded scale."""

def reference_swiglu_quant(x):
    """Independent torch expression of CANN's SwiGLU MXFP8 output quantization."""
    import torch
    values = x.float().reshape(*x.shape[:-1], -1, 32)
    maximum = values.abs().amax(dim=-1).clamp_min(1e-4)
    # Evaluate the logarithm in double precision so a one-ULP excess above
    # a power of two cannot round back onto that boundary before ceil.
    scaled_maximum = maximum * (1.0 / 448.0)
    scale = torch.exp2(torch.ceil(torch.log2(scaled_maximum.double()))).float()
    payload = (values / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return payload.reshape(x.shape), scale
