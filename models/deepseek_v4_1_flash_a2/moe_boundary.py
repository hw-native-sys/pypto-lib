# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native A2 token dispatch and dynamic quantization boundary for PyPTO MoE."""


def dispatch_full384(hidden, route_ids):
    """Dispatch actual product BF16 tokens using actual product top-six indices.

    The registered Torch-NPU custom operator owns expert permutation and INT8
    activation quantization. Routed matrix products, activation, route weighting
    and combination are implemented by the PyPTO consumers. No reference tensor
    enters this boundary. Its outputs preserve native full384 count-mode ABI.
    """
    import torch
    import torch_npu

    if hidden.device.type != "npu" or route_ids.device != hidden.device:
        raise ValueError("Actual product tokens and indices must be on the same NPU")
    if hidden.ndim != 2 or hidden.shape[1] != 5120 or hidden.dtype.is_floating_point is False:
        raise ValueError("Expected actual product hidden [T,5120]")
    if str(hidden.dtype) != "torch.bfloat16" or str(route_ids.dtype) != "torch.int32":
        raise ValueError("Expected BF16 tokens and INT32 indices")
    if tuple(route_ids.shape) != (hidden.shape[0], 6):
        raise ValueError("Full384 MoE requires six actual indices per token")
    if not hidden.is_contiguous() or not route_ids.is_contiguous():
        raise ValueError("Native token dispatch requires contiguous inputs")
    payload, expanded, counts, scales = torch_npu.npu_moe_init_routing_v2(
        hidden, route_ids, scale=None, active_num=hidden.shape[0] * 6,
        expert_num=384, expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[0, 384], quant_mode=1,
    )
    routes = hidden.shape[0] * 6
    if tuple(payload.shape) != (routes, 5120) or payload.dtype != torch.int8:
        raise RuntimeError("Native dispatch payload ABI differs")
    if tuple(expanded.shape) != (routes,) or expanded.dtype != torch.int32:
        raise RuntimeError("Native dispatch permutation ABI differs")
    if tuple(counts.shape) != (384,) or counts.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("Native dispatch expert-count ABI differs")
    if tuple(scales.shape) != (routes,) or scales.dtype != torch.float32:
        raise RuntimeError("Native dispatch scale ABI differs")
    return payload, expanded, counts.to(dtype=torch.int64), scales
