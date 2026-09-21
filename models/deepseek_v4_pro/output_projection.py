# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared host fixtures and reference for native MXFP8 grouped output projections."""


def golden_mxfp8_output_projection(o, tensors):
    import torch
    from utils import decode_e8m0_codes, host_quant_mxfp8, matmul_mx_golden

    tokens, groups, group_in = o.shape
    wo_a = tensors["wo_a"]
    rank = wo_a.shape[-1]
    intermediate = torch.empty(tokens, groups, rank, dtype=torch.bfloat16)
    scale_rows = group_in // 32
    for group in range(groups):
        x, x_scale = host_quant_mxfp8(o[:, group].to(torch.bfloat16))
        scale = decode_e8m0_codes(
            tensors["wo_a_scale"][group * scale_rows:(group + 1) * scale_rows], side="b"
        )
        intermediate[:, group] = matmul_mx_golden(x, x_scale, wo_a[group], scale).to(torch.bfloat16)
    x, x_scale = host_quant_mxfp8(intermediate.reshape(tokens, groups * rank))
    weight_scale = decode_e8m0_codes(tensors["wo_b_scale"], side="b")
    return matmul_mx_golden(x, x_scale, tensors["wo_b"], weight_scale).to(torch.bfloat16)


def build_mxfp8_output_specs(groups, group_in, rank, hidden):
    import torch
    from golden import TensorSpec
    from utils import gen_mxfp8_weight_kn_device

    weights, scales = [], []
    for group in range(groups):
        weight, scale = gen_mxfp8_weight_kn_device(
            rank, group_in, group_in ** -0.5, seed=701 + group
        )
        weights.append(weight)
        scales.append(scale)
    wo_a = torch.stack(weights)
    wo_a_scale = torch.cat(scales, dim=0)
    wo_b, wo_b_scale = gen_mxfp8_weight_kn_device(
        hidden, groups * rank, (groups * rank) ** -0.5, seed=719
    )
    return [
        TensorSpec("wo_a", [groups, group_in, rank], torch.float8_e4m3fn, init_value=lambda: wo_a),
        TensorSpec("wo_a_scale", [groups * (group_in // 32), rank], torch.float8_e8m0fnu,
                   init_value=lambda: wo_a_scale),
        TensorSpec("wo_b", [groups * rank, hidden], torch.float8_e4m3fn, init_value=lambda: wo_b),
        TensorSpec("wo_b_scale", [(groups * rank) // 32, hidden], torch.float8_e8m0fnu,
                   init_value=lambda: wo_b_scale),
    ]
