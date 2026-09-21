# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared MXFP8 attention-weight fixtures in the device KN/scales ABI."""

import torch
from golden import TensorSpec


def mxfp8_weight_specs(name, k, n, *, groups=1, init_value=None, std=None):
    """Build one weight/scales pair; grouped scales pack each matrix separately."""
    from utils import cann_quant_mxfp8_weight_kn

    pair = None

    def materialize():
        nonlocal pair
        if pair is None:
            weight = (init_value() if init_value else
                      torch.randn(*(([groups] if groups > 1 else []) + [n, k]))
                      * (std if std is not None else k ** -0.5))
            data, scale = cann_quant_mxfp8_weight_kn(weight)
            if groups > 1:
                scale = scale.reshape(groups * (k // 32), n)
            pair = data, scale
        return pair

    shape = ([groups] if groups > 1 else []) + [k, n]
    scale_shape = [groups * (k // 32), n]
    return [
        TensorSpec(name, shape, torch.float8_e4m3fn,
                   init_value=lambda: materialize()[0].clone()),
        TensorSpec(name + "_scale", scale_shape, torch.float8_e8m0fnu,
                   init_value=lambda: materialize()[1].clone()),
    ]
