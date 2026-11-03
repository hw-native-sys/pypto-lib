# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""性能版使用自身根签名与共用布局转换，不另维护 NZ 权重名单。"""

from .decode_csa import _decode_csa_tp1_layer
from ..common.weights import repack_weights
from ..common.nz_mode import root_weight_layouts, root_weight_shapes


def pack_args(tensors: dict, *, source_layouts: dict) -> dict:
    return repack_weights(tensors, source_layouts, root_weight_layouts(_decode_csa_tp1_layer),
                          root_weight_shapes(_decode_csa_tp1_layer))
