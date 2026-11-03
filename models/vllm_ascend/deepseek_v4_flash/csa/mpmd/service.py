# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""原生 TND 多 task 服务入口，复用数值无关的 Native/图/缓存生命周期。

输入及回退合同与 TND SPMD 相同；权重按本入口签名准备，资源策略和实际
提交使用本包。真实 Q/L 仍由设备 task 消费，不新增 Host 逐请求计划。
"""

from ..common.weights import prepare_weights
from ..service import TNDServiceRuntime
from .binding import PreparedDynamicCSA
from .entry import decode_csa_tnd
from .policy import select_policy


class CSAServiceRuntime(TNDServiceRuntime):
    @staticmethod
    def _prepare_weights(attention, layer):
        return prepare_weights(attention, None, layer, root_function=decode_csa_tnd)

    @staticmethod
    def _select_policy(**descriptors):
        return select_policy(**descriptors)

    @staticmethod
    def _prepare_binding(native, **options):
        return PreparedDynamicCSA(native, **options)
