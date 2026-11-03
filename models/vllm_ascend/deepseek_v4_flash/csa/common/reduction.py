# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CSA/HCA 共用的编译期规约配置；固定规约用单 K 分片消除跨核累加。"""

import pypto.language as pl

from .options import atomic_add

ATOMIC_ADD = atomic_add()
if ATOMIC_ADD not in (0, 1):
    raise ValueError("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD 只接受 0/1")
STORE_ATOMIC = pl.AtomicType.Add if ATOMIC_ADD else pl.AtomicType.None_


def validate_reduction_mode():
    """初始化后不能让环境开关与已编译的策略不一致。"""
    if atomic_add() != ATOMIC_ADD:
        raise ValueError("PTO CSA 规约配置已在导入时固定，切换 atomic add 需要新进程")
