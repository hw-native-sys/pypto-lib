# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""原生 TND 的 CSA 多 task 实现。

编排以 deepseek_v4_flash_csa 的多 task DAG 为骨架；真实 Q/L、混长、padding
和核内数学借鉴冻结的 spmd_tnd。完整 CSA 不合并为一个 SPMD。
当前设备验证与接入进度见 tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md。
"""
