# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel options shared by standalone validation and vLLM Ascend workers.

These are the existing vLLM environment variables, with the same defaults.
Reading them here keeps kernel imports independent of the serving framework.
"""

import os


def weight_nz_mode() -> int:
    return int(os.getenv("VLLM_ASCEND_ENABLE_NZ", "1"))


def atomic_add() -> int:
    return int(os.getenv("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD", "0"))
