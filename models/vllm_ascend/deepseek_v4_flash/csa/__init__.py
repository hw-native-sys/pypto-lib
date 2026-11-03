# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""vLLM Ascend CSA kernels and native-storage services.

Importing this namespace does not load Torch, PyPTO, or initialize an NPU.
"""


def csa_package(variant: str) -> str:
    """Resolve a supported CSA mode without importing its device dependencies."""
    packages = {
        "spmd_tnd": "spmd",
        "mpmd_tnd": "mpmd",
        "performance": "legacy",
        "perf": "legacy",
    }
    key = variant.strip().lower()
    if key not in packages:
        raise ValueError(f"Unsupported CSA variant: {variant!r}; expected {tuple(packages)}")
    return f"{__name__}.{packages[key]}"
