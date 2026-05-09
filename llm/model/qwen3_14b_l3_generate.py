# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
from pathlib import Path


_KERNEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "qwen3"
    / "14b"
    / "qwen3_14b_l3_generate.py"
)
_SPEC = importlib.util.spec_from_file_location("_qwen3_14b_l3_generate_kernel", _KERNEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load Qwen3-14B l3_generate kernel from {_KERNEL_PATH}")

_KERNEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KERNEL)

build_qwen3_14b_l3_generate_program = _KERNEL.build_qwen3_14b_l3_generate_program
stack_layer_weights_full = _KERNEL.stack_layer_weights_full

__all__ = ["build_qwen3_14b_l3_generate_program", "stack_layer_weights_full"]
