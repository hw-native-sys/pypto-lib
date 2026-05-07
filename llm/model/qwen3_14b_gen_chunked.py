# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib
import pathlib

_kernel = importlib.import_module(
    "examples.models.qwen3.14b.qwen3_14b_gen_chunked",
    package=str(pathlib.Path(__file__).resolve().parents[2]),
)
build_qwen3_14b_gen_chunked_program = _kernel.build_qwen3_14b_gen_chunked_program
stack_layer_weights_chunked = _kernel.stack_layer_weights_chunked
