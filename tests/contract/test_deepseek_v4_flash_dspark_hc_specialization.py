# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for decode-attention HC specializations."""

import ast
from pathlib import Path


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_flash_dspark"


def _normalized_function(filename, function_name):
    tree = ast.parse((MODEL_DIR / filename).read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.name = "hc_specialization"
    return ast.dump(function, include_attributes=False)


def test_decode_attention_hc_specializations_match_shared_bodies():
    pairs = (
        ("hc_pre.py", "hc_pre", "hc_pre_decode_attention"),
        ("hc_post.py", "hc_post", "hc_post_decode_attention"),
    )
    for filename, shared_name, attention_name in pairs:
        shared = _normalized_function(filename, shared_name)
        attention = _normalized_function(filename, attention_name)
        assert attention == shared, f"{attention_name} must remain identical to {shared_name}"
