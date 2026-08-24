# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for deterministic DeepSeek-V4 decode RoPE permutations."""

import ast
from pathlib import Path


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_pro"
DECODE_ROPE_MODULES = (
    "decode_compressor_ratio4.py",
    "decode_compressor_ratio128.py",
    "decode_indexer.py",
    "decode_indexer_compressor.py",
    "decode_sparse_attn.py",
    "decode_sparse_attn_hca.py",
    "decode_sparse_attn_swa.py",
    "qkv_proj_rope.py",
)


def _is_pl_gather(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "gather":
        return False
    owner = call.func.value
    return (isinstance(owner, ast.Name) and owner.id == "pl") or (
        isinstance(owner, ast.Attribute)
        and owner.attr == "tensor"
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "pl"
    )


def test_decode_rope_contract_recognizes_gather_spellings():
    for expression in ("pl.gather(x, index=i)", "pl.tensor.gather(x, index=i)"):
        call = ast.parse(expression, mode="eval").body
        assert isinstance(call, ast.Call)
        assert _is_pl_gather(call)


def test_decode_rope_permutations_use_mask_gather_scatter_only():
    for name in DECODE_ROPE_MODULES:
        source = (MODEL_DIR / name).read_text()
        tree = ast.parse(source)
        indexed_gathers = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _is_pl_gather(node)
            and any(keyword.arg == "index" for keyword in node.keywords)
        ]

        assert not indexed_gathers, (name, [node.lineno for node in indexed_gathers])
        assert "MaskPattern.P0101" in source
        assert "MaskPattern.P1010" in source
        assert "pl.tensor.scatter" in source
