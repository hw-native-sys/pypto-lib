# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static smoke tests for model script command-line contracts used by CI."""

import ast
from pathlib import Path


def test_qwen3_decode_fwd_cli_accepts_ci_platforms_and_omits_disabled_out_window_flag():
    """CI platforms parse and default compile kwargs stay backward-compatible."""
    script = Path(__file__).resolve().parents[2] / "models" / "qwen3" / "14b" / "decode_fwd.py"
    tree = ast.parse(script.read_text())

    platform_choices = None
    num_layers_default = None
    out_window_help = None
    compile_cfg = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
            arg_names = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            if "--platform" in arg_names:
                for keyword in node.keywords:
                    if keyword.arg == "choices" and isinstance(keyword.value, ast.List):
                        platform_choices = [elt.value for elt in keyword.value.elts]
            if "--num-layers" in arg_names:
                for keyword in node.keywords:
                    if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                        num_layers_default = keyword.value.value
            if "--enable-out-window-externalization" in arg_names:
                for keyword in node.keywords:
                    if keyword.arg == "help" and isinstance(keyword.value, ast.Constant):
                        out_window_help = keyword.value.value
        if isinstance(func, ast.Name) and func.id == "run_jit":
            for keyword in node.keywords:
                if keyword.arg == "compile_cfg":
                    compile_cfg = keyword.value

    assert platform_choices == ["a2a3", "a2a3sim", "a5", "a5sim"]
    assert num_layers_default == 2
    assert out_window_help == "Enable out-window externalization compiler pass."
    assert isinstance(compile_cfg, ast.IfExp)
    assert isinstance(compile_cfg.orelse, ast.Dict)
    assert compile_cfg.orelse.keys == []
