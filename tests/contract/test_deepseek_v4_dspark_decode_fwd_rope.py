# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only guards for the two ordinary-token RoPE profiles in decode forward."""

import ast
from pathlib import Path
from types import SimpleNamespace


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_dspark"


def _functions(module):
    tree = ast.parse((MODEL_DIR / f"{module}.py").read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_decode_fwd_routes_rope_by_attention_family():
    forward = _functions("decode_fwd")["decode_fwd"]
    for family, expected_calls in (("swa", 2), ("csa", 2), ("hca", 1)):
        callees = _functions(f"decode_{family}")
        prefix = "" if family == "swa" else "compressed_"
        for tp1 in (False, True):
            name = f"decode_{family}" + ("_tp1" if tp1 else "")
            calls = [
                node for node in ast.walk(forward)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == name
            ]
            assert len(calls) == expected_calls, name
            for call in calls:
                arguments = dict(zip(
                    (arg.arg for arg in callees[name].args.args),
                    (ast.unparse(arg) for arg in call.args),
                ))
                for suffix in ("cos", "sin"):
                    assert arguments[f"freqs_{suffix}"] == (
                        f"{prefix}freqs_{suffix}_local" if tp1 else f"{prefix}freqs_{suffix}"
                    )
                    if not tp1:
                        assert arguments[f"freqs_{suffix}_local"] == f"{prefix}freqs_{suffix}_local"
                    if family != "swa":
                        assert arguments[f"cmp_freqs_{suffix}"] == f"{family}_cmp_freqs_{suffix}"


def test_decode_fwd_binds_and_forwards_both_rope_profiles():
    functions = _functions("decode_fwd")
    rope_names = [
        f"{prefix}freqs_{suffix}{local}"
        for prefix in ("", "compressed_")
        for suffix in ("cos", "sin")
        for local in ("", "_local")
    ]
    for function_name, axis in (("decode_fwd", 0), ("l3_decode_fwd", 1)):
        function = functions[function_name]
        parameters = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
        body = ast.unparse(function)
        for name in rope_names:
            dim = "T_DYN" if name.endswith("_local") else "KV_T_DYN"
            assert dim in parameters[name]
            assert f"{name}.bind_dynamic({axis}, {dim})" in body
    child_call = next(
        node for node in ast.walk(functions["l3_decode_fwd"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "decode_fwd"
    )
    forwarded = dict(zip(
        (arg.arg for arg in functions["decode_fwd"].args.args),
        (ast.unparse(arg) for arg in child_call.args),
    ))
    for name in rope_names:
        assert forwarded[name] == f"{name}[rank]"


def test_decode_fwd_fixture_uses_shared_positions_and_separate_profiles():
    functions = _functions("decode_fwd")
    builder = functions["build_tensor_specs"]
    shared_default = next(
        node for node in builder.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "attention_start_pos is None"
    )
    namespace = {
        "attention_start_pos": None, "TP_SIZE": 4, "active_batch": 3,
        "swa_decode_start_set": lambda batch: SimpleNamespace(tolist=lambda: list(range(batch))),
    }
    exec(compile(ast.Module(body=[shared_default], type_ignores=[]), "<fixture-positions>", "exec"), namespace)
    assert namespace["attention_start_pos"] == list(range(12))
    attention_specs = next(
        node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "attention_specs"
    )
    assert "start_pos=attention_start_pos" in ast.unparse(attention_specs)

    rope_names = ("freqs_cos_local", "freqs_sin_local", "freqs_cos", "freqs_sin")
    namespace = {
        "_SWA_METADATA_NAMES": rope_names,
        "swa_specs": {name: f"swa:{name}" for name in rope_names},
        "csa_specs": {name: f"compressed:{name}" for name in rope_names},
        "specs_by_name": {}, "_copy_spec": lambda name, source: source,
    }
    profile_loops = [
        node for node in builder.body if isinstance(node, ast.For)
        and (ast.unparse(node.iter) == "_SWA_METADATA_NAMES"
             or "public_name = f'compressed_{name}'" in ast.unparse(node))
    ]
    assert len(profile_loops) == 2
    exec(compile(ast.Module(body=profile_loops, type_ignores=[]), "<fixture-profiles>", "exec"), namespace)
    for name in rope_names:
        assert namespace["specs_by_name"][name] == f"swa:{name}"
        assert namespace["specs_by_name"][f"compressed_{name}"] == f"compressed:{name}"
