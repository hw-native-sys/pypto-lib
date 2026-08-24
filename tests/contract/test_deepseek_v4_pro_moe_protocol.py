# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for the DeepSeek-V4 MoE epoch protocol."""

import ast
from pathlib import Path


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_pro"


def _tree(name):
    return ast.parse((MODEL_DIR / name).read_text())


def _function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_all_moe_callers_match_the_typed_window_abi():
    moe_args = _function(_tree("moe.py"), "moe").args.args
    moe_arity = len(moe_args)
    moe_arg_names = [arg.arg for arg in moe_args]
    window_names = (
        "recv_meta", "recv_x", "recv_aux", "recv_route",
        "arrived", "data_arrived", "routed_y_buf", "combine_arrived", "consumed",
    )
    calls = []
    for path in MODEL_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "moe":
                calls.append((path.name, node.lineno, node))

    assert calls
    assert all(len(node.args) == moe_arity and not node.keywords for _, _, node in calls), [
        (name, line, len(node.args), len(node.keywords)) for name, line, node in calls
    ]
    for name, line, node in calls:
        for window_name in window_names:
            arg = node.args[moe_arg_names.index(window_name)]
            assert isinstance(arg, ast.Name) and arg.id == window_name, (
                name, line, window_name, ast.unparse(arg)
            )


def test_moe_readiness_uses_unique_padded_set_epoch_slots():
    source = (MODEL_DIR / "moe.py").read_text()
    tree = ast.parse(source)
    signal_pad = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SIGNAL_PAD"
            for target in node.targets
        )
    )

    assert isinstance(signal_pad.value, ast.Constant) and signal_pad.value.value == 128
    assert "NotifyOp.AtomicAdd" not in source
    assert source.count("NotifyOp.Set") == 4
    assert source.count("WaitCmp.Ge") == 4
    assert "offsets=[my_rank, loc_e, 0]" in source
    assert "offsets=[src, loc_e, 0]" in source
    assert "offsets=[my_rank, e, 0]" in source
    assert "offsets=[src, e, 0]" in source


def test_fixed_epoch_entrypoints_quiesce_before_local_reset():
    for name in ("prefill_layer.py", "decode_mtp.py", "prefill_mtp.py"):
        source = (MODEL_DIR / name).read_text()
        retire_start = source.index('name_hint="moe_signal_retire"')
        retire = source[retire_start:]

        assert "NotifyOp.AtomicAdd" not in retire
        assert "signal=consumed" in retire
        assert retire.index("signal=consumed") < retire.index("target=arrived")
        assert "target=consumed" in retire
        assert "target=data_arrived" in retire
        assert "target=combine_arrived" in retire
        assert "NotifyOp.Set" in retire


def test_decode_layer_cache_specs_match_static_inout_abi():
    tree = _tree("decode_layer.py")
    host = _function(tree, "l3_decode_layer")
    inout_names = {
        arg.arg for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.InOut[")
    }
    out_names = {
        arg.arg for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.Out[")
    }
    build_specs = _function(tree, "build_tensor_specs")
    cache_assignment = next(
        node for node in ast.walk(build_specs)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "mutable_cache_names" for target in node.targets)
    )

    assert inout_names == ast.literal_eval(cache_assignment.value)
    assert out_names == {"x_next"}
    assert "is_output=name in mutable_cache_names" in ast.unparse(build_specs)


def test_prefill_mtp_ranked_specs_preserve_inout_direction():
    tree = _tree("prefill_mtp.py")
    host = _function(tree, "l3_mtp_prefill_fwd")
    inout_names = {
        arg.arg for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.InOut[")
    }
    out_names = {
        arg.arg for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.Out[")
    }

    assert inout_names == {"kv_cache"}
    assert out_names == {"hidden_out", "pre_hc_hidden_out"}
    assert "is_output = spec.is_output" in ast.unparse(_function(tree, "_ranked"))

    main = _function(tree, "main")
    cache_comparators = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "mapped_pool_ratio_reldiff"
    ]
    assert len(cache_comparators) == 1
    comparator = cache_comparators[0]
    assert ast.literal_eval(comparator.args[0]) == "ori_slot_mapping"
    keywords = {keyword.arg: keyword.value for keyword in comparator.keywords}
    assert ast.unparse(keywords["mapping_shape"]) == "(N_RANKS, T)"
    assert ast.unparse(keywords["block_size"]) == "BLOCK_SIZE"
    assert ast.literal_eval(keywords["leading_rank_axis"]) is True
    assert ast.literal_eval(keywords["pool_name"]) == "kv_cache"
    assert ast.literal_eval(keywords["diff_thd"]) == 0.01
    assert ast.literal_eval(keywords["pct_thd"]) == 0.05
