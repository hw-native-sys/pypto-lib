# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for retiring persistent MTP MoE signal credits."""

import ast
from pathlib import Path

import pytest


_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"


def _function_node(path, name):
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _call_name(statement):
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    return ast.unparse(statement.value.func)


def _retirement_scope(function):
    for statement in function.body:
        if not isinstance(statement, ast.With):
            continue
        for item in statement.items:
            call = item.context_expr
            if not isinstance(call, ast.Call) or ast.unparse(call.func) != "pl.at":
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            name_hint = keywords.get("name_hint")
            if isinstance(name_hint, ast.Constant) and name_hint.value == "moe_signal_retire":
                return statement
    raise AssertionError("missing moe_signal_retire scope")


@pytest.mark.parametrize(
    ("filename", "function_name", "anchor_output"),
    [
        ("decode_mtp.py", "mtp_decode_layer", "next_pre_hc_hidden"),
        ("prefill_mtp.py", "mtp_prefill_fwd", "pre_hc_hidden_out"),
    ],
)
def test_mtp_retires_one_epoch_before_consuming_moe_output(
    filename, function_name, anchor_output
):
    function = _function_node(_MODEL_DIR / filename, function_name)
    retire = _retirement_scope(function)

    statements = function.body
    moe_index = next(i for i, statement in enumerate(statements) if _call_name(statement) == "moe")
    retire_index = statements.index(retire)
    head_index = next(
        i for i, statement in enumerate(statements) if _call_name(statement) == "hc_head"
    )
    assert moe_index < retire_index < head_index

    assignments = {
        target.id: ast.unparse(statement.value)
        for statement in statements
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    assert assignments["neg_epochs"] == "pl.cast(0 - MTP_MOE_EPOCH, pl.INT32)"
    assert assignments["neg_data"] == (
        "pl.cast(0 - MTP_MOE_EPOCH * N_LOCAL, pl.INT32)"
    )
    assert assignments["neg_combine"] == (
        "pl.cast(0 - MTP_MOE_EPOCH * (N_LOCAL + 1), pl.INT32)"
    )

    anchor_reads = [
        node.value
        for node in retire.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "pl.read"
    ]
    assert len(anchor_reads) == 1
    assert ast.unparse(anchor_reads[0].args[0]) == anchor_output

    notifications = []
    for call in (
        node
        for node in ast.walk(retire)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pld.system.notify"
    ):
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        notifications.append(
            (keywords["target"], keywords["value"], keywords["op"])
        )

    assert sorted(notifications) == sorted(
        [
            ("combine_arrived", "neg_epochs", "pld.NotifyOp.AtomicAdd"),
            ("arrived", "neg_epochs", "pld.NotifyOp.AtomicAdd"),
            ("data_arrived", "neg_data", "pld.NotifyOp.AtomicAdd"),
            ("combine_arrived", "neg_combine", "pld.NotifyOp.AtomicAdd"),
        ]
    )
