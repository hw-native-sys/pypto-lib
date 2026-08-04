# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for the DeepSeek-V4 MoE pre-normalization fusion.

These checks intentionally parse source instead of importing the model.  Importing
the model freezes the selected EP configuration and requires the PyPTO device
environment, neither of which is needed to protect the fusion ABI and barriers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-flash"
_BRIDGE = _MODEL_DIR / "fused_pre_norm_cce.py"
_MOE = _MODEL_DIR / "moe.py"
_HC_PRE = _MODEL_DIR / "hc_pre.py"
_GATE = _MODEL_DIR / "gate.py"
_KERNEL_DIR = _MODEL_DIR / "kernels" / "fused_pre_norm_cce"
_FUSED_BODY = _KERNEL_DIR / "kernel" / "fused_body.hpp"
_PRODUCTION_ENTRY = _KERNEL_DIR / "entry.cpp"
_DEBUG_ENTRY = _KERNEL_DIR / "debug" / "entry.cpp"
_BASELINE_ENTRY = _KERNEL_DIR / "baseline" / "entry.cpp"
_BASELINE_DEBUG_ENTRY = _KERNEL_DIR / "baseline_debug" / "entry.cpp"

_TENSOR_ARGS = (
    "x_mixed",
    "x_flat",
    "inv_rms",
    "mixes_raw",
    "hc_base",
    "norm_w",
    "pre_val_store",
    "post",
    "xg_buf",
    "ffn_inv_rms_buf",
    "xn_scale_buf",
    "x_norm_scale",
    "sync_workspace",
)
_SCALAR_ARGS = ("scale0", "scale1", "num_tokens")
_PHASE_OUTPUTS = (
    "x_mixed",
    "pre_val_store",
    "post",
    "xg_buf",
    "ffn_inv_rms_buf",
    "xn_scale_buf",
    "x_norm_scale",
)
_DUMP_INPUTS = ("x_flat", "inv_rms", "mixes_raw", "hc_base", "norm_w")
_SYNC_WORKSPACE = "sync_workspace"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _annotation_kind(annotation: ast.AST | None) -> str:
    if isinstance(annotation, ast.Subscript):
        return _qualified_name(annotation.value).rsplit(".", 1)[-1]
    return _qualified_name(annotation).rsplit(".", 1)[-1]


def _extern_decorator(function: ast.FunctionDef) -> ast.Call:
    return next(
        decorator
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        and _qualified_name(decorator.func) == "pl.jit.extern"
    )


def _calls(function: ast.FunctionDef, qualified_name: str) -> list[ast.Call]:
    return sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _qualified_name(node.func) == qualified_name
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )


def _named_spmd_with(function: ast.FunctionDef, name_hint: str) -> list[ast.With]:
    result = []
    for node in ast.walk(function):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        context = node.items[0].context_expr
        if (
            isinstance(context, ast.Call)
            and _qualified_name(context.func) == "pl.spmd"
            and ast.literal_eval(_keyword(context, "name_hint")) == name_hint
        ):
            result.append(node)
    return result


def _named_spmd_calls(
    function: ast.FunctionDef,
    name_hint: str,
) -> list[ast.Call]:
    return [
        call
        for call in _calls(function, "pl.spmd")
        if ast.literal_eval(_keyword(call, "name_hint")) == name_hint
    ]


def _assigned_constant(tree: ast.Module, name: str) -> object:
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == name
        )
    )
    return ast.literal_eval(assignment.value)


def _assigned_value(tree: ast.Module, name: str) -> ast.AST:
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == name
        )
    )
    return assignment.value


def _local_assignment(function: ast.FunctionDef, name: str) -> ast.AST:
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == name
        )
    ]
    assert len(assignments) == 1, (
        f"{function.name} must create exactly one local {name}, "
        f"got {len(assignments)} assignments"
    )
    return assignments[0].value


def _return_names(function: ast.FunctionDef) -> tuple[str, ...]:
    returns = [node for node in function.body if isinstance(node, ast.Return)]
    assert len(returns) == 1
    value = returns[0].value
    assert isinstance(value, ast.Tuple)
    assert all(isinstance(element, ast.Name) for element in value.elts)
    return tuple(element.id for element in value.elts)


def _strip_cpp_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def _enum_symbols(source: str, enum_name: str) -> list[str]:
    match = re.search(
        rf"enum(?:\s+class)?\s+{re.escape(enum_name)}[^{{]*\{{(.*?)\}};",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, f"missing C++ enum {enum_name}"
    return re.findall(r"\b(k[A-Za-z0-9_]+)\s*(?:=[^,\n]+)?\s*,", match.group(1))


def _dump_calls(function: ast.FunctionDef) -> list[tuple[int, str]]:
    result = []
    for call in _calls(function, "pl.dump_tag"):
        if call.args and isinstance(call.args[0], ast.Name):
            result.append((call.lineno, call.args[0].id))
    return result


def _argument_call(tree: ast.Module, option: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _qualified_name(node.func).endswith(".add_argument")
        and any(
            isinstance(arg, ast.Constant) and arg.value == option
            for arg in node.args
        )
    )


def _load_standalone_function(path: Path, name: str):
    """Load one source function without importing its heavyweight model module."""
    function = _function(_parse(path), name)
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[]),
    )
    namespace = {"Path": Path, "re": re}
    exec(compile(module, filename=str(path), mode="exec"), namespace)
    return namespace[name]


def test_fused_extern_python_and_cpp_abis_stay_in_lockstep() -> None:
    tree = _parse(_BRIDGE)
    expected_kinds = (
        "Out",
        "Tensor",
        "Tensor",
        "Tensor",
        "Tensor",
        "Tensor",
        "Out",
        "Out",
        "Out",
        "Out",
        "Out",
        "Out",
        "InOut",
        "Scalar",
        "Scalar",
        "Scalar",
    )

    for function_name, source_name, trailing_args in (
        ("fused_pre_norm_cce", "_ENTRY", ()),
        ("fused_pre_norm_debug_cce", "_DEBUG_ENTRY", ("stop_after",)),
        ("fused_pre_norm_baseline_cce", "_BASELINE_ENTRY", ()),
        (
            "fused_pre_norm_baseline_debug_cce",
            "_BASELINE_DEBUG_ENTRY",
            ("stop_after",),
        ),
    ):
        function = _function(tree, function_name)
        arguments = function.args.args
        assert tuple(argument.arg for argument in arguments) == (
            *_TENSOR_ARGS,
            *_SCALAR_ARGS,
            *trailing_args,
        )
        assert tuple(_annotation_kind(argument.annotation) for argument in arguments) == (
            *expected_kinds,
            *(("Scalar",) if trailing_args else ()),
        )

        output_like = [
            argument.arg
            for argument in arguments
            if _annotation_kind(argument.annotation) in {"Out", "InOut"}
        ]
        assert output_like == [*_PHASE_OUTPUTS, _SYNC_WORKSPACE]
        assert output_like[0] == "x_mixed", (
            "the first extern result must bind to the first pl.Out argument"
        )
        assert _annotation_kind(arguments[0].annotation) == "Out"
        assert _annotation_kind(
            next(
                argument.annotation
                for argument in arguments
                if argument.arg == _SYNC_WORKSPACE
            )
        ) == "InOut"
        workspace_annotation = next(
            argument.annotation
            for argument in arguments
            if argument.arg == _SYNC_WORKSPACE
        )
        assert ast.unparse(workspace_annotation) == (
            "pl.InOut[pl.Tensor[[FUSED_SOFT_SYNC_WORDS], pl.INT32]]"
        )
        assert _return_names(function) == _PHASE_OUTPUTS, (
            "the mutable synchronization workspace is not a public result"
        )

        decorator = _extern_decorator(function)
        assert ast.literal_eval(_keyword(decorator, "core_type")) == "aiv"
        assert ast.unparse(_keyword(decorator, "source")) == source_name

    # Direct tests poison-initialize every result buffer, so their public ABI
    # must preserve those allocations across the extern call.
    for function_name in (
        "fused_pre_norm_test",
        "fused_pre_norm_debug_test",
        "fused_pre_norm_baseline_test",
        "fused_pre_norm_baseline_debug_test",
    ):
        function = _function(tree, function_name)
        annotations = {
            argument.arg: _annotation_kind(argument.annotation)
            for argument in function.args.args
        }
        assert {name: annotations[name] for name in _PHASE_OUTPUTS} == {
            name: "InOut" for name in _PHASE_OUTPUTS
        }

    body = _read(_FUSED_BODY)
    assert _enum_symbols(body, "TensorArg") == [
        "kXMixed",
        "kXFlat",
        "kInvRms",
        "kMixesRaw",
        "kHcBase",
        "kNormWeight",
        "kPreValue",
        "kPost",
        "kXg",
        "kFfnInvRms",
        "kXnScale",
        "kXNormScale",
        "kSyncWorkspace",
        "kTensorArgCount",
    ]
    assert re.search(r"\bkTensorArgCount\s*=\s*13\s*,", body)
    assert _enum_symbols(body, "ScalarArg") == [
        "kScale0",
        "kScale1",
        "kNumTokens",
        "kProductionArgCount",
        "kDebugStopAfter",
        "kDebugArgCount",
    ]
    assert re.search(r"\bkScale0\s*=\s*kTensorArgCount\s*,", body)
    assert re.search(r"\bkDebugStopAfter\s*=\s*kProductionArgCount\s*,", body)
    assert re.search(
        r"int32_t\s*\*\s*sync_workspace\s*="
        r"\s*tensor_data\s*<\s*int32_t\s*>\s*"
        r"\(\s*args\s*,\s*kSyncWorkspace\s*\)\s*;",
        body,
    )


def test_every_fused_launch_is_one_synchronously_started_8_aiv_wave() -> None:
    tree = _parse(_BRIDGE)
    assert _assigned_constant(tree, "FUSED_AIV_CORES") == 8
    assert _assigned_constant(tree, "SOFT_SYNC_COUNTER_INT32") == 16
    assert _assigned_constant(tree, "FUSED_SOFT_SYNC_COUNTERS") == 2
    assert ast.unparse(_assigned_value(tree, "FUSED_SOFT_SYNC_WORDS")) == (
        "FUSED_SOFT_SYNC_COUNTERS * SOFT_SYNC_COUNTER_INT32"
    )
    exports = _assigned_constant(tree, "__all__")
    assert {
        "FUSED_AIV_CORES",
        "SOFT_SYNC_COUNTER_INT32",
        "FUSED_SOFT_SYNC_COUNTERS",
        "FUSED_SOFT_SYNC_WORDS",
    } <= set(exports)
    assert "SOFT_SYNC_SLOT_INT32" not in exports

    for function_name, extern_name in (
        ("fused_pre_norm_test", "fused_pre_norm_cce"),
        ("fused_pre_norm_debug_test", "fused_pre_norm_debug_cce"),
        ("fused_pre_norm_baseline_test", "fused_pre_norm_baseline_cce"),
        (
            "fused_pre_norm_baseline_debug_test",
            "fused_pre_norm_baseline_debug_cce",
        ),
    ):
        launches = _calls(_function(tree, function_name), "pl.spmd_submit")
        assert len(launches) == 1
        launch = launches[0]
        assert ast.unparse(launch.args[0]) == f"self.{extern_name}"
        assert ast.unparse(_keyword(launch, "core_num")) == "FUSED_AIV_CORES"
        assert ast.literal_eval(_keyword(launch, "sync_start")) is True

    model_launch = _calls(_function(_parse(_MOE), "moe"), "pl.spmd_submit")
    assert len(model_launch) == 1
    assert ast.unparse(model_launch[0].args[0]) == "self.fused_pre_norm_cce"
    assert ast.unparse(_keyword(model_launch[0], "core_num")) == (
        "FUSED_AIV_CORES"
    )
    assert ast.literal_eval(_keyword(model_launch[0], "sync_start")) is True
    assert ast.unparse(_keyword(model_launch[0], "deps")) == (
        "[hc_pre_rms_tid, hc_pre_linear_tid]"
    )
    assert ast.literal_eval(
        _keyword(model_launch[0], "allow_early_resolve"),
    ) is True

    body = _read(_FUSED_BODY)
    assert re.search(r"\bkAivLanes\s*=\s*8\s*;", body)

    audit = _function(_parse(_MOE), "audit_fused_pre_norm_codegen")
    audit_source = ast.get_source_segment(_read(_MOE), audit)
    assert audit_source is not None
    audit_patterns = {
        node.value
        for node in ast.walk(audit)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "launch_spec.set_block_num" in audit_patterns
    assert "launch_spec.set_require_sync_start" in audit_patterns
    assert "set_allow_early_resolve" in audit_patterns
    nested_functions = {
        node.name
        for node in ast.walk(audit)
        if isinstance(node, ast.FunctionDef)
    }
    assert {
        "task_params",
        "submitted_task_id",
        "require_task_call",
    } <= nested_functions


def test_codegen_audit_resolves_generated_ids_and_rejects_contract_drift(
    tmp_path: Path,
) -> None:
    audit = _load_standalone_function(_MOE, "audit_fused_pre_norm_codegen")
    orchestration_dir = (
        tmp_path / "next_levels" / "moe_test" / "orchestration"
    )
    orchestration_dir.mkdir(parents=True)
    orchestration = orchestration_dir / "moe_test.cpp"
    valid_source = r"""
uint32_t hc_inv_rms_inline51_ci_shapes[2] = {16, 1};
TensorCreateInfo hc_inv_rms_inline51_ci(hc_inv_rms_inline51_ci_shapes, 2, DataType::FLOAT32, /*manual_dep=*/true);
uint32_t mixes_raw_inline52_ci_shapes[2] = {16, 32};
TensorCreateInfo mixes_raw_inline52_ci(mixes_raw_inline52_ci_shapes, 2, DataType::FLOAT32, /*manual_dep=*/true);
uint32_t other_ci_shapes[1] = {1};
TensorCreateInfo other_ci(other_ci_shapes, 1, DataType::FLOAT32);
uint32_t sync_workspace_inline987_ci_shapes[1] = {32};
TensorCreateInfo sync_workspace_inline987_ci(
    sync_workspace_inline987_ci_shapes, 1, DataType::INT32);
sync_workspace_inline987_ci.set_initial_value(0);
TaskOutputTensors opaque_allocation =
    alloc_tensors(other_ci, sync_workspace_inline987_ci);
const Tensor& sync_workspace_inline987 = opaque_allocation.get_ref(1);

// Spmd arbitrary_rms_group: hc_pre_rms
L0TaskArgs rms_params_alpha;
rms_params_alpha.set_allow_early_resolve(true);
TaskOutputTensors opaque_rms_output =
    rt_submit_aiv_task(73, rms_params_alpha);
PTO2TaskId opaque_rms_runtime_id = opaque_rms_output.task_id();

// Spmd arbitrary_linear_group: hc_pre_linear
L0TaskArgs linear_params_beta;
linear_params_beta.set_allow_early_resolve(true);
TaskOutputTensors opaque_linear_output =
    rt_submit_aic_task(911, linear_params_beta);
PTO2TaskId opaque_linear_runtime_id = opaque_linear_output.task_id();

// Task 4096: fused_pre_norm_cce
L0TaskArgs fused_params_gamma;
fused_params_gamma.add_output(x_mixed_inline1);
fused_params_gamma.add_input(x_flat_inline2);
fused_params_gamma.add_input(hc_inv_rms_inline51);
fused_params_gamma.add_input(mixes_raw_inline52);
fused_params_gamma.add_input(ext_hc_ffn_base);
fused_params_gamma.add_input(ext_norm_w);
fused_params_gamma.add_output(pre_val_store_inline3);
fused_params_gamma.add_output(post_ffn_inline4);
fused_params_gamma.add_output(xg_buf_inline5);
fused_params_gamma.add_output(gate_inv_rms_buf_inline6);
fused_params_gamma.add_output(xn_scale_buf_inline7);
fused_params_gamma.add_output(x_norm_scale_inline8);
fused_params_gamma.add_inout(sync_workspace_inline987);
fused_params_gamma.dump(x_mixed_inline1, sync_workspace_inline987);
PTO2TaskId opaque_fused_deps[2];
uint32_t opaque_fused_dep_count = 0;
opaque_fused_deps[opaque_fused_dep_count++] = opaque_rms_runtime_id;
opaque_fused_deps[opaque_fused_dep_count++] = opaque_linear_runtime_id;
fused_params_gamma.set_dependencies(
    opaque_fused_deps, opaque_fused_dep_count);
fused_params_gamma.launch_spec.set_block_num(8);
fused_params_gamma.launch_spec.set_require_sync_start(true);
fused_params_gamma.set_allow_early_resolve(true);
TaskOutputTensors opaque_fused_output =
    rt_submit_aiv_task(8128, fused_params_gamma);

// Spmd arbitrary_quant_group: x_norm_quant
L0TaskArgs quant_params;
quant_params.add_input(xn_scale_buf_inline7);
quant_params.add_output(x_norm_i8_inline9);
quant_params.add_input(xg_buf_inline5);

// Group gate: MixedKernels
L0TaskArgs gate_params;
gate_params.add_input(ext_gate_bias);
gate_params.add_input(xg_buf_inline5);
gate_params.add_input(ext_gate_w);
gate_params.add_input(gate_inv_rms_buf_inline6);

// Spmd arbitrary_shared_group: sh_gate_up_act_q
L0TaskArgs shared_params;
shared_params.add_input(x_norm_scale_inline8);

// Spmd arbitrary_push_group: dispatch_push
L0TaskArgs push_params;
push_params.add_input(indices_inline10);
push_params.add_output(ext_recv_x);
push_params.add_input(x_norm_i8_inline9);
push_params.add_input(x_norm_scale_inline8);
push_params.add_input(weights_inline11);

// Spmd arbitrary_gather_group: dispatch_gather
L0TaskArgs gather_params;
gather_params.add_output(recv_x_out_inline12);

// Spmd arbitrary_comb_group: comb_sinkhorn
L0TaskArgs comb_params;
comb_params.add_input(hc_inv_rms_inline51);
comb_params.add_input(mixes_raw_inline52);
comb_params.add_input(hc_base_2d_inline13);
comb_params.add_inout(comb_ffn_inline14);
PTO2TaskId comb_params_deps[1];
comb_params_deps[0] = _gather_tid_inline42;

// Spmd arbitrary_post_group: hc_post
L0TaskArgs post_params;
post_params.add_output(y_flat_inline15);
post_params.add_input(post_ffn_inline4);
post_params.add_input(ffn_out_inline16);
post_params.add_input(comb_ffn_inline14);
post_params.add_input(residual_flat_inline17);
"""
    orchestration.write_text(valid_source, encoding="utf-8")
    audit(tmp_path)

    invalid_replacements = (
        (
            "sync_workspace_inline987_ci_shapes[1] = {32}",
            "sync_workspace_inline987_ci_shapes[1] = {31}",
        ),
        (
            "fused_params_gamma.add_inout(sync_workspace_inline987)",
            "fused_params_gamma.add_input(sync_workspace_inline987)",
        ),
        (
            "opaque_fused_deps[2]",
            "opaque_fused_deps[3]",
        ),
        (
            "opaque_fused_deps[opaque_fused_dep_count++] = "
            "opaque_linear_runtime_id",
            "opaque_fused_deps[opaque_fused_dep_count++] = "
            "opaque_rms_runtime_id",
        ),
        (
            "rms_params_alpha.set_allow_early_resolve(true);",
            "rms_params_alpha.set_allow_early_resolve(false);",
        ),
        (
            "fused_params_gamma.launch_spec.set_block_num(8);",
            "fused_params_gamma.launch_spec.set_block_num(7);",
        ),
        (
            "fused_params_gamma.launch_spec.set_require_sync_start(true);",
            "fused_params_gamma.launch_spec.set_require_sync_start(false);",
        ),
        (
            "fused_params_gamma.set_allow_early_resolve(true);",
            "fused_params_gamma.set_allow_early_resolve(false);",
        ),
        (
            "sync_workspace_inline987_ci.set_initial_value(0);",
            "sync_workspace_inline987_ci.set_initial_value(0);\n"
            "sync_workspace_inline987_ci.start_offset = 4;",
        ),
    )
    for valid, invalid in invalid_replacements:
        orchestration.write_text(
            valid_source.replace(valid, invalid),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError):
            audit(tmp_path)


def test_each_python_launch_owns_one_zeroed_inout_sync_workspace() -> None:
    bridge_tree = _parse(_BRIDGE)
    moe_tree = _parse(_MOE)
    for tree in (bridge_tree, moe_tree):
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assert not any(
                isinstance(target, ast.Name)
                and target.id in {_SYNC_WORKSPACE, "_sync_workspace_specialize"}
                for target in targets
            ), "soft-sync workspaces must never be module-global"

    launch_specs = (
        (
            _function(bridge_tree, "fused_pre_norm_test"),
            "fused_pre_norm_cce",
            (
                "x_mixed",
                "x_flat",
                "inv_rms",
                "mixes_raw",
                "hc_base",
                "norm_w",
                "pre_val_store",
                "post",
                "xg_buf",
                "ffn_inv_rms_buf",
                "xn_scale_buf",
                "x_norm_scale",
                "sync_workspace",
                "scale0",
                "scale1",
                "num_tokens",
            ),
        ),
        (
            _function(bridge_tree, "fused_pre_norm_debug_test"),
            "fused_pre_norm_debug_cce",
            (
                "x_mixed",
                "x_flat",
                "inv_rms",
                "mixes_raw",
                "hc_base",
                "norm_w",
                "pre_val_store",
                "post",
                "xg_buf",
                "ffn_inv_rms_buf",
                "xn_scale_buf",
                "x_norm_scale",
                "sync_workspace",
                "scale0",
                "scale1",
                "num_tokens",
                "stop_after",
            ),
        ),
        (
            _function(bridge_tree, "fused_pre_norm_baseline_test"),
            "fused_pre_norm_baseline_cce",
            (
                "x_mixed",
                "x_flat",
                "inv_rms",
                "mixes_raw",
                "hc_base",
                "norm_w",
                "pre_val_store",
                "post",
                "xg_buf",
                "ffn_inv_rms_buf",
                "xn_scale_buf",
                "x_norm_scale",
                "sync_workspace",
                "scale0",
                "scale1",
                "num_tokens",
            ),
        ),
        (
            _function(bridge_tree, "fused_pre_norm_baseline_debug_test"),
            "fused_pre_norm_baseline_debug_cce",
            (
                "x_mixed",
                "x_flat",
                "inv_rms",
                "mixes_raw",
                "hc_base",
                "norm_w",
                "pre_val_store",
                "post",
                "xg_buf",
                "ffn_inv_rms_buf",
                "xn_scale_buf",
                "x_norm_scale",
                "sync_workspace",
                "scale0",
                "scale1",
                "num_tokens",
                "stop_after",
            ),
        ),
        (
            _function(moe_tree, "moe"),
            "fused_pre_norm_cce",
            (
                "x_mixed",
                "x_flat",
                "hc_inv_rms",
                "mixes_raw",
                "hc_ffn_base",
                "norm_w",
                "pre_val_store",
                "post_ffn",
                "xg_buf",
                "gate_inv_rms_buf",
                "xn_scale_buf",
                "x_norm_scale",
                "sync_workspace",
                "scale0",
                "scale1",
                "num_tokens_index",
            ),
        ),
    )

    for function, extern_name, expected_args in launch_specs:
        for workspace_name in (
            _SYNC_WORKSPACE,
            "_sync_workspace_specialize",
        ):
            create = _local_assignment(function, workspace_name)
            assert isinstance(create, ast.Call)
            assert _qualified_name(create.func) == "pl.create_tensor"
            assert len(create.args) == 1
            assert ast.unparse(create.args[0]) == "[FUSED_SOFT_SYNC_WORDS]"
            assert ast.unparse(_keyword(create, "dtype")) == "pl.INT32"
            assert ast.literal_eval(_keyword(create, "init_value")) == 0
            assert not any(
                keyword.arg == "manual_dep" for keyword in create.keywords
            )
            assert not any(
                (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == workspace_name
                )
                or (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == workspace_name
                    and node.func.attr in {"view", "reshape", "slice"}
                )
                for node in ast.walk(function)
            ), f"{workspace_name} must be passed as a fresh, unsliced tensor"

        direct_calls = _calls(function, extern_name)
        launches = _calls(function, "pl.spmd_submit")
        assert len(direct_calls) == len(launches) == 1
        specialize_args = tuple(
            "_sync_workspace_specialize"
            if name == _SYNC_WORKSPACE
            else name
            for name in expected_args
        )
        assert tuple(ast.unparse(arg) for arg in direct_calls[0].args) == (
            specialize_args
        )
        assert tuple(ast.unparse(arg) for arg in launches[0].args[1:]) == (
            expected_args
        )
        assert ast.unparse(_keyword(launches[0], "core_num")) == (
            "FUSED_AIV_CORES"
        )
        assert ast.literal_eval(_keyword(launches[0], "sync_start")) is True

        forbidden_wrappers = {
            _qualified_name(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _qualified_name(node.func) in {"pl.no_dep", "pl.manual_scope"}
            and any(
                isinstance(arg, ast.Name) and arg.id == _SYNC_WORKSPACE
                for arg in node.args
            )
        }
        assert not forbidden_wrappers


def test_moe_passes_producer_and_fused_task_ids_across_scheduler_boundaries() -> None:
    moe_tree = _parse(_MOE)
    function = _function(moe_tree, "moe")
    producer_call = _calls(function, "hc_pre_moe_producers")
    fused_call = _calls(function, "fused_pre_norm_cce")
    gate_call = _calls(function, "gate_precomputed")
    dispatch_call = _calls(function, "dispatch")
    comb_call = _calls(function, "hc_pre_moe_comb")
    assert (
        len(producer_call)
        == len(fused_call)
        == len(gate_call)
        == len(dispatch_call)
        == len(comb_call)
        == 1
    )

    producer_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is producer_call[0]
    )
    assert ast.unparse(producer_assignment.targets[0]) == (
        "(hc_pre_rms_tid, hc_pre_linear_tid)"
    )

    launch = _calls(function, "pl.spmd_submit")
    assert len(launch) == 1
    launch_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is launch[0]
    )
    assert ast.unparse(launch_assignment.targets[0]) == (
        "((x_mixed, pre_val_store, post_ffn, xg_buf, gate_inv_rms_buf, "
        "xn_scale_buf, x_norm_scale), fused_pre_norm_tid)"
    )

    fused_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is fused_call[0]
    )
    assert ast.unparse(fused_assignment.targets[0]) == (
        "(x_mixed, pre_val_store, post_ffn, xg_buf, gate_inv_rms_buf, "
        "xn_scale_buf, x_norm_scale)"
    )
    assert ast.unparse(gate_call[0].args[-1]) == "fused_pre_norm_tid"

    dispatch_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is dispatch_call[0]
    )
    assert ast.unparse(dispatch_assignment.targets[0]) == "dispatch_gather_tid"
    assert ast.unparse(comb_call[0].args[-1]) == "dispatch_gather_tid"
    assert dispatch_call[0].lineno < comb_call[0].lineno

    hc_pre_tree = _parse(_HC_PRE)
    for function_name in ("_hc_pre_separate", "hc_pre_moe_producers"):
        producer_function = _function(hc_pre_tree, function_name)
        for task_name in ("hc_pre_rms", "hc_pre_linear"):
            producer_scopes = _named_spmd_calls(
                producer_function,
                task_name,
            )
            assert len(producer_scopes) == 1
            assert ast.literal_eval(
                _keyword(producer_scopes[0], "allow_early_resolve"),
            ) is True

    dispatch_function = _function(moe_tree, "dispatch")
    gather_scopes = _named_spmd_with(dispatch_function, "dispatch_gather")
    assert len(gather_scopes) == 1
    assert ast.unparse(gather_scopes[0].items[0].optional_vars) == "_gather_tid"
    assert any(
        isinstance(node, ast.Return)
        and ast.unparse(node.value) == "_gather_tid"
        for node in dispatch_function.body
    )


def test_comb_sinkhorn_uses_with_spmd_and_waits_directly_on_dispatch_gather() -> None:
    tree = _parse(_HC_PRE)

    standalone = _function(tree, "_hc_pre_separate")
    standalone_comb = _named_spmd_with(standalone, "comb_sinkhorn")
    assert len(standalone_comb) == 1
    assert isinstance(standalone_comb[0].body[0], ast.Assign)
    assert ast.unparse(standalone_comb[0].body[0]) == (
        "ob = pl.tile.get_block_idx()"
    )

    producers = _function(tree, "hc_pre_moe_producers")
    assert not _named_spmd_with(producers, "comb_sinkhorn")
    assert tuple(argument.arg for argument in producers.args.args) == (
        "x",
        "hc_fn",
        "inv_rms",
        "mixes_raw",
    )

    delayed_comb = _function(tree, "hc_pre_moe_comb")
    comb_scopes = _named_spmd_with(delayed_comb, "comb_sinkhorn")
    assert len(comb_scopes) == 1
    comb_call = comb_scopes[0].items[0].context_expr
    assert isinstance(comb_call, ast.Call)
    assert ast.unparse(_keyword(comb_call, "deps")) == "[dispatch_gather_tid]"
    assert "hc_pre_rms_tid" not in ast.unparse(delayed_comb)
    assert "hc_pre_linear_tid" not in ast.unparse(delayed_comb)

    moe = _function(_parse(_MOE), "moe")
    for tensor_name in ("hc_inv_rms", "mixes_raw"):
        create = next(
            node.value
            for node in ast.walk(moe)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == tensor_name
                for target in node.targets
            )
        )
        assert isinstance(create, ast.Call)
        assert _qualified_name(create.func) == "pl.create_tensor"
        assert ast.literal_eval(_keyword(create, "manual_dep")) is True


def test_aiv_fusion_uses_two_public_atomic_soft_barrier_phases() -> None:
    raw_body = _read(_FUSED_BODY)
    body = _strip_cpp_comments(raw_body)
    assert re.search(
        r"#\s*ifdef\s+__PTO_AUTO__\s*"
        r"#\s*error\s+\"fused_pre_norm soft SYNCALL requires the manual "
        r"extern build path\"\s*"
        r"#\s*endif",
        raw_body,
    )
    assert re.search(
        r"\bvoid\s+soft_sync_aiv\s*\(\s*"
        r"__gm__\s+int32_t\s*\*\s*workspace_base\s*,\s*"
        r"int32_t\s+offset_words\s*,\s*"
        r"int32_t\s+participants\s*\)",
        body,
    )
    assert re.search(
        r"pto::GlobalTensor\s*<\s*int32_t\s*,\s*"
        r"pto::Shape\s*<\s*>\s*,\s*"
        r"pto::Stride\s*<\s*>\s*>\s+workspace\s*"
        r"\(\s*workspace_base\s*\+\s*offset_words\s*\)\s*;",
        body,
    )
    public_calls = re.findall(
        r"pto::SYNCALL\s*<\s*"
        r"pto::SyncAllMode::Soft\s*,\s*"
        r"pto::SyncCoreType::AIVOnly\s*>\s*"
        r"\(\s*workspace\s*,\s*participants\s*\)\s*;",
        body,
    )
    assert len(public_calls) == 1

    phase_calls = re.findall(
        r"\bsoft_sync_aiv\s*\(\s*sync_workspace\s*,\s*"
        r"(kBarrier[12]OffsetWords)\s*,\s*"
        r"(barrier[12]_participants)\s*\)\s*;",
        body,
    )
    assert phase_calls == [
        ("kBarrier1OffsetWords", "barrier1_participants"),
        ("kBarrier2OffsetWords", "barrier2_participants"),
    ]
    for barrier in (1, 2):
        assert re.search(
            rf"if\s*\(\s*barrier{barrier}_participants\s*>\s*0\s*&&\s*"
            rf"lane\s*<\s*barrier{barrier}_participants\s*\)\s*\{{\s*"
            rf"soft_sync_aiv\s*\(\s*sync_workspace\s*,\s*"
            rf"kBarrier{barrier}OffsetWords\s*,\s*"
            rf"barrier{barrier}_participants\s*\)\s*;\s*\}}",
            body,
        )

    for work in ("split", "mix", "ffn"):
        assert re.search(
            rf"const\s+int32_t\s+active_{work}\s*=\s*"
            rf"{work}_work\s*<\s*kAivLanes\s*\?\s*"
            rf"{work}_work\s*:\s*kAivLanes\s*;",
            body,
        )
    assert re.search(
        r"const\s+int32_t\s+dense_barrier1_participants\s*=\s*"
        r"active_split\s*>\s*active_mix\s*\?\s*"
        r"active_split\s*:\s*active_mix\s*;",
        body,
    )
    assert re.search(
        r"const\s+int32_t\s+dense_barrier2_participants\s*=\s*"
        r"active_mix\s*>\s*active_ffn\s*\?\s*"
        r"active_mix\s*:\s*active_ffn\s*;",
        body,
    )
    for barrier in (1, 2):
        assert re.search(
            rf"const\s+int32_t\s+barrier{barrier}_participants\s*=\s*"
            rf"select_barrier_participants\s*<\s*Policy\s*>\s*"
            rf"\(\s*dense_barrier{barrier}_participants\s*\)\s*;",
            body,
        )

    assert _enum_symbols(raw_body, "BarrierPolicy") == [
        "kDenseTarget",
        "kAtomicEightWayBaseline",
    ]
    assert re.search(
        r"template\s*<\s*BarrierPolicy\s+Policy\s*>\s*"
        r"static\s+__aicore__[^;{]*\bselect_barrier_participants\s*"
        r"\(\s*int32_t\s+dense_participants\s*\)",
        body,
    )
    assert re.search(
        r"if\s+constexpr\s*\(\s*Policy\s*==\s*"
        r"BarrierPolicy::kAtomicEightWayBaseline\s*\)\s*\{\s*"
        r"return\s+dense_participants\s*>\s*0\s*\?\s*kAivLanes\s*:\s*0\s*;"
        r"\s*\}\s*return\s+dense_participants\s*;",
        body,
    )
    assert re.search(
        r"template\s*<\s*StopAfter\s+Stop\s*,\s*"
        r"BarrierPolicy\s+Policy\s*=\s*BarrierPolicy::kDenseTarget\s*>\s*"
        r"static\s+__aicore__[^;{]*\brun_fused_pre_norm\s*"
        r"\(\s*__gm__\s+int64_t\s*\*\s*args\s*\)",
        body,
    )

    production_entry = _strip_cpp_comments(_read(_PRODUCTION_ENTRY))
    assert re.search(
        r"run_fused_pre_norm\s*<\s*"
        r"deepseek_fused_pre_norm::StopAfter::kFull\s*>\s*\(\s*args\s*\)",
        production_entry,
    )
    assert "BarrierPolicy" not in production_entry

    production_debug_entry = _strip_cpp_comments(_read(_DEBUG_ENTRY))
    assert "BarrierPolicy" not in production_debug_entry
    assert production_debug_entry.count("run_fused_pre_norm<") == 5

    baseline_entry = _strip_cpp_comments(_read(_BASELINE_ENTRY))
    assert re.search(
        r"run_fused_pre_norm\s*<\s*"
        r"deepseek_fused_pre_norm::StopAfter::kFull\s*,\s*"
        r"deepseek_fused_pre_norm::BarrierPolicy::kAtomicEightWayBaseline\s*>"
        r"\s*\(\s*args\s*\)",
        baseline_entry,
    )
    baseline_debug_entry = _strip_cpp_comments(_read(_BASELINE_DEBUG_ENTRY))
    assert baseline_debug_entry.count("run_fused_pre_norm<") == 5
    assert len(
        re.findall(
            r"run_fused_pre_norm\s*<\s*"
            r"StopAfter::k[A-Za-z0-9_]+\s*,\s*"
            r"BarrierPolicy::kAtomicEightWayBaseline\s*>\s*\(\s*args\s*\)",
            baseline_debug_entry,
        ),
    ) == 5

    assert re.search(
        r"const\s+int32_t\s+lane\s*=\s*"
        r"static_cast\s*<\s*int32_t\s*>\s*"
        r"\(\s*get_block_idx\s*\(\s*args\s*\)\s*\)\s*;",
        body,
    )

    all_kernel_code = "\n".join(
        _strip_cpp_comments(_read(path))
        for path in sorted(_KERNEL_DIR.rglob("*"))
        if path.suffix in {".cpp", ".h", ".hpp"}
    )
    forbidden = {
        "hardware AscendC SyncAll": r"\bAscendC::SyncAll\b",
        "old indexed soft barrier": r"\bSYNCALL_SOFT_AIV_BARRIER\b",
        "old indexed slot constant": r"\bSYNCALL_SOFT_SLOT_INT32\b",
        "soft-sync UB workspace": r"\b(?:kSoftSyncUbAddr|ub_workspace)\b",
        "bare physical get_block_idx": r"\bget_block_idx\s*\(\s*\)",
        "FFTS cross-core synchronization": r"\bffts_cross_core_sync\b",
    }
    for description, pattern in forbidden.items():
        assert re.search(pattern, all_kernel_code, flags=re.IGNORECASE) is None, (
            f"8-lane dynamic scheduling must not use {description}"
        )


def test_soft_barriers_publish_and_acquire_owned_business_ranges() -> None:
    body = _strip_cpp_comments(_read(_FUSED_BODY))
    assert re.search(
        r"for\s*\(\s*uint64_t\s+line\s*=\s*first\s*;"
        r"\s*line\s*<\s*end\s*;"
        r"\s*line\s*\+=\s*kA2A3DcciLineBytes\s*\)",
        body,
    )
    assert re.search(r"dcci\s*\([^;]*CACHELINE_OUT[^;]*\)\s*;", body)
    assert re.search(
        r"dcci\s*\([^;]*SINGLE_CACHE_LINE\s*\)\s*;",
        body,
    )
    for call in (
        "publish_pre_value(pre_value, lane, split_work);",
        "acquire_pre_value(pre_value, lane, mix_work);",
        "publish_x_mixed(x_mixed, lane, mix_work);",
        "acquire_x_mixed(x_mixed, lane, ffn_work);",
    ):
        assert body.count(call) == 1

    positions = {
        call: body.index(call)
        for call in (
            "publish_pre_value(pre_value, lane, split_work);",
            "acquire_pre_value(pre_value, lane, mix_work);",
            "publish_x_mixed(x_mixed, lane, mix_work);",
            "acquire_x_mixed(x_mixed, lane, ffn_work);",
        )
    }
    barrier_positions = [
        match.start()
        for match in re.finditer(
            r"soft_sync_aiv\s*\(\s*sync_workspace\s*,\s*"
            r"kBarrier[12]OffsetWords\s*,\s*"
            r"barrier[12]_participants\s*\)\s*;",
            body,
        )
    ]
    assert len(barrier_positions) == 2
    assert (
        positions["publish_pre_value(pre_value, lane, split_work);"]
        < barrier_positions[0]
        < positions["acquire_pre_value(pre_value, lane, mix_work);"]
        < positions["publish_x_mixed(x_mixed, lane, mix_work);"]
        < barrier_positions[1]
        < positions["acquire_x_mixed(x_mixed, lane, ffn_work);"]
    )


def test_atomic_soft_barriers_use_two_aligned_gm_counters_without_ub_scratch() -> None:
    body = _strip_cpp_comments(_read(_FUSED_BODY))
    assert "#include <pto/pto-inst.hpp>" in _read(_FUSED_BODY)
    assert re.search(
        r"\bkSoftSyncCounterWords\s*=\s*"
        r"pto::SYNCALL_SOFT_WORKSPACE_INT32\s*;",
        body,
    )
    assert re.search(
        r"\bkBarrier1OffsetWords\s*=\s*0\s*;",
        body,
    )
    assert re.search(
        r"\bkBarrier2OffsetWords\s*=\s*kSoftSyncCounterWords\s*;",
        body,
    )
    assert re.search(
        r"\bkSoftSyncWorkspaceWords\s*=\s*"
        r"2\s*\*\s*kSoftSyncCounterWords\s*;",
        body,
    )
    assert re.search(
        r"\bkA2A3DcciLineBytes\s*=\s*64U?\s*;",
        body,
    )
    compact_body = re.sub(r"\s+", "", body)
    assert (
        "static_assert("
        "kSoftSyncCounterWords*sizeof(int32_t)==kA2A3DcciLineBytes"
        ");"
    ) in compact_body
    assert re.search(
        r"static_assert\(\(?kBarrier2OffsetWords\*sizeof\(int32_t\)\)?%"
        r"kA2A3DcciLineBytes==0U\);",
        compact_body,
    )
    assert re.search(
        r"static_assert\s*\(\s*kSoftSyncWorkspaceWords\s*==\s*32\s*\)\s*;",
        body,
    )

    forbidden = (
        "kSoftSyncUbAddr",
        "kSoftSyncWords",
        "SYNCALL_SOFT_SLOT_INT32",
        "SYNCALL_SOFT_AIV_BARRIER",
        "ub_workspace",
        "__ubuf__ int32_t",
    )
    for old_resource in forbidden:
        assert old_resource not in body


def test_generated_phases_keep_grid_stride_mapping_and_barrier_order() -> None:
    body = _strip_cpp_comments(_read(_FUSED_BODY))
    phase_specs = (
        (
            "split_work",
            "deepseek_fused_pre_norm_split_generated::split_pre_post",
        ),
        (
            "mix_work",
            "deepseek_fused_pre_norm_mix_generated::mix_x",
        ),
        (
            "ffn_work",
            "deepseek_fused_pre_norm_ffn_generated::ffn_norm",
        ),
    )

    phase_positions = []
    for work_count, callee in phase_specs:
        loop = re.search(
            rf"for\s*\(\s*int32_t\s+logical_block\s*=\s*lane\s*;"
            rf"\s*logical_block\s*<\s*{work_count}\s*;"
            rf"\s*logical_block\s*\+=\s*kAivLanes\s*\)\s*\{{"
            rf"\s*{re.escape(callee)}\s*\((.*?)\)\s*;\s*\}}",
            body,
            flags=re.DOTALL,
        )
        assert loop is not None, f"{callee} must retain an 8-lane grid-stride loop"
        assert re.search(
            rf"logical_block\s*,\s*{work_count}\s*$",
            loop.group(1),
        ), f"{callee} must receive its original logical block count"
        phase_positions.append(body.index(f"{callee}("))

    barrier_positions = [
        match.start()
        for match in re.finditer(
            r"\bsoft_sync_aiv\s*\(\s*sync_workspace\s*,\s*"
            r"kBarrier[12]OffsetWords\s*,\s*"
            r"barrier[12]_participants\s*\)",
            body,
        )
    ]
    assert len(barrier_positions) == 2
    assert (
        phase_positions[0]
        < barrier_positions[0]
        < phase_positions[1]
        < barrier_positions[1]
        < phase_positions[2]
    )


def test_debug_entry_can_stop_on_each_side_of_both_barriers() -> None:
    bridge_tree = _parse(_BRIDGE)
    expected_python_stops = {
        "STOP_SPLIT_BEFORE_BARRIER1": 0,
        "STOP_AFTER_BARRIER1": 1,
        "STOP_MIX_BEFORE_BARRIER2": 2,
        "STOP_AFTER_BARRIER2": 3,
        "STOP_FULL": 4,
    }
    assert {
        name: _assigned_constant(bridge_tree, name)
        for name in expected_python_stops
    } == expected_python_stops

    body = _read(_FUSED_BODY)
    assert _enum_symbols(body, "StopAfter") == [
        "kSplitBeforeBarrier1",
        "kAfterBarrier1",
        "kMixBeforeBarrier2",
        "kAfterBarrier2",
        "kFull",
    ]
    cpp_stops = dict(
        re.findall(
            r"\b(k(?:SplitBeforeBarrier1|AfterBarrier1|MixBeforeBarrier2|"
            r"AfterBarrier2|Full))\s*=\s*(\d+)",
            body,
        )
    )
    assert cpp_stops == {
        "kSplitBeforeBarrier1": "0",
        "kAfterBarrier1": "1",
        "kMixBeforeBarrier2": "2",
        "kAfterBarrier2": "3",
        "kFull": "4",
    }

    production_entry = _strip_cpp_comments(_read(_PRODUCTION_ENTRY))
    assert "StopAfter::kFull" in production_entry
    assert "kDebugStopAfter" not in production_entry
    assert "switch" not in production_entry

    debug_entry = _strip_cpp_comments(_read(_DEBUG_ENTRY))
    assert re.search(r"args\s*\[\s*deepseek_fused_pre_norm::kDebugStopAfter\s*\]", debug_entry)
    assert re.search(r"switch\s*\(\s*stop_after\s*\)", debug_entry)
    for stop_name in cpp_stops:
        assert f"StopAfter::{stop_name}" in debug_entry

    debug_map = next(
        node.value
        for node in ast.walk(bridge_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "debug_stops"
            for target in node.targets
        )
    )
    assert isinstance(debug_map, ast.Dict)
    assert {
        ast.literal_eval(key): ast.unparse(value)
        for key, value in zip(debug_map.keys, debug_map.values)
    } == {
        "split": "STOP_SPLIT_BEFORE_BARRIER1",
        "barrier1": "STOP_AFTER_BARRIER1",
        "mix": "STOP_MIX_BEFORE_BARRIER2",
        "barrier2": "STOP_AFTER_BARRIER2",
        "full": "STOP_FULL",
    }


def test_dump_tags_bracket_the_extern_and_cover_consumer_side_buffers() -> None:
    bridge_tree = _parse(_BRIDGE)
    tag_helper = _function(bridge_tree, "_tag_test_tensors")
    assert {name for _, name in _dump_calls(tag_helper)} == set(
        (*_DUMP_INPUTS, *_PHASE_OUTPUTS),
    )
    assert _SYNC_WORKSPACE not in {
        argument.arg for argument in tag_helper.args.args
    }
    for function_name, extern_name in (
        ("fused_pre_norm_test", "fused_pre_norm_cce"),
        ("fused_pre_norm_debug_test", "fused_pre_norm_debug_cce"),
        ("fused_pre_norm_baseline_test", "fused_pre_norm_baseline_cce"),
        (
            "fused_pre_norm_baseline_debug_test",
            "fused_pre_norm_baseline_debug_cce",
        ),
    ):
        function = _function(bridge_tree, function_name)
        tag_calls = _calls(function, "_tag_test_tensors")
        extern_calls = _calls(function, extern_name)
        launches = _calls(function, "pl.spmd_submit")
        assert len(tag_calls) == 2
        assert len(extern_calls) == 1
        assert len(launches) == 1
        workspace_dump_lines = [
            line
            for line, name in _dump_calls(function)
            if name == _SYNC_WORKSPACE
        ]
        assert len(workspace_dump_lines) == 1, (
            "the consumed InOut workspace may only be tagged before submit"
        )
        assert (
            tag_calls[0].lineno
            < workspace_dump_lines[0]
            < extern_calls[0].lineno
            < launches[0].lineno
            < tag_calls[1].lineno
        )
        assert "_sync_workspace_specialize" in {
            ast.unparse(argument) for argument in extern_calls[0].args
        }
        assert _SYNC_WORKSPACE in {
            ast.unparse(argument) for argument in launches[0].args[1:]
        }

    model = _function(_parse(_MOE), "moe")
    model_extern = _calls(model, "fused_pre_norm_cce")
    model_launch = _calls(model, "pl.spmd_submit")
    assert len(model_extern) == 1
    assert len(model_launch) == 1
    model_launch_line = model_launch[0].lineno
    model_dumps = _dump_calls(model)
    workspace_dump_lines = [
        line for line, name in model_dumps if name == _SYNC_WORKSPACE
    ]
    assert len(workspace_dump_lines) == 1
    assert workspace_dump_lines[0] < model_extern[0].lineno < model_launch_line
    assert "_sync_workspace_specialize" in {
        ast.unparse(argument) for argument in model_extern[0].args
    }
    assert _SYNC_WORKSPACE in {
        ast.unparse(argument) for argument in model_launch[0].args[1:]
    }
    before = {name for line, name in model_dumps if line < model_launch_line}
    after = {name for line, name in model_dumps if line > model_launch_line}
    assert {
        "x_flat",
        "hc_inv_rms",
        "mixes_raw",
        "hc_ffn_base",
        "norm_w",
        "x_mixed",
        "pre_val_store",
        "post_ffn",
        "xg_buf",
        "gate_inv_rms_buf",
        "xn_scale_buf",
        "x_norm_scale",
        _SYNC_WORKSPACE,
    } <= before
    assert {
        "x_mixed",
        "pre_val_store",
        "post_ffn",
        "xg_buf",
        "gate_inv_rms_buf",
        "xn_scale_buf",
        "x_norm_scale",
    } <= after
    assert _SYNC_WORKSPACE not in after

    gate = _function(_parse(_GATE), "gate_precomputed")
    assert {
        "xg_buf",
        "inv_rms_buf",
        "xn_scale_buf",
        "x_norm_scale",
    } <= {name for _, name in _dump_calls(gate)}


def test_model_and_standalone_clis_forward_partial_dump_level() -> None:
    for path in (_MOE, _BRIDGE):
        tree = _parse(path)
        argument = _argument_call(tree, "--dump-args")
        assert ast.literal_eval(_keyword(argument, "nargs")) == "?"
        assert ast.literal_eval(_keyword(argument, "const")) == 1
        assert ast.literal_eval(_keyword(argument, "default")) == 0
        assert ast.unparse(_keyword(argument, "type")) == "int"
        assert ast.literal_eval(_keyword(argument, "choices")) == (0, 1, 2, 3)

        forwarded = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "enable_dump_args"
        ]
        assert forwarded
        assert {
            ast.unparse(value)
            for value in forwarded
        } == {"args.dump_args"}


def test_ep_expert_formula_keeps_sixteen_experts_per_rank() -> None:
    tree = _parse(_MOE)
    replacement = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and _qualified_name(target) == "config.FLASH"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _qualified_name(node.value.func) == "dataclasses.replace"
    )
    assert isinstance(replacement, ast.Call)
    expert_count = _keyword(replacement, "n_routed_experts")
    assert ast.unparse(expert_count) == (
        "config.FLASH.n_routed_experts // 16 * EP"
    )


def test_balanced_routing_default_has_three_routes_per_expert() -> None:
    for ep in (2, 4, 8):
        expert_count = 16 * ep
        route_count = ep * 8 * 6
        routes_per_expert, remainder = divmod(route_count, expert_count)
        assert remainder == 0
        assert routes_per_expert == 3
