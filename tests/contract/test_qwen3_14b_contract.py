# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from contract.registry import find_contract_for_model_config, get_contract


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_qwen3_14b_public_abi_fingerprint_is_frozen() -> None:
    """Backend selection is a build concern and must not change the serving ABI."""
    contract = get_contract("qwen3", "14b")

    assert contract.abi_fingerprint() == ("00fb898b160b00e9448d18c1df30e25215d5fc4f98ec2f0ed9ee6da34c59826f")
    assert [(argument.name, argument.direction) for argument in contract.kernels["decode"].args] == [
        ("input_rms_weight", "in"),
        ("wq", "in"),
        ("wk", "in"),
        ("wv", "in"),
        ("q_norm_weight", "in"),
        ("k_norm_weight", "in"),
        ("seq_lens", "in"),
        ("block_table", "in"),
        ("slot_mapping", "in"),
        ("rope_cos", "in"),
        ("rope_sin", "in"),
        ("k_cache", "inout"),
        ("v_cache", "inout"),
        ("wo", "in"),
        ("w_gate", "in"),
        ("w_up", "in"),
        ("w_down", "in"),
        ("post_rms_weight", "in"),
        ("final_norm_weight", "in"),
        ("lm_head_weight", "in"),
        ("out", "out"),
        ("embed_weight", "in"),
        ("sampled_ids_in", "in"),
        ("sampled_ids", "out"),
        ("next_hidden", "out"),
    ]


def _ast_function_source(source: str, function: ast.FunctionDef) -> str:
    return ast.get_source_segment(source, function) or ""


def _ast_function_signature(function: ast.FunctionDef) -> tuple[tuple[str, str | None], ...]:
    return tuple(
        (
            argument.arg,
            ast.unparse(argument.annotation) if argument.annotation is not None else None,
        )
        for argument in function.args.args
    )


def test_step9_decode_is_sole_pypto_and_preserves_task_graph() -> None:
    """Pin the sole native implementation and the public entry signatures."""
    path = _REPO_ROOT / "models" / "qwen3_14b" / "decode_fwd.py"
    source = path.read_text()
    tree = ast.parse(source)

    assert "from paged_attention_pypto import" in source
    forbidden = (
        "PYPTO_QWEN3_PA_IMPL",
        "expect_pa_impl",
        "pa_backend",
        "paged_attention_cce",
        "legacy_cce",
        "fa_fused",
        "PA_METADATA_BYTES",
        "PA_WORKSPACE_BYTES",
        "build_paged_attention_metadata",
    )
    residue = [marker for marker in forbidden if marker in source]
    assert residue == [], f"decode_fwd.py still references retired CCE markers: {residue}"

    decode_layer = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_decode_layer"
    )
    decode_layer_source = _ast_function_source(source, decode_layer)
    assert decode_layer_source.count("_run_paged_attention(") == 1
    assert "rope_qkv_pypto(" not in decode_layer_source
    assert "paged_attention_pypto_swpipe(" not in decode_layer_source
    assert "scratch_ready[0] = attn_done_tid" in decode_layer_source
    assert "out_proj_dummy = pl.system.task_dummy(deps=[attn_done_tid])" in decode_layer_source
    assert 'name_hint="out_proj"' in decode_layer_source
    assert "deps=[attn_done_tid]" in decode_layer_source

    adapters = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_paged_attention"
    ]
    assert len(adapters) == 1
    adapter_source = _ast_function_source(source, adapters[0])
    assert adapter_source.count("paged_attention_pypto_swpipe(") == 1
    assert "rope_qkv_pypto" not in adapter_source
    assert "rope_tid" not in adapter_source
    assert "sync_start=True" not in adapter_source

    for entry_name, expected_arg_count, body_name in (
        ("decode_fwd", 25, "_decode_fwd_body"),
        ("decode_fwd_layers", 20, "_decode_fwd_layers_body"),
    ):
        entries = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_name
        ]
        assert len(entries) == 1
        entry = entries[0]
        assert len(entry.args.args) == expected_arg_count
        entry_source = _ast_function_source(source, entry)
        assert f"{body_name}(" in entry_source
        assert entry_source.count("pl.create_tensor(") == 4
        assert "score_transfer" in entry_source
        assert "probability_transfer" in entry_source
        assert "pv_transfer" in entry_source
        assert "ffts_workspace" in entry_source
        assert "scratch_ready = pl.array.create(1, pl.TASK_ID)" in entry_source
        assert "scratch_ready[0] = pl.system.task_dummy" not in entry_source


def test_step9_decode_wrapper_materializes_public_out_tuple() -> None:
    """Keep serving L3 return discovery intact on the sole PyPTO entry."""
    path = _REPO_ROOT / "models" / "qwen3_14b" / "decode_fwd.py"
    tree = ast.parse(path.read_text())
    entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_fwd"]
    assert len(entries) == 1
    entry = entries[0]

    expected_outputs = ("out", "sampled_ids_out", "next_hidden")
    declared_outputs = tuple(
        argument.arg
        for argument in entry.args.args
        if argument.annotation is not None and ast.unparse(argument.annotation).startswith("pl.Out[")
    )
    assert declared_outputs == expected_outputs

    body_calls = [
        statement
        for statement in entry.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_decode_fwd_body"
    ]
    assert len(body_calls) == 1
    assignment = body_calls[0]
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Tuple)
    assert all(isinstance(element, ast.Name) for element in assignment.targets[0].elts)
    assert tuple(element.id for element in assignment.targets[0].elts) == expected_outputs

    assert isinstance(entry.body[-1], ast.Return)
    assert isinstance(entry.body[-1].value, ast.Tuple)
    assert all(isinstance(element, ast.Name) for element in entry.body[-1].value.elts)
    assert tuple(element.id for element in entry.body[-1].value.elts) == expected_outputs

    serving_entries = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qwen3_decode_host"
    ]
    assert len(serving_entries) == 1
    serving_entry = serving_entries[0]
    assert any(ast.unparse(decorator) == "pl.jit.host" for decorator in serving_entry.decorator_list)

    serving_call = next(
        statement
        for statement in serving_entry.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "decode_fwd"
    )
    assert isinstance(serving_call.targets[0], ast.Tuple)
    assert tuple(element.id for element in serving_call.targets[0].elts) == (
        "logits",
        "sampled_ids",
        "next_hidden",
    )
    assert isinstance(serving_entry.body[-1], ast.Return)
    assert isinstance(serving_entry.body[-1].value, ast.Tuple)
    assert tuple(element.id for element in serving_entry.body[-1].value.elts) == (
        "logits",
        "sampled_ids",
        "next_hidden",
    )


_DECODE_IMPORT_PROBE_PREFIX = "__QWEN3_DECODE_IMPORT_PROBE__="


def _run_decode_import_probe(build_dir: Path) -> subprocess.CompletedProcess[str]:
    model_dir = _REPO_ROOT / "models" / "qwen3_14b"
    environment = os.environ.copy()
    environment.update(
        {
            "PYPTO_PROG_BUILD_DIR": str(build_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(model_dir), str(_REPO_ROOT), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    probe = f"""
import inspect
import json
import sys

import decode_fwd


def jit_source(value):
    return inspect.getsource(getattr(value, "_func", value))


backend_modules = sorted(
    {{name.rsplit(".", 1)[-1] for name in sys.modules}}
    & {{"paged_attention_cce", "paged_attention_pypto"}}
)
payload = {{
    "backend_modules": backend_modules,
    "adapter_source": jit_source(decode_fwd._run_paged_attention),
    "decode_source": jit_source(decode_fwd.decode_fwd),
    "layers_source": jit_source(decode_fwd.decode_fwd_layers),
    "decode_args": list(inspect.signature(decode_fwd.decode_fwd._func).parameters),
    "layers_args": list(inspect.signature(decode_fwd.decode_fwd_layers._func).parameters),
}}
print({_DECODE_IMPORT_PROBE_PREFIX!r} + json.dumps(payload, sort_keys=True))
"""
    return subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_step9_production_decode_fresh_process_is_sole_pypto(
    tmp_path: Path,
) -> None:
    """A fresh process exposes only native PyPTO."""
    completed = _run_decode_import_probe(tmp_path / "build")

    assert completed.returncode == 0, completed.stderr
    probe_lines = [
        line for line in completed.stdout.splitlines() if line.startswith(_DECODE_IMPORT_PROBE_PREFIX)
    ]
    assert len(probe_lines) == 1, completed.stdout
    payload = json.loads(probe_lines[0].removeprefix(_DECODE_IMPORT_PROBE_PREFIX))
    assert payload["backend_modules"] == ["paged_attention_pypto"]
    assert len(payload["decode_args"]) == 25
    assert len(payload["layers_args"]) == 20

    compile_source = "\n".join(
        (payload["adapter_source"], payload["decode_source"], payload["layers_source"])
    )
    assert compile_source.count("paged_attention_pypto_swpipe(") == 1
    assert "rope_qkv_pypto" not in compile_source
    assert "rope_tid" not in compile_source
    assert "score_transfer = pl.create_tensor" in compile_source
    assert "ffts_workspace = pl.create_tensor" in compile_source
    for forbidden in (
        "paged_attention_rope_cce(",
        "build_paged_attention_metadata(",
        "PA_WORKSPACE_BYTES",
        "legacy_cce",
        "PYPTO_QWEN3_PA_IMPL",
    ):
        assert forbidden not in compile_source


def test_qwen_pa_driver_checks_codegen_scratch_and_no_cce() -> None:
    component = (_REPO_ROOT / "models" / "qwen3_14b" / "test_paged_attention_pypto.py").read_text()
    assert "_resolve_pass_dump" in component
    assert "after_ExpandMixedKernel" in component
    assert "fused PA must allocate q_tnd plus one four-tensor scratch set" in component
    assert "PyPTO PA artifact contains a legacy CCE marker" in component
    assert "fused PA artifact is missing the Phase-0 GM fence + hard mixed-core barrier" in component
    assert "375-case Cartesian product" in component


def _tiny_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=17,
    )


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        max_batch_size=16,
        max_seq_len=2,
        page_size=128,
        vocab_pad_multiple=512,
        total_kv_pages=16,
    )


def test_registry_resolves_explicit_qwen3_14b_contract() -> None:
    contract = get_contract("qwen3", "14b")

    assert contract.model.family == "qwen3"
    assert contract.model.variant == "14b"
    assert sorted(contract.kernels) == ["decode", "greedy_sample", "prefill"]
    assert contract.execution == {"prefill": ("prefill",), "decode": ("decode",)}
    assert contract.abi_fingerprint()


def test_registry_matches_qwen3_14b_model_config() -> None:
    model_config = SimpleNamespace(
        model_id="local-served-name",
        architecture="Qwen3ForCausalLM",
        architectures=("Qwen3ForCausalLM",),
        model_type="qwen3",
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
    )

    contract = find_contract_for_model_config(model_config)

    assert contract.model.family == "qwen3"
    assert contract.model.variant == "14b"


def test_registry_matches_qwen3_14b_model_config_with_null_architectures() -> None:
    model_config = SimpleNamespace(
        model_id="local-served-name",
        architecture="Qwen3ForCausalLM",
        architectures=None,
        model_type="qwen3",
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
    )

    contract = find_contract_for_model_config(model_config)

    assert contract.model.family == "qwen3"
    assert contract.model.variant == "14b"


def test_loaded_kernel_modules_match_current_qwen3_files() -> None:
    contract = get_contract("qwen3", "14b")
    loaded = contract.load_kernels()
    model = _qwen3_14b_model()

    assert sorted(loaded.functions) == ["decode_fwd", "greedy_sample_fwd", "prefill_fwd"]
    assert sorted(contract.kernels) == ["decode", "greedy_sample", "prefill"]
    assert set(contract.kernels) <= {name.removesuffix("_fwd") for name in loaded.functions}
    contract.validate_kernels(contract, loaded, model)


def test_loaded_kernel_signatures_match_contract_arg_counts() -> None:
    contract = get_contract("qwen3", "14b")
    loaded = contract.load_kernels()

    for stage_name, stage in contract.kernels.items():
        kernel_fn = loaded.functions[f"{stage_name}_fwd"]
        kernel_params = tuple(inspect.signature(kernel_fn._func).parameters)
        assert len(kernel_params) == len(stage.args)


def test_contract_args_match_public_host_signatures() -> None:
    contract = get_contract("qwen3", "14b")

    for stage in contract.kernels.values():
        host_params = tuple(inspect.signature(stage.host_jit_fn._func).parameters)
        assert tuple(arg.name for arg in stage.args) == host_params

    for stage_name, module_name in (("prefill", "prefill_fwd.py"), ("decode", "decode_fwd.py")):
        source = (_REPO_ROOT / "models" / "qwen3_14b" / module_name).read_text()
        tree = ast.parse(source)
        serving_host = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == f"qwen3_{stage_name}_host"
        )
        assert any(ast.unparse(decorator) == "pl.jit.host" for decorator in serving_host.decorator_list)
        assert tuple(arg.name for arg in contract.kernels[stage_name].args) == tuple(
            argument.arg for argument in serving_host.args.args
        )


def test_prefill_host_maps_arguments_to_loaded_kernel_order() -> None:
    contract_tree = ast.parse((_REPO_ROOT / "models" / "qwen3_14b" / "contract.py").read_text())
    prefill_source = (_REPO_ROOT / "models" / "qwen3_14b" / "prefill_fwd.py").read_text()
    prefill_tree = ast.parse(prefill_source)
    contract_host = next(
        node
        for node in contract_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "qwen3_prefill_host"
    )
    kernel = next(
        node for node in prefill_tree.body if isinstance(node, ast.FunctionDef) and node.name == "prefill_fwd"
    )
    serving_host = next(
        node
        for node in prefill_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "qwen3_prefill_host"
    )

    kernel_args = [argument.arg for argument in kernel.args.args]
    for host in (contract_host, serving_host):
        call = next(
            node.value
            for node in host.body
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
        )
        assert [arg.id for arg in call.args if isinstance(arg, ast.Name)] == kernel_args

    serving_annotations = dict(_ast_function_signature(serving_host))
    assert serving_annotations["rope_cos"] == "pl.Tensor[[D.rope_seq, HEAD_DIM], pl.FP32]"
    assert serving_annotations["rope_sin"] == "pl.Tensor[[D.rope_seq, HEAD_DIM], pl.FP32]"


def test_native_paged_attention_source_has_phase0_and_fixed_mixed_groups() -> None:
    source_path = _REPO_ROOT / "models" / "qwen3_14b" / "paged_attention_pypto.py"
    source = source_path.read_text()
    tree = ast.parse(source)

    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "paged_attention_pypto_swpipe" in functions
    assert "rope_qkv_pypto" not in functions
    assert "rope_tid" not in source
    attention_function = functions["paged_attention_pypto_swpipe"]
    attention_source = _ast_function_source(source, attention_function)
    assert "q2d[" in attention_source
    assert "key_cache_bsnd = pl.reshape(key_cache," in attention_source
    assert "value_cache_bsnd = pl.reshape(value_cache," in attention_source
    assert "key_cache_bsnd[" in attention_source
    assert "value_cache_bsnd[" in attention_source
    assert "with pl.spmd(" in attention_source
    assert "ATTN_SPMD_BLOCKS" in attention_source
    spmd_calls = [
        node
        for node in ast.walk(attention_function)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.spmd"
    ]
    assert len(spmd_calls) == 1
    spmd_keywords = {keyword.arg: keyword.value for keyword in spmd_calls[0].keywords}
    assert isinstance(spmd_keywords["sync_start"], ast.Constant)
    assert spmd_keywords["sync_start"].value is True
    assert isinstance(spmd_keywords["allow_early_resolve"], ast.Constant)
    assert spmd_keywords["allow_early_resolve"].value is True
    assert isinstance(spmd_keywords["deps"], (ast.List, ast.Tuple))
    assert {ast.unparse(dependency) for dependency in spmd_keywords["deps"].elts} == {
        "q_proj_tid",
        "k_proj_tid",
        "v_proj_tid",
        "rms_tid",
        "attn_out_seed_tid",
        "mlp_out_seed_tid",
        "scratch_ready_tid",
    }
    assert "pl.system.set_ffts(ffts_workspace)" in attention_source
    syncall_calls = [
        node
        for node in ast.walk(attention_function)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.system.syncall"
    ]
    assert len(syncall_calls) == 1
    syncall_keywords = {keyword.arg: keyword.value for keyword in syncall_calls[0].keywords}
    assert isinstance(syncall_keywords["core_type"], ast.Constant)
    assert syncall_keywords["core_type"].value == "mix"
    fence_calls = [
        node
        for node in ast.walk(attention_function)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.system.fence"
    ]
    assert len(fence_calls) == 1
    assert attention_source.index("pl.system.fence()") < attention_source.index(
        'pl.system.syncall(core_type="mix")'
    )
    assert "pl.system.sync_set(" in attention_source
    assert "pl.system.sync_wait(" in attention_source
    assert "block_table_2d = pl.reshape(block_table, [active_batch, max_blocks_per_seq])" in attention_source
    assert "pl.tensor.read(block_table_2d, [batch," in attention_source
    assert "base = batch * max_blocks_per_seq" not in attention_source


def test_decode_contract_uses_dynamic_batch() -> None:
    """decode serves any public batch >= 1 from one compiled program.

    Every host-visible batch axis is the same ``BATCH`` dim (prefill uses it too,
    so the two stages no longer spell the same concept differently), and
    ``limits["batch"]`` is the padded pipeline WIDTH -- one decode row window --
    not a required exact value and no longer decode's ceiling.
    """
    contract = get_contract("qwen3", "14b")
    decode_args = {arg.name: arg.shape for arg in contract.kernels["decode"].args}

    assert contract.limits["batch"] == 16
    assert decode_args["seq_lens"] == ("BATCH",)
    assert decode_args["slot_mapping"] == ("BATCH",)
    assert decode_args["out"] == ("BATCH", "VOCAB")
    assert decode_args["sampled_ids_in"] == ("BATCH", "SAMPLED_IDS_PAD")
    assert decode_args["sampled_ids"] == ("BATCH", "SAMPLED_IDS_PAD")
    assert decode_args["next_hidden"] == ("BATCH", "H")

    # prefill already used a dynamic batch axis; decode must now name it the same.
    prefill_args = {arg.name: arg.shape for arg in contract.kernels["prefill"].args}
    assert prefill_args["seq_lens"] == ("BATCH",)

    # Compile-time dummies stay sized at the padded width -- they bound buffer
    # capacity, not the runtime shape.
    compile_args = contract.kernels["decode"].compile_args_builder(
        _tiny_model_config(),
        _runtime_config(),
    )
    assert compile_args[6].shape == (16,)
    assert compile_args[-1].shape == (16, 8)


@pytest.mark.parametrize("batch", [1, 2, 8, 15, 16, 17, 32, 33, 100])
def test_decode_contract_accepts_any_batch(batch: int) -> None:
    runtime = _runtime_config()
    runtime.max_batch_size = batch
    contract = get_contract("qwen3", "14b")
    # Must not raise at ANY batch: decode_fwd runs a batch above the padded width
    # as ceil(batch / batch_pad) row windows, so limits["batch"] bounds one window,
    # not the public batch.
    contract.kernels["decode"].compile_args_builder(_tiny_model_config(), runtime)


@pytest.mark.parametrize("batch", [0, -1])
def test_decode_contract_rejects_empty_batch(batch: int) -> None:
    runtime = _runtime_config()
    runtime.max_batch_size = batch
    contract = get_contract("qwen3", "14b")
    with pytest.raises(ValueError, match="max_batch_size"):
        contract.kernels["decode"].compile_args_builder(_tiny_model_config(), runtime)


@pytest.mark.parametrize("batch", [17, 32])
def test_prefill_and_greedy_sample_stay_capped_at_pad(batch: int) -> None:
    """Only decode chunks. The other two stages still bound the batch by the pad."""
    runtime = _runtime_config()
    runtime.max_batch_size = batch
    contract = get_contract("qwen3", "14b")
    for stage in ("prefill", "greedy_sample"):
        with pytest.raises(ValueError, match="max_batch_size"):
            contract.kernels[stage].compile_args_builder(_tiny_model_config(), runtime)


def test_runtime_arg_builders_follow_host_order() -> None:
    contract = get_contract("qwen3", "14b")
    static = SimpleNamespace(
        decode_weights={
            "decode_input_rms_weight": "input_rms_weight",
            "decode_wq": "wq",
            "decode_wk": "wk",
            "decode_wv": "wv",
            "decode_q_norm_weight": "q_norm_weight",
            "decode_k_norm_weight": "k_norm_weight",
            "decode_wo": "wo",
            "decode_w_gate": "w_gate",
            "decode_w_up": "w_up",
            "decode_w_down": "w_down",
            "decode_post_rms_weight": "post_rms_weight",
        },
        rope_cos="rope_cos",
        rope_sin="rope_sin",
        final_norm_weight="final_norm_weight",
        padded_lm_head_weight="lm_head",
        padded_embed_weight="embed",
    )
    prefill_inputs = SimpleNamespace(
        token_ids="token_ids",
        seq_lens="seq_lens",
        chunk_lens="chunk_lens",
        chunk_offsets="chunk_offsets",
        block_table="block_table",
        slot_mapping="slot_mapping",
    )
    decode_inputs = SimpleNamespace(
        seq_lens="seq_lens",
        block_table="block_table",
        slot_mapping="slot_mapping",
        logits="logits",
        token_ids="token_ids",
    )

    prefill_args = contract.kernels["prefill"].runtime_args_builder(
        prefill_inputs,
        static,
        k_cache="k_cache",
        v_cache="v_cache",
        logits="logits",
    )
    decode_args = contract.kernels["decode"].runtime_args_builder(
        decode_inputs,
        static,
        k_cache="k_cache",
        v_cache="v_cache",
        sampled_ids_buffer="sampled_ids",
        next_hidden_buffer="next_hidden",
    )

    assert prefill_args == (
        "token_ids",
        "seq_lens",
        "chunk_lens",
        "chunk_offsets",
        "input_rms_weight",
        "wq",
        "wk",
        "wv",
        "q_norm_weight",
        "k_norm_weight",
        "rope_cos",
        "rope_sin",
        "block_table",
        "slot_mapping",
        "k_cache",
        "v_cache",
        "wo",
        "w_gate",
        "w_up",
        "w_down",
        "post_rms_weight",
        "final_norm_weight",
        "lm_head",
        "embed",
        "logits",
    )
    assert decode_args[:4] == ("input_rms_weight", "wq", "wk", "wv")
    assert decode_args[-5:] == ("logits", "embed", "token_ids", "sampled_ids", "next_hidden")


def test_prepare_weights_rejects_oversized_lm_head_vocab() -> None:
    contract = get_contract("qwen3", "14b")
    model = SimpleNamespace(
        lm_head=torch.zeros((5, 3)),
        embed_tokens=torch.zeros((4, 3)),
        layers=(),
        final_norm_weight=torch.ones(3),
    )

    with pytest.raises(ValueError, match=r"Model vocabulary size 5 exceeds"):
        contract.prepare_weights(model, lambda tensor: tensor, padded_vocab=4)


def test_prepare_weights_rejects_oversized_embedding_vocab() -> None:
    contract = get_contract("qwen3", "14b")
    model = SimpleNamespace(
        lm_head=torch.zeros((4, 3)),
        embed_tokens=torch.zeros((5, 3)),
        layers=(),
        final_norm_weight=torch.ones(3),
    )

    with pytest.raises(ValueError, match=r"Model embedding vocabulary size 5 exceeds"):
        contract.prepare_weights(model, lambda tensor: tensor, padded_vocab=4)


def test_prepare_weights_exports_stacked_decode_weights_once() -> None:
    contract = get_contract("qwen3", "14b")
    layer = SimpleNamespace(
        input_rms_weight=torch.ones(3),
        wq=torch.ones((3, 3)),
        wk=torch.ones((2, 3)),
        wv=torch.ones((2, 3)),
        q_norm_weight=torch.ones(2),
        k_norm_weight=torch.ones(2),
        wo=torch.ones((3, 3)),
        post_rms_weight=torch.ones(3),
        w_gate=torch.ones((4, 3)),
        w_up=torch.ones((4, 3)),
        w_down=torch.ones((3, 4)),
    )
    model = SimpleNamespace(
        lm_head=torch.zeros((4, 3)),
        embed_tokens=torch.zeros((4, 3)),
        layers=(layer,),
        final_norm_weight=torch.ones(3),
    )
    exported = []

    def export(tensor: torch.Tensor) -> torch.Tensor:
        exported.append(tensor)
        return tensor

    contract.prepare_weights(model, export, padded_vocab=5, release_layers=False)

    assert len(exported) == 14


def _qwen3_14b_model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=5120,
            intermediate_size=17408,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
        ),
        runtime=SimpleNamespace(
            max_batch_size=16,
            max_seq_len=4096,
            page_size=128,
            vocab_pad_multiple=512,
            total_kv_pages=16,
        ),
    )
