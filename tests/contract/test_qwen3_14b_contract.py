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
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from contract.registry import find_contract_for_model_config, get_contract


_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_fused_attention_declares_real_output_first() -> None:
    source = _REPO_ROOT / "models" / "qwen3_14b" / "paged_attention_cce.py"
    tree = ast.parse(source.read_text())
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "paged_attention_rope_cce"
    )
    output_like = [
        arg.arg
        for arg in func.args.args
        if isinstance(arg.annotation, ast.Subscript)
        and isinstance(arg.annotation.value, ast.Attribute)
        and arg.annotation.value.attr in {"Out", "InOut"}
    ]

    assert output_like[0] == "out", (
        "single-result extern binds its return to the first Out/InOut parameter"
    )


def test_fused_attention_uses_standalone_rope_worker_count() -> None:
    decode_source = _REPO_ROOT / "models" / "qwen3_14b" / "decode_fwd.py"
    decode_tree = ast.parse(decode_source.read_text())
    rope_cores = next(
        node.value.value
        for node in decode_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ROPE_CORES"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
    )

    kernel_dir = (
        _REPO_ROOT
        / "models"
        / "qwen3_14b"
        / "kernels"
        / "paged_attention_cce"
        / "kernel"
    )
    fai_body = (kernel_dir / "fai_body.hpp").read_text()
    assert f"constexpr uint32_t kQwenRopeCores = {rope_cores};" in fai_body
    # Match the lane guard and the guarded call, but NOT the call's trailing
    # arguments: regenerating the RoPE body can change the parameter list (e.g.
    # adding a dynamic-dim scalar), and that is a legitimate change this test
    # should not block. The arg mapping itself is covered by the static_assert
    # on the function-pointer type in fai_body.hpp.
    guarded_call = re.search(
        r"uint32_t rope_lane = block_idx \* 2 \+ sub_block_idx;\s*"
        r"if \(rope_lane < kQwenRopeCores\) \{\s*"
        r"qwen_rope_gen::rope_qkv\(",
        fai_body,
        flags=re.DOTALL,
    )
    assert guarded_call is not None

    # Anchor on the hand-written provenance banner, not on a generated
    # `const int64_t vNN = 32;` line -- SSA constants are renumbered by every
    # regeneration, so pinning one makes any regen look like a real failure.
    generated_rope = (kernel_dir / "rope_qkv_generated.hpp").read_text()
    assert f"// ROPE_CORES: {rope_cores}" in generated_rope, (
        "update the ROPE_CORES provenance banner in rope_qkv_generated.hpp's "
        "hand-written preamble when regenerating the specialized RoPE body"
    )


def test_compile_arg_builders_follow_loaded_stage_specs() -> None:
    contract = get_contract("qwen3", "14b")
    loaded = contract.load_kernels()
    model_config = _tiny_model_config()
    runtime_config = _runtime_config()

    prefill_args = contract.kernels["prefill"].compile_args_builder(model_config, runtime_config)
    decode_args = contract.kernels["decode"].compile_args_builder(model_config, runtime_config)
    greedy_args = contract.kernels["greedy_sample"].compile_args_builder(model_config, runtime_config)

    assert len(prefill_args) == len(contract.kernels["prefill"].args)
    assert len(prefill_args) == len(inspect.signature(loaded.functions["prefill_fwd"]._func).parameters)
    assert prefill_args[0].shape == (32, 8)
    assert prefill_args[-1].shape == (16, 512)
    assert prefill_args[-1].dtype == torch.float32

    assert len(decode_args) == len(contract.kernels["decode"].args)
    assert len(decode_args) == len(inspect.signature(loaded.functions["decode_fwd"]._func).parameters)
    assert decode_args[0].shape == (2, 8)
    assert decode_args[-3].shape == (16, 8)
    assert decode_args[-2].shape == (16, 8)
    assert decode_args[-1].shape == (16, 8)

    assert [tuple(arg.shape) for arg in greedy_args] == [(16, 512), (16, 8)]
    assert len(greedy_args) == len(inspect.signature(loaded.functions["greedy_sample_fwd"]._func).parameters)


def _rope_qkv_function_body() -> str:
    path = (
        _REPO_ROOT / "models" / "qwen3_14b" / "kernels" / "paged_attention_cce"
        / "kernel" / "rope_qkv_generated.hpp"
    )
    src = path.read_text()
    start = src.index("static __aicore__ void rope_qkv(")
    depth, idx = 0, src.index("{", start)
    while True:
        if src[idx] == "{":
            depth += 1
        elif src[idx] == "}":
            depth -= 1
            if depth == 0:
                return src[start:idx + 1]
        idx += 1


_FLAG_RE = re.compile(r"(set_flag|wait_flag)\((\w+),\s*(\w+),\s*(\w+)\)")


def _flag_balance(text: str) -> tuple[Counter, Counter]:
    sets: Counter = Counter()
    waits: Counter = Counter()
    for kind, src_pipe, dst_pipe, event in _FLAG_RE.findall(text):
        (sets if kind == "set_flag" else waits)[(src_pipe, dst_pipe, event)] += 1
    return sets, waits


def test_generated_rope_every_feasible_path_is_sync_safe() -> None:
    """Every executable path through the guarded RoPE items must not deadlock.

    The generated body runs two guarded item blocks per pipeline iteration:
    ``if (item < NUM_KV_HEADS * batch) { ... }``. At the padded batch both
    guards are always true (max item id 127 < 128), so the skip path never runs
    today -- but a runtime batch < BATCH_PAD makes it live, and a skipped block
    owning a ``set_flag`` whose ``wait_flag`` still executes would hang the AIV.

    Model the feasible paths rather than asserting per-block balance: the two
    blocks handle items ``L`` and ``L + ROPE_CORES``, so the guard can only ever
    drop the *later* one. Executing block 1 without block 0 is unreachable, and
    a future codegen where block 0 legitimately hands a credit to block 1 must
    not be rejected. Simulate each reachable path in source order and require
    that no wait ever runs without an outstanding set, and that the epilogue
    drains every credit.
    """
    body = _rope_qkv_function_body()
    lines = body.split("\n")

    blocks: list[tuple[int, int]] = []
    for n, line in enumerate(lines):
        if not re.match(r"\s*if \(v\d+ < v\d+\) \{", line):
            continue
        depth = 0
        for end in range(n, len(lines)):
            depth += lines[end].count("{") - lines[end].count("}")
            if depth == 0 and end > n:
                blocks.append((n, end))
                break
    assert len(blocks) == 2, f"expected 2 guarded item blocks, found {len(blocks)}"

    def line_block(index: int) -> int | None:
        for b, (start, end) in enumerate(blocks):
            if start <= index <= end:
                return b
        return None

    # Reachable prefixes only: item L+ROPE_CORES cannot pass a guard that item L
    # failed, so {block 1 alone} is not a reachable state.
    for taken in ((), (0,), (0, 1)):
        credits: Counter = Counter()
        for index, line in enumerate(lines):
            owner = line_block(index)
            if owner is not None and owner not in taken:
                continue
            for kind, src_pipe, dst_pipe, event in _FLAG_RE.findall(line):
                key = (src_pipe, dst_pipe, event)
                if kind == "set_flag":
                    credits[key] += 1
                else:
                    assert credits[key] > 0, (
                        f"path {taken or '(no guarded block)'}: wait_flag{key} at "
                        f"generated line {index} has no outstanding set_flag -- "
                        f"this path would hang the AIV at a runtime batch < BATCH_PAD"
                    )
                    credits[key] -= 1
        outstanding = {k: v for k, v in credits.items() if v}
        assert not outstanding, (
            f"path {taken or '(no guarded block)'} leaves undrained sync credits "
            f"{outstanding}; the epilogue must consume exactly what the prologue set"
        )


def test_decode_contract_uses_dynamic_batch() -> None:
    """decode serves any public batch in [1, batch_pad] from one compiled program.

    Every host-visible batch axis is the same ``BATCH`` dim (prefill uses it too,
    so the two stages no longer spell the same concept differently), and
    ``limits["batch"]`` is the UPPER BOUND -- the padded pipeline width -- not a
    required exact value.
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


@pytest.mark.parametrize("batch", [1, 2, 8, 15, 16])
def test_decode_contract_accepts_batch_within_pad(batch: int) -> None:
    runtime = _runtime_config()
    runtime.max_batch_size = batch
    contract = get_contract("qwen3", "14b")
    # Must not raise: any batch up to the padded width is servable.
    contract.kernels["decode"].compile_args_builder(_tiny_model_config(), runtime)


@pytest.mark.parametrize("batch", [0, 17, 32])
def test_decode_contract_rejects_batch_outside_pad(batch: int) -> None:
    runtime = _runtime_config()
    runtime.max_batch_size = batch
    contract = get_contract("qwen3", "14b")
    with pytest.raises(ValueError, match="max_batch_size"):
        contract.kernels["decode"].compile_args_builder(_tiny_model_config(), runtime)


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
        hidden="hidden",
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

    assert prefill_args[:6] == ("hidden", "seq_lens", "chunk_lens", "chunk_offsets", "input_rms_weight", "wq")
    assert prefill_args[-4:] == ("post_rms_weight", "final_norm_weight", "lm_head", "logits")
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
