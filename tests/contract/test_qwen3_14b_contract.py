# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from contract.registry import find_contract_for_model_config, get_contract


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
    assert set(contract.kernels) <= {
        name.removesuffix("_fwd")
        for name in loaded.functions
    }
    contract.validate_kernels(contract, loaded, model)


def test_loaded_kernel_signatures_match_contract_arg_counts() -> None:
    contract = get_contract("qwen3", "14b")
    loaded = contract.load_kernels()

    for stage_name, stage in contract.kernels.items():
        kernel_fn = loaded.functions[f"{stage_name}_fwd"]
        kernel_params = tuple(inspect.signature(kernel_fn._func).parameters)
        assert len(kernel_params) == len(stage.args)


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
