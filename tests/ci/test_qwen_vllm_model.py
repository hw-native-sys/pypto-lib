# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contracts for Lib's Qwen/vLLM model adapter."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("qwen_model", ROOT / ".github/scripts/qwen-vllm-kernel/qwen_model.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.fixture
def config(monkeypatch):
    versions = {"vllm": "0.20.2", "vllm-ascend": "0.20.2rc1"}
    monkeypatch.setattr(adapter.importlib.metadata, "version", versions.__getitem__)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3", num_hidden_layers=40, hidden_size=5120, num_key_value_heads=8),
            dtype=torch.bfloat16, quantization=None, enable_sleep_mode=False, enforce_eager=True, max_model_len=4096,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False, async_scheduling=False, max_num_seqs=16),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None, lora_config=None, kv_transfer_config=None,
        additional_config={"simpler": {"artifact": "source-built-artifact"}},
    )


@pytest.mark.parametrize("capture", [False, True])
def test_config_accepts_the_two_execution_modes(config, capture):
    config.model_config.enforce_eager = not capture
    adapter.validate_config(config, capture=capture)


@pytest.mark.parametrize("fault", ["layers", "dtype", "quantization", "tp", "prefix", "chunking", "batch", "mode"])
def test_config_rejects_unsupported_execution(config, fault):
    if fault == "layers":
        config.model_config.hf_config.num_hidden_layers = 1
    elif fault == "dtype":
        config.model_config.dtype = torch.float32
    elif fault == "quantization":
        config.model_config.quantization = "int8"
    elif fault == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif fault == "prefix":
        config.cache_config.enable_prefix_caching = True
    elif fault == "chunking":
        config.scheduler_config.enable_chunked_prefill = True
    elif fault == "batch":
        config.scheduler_config.max_num_seqs = 1
    else:
        config.model_config.enforce_eager = False
    with pytest.raises(ValueError):
        adapter.validate_config(config, capture=False)


def test_layer_slices_and_flat_kernel_views_share_disjoint_storage(monkeypatch):
    for name, value in {"LAYERS": 2, "PAGE": 2, "HEADS": 1, "HEAD_DIM": 2, "MIN_CACHE_PAGES": 2}.items():
        monkeypatch.setattr(adapter, name, value)
    key, value = adapter.allocate_layer_caches(pages=3, device="cpu")
    assert key.shape == value.shape == (2, 3, 2, 1, 2)
    flat = key.view(-1, 2)
    per_layer = key[1].view(torch.int8).flatten().view(torch.bfloat16).view(3, 2, 1, 2)
    per_layer[2, 1, 0, 1] = 17
    assert flat[-1, 1].item() == 17
    assert key[0].count_nonzero() == 0 and value.count_nonzero() == 0
    assert per_layer.data_ptr() == key.data_ptr() + key.stride(0) * key.element_size()
    assert flat.data_ptr() == key.data_ptr() and value.data_ptr() != key.data_ptr()
    with pytest.raises(ValueError, match="at least"):
        adapter.allocate_layer_caches(pages=1, device="cpu")


def test_packing_preserves_native_projections_and_checkpoint_fp32_norms(tmp_path, monkeypatch):
    for name, value in {"LAYERS": 2, "HIDDEN": 4, "PADDED_VOCAB": 8}.items():
        monkeypatch.setattr(adapter, name, value)
    layers = []
    for _ in range(2):
        attention = SimpleNamespace(q_size=4, kv_size=2, qkv_proj=torch.nn.Linear(4, 8, bias=False))
        attention.o_proj = torch.nn.Linear(4, 4, bias=False)
        mlp = SimpleNamespace(gate_up_proj=torch.nn.Linear(4, 6, bias=False), down_proj=torch.nn.Linear(3, 4, bias=False))
        layers.append(SimpleNamespace(self_attn=attention, mlp=mlp))
    inner = SimpleNamespace(layers=layers, embed_tokens=torch.nn.Embedding(6, 4))
    model = SimpleNamespace(
        model=inner, lm_head=torch.nn.Linear(4, 6, bias=False), config=SimpleNamespace(vocab_size=6),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model=str(tmp_path))),
    )
    tensors = {"model.norm.weight": torch.full((4,), 1.00012345)}
    for index in range(2):
        for suffix, size in (
            ("input_layernorm.weight", 4), ("post_attention_layernorm.weight", 4),
            ("self_attn.q_norm.weight", 2), ("self_attn.k_norm.weight", 2),
        ):
            tensors[f"model.layers.{index}.{suffix}"] = torch.full((size,), 1.00012345 + index)
    save_file(tensors, str(tmp_path / "norms.safetensors"))
    index = {"weight_map": {name: "norms.safetensors" for name in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    inputs = torch.randn(3, 4)
    original = [layer.self_attn.qkv_proj(inputs).detach().clone() for layer in layers]
    projections = [layer.self_attn.qkv_proj.weight.detach().clone() for layer in layers]
    packed = adapter.pack_model_weights(model)
    for layer_index, layer in enumerate(layers):
        actual, bias = layer.self_attn.qkv_proj(inputs)
        assert bias is None
        torch.testing.assert_close(actual, original[layer_index])
        q = packed["wq"].view(2, 4, 4)[layer_index]
        torch.testing.assert_close(q, projections[layer_index][:4].T)
        assert layer.self_attn.qkv_proj.weight_0.data_ptr() == q.data_ptr()
    assert packed["input_rms_weight"].dtype == torch.float32
    assert packed["input_rms_weight"][0, 0].item() == tensors["model.layers.0.input_layernorm.weight"][0].item()
    assert packed["input_rms_weight"][0, 0].item() != tensors["model.layers.0.input_layernorm.weight"][0].bfloat16().item()
    assert model.lm_head.weight.data_ptr() == packed["lm_head_weight"].data_ptr()
    torch.testing.assert_close(packed["lm_head_weight"][6:], packed["lm_head_weight"][:1].expand(2, -1))
    assert packed["embed_weight"][6:].count_nonzero() == 0


def test_rope_tables_keep_split_half_order(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "HEAD_DIM", 4)
    monkeypatch.setattr(adapter, "MAX_SEQ_LEN", 3)
    (tmp_path / "config.json").write_text(json.dumps({"rope_theta": 100.0}))
    cos, sin = adapter.rope_tables(tmp_path)
    torch.testing.assert_close(cos[1], torch.tensor([1.0, 0.1, 1.0, 0.1]).cos())
    torch.testing.assert_close(sin[1], torch.tensor([1.0, 0.1, 1.0, 0.1]).sin())
    assert cos.is_contiguous() and sin.is_contiguous()
