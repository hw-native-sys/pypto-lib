# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B weights, KV storage and vLLM configuration for the Kernel smoke."""

import importlib.metadata
import json
from pathlib import Path

import torch

from models.qwen3_14b.constants import QWEN3_14B as M, QWEN3_14B_TILING as T

BATCH, LAYERS = M.batch_pad, M.num_layers
HEADS, HEAD_DIM, HIDDEN = M.num_kv_heads, M.head_dim, M.hidden
PAGE, MAX_SEQ_LEN, PADDED_VOCAB = T.seq_tile, M.max_seq, M.vocab
MIN_CACHE_PAGES = BATCH * ((3338 + 128 + PAGE - 1) // PAGE)


def validate_config(config, *, capture):
    for package, expected in (("vllm", "0.20.2"), ("vllm-ascend", "0.20.2rc1")):
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"{package} differs from the pinned Kernel smoke version")
    model = config.model_config
    expected = {"model_type": "qwen3", "num_hidden_layers": LAYERS, "hidden_size": HIDDEN, "num_key_value_heads": HEADS}
    if any(getattr(model.hf_config, name, None) != value for name, value in expected.items()):
        raise ValueError("Kernel smoke requires the Qwen3-14B geometry")
    parallel = config.parallel_config
    if any(getattr(parallel, name) != 1 for name in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
    )):
        raise ValueError("Kernel smoke requires single-device execution")
    if model.dtype != torch.bfloat16 or model.quantization is not None or config.speculative_config is not None:
        raise ValueError("Kernel smoke requires BF16 without quantization or speculative decoding")
    if config.lora_config is not None or model.enable_sleep_mode:
        raise ValueError("Kernel smoke requires stable weights and context")
    if model.enforce_eager == capture or model.max_model_len != MAX_SEQ_LEN:
        raise ValueError("Kernel smoke mode or context differs from the configured worker")
    if config.cache_config.enable_prefix_caching or config.kv_transfer_config is not None:
        raise ValueError("Kernel smoke requires private local KV pages")
    scheduler = config.scheduler_config
    if scheduler.enable_chunked_prefill or scheduler.async_scheduling or scheduler.max_num_seqs != BATCH:
        raise ValueError("Kernel smoke requires synchronous, unchunked B16 requests")
    if (config.additional_config or {}).get("simpler", {}).get("artifact") is None:
        raise ValueError("additional_config.simpler.artifact is required")


def allocate_layer_caches(*, pages, device):
    if pages < MIN_CACHE_PAGES:
        raise ValueError(f"Kernel smoke requires at least {MIN_CACHE_PAGES} KV pages")
    shape = (LAYERS, pages, PAGE, HEADS, HEAD_DIM)
    return tuple(torch.zeros(shape, dtype=torch.bfloat16, device=device) for _ in range(2))


def rope_tables(model_dir):
    config = json.loads((Path(model_dir) / "config.json").read_text())
    theta = float(config.get("rope_theta", 1_000_000.0))
    half = HEAD_DIM // 2
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32).unsqueeze(1)
    inv_freq = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
    angles = positions * inv_freq.unsqueeze(0)
    cos, sin = angles.cos(), angles.sin()
    return torch.cat((cos, cos), dim=1).contiguous(), torch.cat((sin, sin), dim=1).contiguous()


class PackedLinear(torch.nn.Module):
    def __init__(self, *weights):
        super().__init__()
        for index, weight in enumerate(weights):
            self.register_buffer(f"weight_{index}", weight, persistent=False)

    def forward(self, hidden):
        parts = [torch.matmul(hidden, weight) for weight in self.buffers()]
        return (parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)), None


@torch.no_grad()
def pack_model_weights(model):
    from safetensors import safe_open

    layers = model.model.layers
    if len(layers) != LAYERS:
        raise ValueError("native model layer count differs from the Lib decode ABI")
    packed = {}
    attention = layers[0].self_attn
    intermediate = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    groups = (
        ("self_attn", "qkv_proj", ("wq", "wk", "wv"), (attention.q_size, attention.kv_size, attention.kv_size)),
        ("self_attn", "o_proj", ("wo",), (HIDDEN,)),
        ("mlp", "gate_up_proj", ("w_gate", "w_up"), (intermediate, intermediate)),
        ("mlp", "down_proj", ("w_down",), (HIDDEN,)),
    )
    for parent_name, projection_name, names, sizes in groups:
        for part, name in enumerate(names):
            start, count = sum(sizes[:part]), sizes[part]
            weight = getattr(getattr(layers[0], parent_name), projection_name).weight
            packed[name] = weight.new_empty((LAYERS * weight.shape[1], count))
            for index, layer in enumerate(layers):
                weight = getattr(getattr(layer, parent_name), projection_name).weight
                packed[name].view(LAYERS, -1, count)[index].copy_(weight[start : start + count].T)
            del weight
        for index, layer in enumerate(layers):
            views = [packed[name].view(LAYERS, -1, count)[index] for name, count in zip(names, sizes)]
            setattr(getattr(layer, parent_name), projection_name, PackedLinear(*views))
        if packed[names[0]].device.type == "npu":
            torch.npu.synchronize()
            torch.npu.empty_cache()
    for name, module, repeat_padding in (
        ("embed_weight", model.model.embed_tokens, False),
        ("lm_head_weight", model.lm_head, True),
    ):
        value = module.weight[: model.config.vocab_size]
        missing = PADDED_VOCAB - value.shape[0]
        if missing < 0:
            raise ValueError("vocabulary exceeds the decode ABI")
        padding = value[:1].expand(missing, -1) if repeat_padding else value.new_zeros((missing, HIDDEN))
        packed[name] = torch.cat((value, padding)).contiguous()
        module.weight = torch.nn.Parameter(packed[name], requires_grad=False)
    device = value.device
    del value, padding
    root = Path(model.vllm_config.model_config.model)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def read_tensor(name):
        with safe_open(root / weight_map[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    for name, suffix in (
        ("input_rms_weight", "input_layernorm.weight"),
        ("post_rms_weight", "post_attention_layernorm.weight"),
        ("q_norm_weight", "self_attn.q_norm.weight"),
        ("k_norm_weight", "self_attn.k_norm.weight"),
    ):
        tensors = [read_tensor(f"model.layers.{index}.{suffix}").view(1, -1).float() for index in range(LAYERS)]
        packed[name] = torch.cat(tensors, dim=0).contiguous().to(device)
    packed["final_norm_weight"] = read_tensor("model.norm.weight").view(1, -1).float().to(device)
    return packed
