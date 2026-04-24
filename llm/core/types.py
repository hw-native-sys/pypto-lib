# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from .tokenizer import TokenizerAdapter


BATCH_SIZE_ALIGNMENT = 16


def padded_batch_size(batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rounded = ((batch_size + BATCH_SIZE_ALIGNMENT - 1) // BATCH_SIZE_ALIGNMENT) * BATCH_SIZE_ALIGNMENT
    return max(BATCH_SIZE_ALIGNMENT, rounded)


@dataclass(frozen=True)
class GenerateConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    stream: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    torch_dtype: str


@dataclass(frozen=True)
class RuntimeConfig:
    page_size: int = 64
    max_batch_size: int = 1
    max_seq_len: int = 4096
    device: str = "cpu"
    kv_dtype: str = "float32"
    weight_dtype: str = "float32"
    total_kv_pages: int | None = None


@dataclass(frozen=True)
class LayerSpec:
    layer_idx: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


@dataclass
class LayerWeights:
    input_rms_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    wo: torch.Tensor
    post_rms_weight: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class RuntimeModel:
    config: ModelConfig
    runtime: RuntimeConfig
    embed_tokens: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head: torch.Tensor
    layers: list[LayerWeights]


@dataclass
class ModelRecord:
    config: ModelConfig
    runtime: RuntimeConfig
    tokenizer: TokenizerAdapter
    layer_specs: list[LayerSpec]
    runtime_model: RuntimeModel


@dataclass
class LoadedModel:
    model_id: str
    model_dir: str
    config: ModelConfig
    tokenizer: TokenizerAdapter
    layer_specs: list[LayerSpec]
    runtime_model: RuntimeModel


@dataclass
class SamplingParams:
    temperature: float
    top_p: float
    top_k: int | None = None


@dataclass
class RequestState:
    request_id: str
    model_id: str
    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams | None = None
    status: Literal["waiting", "prefill", "decode", "finished", "aborted", "error"] = "waiting"
    max_new_tokens: int = 0
    stop_strings: tuple[str, ...] = ()
    eos_token_id: int | None = None
    seq_len: int = 0
    num_prompt_tokens: int = 0
    kv_allocation: "KvAllocation | None" = None
    output_text: str = ""


@dataclass
class KvAllocation:
    request_id: str
    model_id: str
    page_ids: list[int]
    tokens_capacity: int
    tokens_used: int = 0


@dataclass
class PrefillBatch:
    request_ids: list[str]
    token_ids: torch.Tensor
    input_embeddings: torch.Tensor
    seq_lens: torch.Tensor
    kv_allocations: list[KvAllocation]


@dataclass
class PrefillResult:
    last_hidden: torch.Tensor
    logits: torch.Tensor


@dataclass
class DecodeBatch:
    request_ids: list[str]
    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    seq_lens: torch.Tensor
    kv_allocations: list[KvAllocation]
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class DecodeResult:
    hidden_states: torch.Tensor
    logits: torch.Tensor


@dataclass
class GenerateResult:
    text: str
    token_ids: list[int]
    finish_reason: str
