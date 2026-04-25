# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

from core.engine import LLMEngine
from core.executor import ModelExecutor
from core.kv_cache import KvCacheManager
from core.pypto_executor import PyptoQwen14BExecutor
from core.types import (
    DecodeBatch,
    GenerateConfig,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    RuntimeConfig,
    RuntimeModel,
)


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [max(1, len(text))]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _model(
    max_batch_size: int,
    max_seq_len: int = 128,
    page_size: int = 64,
    eos_token_id: int | None = None,
) -> RuntimeModel:
    config = ModelConfig(
        model_id="test-model",
        architecture="qwen3",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=max_seq_len,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=None,
        eos_token_id=eos_token_id,
        pad_token_id=None,
        torch_dtype="float32",
    )
    runtime = RuntimeConfig(
        page_size=page_size,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device="cpu",
    )
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=torch.zeros(config.vocab_size, config.hidden_size),
        final_norm_weight=torch.ones(config.hidden_size),
        lm_head=torch.zeros(config.vocab_size, config.hidden_size),
        layers=[],
    )


def test_kv_cache_capacity_uses_actual_runtime_batch_size():
    model = _model(max_batch_size=1, max_seq_len=128, page_size=64)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)

    k_cache, _ = manager.materialize_decode_cache(model.config.model_id, 0)
    assert k_cache.shape[0] == 1 * 2 * model.config.num_key_value_heads * model.runtime.page_size


def test_prefill_inputs_use_actual_user_batch_without_padding_lanes():
    model = _model(max_batch_size=15)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    executor = PyptoQwen14BExecutor(manager)
    allocations = [
        manager.allocate_for_prompt(model.config.model_id, f"req-{idx}", idx + 1)
        for idx in range(2)
    ]
    seq_lens = torch.tensor(
        [idx + 1 for idx in range(len(allocations))],
        dtype=torch.int32,
    )
    embeddings = torch.ones(
        len(allocations),
        int(seq_lens.max().item()),
        model.config.hidden_size,
    )

    prepared = executor._prepare_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=[alloc.request_id for alloc in allocations],
            token_ids=torch.zeros(
                len(allocations),
                int(seq_lens.max().item()),
                dtype=torch.long,
            ),
            input_embeddings=embeddings,
            seq_lens=seq_lens,
            kv_allocations=allocations,
        ),
    )

    assert prepared.actual_batch == 2
    assert prepared.hidden.shape == (2, model.runtime.max_seq_len, model.config.hidden_size)
    assert prepared.seq_lens.tolist() == [1, 2]
    assert prepared.block_table.shape == (2 * 2,)
    assert prepared.block_table[0].item() == allocations[0].page_ids[0]
    assert prepared.slot_mapping.shape == (2 * model.runtime.max_seq_len,)
    assert prepared.slot_mapping[
        model.runtime.max_seq_len + 1
    ].item() == manager.slot_mapping_for_request(allocations[1], 1)
    assert prepared.slot_mapping[-1].item() == -1


def test_decode_inputs_use_actual_user_batch_without_padding_lanes():
    model = _model(max_batch_size=1)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    executor = PyptoQwen14BExecutor(manager)
    alloc = manager.allocate_for_prompt(model.config.model_id, "req-0", 1)
    hidden_states = torch.ones(1, model.config.hidden_size)

    prepared = executor._prepare_decode_inputs(
        model,
        DecodeBatch(
            request_ids=[alloc.request_id],
            token_ids=torch.zeros(1, 1, dtype=torch.long),
            hidden_states=hidden_states,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            kv_allocations=[alloc],
            block_table=manager.block_table_for_batch([alloc]),
            slot_mapping=manager.slot_mapping_for_batch([alloc]),
        ),
    )

    assert prepared.actual_batch == 1
    assert prepared.hidden.shape == (1, model.config.hidden_size)
    assert prepared.seq_lens.tolist() == [1]
    assert prepared.block_table.shape == (2,)
    assert prepared.block_table[0].item() == alloc.page_ids[0]
    assert prepared.slot_mapping.tolist() == [manager.slot_mapping_for_request(alloc)]


def test_engine_generate_batch_uses_batched_executor_results():
    model = _model(max_batch_size=2, eos_token_id=0)
    manager = KvCacheManager()
    executor = ModelExecutor(manager)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    results = engine.generate_batch(
        model.config.model_id,
        ["a", "abcd"],
        GenerateConfig(max_new_tokens=2, temperature=0.0),
    )

    assert [result.token_ids for result in results] == [[0], [0]]
    assert [result.finish_reason for result in results] == ["eos", "eos"]
