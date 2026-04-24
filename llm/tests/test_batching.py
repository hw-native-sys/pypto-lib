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
    padded_batch_size,
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


def test_padded_batch_size_uses_multiples_of_16():
    assert padded_batch_size(1) == 16
    assert padded_batch_size(15) == 16
    assert padded_batch_size(16) == 16
    assert padded_batch_size(17) == 32
    assert padded_batch_size(32) == 32


def test_kv_cache_capacity_uses_padded_runtime_batch_size():
    model = _model(max_batch_size=1, max_seq_len=128, page_size=64)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)

    k_cache, _ = manager.materialize_decode_cache(model.config.model_id, 0)
    assert k_cache.shape[0] == 16 * 2 * model.config.num_key_value_heads * model.runtime.page_size


def test_prefill_inputs_are_padded_to_compiled_batch_and_flattened():
    model = _model(max_batch_size=15)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    executor = PyptoQwen14BExecutor(manager)
    allocations = [
        manager.allocate_for_prompt(model.config.model_id, f"req-{idx}", idx + 1)
        for idx in range(model.runtime.max_batch_size)
    ]
    seq_lens = torch.tensor(
        [idx + 1 for idx in range(model.runtime.max_batch_size)],
        dtype=torch.int32,
    )
    embeddings = torch.ones(
        model.runtime.max_batch_size,
        int(seq_lens.max().item()),
        model.config.hidden_size,
    )

    padded = executor._pad_prefill_inputs(
        model,
        PrefillBatch(
            request_ids=[alloc.request_id for alloc in allocations],
            token_ids=torch.zeros(model.runtime.max_batch_size, int(seq_lens.max().item()), dtype=torch.long),
            input_embeddings=embeddings,
            seq_lens=seq_lens,
            kv_allocations=allocations,
        ),
        compile_batch=16,
    )

    assert padded.actual_batch == 15
    assert padded.hidden.shape == (16, model.runtime.max_seq_len, model.config.hidden_size)
    assert padded.seq_lens.tolist() == list(range(1, 16)) + [0]
    assert padded.block_table.shape == (16 * 2,)
    assert padded.block_table[0].item() == allocations[0].page_ids[0]
    assert padded.slot_mapping.shape == (16 * model.runtime.max_seq_len,)
    assert padded.slot_mapping[-1].item() == -1


def test_decode_inputs_use_scratch_page_for_padding_lanes():
    model = _model(max_batch_size=1)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    executor = PyptoQwen14BExecutor(manager)
    alloc = manager.allocate_for_prompt(model.config.model_id, "req-0", 1)
    hidden_states = torch.ones(1, model.config.hidden_size)

    padded = executor._pad_decode_inputs(
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
        compile_batch=16,
    )

    assert padded.hidden.shape == (16, model.config.hidden_size)
    assert padded.seq_lens.tolist() == [1] * 16
    assert padded.pad_allocation is not None
    assert padded.slot_mapping[0].item() == manager.slot_mapping_for_request(alloc)
    assert padded.slot_mapping[1].item() == manager.slot_mapping_for_request(padded.pad_allocation, 0)
    assert padded.slot_mapping[1].item() != padded.slot_mapping[0].item()
    manager.free(padded.pad_allocation)


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
