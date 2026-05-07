# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests: PyptoQwen14BExecutor l3_mode=True vs baseline.

Validates that:
  1. L3 gen_chunked decode (has_prefill=0) produces logits equivalent to the
     baseline per-layer loop.
  2. L3 gen_chunked combined prefill+decode (has_prefill=1) produces logits
     within tolerance of the baseline prefill path.  Note: the L3 path derives
     the first-token logit via the decode stream (last prompt token as query,
     standard decode semantics without causal self-attention at position N-1),
     so a small numerical difference vs the baseline is expected and acceptable.
  3. Both tests exercise the shared session() context manager path.

Architecture under test (l3_mode=True):
  - ONE host_orch (L3) per chunk containing both qwen3_prefill_layer (L2) and
    qwen3_decode_layer (L2), interleaved per layer within the chunk.

Runs on NPU; not part of CPU-only batching tests.
"""

from __future__ import annotations

import os

import pytest
import torch

from core.kv_cache import KvCacheManager
from core.pypto_executor import PyptoQwen14BExecutor
from core.types import (
    DecodeBatch,
    LayerWeights,
    ModelConfig,
    ModelRecord,
    PrefillBatch,
    RuntimeConfig,
    RuntimeModel,
)


_DEVICE_ID = int(os.environ.get("PYPTO_QWEN3_L3_DEVICE_ID", "10"))
_PLATFORM = os.environ.get("PYPTO_QWEN3_L3_PLATFORM", "a2a3")


# Qwen3-14B fixed shape required by PyptoQwen14BExecutor._validate_supported_shape.
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

# Test sizing (small to keep runtime modest).
NUM_LAYERS = 2
BATCH = 16
MAX_SEQ = 128
PAGE_SIZE = 64


def _make_layer_weights(seed: int) -> LayerWeights:
    g = torch.Generator().manual_seed(seed)
    scale_in = HIDDEN_SIZE ** -0.5
    scale_inter = INTERMEDIATE_SIZE ** -0.5

    def _kernel_w(rows: int, cols: int, scale: float, gen_seed_offset: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed + gen_seed_offset)
        return ((torch.rand(rows, cols, generator=gen) - 0.5) * scale).to(torch.bfloat16)

    return LayerWeights(
        input_rms_weight=(torch.rand(1, HIDDEN_SIZE, generator=g) * 0.5 + 0.5).float(),
        wq=_kernel_w(HIDDEN_SIZE, HIDDEN_SIZE, scale_in, 1),
        wk=_kernel_w(KV_HIDDEN, HIDDEN_SIZE, scale_in, 2),
        wv=_kernel_w(KV_HIDDEN, HIDDEN_SIZE, scale_in, 3),
        q_norm_weight=torch.ones(1, HEAD_DIM).float(),
        k_norm_weight=torch.ones(1, HEAD_DIM).float(),
        wo=_kernel_w(HIDDEN_SIZE, HIDDEN_SIZE, scale_in, 4),
        post_rms_weight=(torch.rand(1, HIDDEN_SIZE, generator=g) * 0.5 + 0.5).float(),
        w_gate=_kernel_w(INTERMEDIATE_SIZE, HIDDEN_SIZE, scale_in, 5),
        w_up=_kernel_w(INTERMEDIATE_SIZE, HIDDEN_SIZE, scale_in, 6),
        w_down=_kernel_w(HIDDEN_SIZE, INTERMEDIATE_SIZE, scale_inter, 7),
    )


def _make_runtime_model(model_id: str) -> RuntimeModel:
    config = ModelConfig(
        model_id=model_id,
        architecture="qwen3",
        vocab_size=512,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=MAX_SEQ,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        torch_dtype="bfloat16",
    )
    max_blocks_per_seq = (MAX_SEQ + PAGE_SIZE - 1) // PAGE_SIZE
    runtime = RuntimeConfig(
        page_size=PAGE_SIZE,
        max_batch_size=BATCH,
        max_seq_len=MAX_SEQ,
        device="cpu",
        kv_dtype="bfloat16",
        weight_dtype="bfloat16",
        total_kv_pages=BATCH * max_blocks_per_seq,
    )
    layers = [_make_layer_weights(seed=100 + i) for i in range(NUM_LAYERS)]
    return RuntimeModel(
        config=config,
        runtime=runtime,
        embed_tokens=torch.zeros(config.vocab_size, HIDDEN_SIZE),
        final_norm_weight=torch.ones(HIDDEN_SIZE).float(),
        lm_head=(
            torch.randn(config.vocab_size, HIDDEN_SIZE, generator=torch.Generator().manual_seed(7))
            * 0.02
        ).to(torch.bfloat16),
        layers=layers,
    )


def _make_decode_batch(model: RuntimeModel, manager: KvCacheManager, *, ctx_len: int = 32) -> DecodeBatch:
    g = torch.Generator().manual_seed(42)
    actual_batch = BATCH
    hidden_states = ((torch.rand(actual_batch, HIDDEN_SIZE, generator=g) - 0.5) * 0.5).to(torch.bfloat16)
    seq_lens = torch.full((actual_batch,), ctx_len, dtype=torch.int32)

    allocations = []
    for batch_idx in range(actual_batch):
        alloc = manager.allocate_for_prompt(model.config.model_id, f"req-{batch_idx}", ctx_len)
        alloc.tokens_used = ctx_len
        allocations.append(alloc)

    block_table = manager.block_table_for_batch(allocations)
    slot_mapping = manager.slot_mapping_for_batch(allocations)
    return DecodeBatch(
        request_ids=[alloc.request_id for alloc in allocations],
        token_ids=torch.zeros(actual_batch, 1, dtype=torch.long),
        hidden_states=hidden_states,
        seq_lens=seq_lens,
        kv_allocations=allocations,
        block_table=block_table,
        slot_mapping=slot_mapping,
    )


def _make_prefill_batch(model: RuntimeModel, manager: KvCacheManager, *, seq_len: int = 16) -> PrefillBatch:
    g = torch.Generator().manual_seed(99)
    actual_batch = BATCH
    # Input embeddings: [batch, max_seq, hidden] — only first seq_len tokens are valid.
    input_embeddings = ((torch.rand(actual_batch, MAX_SEQ, HIDDEN_SIZE, generator=g) - 0.5) * 0.5).to(
        torch.bfloat16
    )
    seq_lens = torch.full((actual_batch,), seq_len, dtype=torch.int32)
    token_ids = torch.zeros(actual_batch, MAX_SEQ, dtype=torch.long)

    allocations = []
    for batch_idx in range(actual_batch):
        alloc = manager.allocate_for_prompt(model.config.model_id, f"prefill-{batch_idx}", seq_len)
        allocations.append(alloc)

    return PrefillBatch(
        request_ids=[alloc.request_id for alloc in allocations],
        token_ids=token_ids,
        input_embeddings=input_embeddings,
        seq_lens=seq_lens,
        kv_allocations=allocations,
    )


def _make_executor_pair(model_id_base: str, *, l3_mode: bool) -> tuple:
    """Return (model, manager, executor) for either baseline or L3 mode."""
    model = _make_runtime_model(model_id_base)
    manager = KvCacheManager()
    manager.register_model(model.config.model_id, model.config, model.runtime)
    executor = PyptoQwen14BExecutor(
        manager, platform=_PLATFORM, device_id=_DEVICE_ID, l3_mode=l3_mode,
    )
    executor.register_model(
        model.config.model_id,
        ModelRecord(
            config=model.config, runtime=model.runtime,
            tokenizer=None, layer_specs=[], runtime_model=model,
        ),
    )
    return model, manager, executor


_SKIP = pytest.mark.skipif(
    os.environ.get("PYPTO_QWEN3_L3_RUN") != "1",
    reason="NPU e2e test; set PYPTO_QWEN3_L3_RUN=1 to enable",
)


@_SKIP
def test_l3_decode_logits_match_baseline():
    """L3 gen_chunked decode (has_prefill=0) logits must match baseline within tolerance."""
    baseline_model, baseline_manager, baseline_executor = _make_executor_pair("qwen3-baseline", l3_mode=False)
    l3_model, l3_manager, l3_executor = _make_executor_pair("qwen3-l3", l3_mode=True)

    for layer_b, layer_l in zip(baseline_model.layers, l3_model.layers):
        assert torch.equal(layer_b.wq, layer_l.wq), "weight mismatch — test setup error"

    baseline_batch = _make_decode_batch(baseline_model, baseline_manager)
    l3_batch = _make_decode_batch(l3_model, l3_manager)
    assert torch.equal(baseline_batch.hidden_states, l3_batch.hidden_states)

    with baseline_executor.session():
        baseline_result = baseline_executor.run_decode(baseline_model, baseline_batch)
    with l3_executor.session():
        l3_result = l3_executor.run_decode(l3_model, l3_batch)

    max_diff = (baseline_result.logits - l3_result.logits).abs().max().item()
    assert torch.allclose(
        baseline_result.logits, l3_result.logits, rtol=5e-2, atol=5e-2,
    ), f"L3 decode logits diverge from baseline: max diff = {max_diff}"


@_SKIP
def test_l3_prefill_logits_match_baseline():
    """L3 gen_chunked combined prefill+decode (has_prefill=1) logits vs baseline.

    The L3 path produces first-token logits via the decode stream (last prompt
    token as query, RoPE at position N-1, attends to KV[0..N-1]).  The baseline
    uses the prefill output at the last position (includes self-attention at N-1).
    A small numerical difference is expected; tolerance is relaxed to 1e-1.
    """
    baseline_model, baseline_manager, baseline_executor = _make_executor_pair(
        "qwen3-prefill-baseline", l3_mode=False
    )
    l3_model, l3_manager, l3_executor = _make_executor_pair("qwen3-prefill-l3", l3_mode=True)

    for layer_b, layer_l in zip(baseline_model.layers, l3_model.layers):
        assert torch.equal(layer_b.wq, layer_l.wq), "weight mismatch — test setup error"

    baseline_batch = _make_prefill_batch(baseline_model, baseline_manager)
    l3_batch = _make_prefill_batch(l3_model, l3_manager)
    assert torch.equal(baseline_batch.input_embeddings, l3_batch.input_embeddings)

    with baseline_executor.session():
        baseline_result = baseline_executor.run_prefill(baseline_model, baseline_batch)
    with l3_executor.session():
        l3_result = l3_executor.run_prefill(l3_model, l3_batch)

    max_diff = (baseline_result.logits - l3_result.logits).abs().max().item()
    assert torch.allclose(
        baseline_result.logits, l3_result.logits, rtol=1e-1, atol=1e-1,
    ), f"L3 combined prefill+decode logits diverge from baseline: max diff = {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
