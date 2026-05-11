# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comparison case: Qwen3-14B end-to-end generation via the production LLMEngine.

This case drives the same code path as ``llm.examples.qwen3_14b_npu_generate``:
``LLMEngine`` + ``PyptoQwen14BExecutor``, which compiles the per-layer prefill
kernel and the fused multi-layer decode kernel and manages the paged KV cache.

The HF reference side is unchanged: a stock ``Qwen3ForCausalLM`` runs greedy
decoding on the same prompt with the same number of layers (40, the full
model -- the kernel is compiled for exactly that depth).

Run via::

    python -m llm.testing.hf_compare run qwen3_14b.e2e_fused \\
        -k hf_model_path=/data/linyifan/models/Qwen3-14B \\
        -k platform=a2a3 \\
        -k max_new_tokens=16 \\
        -k max_seq=512
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..base import ComparisonCase, InputSpec, OutputSelector, TensorSpec, Tolerance, register_case
from ..reference import CallableReference
from ..target import CallableTarget, _resolve_device_id
from ..weight_adapter import PassthroughAdapter

# Reuse the HF reference forward + prompt parsing verbatim from the per-layer
# e2e case so behaviour stays in sync.
from .qwen3_14b_e2e import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT_IDS,
    _hf_e2e_forward,
    _parse_prompt_ids,
)


# Qwen3-14B is hardcoded into the fused decode kernel and the bundled weight
# stacking path (see llm/core/pypto_executor.py). Smaller depths require a
# kernel rebuild and are not supported by this case.
NUM_LAYERS = 40


def _run_prefill_step(
    executor: Any,
    runtime_model: Any,
    *,
    request_id: str,
    prompt_token_ids: list[int],
    alloc: Any,
    device: torch.device,
) -> torch.Tensor:
    """Run one prefill step, return the final-token logits."""
    from llm.core.types import PrefillBatch

    prompt_len = len(prompt_token_ids)
    token_tensor = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).view(1, -1)
    embeddings = executor.lookup_embeddings(runtime_model, token_tensor.view(-1)).view(
        1, prompt_len, -1
    )
    seq_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)

    result = executor.run_prefill(
        runtime_model,
        PrefillBatch(
            request_ids=[request_id],
            token_ids=token_tensor,
            input_embeddings=embeddings,
            seq_lens=seq_lens,
            kv_allocations=[alloc],
        ),
    )
    return result.logits


def _run_decode_step(
    executor: Any,
    runtime_model: Any,
    kv_cache_manager: Any,
    *,
    request_id: str,
    next_token: int,
    cur_seq_len: int,
    alloc: Any,
    device: torch.device,
) -> torch.Tensor:
    """Run one decode step for a single batch=1 sequence, return logits."""
    from llm.core.types import DecodeBatch

    decode_token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
    decode_embeddings = executor.lookup_embeddings(runtime_model, decode_token_tensor)
    decode_seq_lens = torch.tensor([cur_seq_len], dtype=torch.int32, device=device)
    block_table = kv_cache_manager.block_table_for_batch_padded([alloc]).to(device)
    slot_mapping = kv_cache_manager.slot_mapping_for_batch([alloc]).to(device)

    result = executor.run_decode(
        runtime_model,
        DecodeBatch(
            request_ids=[request_id],
            token_ids=decode_token_tensor.unsqueeze(1),
            hidden_states=decode_embeddings,
            seq_lens=decode_seq_lens,
            kv_allocations=[alloc],
            block_table=block_table,
            slot_mapping=slot_mapping,
        ),
    )
    return result.logits


def _run_e2e_fused_target(
    inputs: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],  # noqa: ARG001 -- engine loads weights itself
    *,
    model_dir: str,
    max_new_tokens: int,
    max_seq: int,
    platform: str,
    device_id: int | str | None,
) -> dict[str, torch.Tensor]:
    """Drive `LLMEngine` + `PyptoQwen14BExecutor` for one greedy generation.

    Mirrors `LLMEngine.generate_batch` (llm/core/engine.py) but bypasses the
    tokenizer to feed pre-tokenized prompt IDs supplied by the HF-compare
    fixture, and captures the final decode logits for downstream comparison.
    """
    from llm.core import LLMEngine, RuntimeConfig
    from llm.core.kv_cache import KvCacheManager
    from llm.core.pypto_executor import PyptoQwen14BExecutor

    resolved_device_id = _resolve_device_id(device_id)

    kv_cache_manager = KvCacheManager()
    executor = PyptoQwen14BExecutor(
        kv_cache_manager,
        platform=platform,
        device_id=resolved_device_id,
    )
    engine = LLMEngine(kv_cache_manager=kv_cache_manager, executor=executor)

    model_id = "qwen3_14b"
    engine.init_model(
        model_id=model_id,
        model_dir=str(model_dir),
        model_format="huggingface",
        runtime_config=RuntimeConfig(
            page_size=256,
            max_batch_size=16,
            max_seq_len=max_seq,
            device="cpu",
            kv_dtype="bfloat16",
            weight_dtype="float32",
        ),
    )

    record = engine._models[model_id]  # noqa: SLF001 -- intentional engine introspection
    runtime_model = record.runtime_model
    device = runtime_model.runtime.device

    input_ids = inputs["input_ids"].to(torch.long)
    if input_ids.shape[0] != 1:
        raise ValueError(f"qwen3_14b.e2e_fused expects batch=1, got {input_ids.shape[0]}")
    prompt_token_ids: list[int] = input_ids[0].tolist()
    prompt_len = len(prompt_token_ids)
    if prompt_len + max_new_tokens - 1 > max_seq:
        raise ValueError(
            f"prompt_len + max_new_tokens - 1 = {prompt_len + max_new_tokens - 1} "
            f"exceeds max_seq={max_seq}"
        )

    request_id = "req-0"
    alloc = kv_cache_manager.allocate_for_prompt(model_id, request_id, prompt_len)
    generated: list[int] = []

    try:
        last_logits = _run_prefill_step(
            executor, runtime_model,
            request_id=request_id, prompt_token_ids=prompt_token_ids,
            alloc=alloc, device=device,
        )
        next_token = int(torch.argmax(last_logits[0]).item())
        generated.append(next_token)

        cur_seq_len = prompt_len
        for _ in range(1, max_new_tokens):
            kv_cache_manager.ensure_one_more_slot(alloc)
            cur_seq_len += 1
            last_logits = _run_decode_step(
                executor, runtime_model, kv_cache_manager,
                request_id=request_id, next_token=next_token,
                cur_seq_len=cur_seq_len, alloc=alloc, device=device,
            )
            next_token = int(torch.argmax(last_logits[0]).item())
            generated.append(next_token)
    finally:
        kv_cache_manager.free(alloc)

    return {
        "generated_ids": torch.tensor([generated], dtype=torch.int64),
        "last_logits": last_logits.float(),
    }


@register_case("qwen3_14b.e2e_fused")
def build(
    hf_model_path: str = DEFAULT_MODEL_PATH,
    prompt_ids: str = DEFAULT_PROMPT_IDS,
    platform: str = "a2a3",
    device_id: int | str | None = None,
    max_new_tokens: int = 16,
    max_seq: int = 512,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    hf_dtype: str = "fp32",
) -> ComparisonCase:
    device_id = int(device_id) if device_id is not None else None
    max_new_tokens = int(max_new_tokens)
    max_seq = int(max_seq)
    atol = float(atol)
    rtol = float(rtol)
    hf_dtype_t = torch.float32 if hf_dtype == "fp32" else torch.bfloat16

    prompt = _parse_prompt_ids(prompt_ids)
    if len(prompt) + max_new_tokens - 1 > max_seq:
        raise ValueError(
            f"prompt length {len(prompt)} with max_new_tokens={max_new_tokens} "
            f"exceeds max_seq={max_seq}"
        )

    prompt_tensor = torch.tensor(prompt, dtype=torch.int64).view(1, -1)
    input_spec = InputSpec(
        tensors={
            "input_ids": TensorSpec(
                tuple(prompt_tensor.shape),
                torch.int64,
                sampler=lambda s, d, g, t=prompt_tensor: t.clone().to(d),
            ),
        }
    )

    reference = CallableReference(
        name="hf.Qwen3ForCausalLM",
        fn=lambda inp, st: _hf_e2e_forward(
            inp, st,
            model_path=hf_model_path,
            num_layers=NUM_LAYERS,
            max_new_tokens=max_new_tokens,
            hf_dtype=hf_dtype_t,
        ),
    )
    target = CallableTarget(
        name=f"pypto.qwen3_14b_e2e_fused[{platform}]",
        fn=lambda inp, w: _run_e2e_fused_target(
            inp, w,
            model_dir=hf_model_path,
            max_new_tokens=max_new_tokens,
            max_seq=max_seq,
            platform=platform,
            device_id=device_id,
        ),
    )

    return ComparisonCase(
        name="qwen3_14b.e2e_fused",
        reference=reference,
        target=target,
        input_spec=input_spec,
        weight_adapter=PassthroughAdapter(),
        selectors=[
            OutputSelector(
                name="generated_ids",
                ref_key="generated_ids",
                tgt_key="generated_ids",
                cast_to=torch.float32,
            ),
            OutputSelector(
                name="last_logits",
                ref_key="last_logits",
                tgt_key="last_logits",
                cast_to=torch.float32,
            ),
        ],
        tolerance=Tolerance(atol=atol, rtol=rtol, pass_rate_threshold=1.0),
        hf_weights=hf_model_path,
    )
