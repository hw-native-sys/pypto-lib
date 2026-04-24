# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
from collections.abc import Iterator

import torch

from .executor import ModelExecutor
from .kv_cache import KvCacheManager
from .model_loader import ModelLoader
from .sampler import Sampler
from .types import (
    DecodeBatch,
    GenerateConfig,
    GenerateResult,
    ModelRecord,
    PrefillBatch,
    RequestState,
    RuntimeConfig,
)


class LLMEngine:
    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        kv_cache_manager: KvCacheManager | None = None,
        executor: ModelExecutor | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        self._model_loader = model_loader or ModelLoader()
        self._kv_cache_manager = kv_cache_manager or KvCacheManager()
        self._executor = executor or ModelExecutor(self._kv_cache_manager)
        self._sampler = sampler or Sampler()
        self._models: dict[str, ModelRecord] = {}
        self._request_counter = itertools.count()

    def init_model(
        self,
        model_id: str,
        model_dir: str,
        runtime_config: RuntimeConfig | None = None,
        model_format: str | None = None,
        **loader_options: object,
    ) -> None:
        loaded = self._model_loader.load(
            model_id=model_id,
            model_dir=model_dir,
            runtime_config=runtime_config,
            model_format=model_format,
            **loader_options,
        )
        config = loaded.config
        runtime = loaded.runtime_model.runtime
        self._kv_cache_manager.register_model(model_id, config, runtime)
        self._models[model_id] = ModelRecord(
            config=config,
            runtime=runtime,
            tokenizer=loaded.tokenizer,
            layer_specs=loaded.layer_specs,
            runtime_model=loaded.runtime_model,
        )
        register_model = getattr(self._executor, "register_model", None)
        if callable(register_model):
            register_model(model_id, self._models[model_id])

    def generate(self, model_id: str, prompt: str, config: GenerateConfig | None = None) -> str | Iterator[str]:
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            return self._generate_stream(model_id, prompt, generate_config)
        return self._generate_result(model_id, prompt, generate_config).text

    def _generate_non_stream(self, model_id: str, prompt: str, config: GenerateConfig) -> str:
        return self._generate_result(model_id, prompt, config).text

    def generate_batch(
        self,
        model_id: str,
        prompts: list[str] | tuple[str, ...],
        config: GenerateConfig | None = None,
    ) -> list[GenerateResult]:
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            raise ValueError("generate_batch requires stream=False")
        if not prompts:
            return []
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} is not initialized.")
        record = self._models[model_id]
        if len(prompts) > record.runtime.max_batch_size:
            max_batch_size = record.runtime.max_batch_size
            raise ValueError(
                f"batch has {len(prompts)} prompts, but runtime max_batch_size is {max_batch_size}"
            )

        runtime_model = record.runtime_model
        tokenizer = record.tokenizer
        prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
        for token_ids in prompt_token_ids:
            if not token_ids and record.config.bos_token_id is not None:
                token_ids.append(record.config.bos_token_id)
            if not token_ids:
                raise ValueError("Prompt tokenization produced no tokens.")

        requests: list[RequestState] = []
        allocations = []
        try:
            for prompt, token_ids in zip(prompts, prompt_token_ids, strict=True):
                request_id = f"req-{next(self._request_counter)}"
                alloc = self._kv_cache_manager.allocate_for_prompt(model_id, request_id, len(token_ids))
                allocations.append(alloc)
                requests.append(
                    RequestState(
                        request_id=request_id,
                        model_id=model_id,
                        prompt=prompt,
                        prompt_token_ids=token_ids,
                        max_new_tokens=generate_config.max_new_tokens,
                        stop_strings=generate_config.stop,
                        eos_token_id=record.config.eos_token_id,
                        seq_len=len(token_ids),
                        num_prompt_tokens=len(token_ids),
                        kv_allocation=alloc,
                    )
                )

            max_prompt_len = max(len(token_ids) for token_ids in prompt_token_ids)
            token_tensor = torch.zeros(
                (len(prompt_token_ids), max_prompt_len),
                dtype=torch.long,
                device=runtime_model.runtime.device,
            )
            embeddings = torch.zeros(
                (len(prompt_token_ids), max_prompt_len, record.config.hidden_size),
                dtype=runtime_model.embed_tokens.dtype,
                device=runtime_model.runtime.device,
            )
            for batch_idx, token_ids in enumerate(prompt_token_ids):
                row_tokens = torch.tensor(token_ids, dtype=torch.long, device=runtime_model.runtime.device)
                token_tensor[batch_idx, : len(token_ids)] = row_tokens
                embeddings[batch_idx, : len(token_ids), :] = self._executor.lookup_embeddings(
                    runtime_model,
                    row_tokens,
                )

            prefill_result = self._executor.run_prefill(
                runtime_model,
                PrefillBatch(
                    request_ids=[request.request_id for request in requests],
                    token_ids=token_tensor,
                    input_embeddings=embeddings,
                    seq_lens=torch.tensor(
                        [len(token_ids) for token_ids in prompt_token_ids],
                        dtype=torch.int32,
                        device=runtime_model.runtime.device,
                    ),
                    kv_allocations=allocations,
                ),
            )

            sampling_params = self._sampler.from_generate_config(generate_config)
            current_tokens = []
            for batch_idx in range(len(requests)):
                current_tokens.append(
                    self._sampler.sample(
                        self._select_batch_row(prefill_result.logits, batch_idx),
                        sampling_params,
                    )
                )
            active_indices = list(range(len(requests)))
            finish_reasons = ["length"] * len(requests)

            for _ in range(generate_config.max_new_tokens):
                next_active: list[int] = []
                decode_tokens: list[int] = []
                for request_idx in active_indices:
                    request = requests[request_idx]
                    current_token = current_tokens[request_idx]
                    request.generated_token_ids.append(current_token)
                    request.output_text = tokenizer.decode(request.generated_token_ids)

                    if record.config.eos_token_id is not None and current_token == record.config.eos_token_id:
                        finish_reasons[request_idx] = "eos"
                        continue
                    if any(stop and request.output_text.endswith(stop) for stop in generate_config.stop):
                        finish_reasons[request_idx] = "stop"
                        continue
                    if len(request.generated_token_ids) >= generate_config.max_new_tokens:
                        finish_reasons[request_idx] = "length"
                        continue

                    alloc = request.kv_allocation
                    if alloc is None:
                        raise RuntimeError("Request is missing KV allocation.")
                    self._kv_cache_manager.ensure_one_more_slot(alloc)
                    request.seq_len += 1
                    next_active.append(request_idx)
                    decode_tokens.append(current_token)

                if not next_active:
                    break

                decode_token_tensor = torch.tensor(
                    decode_tokens,
                    dtype=torch.long,
                    device=runtime_model.runtime.device,
                )
                decode_embeddings = self._executor.lookup_embeddings(runtime_model, decode_token_tensor)
                active_allocations = []
                for idx in next_active:
                    alloc = requests[idx].kv_allocation
                    if alloc is None:
                        raise RuntimeError("Request is missing KV allocation.")
                    active_allocations.append(alloc)
                decode_result = self._executor.run_decode(
                    runtime_model,
                    DecodeBatch(
                        request_ids=[requests[idx].request_id for idx in next_active],
                        token_ids=decode_token_tensor.unsqueeze(1),
                        hidden_states=decode_embeddings,
                        seq_lens=torch.tensor(
                            [requests[idx].seq_len for idx in next_active],
                            dtype=torch.int32,
                            device=runtime_model.runtime.device,
                        ),
                        kv_allocations=active_allocations,
                        block_table=self._kv_cache_manager.block_table_for_batch(active_allocations).to(
                            runtime_model.runtime.device
                        ),
                        slot_mapping=self._kv_cache_manager.slot_mapping_for_batch(active_allocations).to(
                            runtime_model.runtime.device
                        ),
                    ),
                )
                for row_idx, request_idx in enumerate(next_active):
                    current_tokens[request_idx] = self._sampler.sample(
                        self._select_batch_row(decode_result.logits, row_idx),
                        sampling_params,
                    )
                active_indices = next_active
        finally:
            for alloc in allocations:
                self._kv_cache_manager.free(alloc)

        return [
            GenerateResult(
                text=request.output_text,
                token_ids=list(request.generated_token_ids),
                finish_reason=finish_reasons[request_idx],
            )
            for request_idx, request in enumerate(requests)
        ]

    def _generate_stream(self, model_id: str, prompt: str, config: GenerateConfig) -> Iterator[str]:
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} is not initialized.")
        record = self._models[model_id]
        runtime_model = record.runtime_model
        tokenizer = record.tokenizer
        prompt_token_ids = tokenizer.encode(prompt)
        if not prompt_token_ids and record.config.bos_token_id is not None:
            prompt_token_ids = [record.config.bos_token_id]
        if not prompt_token_ids:
            raise ValueError("Prompt tokenization produced no tokens.")

        request_id = f"req-{next(self._request_counter)}"
        alloc = self._kv_cache_manager.allocate_for_prompt(model_id, request_id, len(prompt_token_ids))
        request = RequestState(
            request_id=request_id,
            model_id=model_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=config.max_new_tokens,
            stop_strings=config.stop,
            eos_token_id=record.config.eos_token_id,
            seq_len=len(prompt_token_ids),
            num_prompt_tokens=len(prompt_token_ids),
            kv_allocation=alloc,
        )

        try:
            token_tensor = torch.tensor(prompt_token_ids, dtype=torch.long, device=runtime_model.runtime.device)
            embeddings = self._executor.lookup_embeddings(runtime_model, token_tensor).unsqueeze(0)
            prefill_result = self._executor.run_prefill(
                runtime_model,
                PrefillBatch(
                    request_ids=[request.request_id],
                    token_ids=token_tensor.unsqueeze(0),
                    input_embeddings=embeddings,
                    seq_lens=torch.tensor(
                        [len(prompt_token_ids)],
                        dtype=torch.int32,
                        device=runtime_model.runtime.device,
                    ),
                    kv_allocations=[alloc],
                ),
            )

            logits = self._select_batch_row(prefill_result.logits, 0)
            generated: list[int] = []
            emitted_text = ""
            sampling_params = self._sampler.from_generate_config(config)
            current_token = self._sampler.sample(logits, sampling_params)

            for _ in range(config.max_new_tokens):
                generated.append(current_token)
                text = tokenizer.decode(generated)
                delta = text[len(emitted_text) :]
                emitted_text = text
                if delta:
                    yield delta
                if self._should_stop(record, config, generated, emitted_text, current_token):
                    break

                self._kv_cache_manager.ensure_one_more_slot(alloc)
                request.seq_len += 1
                decode_token = torch.tensor([current_token], dtype=torch.long, device=runtime_model.runtime.device)
                decode_embeddings = self._executor.lookup_embeddings(runtime_model, decode_token)
                decode_result = self._executor.run_decode(
                    runtime_model,
                    DecodeBatch(
                        request_ids=[request.request_id],
                        token_ids=decode_token.unsqueeze(0),
                        hidden_states=decode_embeddings,
                        seq_lens=torch.tensor(
                            [request.seq_len],
                            dtype=torch.int32,
                            device=runtime_model.runtime.device,
                        ),
                        kv_allocations=[alloc],
                        block_table=self._kv_cache_manager.block_table_for_batch([alloc]).to(
                            runtime_model.runtime.device
                        ),
                        slot_mapping=self._kv_cache_manager.slot_mapping_for_batch([alloc]).to(
                            runtime_model.runtime.device
                        ),
                    ),
                )
                logits = self._select_batch_row(decode_result.logits, 0)
                current_token = self._sampler.sample(logits, sampling_params)
        finally:
            self._kv_cache_manager.free(alloc)

    def generate_result(self, model_id: str, prompt: str, config: GenerateConfig | None = None) -> GenerateResult:
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            raise ValueError("generate_result requires stream=False")
        return self._generate_result(model_id, prompt, generate_config)

    def _generate_result(self, model_id: str, prompt: str, config: GenerateConfig) -> GenerateResult:
        return self.generate_batch(model_id, [prompt], config)[0]

    @staticmethod
    def _select_batch_row(tensor: torch.Tensor, row_idx: int) -> torch.Tensor:
        return tensor[row_idx] if tensor.dim() > 1 else tensor

    @staticmethod
    def _should_stop(
        record: ModelRecord,
        config: GenerateConfig,
        generated: list[int],
        emitted_text: str,
        current_token: int,
    ) -> bool:
        if record.config.eos_token_id is not None and current_token == record.config.eos_token_id:
            return True
        if len(generated) >= config.max_new_tokens:
            return True
        return any(stop and emitted_text.endswith(stop) for stop in config.stop)
