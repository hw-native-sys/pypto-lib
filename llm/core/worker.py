# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field

import torch

from .types import (
    DecodeBatch,
    KvAllocation,
    PrefillBatch,
    RuntimeConfig,
    SamplingParams,
    StepOutput,
    WorkerCommand,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    model_id: str = ""
    model_dir: str = ""
    platform: str = "a2a3"
    device_id: int = 0
    runtime_config: RuntimeConfig | None = None
    executor_cls: str = "PyptoQwen14BExecutor"
    executor_kwargs: dict = field(default_factory=dict)


class WorkerProcess:
    """Dedicated process that owns a single NPU device and executes model inference.

    Architecture (single-card, extensible to multi-card by spawning multiple workers):
      Main Process  --[input_queue]--> WorkerProcess --[output_queue]--> Main Process

    The worker loads the model fresh in its own process, avoiding cross-process
    tensor transfer and NPU thread-affinity issues.
    """

    def __init__(
        self,
        config: WorkerConfig,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
    ):
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.engine = None
        self.executor = None
        self.kv_cache_manager = None
        self.sampler = None
        self.model_record = None
        self._allocations: dict[str, KvAllocation] = {}  # persists across steps

    def init_device_and_model(self) -> None:
        from .engine import LLMEngine
        from .kv_cache import KvCacheManager
        from .sampler import Sampler

        logger.info(
            f"Worker initializing: platform={self.config.platform}, "
            f"device={self.config.device_id}"
        )

        self.kv_cache_manager = KvCacheManager()
        self.sampler = Sampler()

        executor_cls = self._resolve_executor_cls()
        self.executor = executor_cls(
            self.kv_cache_manager,
            platform=self.config.platform,
            device_id=self.config.device_id,
            **self.config.executor_kwargs,
        )

        self.engine = LLMEngine(
            kv_cache_manager=self.kv_cache_manager,
            executor=self.executor,
            sampler=self.sampler,
        )
        self.engine.init_model(
            self.config.model_id,
            self.config.model_dir,
            runtime_config=self.config.runtime_config,
        )
        self.model_record = self.engine._models[self.config.model_id]
        logger.info("Worker model loaded and ready")

    def _resolve_executor_cls(self):
        if self.config.executor_cls == "PyptoQwen14BExecutor":
            from .pypto_executor import PyptoQwen14BExecutor
            return PyptoQwen14BExecutor
        from .executor import ModelExecutor
        return ModelExecutor

    def busy_loop(self) -> None:
        logger.info("Worker entering busy loop")
        while True:
            try:
                cmd: WorkerCommand = self.input_queue.get()
            except Exception:
                break

            if cmd.type == "shutdown":
                logger.info("Worker received shutdown command")
                break
            elif cmd.type == "step":
                # Free allocations for finished requests first
                if cmd.finished_request_ids:
                    for req_id in cmd.finished_request_ids:
                        self.free_allocation(req_id)

                try:
                    result = self._execute_step(cmd.scheduler_output)
                    self.output_queue.put(result)
                except Exception as e:
                    logger.error(f"Worker step failed: {e}", exc_info=True)
                    self.output_queue.put(StepOutput(new_tokens={}, error=str(e)))
            else:
                logger.warning(f"Worker received unknown command: {cmd.type}")

        logger.info("Worker exiting")

    def _execute_step(self, scheduler_output) -> StepOutput:
        """Execute one batch step (may contain prefill + decode requests)."""
        runtime_model = self.model_record.runtime_model
        new_tokens: dict[str, int] = {}

        prefill_requests = [
            sr for sr in scheduler_output.scheduled_requests if sr.is_prefill
        ]
        decode_requests = [
            sr for sr in scheduler_output.scheduled_requests if not sr.is_prefill
        ]

        with self.executor.session():
            if prefill_requests:
                self._batch_prefill(prefill_requests, runtime_model, new_tokens)
            if decode_requests:
                self._batch_decode(decode_requests, runtime_model, new_tokens)

        return StepOutput(new_tokens=new_tokens)

    def _batch_prefill(
        self, scheduled: list, runtime_model, new_tokens: dict[str, int]
    ) -> None:
        device = runtime_model.runtime.device
        batch_size = len(scheduled)

        token_ids_list = []
        seq_lens = []
        allocations = []

        for sr in scheduled:
            request = sr.request
            prompt_tokens = request.prompt_token_ids
            token_ids_list.append(prompt_tokens)
            seq_lens.append(len(prompt_tokens))
            alloc = self._get_or_create_allocation(request, len(prompt_tokens))
            allocations.append(alloc)

        max_seq = max(seq_lens)
        token_tensor = torch.zeros((batch_size, max_seq), dtype=torch.long, device=device)
        embeddings = torch.zeros(
            (batch_size, max_seq, self.model_record.config.hidden_size),
            dtype=runtime_model.embed_tokens.dtype,
            device=device,
        )

        for i, tokens in enumerate(token_ids_list):
            row = torch.tensor(tokens, dtype=torch.long, device=device)
            token_tensor[i, : len(tokens)] = row
            embeddings[i, : len(tokens), :] = self.executor.lookup_embeddings(
                runtime_model, row
            )

        prefill_result = self.executor.run_prefill(
            runtime_model,
            PrefillBatch(
                request_ids=[sr.request.request_id for sr in scheduled],
                token_ids=token_tensor,
                input_embeddings=embeddings,
                seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
                kv_allocations=allocations,
            ),
        )

        for i, sr in enumerate(scheduled):
            request = sr.request
            will_be_computed = request.num_computed_tokens + sr.num_new_tokens
            if will_be_computed >= request.num_prompt_tokens:
                logits = (
                    prefill_result.logits[i]
                    if prefill_result.logits.dim() > 1
                    else prefill_result.logits
                )
                params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                )
                token_id = self.sampler.sample(logits, params)
                new_tokens[request.request_id] = token_id


    def _batch_decode(
        self, scheduled: list, runtime_model, new_tokens: dict[str, int]
    ) -> None:
        device = runtime_model.runtime.device

        decode_tokens = []
        allocations = []
        seq_lens = []

        for sr in scheduled:
            request = sr.request
            last_token = (
                request.output_token_ids[-1]
                if request.output_token_ids
                else request.prompt_token_ids[-1]
            )
            decode_tokens.append(last_token)

            alloc = self._get_or_create_allocation(request, request.num_prompt_tokens)
            self.kv_cache_manager.ensure_one_more_slot(alloc)
            allocations.append(alloc)
            seq_lens.append(request.num_tokens)

        decode_token_tensor = torch.tensor(decode_tokens, dtype=torch.long, device=device)
        decode_embeddings = self.executor.lookup_embeddings(runtime_model, decode_token_tensor)

        block_table = self.kv_cache_manager.block_table_for_batch(allocations).to(device)
        slot_mapping = self.kv_cache_manager.slot_mapping_for_batch(allocations).to(device)

        decode_result = self.executor.run_decode(
            runtime_model,
            DecodeBatch(
                request_ids=[sr.request.request_id for sr in scheduled],
                token_ids=decode_token_tensor.unsqueeze(1),
                hidden_states=decode_embeddings,
                seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
                kv_allocations=allocations,
                block_table=block_table,
                slot_mapping=slot_mapping,
            ),
        )

        for i, sr in enumerate(scheduled):
            request = sr.request
            logits = (
                decode_result.logits[i]
                if decode_result.logits.dim() > 1
                else decode_result.logits
            )
            params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            token_id = self.sampler.sample(logits, params)
            new_tokens[request.request_id] = token_id

    def _get_or_create_allocation(self, request, prompt_len: int) -> KvAllocation:
        """Get existing KV allocation or create one. Persists across steps."""
        req_id = request.request_id
        if req_id in self._allocations:
            return self._allocations[req_id]
        alloc = self.kv_cache_manager.allocate_for_prompt(
            self.config.model_id, req_id, prompt_len
        )
        self._allocations[req_id] = alloc
        return alloc

    def free_allocation(self, request_id: str) -> None:
        """Free KV allocation when request finishes."""
        alloc = self._allocations.pop(request_id, None)
        if alloc is not None:
            self.kv_cache_manager.free(alloc)


def _worker_entry(
    config: WorkerConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ready_event,
):
    """Entry point for the worker subprocess."""
    worker = WorkerProcess(config, input_queue, output_queue)
    try:
        worker.init_device_and_model()
        ready_event.set()
        worker.busy_loop()
    except Exception as e:
        logger.error(f"Worker process failed: {e}", exc_info=True)
        ready_event.set()


def spawn_worker(config: WorkerConfig) -> tuple[mp.Process, mp.Queue, mp.Queue, mp.Event]:
    """Spawn a worker process and return (process, input_queue, output_queue, ready_event).

    Extensible to multi-card: call spawn_worker() per device_id.
    """
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    ready_event = ctx.Event()

    process = ctx.Process(
        target=_worker_entry,
        args=(config, input_queue, output_queue, ready_event),
        daemon=False,  # non-daemon for Ascend compatibility (like vllm-ascend)
    )
    process.start()
    return process, input_queue, output_queue, ready_event
