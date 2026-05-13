# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from .block_pool import BlockPool
from .scheduler import Request, RequestStatus, Scheduler, SchedulerConfig, SchedulerOutput
from .types import StepOutput, WorkerCommand
from .worker import WorkerConfig, WorkerProcess, spawn_worker

logger = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    max_num_running_reqs: int = 32
    max_num_scheduled_tokens: int = 4096
    long_prefill_token_threshold: int = 2048
    max_seq_len: int = 4096
    engine_loop_interval: float = 0.001
    block_size: int = 256
    num_blocks: int | None = None


@dataclass
class _RequestContext:
    request: Request
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


@dataclass
class TokenOutput:
    token_id: int | None = None
    text: str = ""
    finished: bool = False
    finish_reason: str = ""


class AsyncLLMEngine:
    """Async engine with multiprocess worker for NPU execution.

    Architecture:
      Main process: scheduler + API serving + output processing
      Worker process: NPU device + model execution (single-card, extensible to multi-card)
    """

    def __init__(
        self,
        worker_config: WorkerConfig,
        serving_config: ServingConfig | None = None,
        tokenizer=None,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None,
        in_process: bool = False,
    ) -> None:
        self.worker_config = worker_config
        self.config = serving_config or ServingConfig()
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self._in_process = in_process

        num_blocks = self.config.num_blocks or (
            self.config.max_num_running_reqs
            * (self.config.max_seq_len // self.config.block_size + 1)
        )
        self.block_pool = BlockPool(
            num_blocks=num_blocks, block_size=self.config.block_size
        )

        scheduler_config = SchedulerConfig(
            max_num_running_reqs=self.config.max_num_running_reqs,
            max_num_scheduled_tokens=self.config.max_num_scheduled_tokens,
            long_prefill_token_threshold=self.config.long_prefill_token_threshold,
            max_seq_len=self.config.max_seq_len,
        )
        self.scheduler = Scheduler(config=scheduler_config, block_pool=self.block_pool)

        self._request_contexts: dict[str, _RequestContext] = {}
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._request_counter = 0
        self._pending_free_ids: list[str] = []  # request IDs to free in worker

        # Worker process state
        self._worker_process = None
        self._input_queue = None
        self._output_queue = None

    async def start(self) -> None:
        """Start worker process and engine loop."""
        import queue
        import threading

        if self._in_process:
            # In-process mode: use threading queues and run worker in a thread.
            # Useful for testing and CPU-only mode.
            self._input_queue = queue.Queue()
            self._output_queue = queue.Queue()
            worker = WorkerProcess(
                self.worker_config, self._input_queue, self._output_queue
            )
            worker.init_device_and_model()
            self._worker_thread = threading.Thread(
                target=worker.busy_loop, daemon=True
            )
            self._worker_thread.start()
            logger.info("Worker started in-process (thread mode)")
        else:
            process, input_q, output_q, ready_event = spawn_worker(self.worker_config)
            self._worker_process = process
            self._input_queue = input_q
            self._output_queue = output_q

            logger.info("Waiting for worker to initialize model...")
            await asyncio.to_thread(ready_event.wait, timeout=600)
            if not ready_event.is_set():
                raise RuntimeError("Worker failed to initialize within timeout")
            logger.info("Worker ready")

        self._running = True
        self._loop_task = asyncio.create_task(self._engine_loop())
        logger.info("AsyncLLMEngine started")

    async def stop(self) -> None:
        """Stop engine loop and worker process."""
        self._running = False
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

        if self._input_queue is not None:
            self._input_queue.put(WorkerCommand(type="shutdown"))

        if self._in_process:
            if hasattr(self, "_worker_thread") and self._worker_thread is not None:
                self._worker_thread.join(timeout=10)
                self._worker_thread = None
        else:
            if self._worker_process is not None:
                self._worker_process.join(timeout=30)
                if self._worker_process.is_alive():
                    self._worker_process.terminate()
                self._worker_process = None
        logger.info("AsyncLLMEngine stopped")

    def generate_request_id(self) -> str:
        self._request_counter += 1
        return f"serving-req-{self._request_counter}"

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        config,
    ) -> AsyncGenerator[TokenOutput, None]:
        """Add a request and yield token outputs as they are generated."""
        prompt_token_ids = self.tokenizer.encode(prompt)
        if not prompt_token_ids and self.bos_token_id is not None:
            prompt_token_ids = [self.bos_token_id]
        if not prompt_token_ids:
            raise ValueError("Prompt tokenization produced no tokens.")

        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=config.max_new_tokens,
            arrival_time=time.time(),
            stop_strings=tuple(config.stop) if config.stop else (),
            eos_token_id=self.eos_token_id,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        ctx = _RequestContext(request=request)
        self._request_contexts[request_id] = ctx
        self.scheduler.add_request(request)

        try:
            while True:
                output: TokenOutput = await ctx.queue.get()
                yield output
                if output.finished:
                    break
        finally:
            self._request_contexts.pop(request_id, None)

    async def abort_request(self, request_id: str) -> None:
        self.scheduler.abort_request(request_id)
        ctx = self._request_contexts.pop(request_id, None)
        if ctx is not None:
            await ctx.queue.put(
                TokenOutput(finished=True, finish_reason="FINISHED_ABORTED")
            )

    async def _engine_loop(self) -> None:
        """Main loop: schedule → send to worker → receive results → dispatch."""
        logger.info("Engine loop started")
        while self._running:
            if not self.scheduler.has_work():
                await asyncio.sleep(self.config.engine_loop_interval)
                continue

            scheduler_output = self.scheduler.schedule()
            if scheduler_output.is_empty:
                await asyncio.sleep(self.config.engine_loop_interval)
                continue

            # Send batch to worker (include any pending free IDs)
            finished_ids = self._pending_free_ids.copy()
            self._pending_free_ids.clear()
            self._input_queue.put(
                WorkerCommand(
                    type="step",
                    scheduler_output=scheduler_output,
                    finished_request_ids=finished_ids or None,
                )
            )

            # Wait for result (non-blocking via thread — this is safe because
            # we're waiting on a multiprocessing.Queue from a separate process,
            # not doing NPU work in this thread)
            step_output: StepOutput = await asyncio.to_thread(
                self._output_queue.get, timeout=300
            )

            if step_output.error:
                logger.error(f"Worker returned error: {step_output.error}")
                self._handle_step_error(scheduler_output)
                continue

            # Process results
            self._process_step_output(scheduler_output, step_output)

        logger.info("Engine loop stopped")

    def _process_step_output(
        self, scheduler_output: SchedulerOutput, step_output: StepOutput
    ) -> None:
        """Process worker results: update scheduler state, push tokens to request queues."""
        request_outputs = self.scheduler.update_from_output(
            scheduler_output, step_output.new_tokens
        )

        for req_output in request_outputs:
            ctx = self._request_contexts.get(req_output.request_id)
            if ctx is None:
                continue

            # Decode current output text
            text = ""
            if ctx.request.output_token_ids:
                text = self.tokenizer.decode(ctx.request.output_token_ids)

            # Check stop_strings (scheduler doesn't handle these)
            if not req_output.finished and ctx.request.stop_strings:
                for stop in ctx.request.stop_strings:
                    if stop and text.endswith(stop):
                        req_output.finished = True
                        req_output.finish_reason = "FINISHED_STOP"
                        ctx.request.status = RequestStatus.FINISHED_STOP
                        self.scheduler._free_request_blocks(ctx.request)
                        self.scheduler.running = [
                            r
                            for r in self.scheduler.running
                            if r.request_id != req_output.request_id
                        ]
                        break

            # Track finished requests for worker KV cache cleanup
            if req_output.finished:
                self._pending_free_ids.append(req_output.request_id)

            token_output = TokenOutput(
                token_id=req_output.new_token_id,
                text=text,
                finished=req_output.finished,
                finish_reason=req_output.finish_reason,
            )
            ctx.queue.put_nowait(token_output)

    def _handle_step_error(self, scheduler_output: SchedulerOutput) -> None:
        """On worker error, abort all requests in the failed batch."""
        for sr in scheduler_output.scheduled_requests:
            ctx = self._request_contexts.get(sr.request.request_id)
            if ctx is not None:
                ctx.queue.put_nowait(
                    TokenOutput(finished=True, finish_reason="error")
                )
            self.scheduler.abort_request(sr.request.request_id)
