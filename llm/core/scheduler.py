# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from .block_pool import BlockPool


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED_EOS = auto()
    FINISHED_LENGTH = auto()
    FINISHED_STOP = auto()
    FINISHED_ABORTED = auto()

    @property
    def is_finished(self) -> bool:
        return self in (
            RequestStatus.FINISHED_EOS,
            RequestStatus.FINISHED_LENGTH,
            RequestStatus.FINISHED_STOP,
            RequestStatus.FINISHED_ABORTED,
        )


@dataclass
class SchedulerConfig:
    max_num_running_reqs: int = 32
    max_num_scheduled_tokens: int = 4096
    long_prefill_token_threshold: int = 2048
    max_seq_len: int = 4096


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    arrival_time: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    stop_strings: tuple[str, ...] = ()
    eos_token_id: int | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    cached_block_ids: list[int] = field(default_factory=list)
    allocated_block_ids: list[int] = field(default_factory=list)
    block_hashes: list[int] = field(default_factory=list)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    @property
    def num_new_tokens_needed(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids


@dataclass
class ScheduledRequest:
    request: Request
    num_new_tokens: int
    is_prefill: bool
    resumed_from_preemption: bool = False


@dataclass
class SchedulerOutput:
    scheduled_requests: list[ScheduledRequest] = field(default_factory=list)
    preempted_requests: list[Request] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0


@dataclass
class RequestOutput:
    request_id: str
    new_token_id: int | None = None
    finished: bool = False
    finish_reason: str = ""


class Scheduler:
    """Continuous batching scheduler with chunked prefill and preemption."""

    def __init__(self, config: SchedulerConfig, block_pool: BlockPool) -> None:
        self.config = config
        self.block_pool = block_pool
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        request.block_hashes = self.block_pool.compute_block_hashes(request.prompt_token_ids)
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def abort_request(self, request_id: str) -> None:
        request = self.requests.get(request_id)
        if request is None:
            return
        request.status = RequestStatus.FINISHED_ABORTED
        self._free_request_blocks(request)
        self.running = [r for r in self.running if r.request_id != request_id]
        self.waiting = deque(r for r in self.waiting if r.request_id != request_id)
        del self.requests[request_id]

    def has_work(self) -> bool:
        return len(self.running) > 0 or len(self.waiting) > 0

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        token_budget = self.config.max_num_scheduled_tokens

        # Phase 1: schedule RUNNING requests (decode or resumed prefill)
        running_to_keep: list[Request] = []
        for request in self.running:
            num_new = request.num_new_tokens_needed
            if num_new <= 0:
                running_to_keep.append(request)
                continue

            if self.config.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.config.long_prefill_token_threshold)
            num_new = min(num_new, token_budget)

            if num_new <= 0:
                running_to_keep.append(request)
                continue

            num_blocks_needed = self._blocks_needed(request, num_new)
            if not self._try_allocate_blocks(request, num_blocks_needed):
                preempted = self._preempt_lowest_priority(request)
                if preempted is None:
                    running_to_keep.append(request)
                    continue
                output.preempted_requests.append(preempted)
                if not self._try_allocate_blocks(request, num_blocks_needed):
                    running_to_keep.append(request)
                    continue

            is_prefill = request.is_prefill
            output.scheduled_requests.append(
                ScheduledRequest(request=request, num_new_tokens=num_new, is_prefill=is_prefill)
            )
            if is_prefill:
                output.num_prefill_tokens += num_new
            else:
                output.num_decode_tokens += num_new
            token_budget -= num_new
            running_to_keep.append(request)

        self.running = running_to_keep

        # Phase 2: schedule WAITING requests (new prefill)
        remaining_waiting: deque[Request] = deque()
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.config.max_num_running_reqs:
                break

            request = self.waiting.popleft()

            # Prefix cache lookup
            cached_blocks = self.block_pool.get_computed_blocks(request.prompt_token_ids)
            if cached_blocks:
                request.cached_block_ids = [b.block_id for b in cached_blocks]
                request.num_computed_tokens = len(cached_blocks) * self.block_pool.block_size

            num_new = request.num_new_tokens_needed
            if self.config.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.config.long_prefill_token_threshold)
            num_new = min(num_new, token_budget)

            if num_new <= 0:
                remaining_waiting.append(request)
                continue

            num_blocks_needed = self._blocks_needed(request, num_new)
            if not self._try_allocate_blocks(request, num_blocks_needed):
                # Release cached blocks we just acquired
                for block in cached_blocks:
                    self.block_pool.release(block)
                request.cached_block_ids = []
                request.num_computed_tokens = 0
                remaining_waiting.append(request)
                break

            request.status = RequestStatus.RUNNING
            self.running.append(request)
            output.scheduled_requests.append(
                ScheduledRequest(request=request, num_new_tokens=num_new, is_prefill=True)
            )
            output.num_prefill_tokens += num_new
            token_budget -= num_new

        # Put back requests that couldn't be scheduled
        remaining_waiting.extend(self.waiting)
        self.waiting = remaining_waiting

        return output

    def update_from_output(
        self, scheduler_output: SchedulerOutput, new_token_ids: dict[str, int]
    ) -> list[RequestOutput]:
        """Update request states after model execution. Returns outputs for finished/streaming."""
        outputs: list[RequestOutput] = []

        for scheduled in scheduler_output.scheduled_requests:
            request = scheduled.request
            request.num_computed_tokens += scheduled.num_new_tokens

            # Cache completed blocks
            self._cache_completed_blocks(request)

            if request.is_prefill:
                # Still in prefill (chunked), no output token yet
                if request.num_computed_tokens < request.num_prompt_tokens:
                    continue
                # Prefill just completed — the model produced a token
                token_id = new_token_ids.get(request.request_id)
                if token_id is not None:
                    request.output_token_ids.append(token_id)
                    outputs.append(RequestOutput(request_id=request.request_id, new_token_id=token_id))
            else:
                # Decode step
                token_id = new_token_ids.get(request.request_id)
                if token_id is not None:
                    request.output_token_ids.append(token_id)
                    outputs.append(RequestOutput(request_id=request.request_id, new_token_id=token_id))

        # Check finish conditions
        finished_ids: list[str] = []
        for request in self.running:
            if request.status.is_finished:
                continue
            finish_reason = self._check_finish(request)
            if finish_reason is not None:
                request.status = finish_reason
                finished_ids.append(request.request_id)
                for out in outputs:
                    if out.request_id == request.request_id:
                        out.finished = True
                        out.finish_reason = finish_reason.name
                        break
                else:
                    outputs.append(RequestOutput(
                        request_id=request.request_id,
                        finished=True,
                        finish_reason=finish_reason.name,
                    ))

        # Remove finished requests from running
        for req_id in finished_ids:
            request = self.requests.get(req_id)
            if request is not None:
                self._free_request_blocks(request)
            self.running = [r for r in self.running if r.request_id != req_id]

        return outputs

    def _check_finish(self, request: Request) -> RequestStatus | None:
        if not request.output_token_ids:
            return None
        last_token = request.output_token_ids[-1]
        if request.eos_token_id is not None and last_token == request.eos_token_id:
            return RequestStatus.FINISHED_EOS
        if len(request.output_token_ids) >= request.max_new_tokens:
            return RequestStatus.FINISHED_LENGTH
        return None

    def _blocks_needed(self, request: Request, num_new_tokens: int) -> int:
        current_total_tokens = request.num_computed_tokens + num_new_tokens
        current_blocks = len(request.cached_block_ids) + len(request.allocated_block_ids)
        needed_blocks = (current_total_tokens + self.block_pool.block_size - 1) // self.block_pool.block_size
        return max(0, needed_blocks - current_blocks)

    def _try_allocate_blocks(self, request: Request, num_blocks: int) -> bool:
        if num_blocks <= 0:
            return True
        if self.block_pool.num_free_blocks < num_blocks:
            return False
        for _ in range(num_blocks):
            block = self.block_pool.allocate()
            if block is None:
                return False
            request.allocated_block_ids.append(block.block_id)
        return True

    def _preempt_lowest_priority(self, exclude: Request) -> Request | None:
        if not self.running:
            return None
        # Preempt the most recently added running request (lowest priority = latest arrival)
        candidates = [r for r in self.running if r.request_id != exclude.request_id]
        if not candidates:
            return None
        victim = max(candidates, key=lambda r: r.arrival_time)
        self._free_request_blocks(victim)
        victim.status = RequestStatus.PREEMPTED
        victim.num_computed_tokens = 0
        victim.cached_block_ids = []
        victim.allocated_block_ids = []
        self.running = [r for r in self.running if r.request_id != victim.request_id]
        self.waiting.appendleft(victim)
        return victim

    def _free_request_blocks(self, request: Request) -> None:
        for block_id in request.cached_block_ids:
            block = self.block_pool.blocks[block_id]
            self.block_pool.release(block)
        for block_id in request.allocated_block_ids:
            block = self.block_pool.blocks[block_id]
            self.block_pool.release(block)
        request.cached_block_ids = []
        request.allocated_block_ids = []

    def _cache_completed_blocks(self, request: Request) -> None:
        """Register completed blocks in the prefix cache."""
        total_blocks_computed = request.num_computed_tokens // self.block_pool.block_size
        already_cached = len(request.cached_block_ids)
        all_block_ids = request.cached_block_ids + request.allocated_block_ids

        for i in range(already_cached, total_blocks_computed):
            if i >= len(request.block_hashes) or i >= len(all_block_ids):
                break
            block_id = all_block_ids[i]
            block = self.block_pool.blocks[block_id]
            self.block_pool.cache_block(block, request.block_hashes[i])
