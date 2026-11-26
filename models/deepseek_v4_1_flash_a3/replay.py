# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate tokens with resident weights and seven persistent cache allocations."""

import dataclasses
import json
import time

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import config as C
from .host import prepare_resident_inputs
from .reference import compare_logits
from .swimlane import step_name
from .telemetry import route_distribution
from .device_views import _device_view
from .utils import DECODE_CACHE_NAMES, paged_slots, window_metadata


class DecodeInputs:
    """Copy one aligned input image per rank and bind its fields as typed device views."""

    def __init__(self, template):
        offsets = {}
        size = 0
        for name, value in template.items():
            offsets[name] = size
            size = (size + value.nbytes + 511) // 512 * 512
        self.images = torch.zeros((3, size), dtype=torch.uint8).share_memory_()
        self.fields = []
        for image in self.images:
            fields = {}
            for name, value in template.items():
                offset = offsets[name]
                fields[name] = image.narrow(0, offset, value.nbytes).view(value.dtype).reshape(value.shape)
                fields[name].copy_(value)
            self.fields.append(fields)
        self.devices = {}

    def bind(self, runtime, rank):
        image = self.images[0]
        allocation = runtime.alloc_tensor(image.shape, image.dtype, init=image, worker_id=rank)
        self.devices[rank] = allocation
        return {name: _device_view(allocation, field) for name, field in self.fields[0].items()}

    def stage(self, inputs, logit_rows):
        for slot, fields in enumerate(self.fields):
            for name, value in inputs[slot != 0].items():
                if slot == 1 and name == logit_rows:
                    value = inputs[False][name]
                fields[name].copy_(value)

    def stage_token(self, weights, engram, token, position, owner):
        """Fill the live row in the two active TP input images for a single Decode token."""
        ids = torch.tensor([token], dtype=torch.int64)
        pos = torch.tensor([position], dtype=torch.int32)
        req = torch.zeros(1, dtype=torch.int32)
        table = self.fields[0]["c1a_index_block_table"]
        c2a_table = self.fields[0]["c2a_index_block_table"]
        slot, window, _ = window_metadata(pos, req, table)
        compressed_slot = paged_slots(pos, req, c2a_table, C.BLOCK_SIZE, 2, True)
        embedding = weights.lookup_rows(ids)
        lookup = engram.lookup(ids, 1, position, request_id=owner)
        for image, fields in enumerate(self.fields[1:], 1):
            fields["x_hc"][0].copy_(embedding)
            fields["engram_lookup"][:, :1].copy_(lookup)
            fields["position_ids"][:1].copy_(pos)
            fields["window_slots"][:1].copy_(slot)
            fields["window_indices"][:1].copy_(window)
            fields["c1a_compressed_slots"][:1].copy_(slot)
            fields["c2a_compressed_slots"][:1].copy_(compressed_slot)
            fields["token_to_req_indices"][0] = 0
            fields["query_start_loc"][1] = 1
            fields["logit_row_indices"][0] = 0 if image == 2 else -1

    def upload(self, runtime, rank, owner):
        slot = 2 if rank == owner else int(rank // C.TP_SIZE == owner // C.TP_SIZE)
        image = self.images[slot]
        runtime.copy_to(self.devices[rank].data_ptr, image.data_ptr(), image.nbytes, worker_id=rank)


class LogitValidationError(ValueError):
    """Report completed requests whose logits did not satisfy the frozen policy."""


def reset_caches(runtime, ranks, host_cache):
    """Reset the same seven resident allocations for the next request-slot reuse."""
    for rank in range(16):
        for name, value in host_cache[rank].items():
            runtime.copy_to(ranks[0][rank][name].data_ptr, value.data_ptr(),
                            value.numel() * value.element_size(), worker_id=rank)


def run_request(
    runtime, programs, arguments, names, ranks,
    weights, engram, tokenizer, request, host_cache, rope_tables,
    config, memory, report, *, options, host_io, decode_inputs, prefill_slab_rows, reference=None, teacher_forced=False,
    audit=None, save_logits=None, label="greedy", profile=False, profile_dir=None,
):
    """Submit prompt chunks followed by one Decode call per subsequent token."""
    if teacher_forced and reference is None:
        raise ValueError("teacher forcing requires the frozen reference")
    trace = reference["token_ids"] if teacher_forced else request.get("trace_token_ids")
    owner = request["dp_rank"] * 4
    constants = set(weights.tensor_layouts) | set(DECODE_CACHE_NAMES.values()) | {"freqs_cos", "freqs_sin"}
    counts = {name: arguments[1][name] for name in ("moe_num_tokens", "attention_num_tokens")}
    logit_buffer = host_io["logits"]
    sampled = host_io["sampled_ids"]
    report.update(prefill_calls=0, decode_calls=0, output_sequences=[], logit_checks=[], cache_checks=[],
                  teacher_forced=trace is not None, label=label, instrumented_cache_audits=audit is not None,
                  profiled=profile)
    if trace is not None:
        report["decode_input_ids"] = []
    committed = 0
    generated, token_times, engine_times, observed_logits = [], [], [], []
    tokenizer_config = json.loads((weights.checkpoint.root / "tokenizer_config.json").read_text())
    eos = tokenizer_config["eos_token"]
    eos_id = tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else eos)
    start = time.monotonic()
    prompt = request["input_ids"]
    chunks = (len(prompt) + options.chunk_size - 1) // options.chunk_size
    idle_rank = (owner + C.TP_SIZE) % 16
    idle_inputs = host_io.get("idle_inputs")
    if idle_inputs is not None:
        # The idle TP group has no token history for this request.
        engram.hash_ids([], 0, request_id=idle_rank)
    with ThreadPoolExecutor(max_workers=8) as uploads:
        for call in range(chunks + options.max_new_tokens - 1):
            step = call - chunks + 1
            index = int(step > 0)
            capacity = options.decode_capacity if index else options.capacity
            if index:
                current_ids = [trace[step - 1] if trace is not None else generated[-1]]
            else:
                start_row = call * options.chunk_size
                current_ids = prompt[start_row:start_row + options.chunk_size]
            if index:
                idle_uploads = [uploads.submit(decode_inputs.upload, runtime, rank, owner)
                                for rank in range(16) if rank // C.TP_SIZE != request["dp_rank"]]
                decode_inputs.stage_token(weights, engram, current_ids[0], committed, owner)
                if trace is not None:
                    report["decode_input_ids"].append(current_ids[0])
            else:
                inputs = {}
                for active, rank, ids, position in ((False, idle_rank, [], 0), (True, owner, current_ids, committed)):
                    if not active and idle_inputs is not None:
                        inputs[False] = idle_inputs[0].copy()
                        continue
                    values = prepare_resident_inputs(
                        weights, ids, capacity, start_pos=position, max_seq_len=options.context_capacity,
                        cache_state=host_cache[rank], rope_tables=rope_tables, engram=engram, request_id=rank,
                        moe_slab_rows=prefill_slab_rows,
                    )
                    inputs[active] = {name: value for name, value in values.items()
                                      if name in ranks[0][rank] and name not in constants}
                # Both tables are read-only. An idle TP group retains the empty input
                # image until it becomes active. An interrupted upload invalidates the record.
                dirty_groups = host_io.pop("prefill_dirty_groups", range(C.DP_SIZE))
                for active, values in inputs.items():
                    for name, value in values.items():
                        buffers = host_io.get("prefill_" + name)
                        if buffers is None:
                            continue
                        buffer = buffers[int(active)]
                        if active or idle_inputs is None or not host_io.get("idle_metadata_staged"):
                            buffer.copy_(value)
                        values[name] = buffer
                if idle_inputs is not None:
                    host_io["idle_metadata_staged"] = True

            def upload_rank(rank):
                if index:
                    decode_inputs.upload(runtime, rank, owner)
                    return
                active = rank // C.TP_SIZE == request["dp_rank"]
                for name, value in inputs[active].items():
                    if not active and rank // C.TP_SIZE not in dirty_groups:
                        if name in ("embedding", "engram_lookup"):
                            continue
                    if name == "logit_rows" and rank != owner:
                        value = inputs[False][name]
                    runtime.copy_to(ranks[index][rank][name].data_ptr, value.data_ptr(),
                                    value.numel() * value.element_size(), worker_id=rank)

            if index:
                for tensor in counts.values():
                    tensor.zero_()
                    tensor[owner:owner + C.TP_SIZE] = len(current_ids)
                list(uploads.map(upload_rank, range(owner, owner + C.TP_SIZE)))
                for completed in idle_uploads:
                    completed.result()
            else:
                list(uploads.map(upload_rank, range(16)))
            if not index:
                host_io["prefill_dirty_groups"] = (request["dp_rank"],)
                del inputs, values
            engine_start = time.monotonic()
            memory.phase = "decode" if index else "prefill"
            marker = dict(label=label, step=call + 1, mode=memory.phase)
            if profile_dir is not None:
                marker["profiled"] = True
            print("A3_ENGINE_BEGIN", json.dumps(marker), flush=True)
            program_args = [arguments[index][name] for name in names[index]]
            if profile_dir is not None:
                runtime.run_profiled_step(programs[index], *program_args, config=config, directory=profile_dir / step_name(step))
            else:
                dispatch_config = dataclasses.replace(config, enable_chip_swimlane=4) if profile and index and step == options.max_new_tokens - 1 else config
                runtime.run(programs[index], *program_args, config=dispatch_config)
            engine_times.append(time.monotonic() - engine_start)
            print("A3_ENGINE_END", json.dumps(marker), flush=True)
            committed += len(current_ids)
            report["decode_calls" if index else "prefill_calls"] += 1
            if step < 0:
                continue
            logits = logit_buffer[step]
            for name, value in (("logits", logits), ("sampled_ids", sampled)):
                runtime.copy_from(value.data_ptr(), ranks[index][owner][name].data_ptr,
                                  value.numel() * value.element_size(), worker_id=owner)
            row = logits.numpy()
            if not np.isfinite(row).all():
                raise FloatingPointError("nonfinite owner logits")
            token = int(row.argmax())
            if token != int(sampled[0]):
                raise RuntimeError("device greedy token differs from the complete vocabulary argmax")
            generated.append(token)
            observed_logits.append(logits)
            token_times.append(time.monotonic())
            report["output_sequences"] = [{"request_id": request["request_id"], "token_ids": generated.copy()}]
            print("A3_TOKEN", json.dumps(dict(label=label, step=step + 1, token_id=token, committed=committed)), flush=True)
            if audit is not None:
                report["cache_checks"].append(audit(runtime, ranks, host_cache, owner, step))
            if token == eos_id and not options.ignore_eos and request.get("trace_token_ids") is None:
                if (reference is not None or profile_dir is not None) and step + 1 < options.max_new_tokens:
                    raise RuntimeError("Reference replay reached EOS before its last step")
                break
    if save_logits is not None:
        torch.save(dict(logits=torch.stack(observed_logits), token_ids=generated,
                        checkpoint_identity=weights.checkpoint.identity), save_logits)
    report.update(status="passed" if reference is not None else "executed_pending_numerical_acceptance",
                  committed_positions=committed, ttft_seconds=token_times[0] - start,
                  tpot_seconds=(token_times[-1] - token_times[0]) / max(1, len(generated) - 1),
                  end_to_end_seconds=token_times[-1] - start,
                  token_intervals_seconds=[b - a for a, b in zip(token_times, token_times[1:])],
                  engine_seconds=engine_times, text=tokenizer.decode(generated, skip_special_tokens=False))
    report["prefill_routing"] = route_distribution(runtime, ranks, owner, len(prompt) - (chunks - 1) * options.chunk_size)
    if reference is not None:
        report["numerical_acceptance"] = reference["policy"]
        for step, value in enumerate(observed_logits):
            checked = compare_logits(value, reference["logits"][step], reference["policy"])
            report["logit_checks"].append(checked)
        failed_steps = [step + 1 for step, checked in enumerate(report["logit_checks"]) if not checked["passed"]]
        if failed_steps:
            report["status"] = "failed"
            raise LogitValidationError(f"frozen W8A8 logit acceptance failed at steps {failed_steps}")
    return report
