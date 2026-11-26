# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one TP-group request through resident A3 prefill and seven decode calls."""

import dataclasses
import hashlib
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
from tokenizers import Tokenizer
from pypto.backend import BackendType
from pypto.ir import DistributedConfig
from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
from pypto.runtime import DeviceTensor, ExecutionMode, RunConfig
from pypto.runtime.distributed_runner import DistributedWorker

from . import config as C
from .config import FLASH as M
from .device_views import resident_graph_argument
from .engram import HostEngram
from .host import prepare_resident_inputs
from .utils import DECODE_CACHE_NAMES, decode_cache_views, prepare_decode_inputs
from .weights import Checkpoint, ResidentWeights
from .replay import DecodeInputs, LogitValidationError, reset_caches, run_request
from .swimlane import profile_program_dirs, summarize_swimlanes


class SwimlaneWorker(DistributedWorker):
    """Write Level 1 records and task dependencies into the same step directory."""

    _capture_dir = ContextVar("a3_step_capture_dir", default=None)

    def run_profiled_step(self, compiled, *arguments, config, directory):
        directory.mkdir()
        token = self._capture_dir.set(directory.resolve())
        try:
            config = dataclasses.replace(config, enable_dep_gen=True, enable_chip_swimlane=0)
            self.run(compiled, *arguments, config=config)
        finally:
            self._capture_dir.reset(token)

    def _prepare_dispatch_config(self, compiled, config):
        call_config, directory, two_pass = super()._prepare_dispatch_config(compiled, config)
        capture = self._capture_dir.get()
        if capture is not None:
            # Enable timing on the native dispatch. The public swimlane flag requests two passes.
            call_config.enable_chip_swimlane = 1
            call_config.output_prefix = str(capture)
        return call_config, directory, two_pass


class HbmSampler:
    """Sample board HBM occupancy and retain the largest observation per phase."""

    def __init__(self, devices):
        self.devices = devices
        self.phase = "prepare"
        self.samples = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        completed = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, timeout=20, check=True)
        values = {}
        for line in completed.stdout.splitlines():
            fields = [part.strip() for part in line.split("|")[1:-1]]
            if len(fields) != 3:
                continue
            identity = fields[0].split()
            usage = re.search(r"(\d+)\s*/\s*(\d+)\s*$", fields[2])
            if len(identity) == 2 and all(part.isdigit() for part in identity) and usage:
                values[int(identity[1])] = int(usage[1]) * 1024**2
        if any(device not in values for device in self.devices):
            raise RuntimeError("npu-smi did not report every allocated physical chip")
        self.samples.append(dict(phase=self.phase, monotonic=time.monotonic(),
                                 bytes_per_rank=[values[device] for device in self.devices]))

    def _run(self):
        while not self.stopped.is_set():
            try:
                self.sample()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.samples.append(dict(phase=self.phase, error=str(error)))
            self.stopped.wait(2.0)

    def report(self):
        valid = [sample for sample in self.samples if "bytes_per_rank" in sample]
        if not valid:
            raise RuntimeError("no HBM occupancy sample was collected")
        phases = {}
        for sample in valid:
            phase = phases.setdefault(sample["phase"], [0] * len(self.devices))
            for rank, value in enumerate(sample["bytes_per_rank"]):
                phase[rank] = max(phase[rank], value)
        peak = max((value, rank, name) for name, values in phases.items() for rank, value in enumerate(values))
        return dict(method="npu-smi physical-chip occupancy sampled every 2 s", includes_board_baseline=True,
                    peak_observed_bytes=peak[0], peak_rank=peak[1], peak_phase=peak[2],
                    phases=phases, samples=self.samples)


def audit_caches(runtime, ranks, host_cache, owner, step, directory=None):
    """Check TP replicas, inactive request preservation and retained cache addresses."""
    checks = {}
    for name, initial in host_cache[0].items():
        actual = []
        for rank in range(16):
            value = torch.empty_like(initial)
            handle = ranks[0][rank][name]
            runtime.copy_from(value.data_ptr(), handle.data_ptr, value.numel() * value.element_size(), worker_id=rank)
            actual.append(value)
        active_equal = all(torch.equal(actual[owner].view(torch.uint8), actual[rank].view(torch.uint8))
                           for rank in range(owner, owner + 4))
        inactive_equal = all(torch.equal(actual[rank].view(torch.uint8), host_cache[rank][name].view(torch.uint8))
                             for rank in range(16) if rank // 4 != owner // 4)
        checks[name] = dict(tp_replicas_equal=active_equal, inactive_preserved=inactive_equal)
        if directory is not None:
            torch.save(actual[owner], directory / f"cache_step{step + 1}_{name}.pt")
        if not active_equal or not inactive_equal:
            raise RuntimeError(f"cache ownership validation failed for {name}: {checks[name]}")
    return checks


def read_request(args, tokenizer):
    """Read a request replay or tokenize application input before allocating device state."""
    path = args.input_ids_json or args.tokens_file
    development = path is None and args.tokens is None and args.prompt is None
    if path is not None:
        payload = path.read_bytes()
        record = json.loads(payload)
        if args.input_ids_json and isinstance(record, list) and len(record) == 1 and isinstance(record[0], dict):
            record = record[0]
        if isinstance(record, list):
            record = dict(request_id="request", dp_rank=0, input_ids=record)
    else:
        if development:
            block = tokenizer.encode("Explain the mathematical reasoning step by step. ", add_special_tokens=False).ids
            if not block:
                raise ValueError("Development text produced an empty sequence")
            ids = (block * ((8192 + len(block) - 1) // len(block)))[:8192]
        elif args.prompt is not None:
            tokenizer.no_padding()
            tokenizer.no_truncation()
            ids = tokenizer.encode(args.prompt).ids
        else:
            ids = [int(value) for value in args.tokens.split(",")]
        record = dict(request_id="development-8192" if development else "request", dp_rank=0, input_ids=ids)
        payload = (json.dumps(record) + "\n").encode()
        if development:
            path = Path(args.output or args.runtime_dir) / "development_input_ids.json"
            path.write_bytes(payload)
    if not isinstance(record, dict) or not {"request_id", "dp_rank", "input_ids"} <= record.keys():
        raise ValueError("A request requires request_id, dp_rank and input_ids")
    if type(record["dp_rank"]) is not int or record["dp_rank"] not in range(4):
        raise ValueError("dp_rank must be in 0..3")
    ids = record["input_ids"]
    if not isinstance(ids, list) or not ids or any(type(v) is not int or not 0 <= v < M.vocab_size for v in ids):
        raise ValueError("Expected a nonempty array of valid integer token IDs")
    if "trace_token_ids" in record:
        trace = record["trace_token_ids"]
        if not isinstance(trace, list) or len(trace) != args.max_new_tokens or any(
            type(token) is not int or not 0 <= token < M.vocab_size for token in trace
        ):
            raise ValueError("trace_token_ids must contain one in-vocabulary token per requested output")
        if args.reference_dir:
            raise ValueError("A fixed-trace capture cannot also select a frozen reference replay")
    if len(ids) + args.max_new_tokens > args.context_capacity:
        raise ValueError("Prompt and reserved output positions exceed the compiled context")
    return record, dict(path=str(path) if path else None, sha256=hashlib.sha256(payload).hexdigest(),
                        development_replay=development)


@dataclasses.dataclass
class ResidentGraphs:
    """Live program bindings shared by offline generation and HTTP requests."""

    runtime: DistributedWorker
    programs: list
    arguments: list
    names: list
    ranks: list
    weights: ResidentWeights
    engram: HostEngram
    host_cache: list
    rope_tables: tuple
    config: RunConfig
    host_io: dict
    decode_inputs: DecodeInputs
    prefill_slab_rows: int

    def reset(self):
        reset_caches(self.runtime, self.ranks, self.host_cache)
        self.engram.history.clear()

    def replay(self, tokenizer, request, memory, report, **kwargs):
        return run_request(
            self.runtime, self.programs, self.arguments, self.names, self.ranks, self.weights, self.engram,
            tokenizer, request, self.host_cache, self.rope_tables, self.config, memory, report,
            host_io=self.host_io, decode_inputs=self.decode_inputs, prefill_slab_rows=self.prefill_slab_rows, **kwargs,
        )


@contextmanager
def open_resident_graphs(
    args, devices, weights, engram, memory, report, *,
    prefill_slab_rows, selected_layers=None, load_start=None, request_slots=1,
):
    """Load weights and register the two forward programs for the lifetime of a worker."""
    if load_start is None:
        load_start = time.monotonic()
    config = RunConfig(backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                       distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
                       ring_heap=(4 * 1024**3, 4 * 1024**3, (4 if prefill_slab_rows >= 128 else 2) * 1024**3, 1024**3), ring_dep_pool=131072)
    program_dirs = (args.runtime_dir, args.decode_runtime_dir)
    if args.profile_all_steps is not None:
        program_dirs = profile_program_dirs(args.profile_all_steps, program_dirs)
    programs = [DistributedCompiledProgram.from_dir(path, backend_type=config.backend_type,
                execution_mode=config.execution_mode, distributed_config=config.distributed_config)
                for path in program_dirs]
    names = [[parameter.name.removesuffix("__ssa_v0") for parameter in program._get_metadata()[0]]
             for program in programs]
    prepare_windows = [phase_names[-1] == "execute_layers" for phase_names in names]
    for phase_names, supported in zip(names, prepare_windows, strict=True):
        if supported:
            phase_names.pop()
    cache_names = set(DECODE_CACHE_NAMES.values())
    constants = set(weights.tensor_layouts) | cache_names | {"freqs_cos", "freqs_sin"}
    counts = {name: torch.zeros(16, dtype=torch.int32).share_memory_()
              for name in ("moe_num_tokens", "attention_num_tokens")}
    outputs = {"x_hc_out": ((args.capacity, M.hc_mult, M.hidden_size), torch.float32),
               "attn_hc_out": ((args.capacity, M.hc_mult, M.hidden_size), torch.float32),
               "pre_mix_out": ((args.capacity, M.hc_mult), torch.float32),
               "logits": ((16, M.vocab_size), torch.float32), "sampled_ids": ((16, 8), torch.int32)}

    saved_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        prefill_template = prepare_resident_inputs(weights, [], args.capacity, max_seq_len=args.context_capacity,
                                                  engram=engram, request_slots=request_slots,
                                                  moe_slab_rows=prefill_slab_rows)
        rope_tables = prefill_template["freqs_cos"], prefill_template["freqs_sin"]
        initial_cache = {name: prefill_template[name] for name in cache_names}
        decode_values = prepare_resident_inputs(weights, [], args.decode_capacity, max_seq_len=args.context_capacity,
                                               cache_state=initial_cache, rope_tables=rope_tables, engram=engram,
                                               request_slots=request_slots)
        decode_template = prepare_decode_inputs(decode_values)
        # TP peers read the same active or idle inputs until the synchronous forward completes.
        decode_inputs = DecodeInputs({name: value for name, value in decode_template.items()
                                      if name in names[1] and name not in constants})
        host_io = {"logits": torch.empty((args.max_new_tokens, M.vocab_size), dtype=torch.float32).share_memory_(),
                   "sampled_ids": torch.empty(8, dtype=torch.int32).share_memory_()}
        if request_slots > 1:
            # All blocks of a native cache start with the same payload or scale fill.
            for name, initial in initial_cache.items():
                host_io["reset_" + name] = initial[0, 0].clone().share_memory_()
        for name in names[0]:
            if name in prefill_template and name not in constants:
                template = prefill_template[name]
                host_io["prefill_" + name] = torch.empty((2, *template.shape), dtype=template.dtype).share_memory_()
        worker = SwimlaneWorker if args.profile_all_steps is not None else DistributedWorker
        with worker(programs, config, persistent=True, reset_persistent_windows=False,
                    inherited_host_tensors=[*counts.values(), *host_io.values(), decode_inputs.images]) as runtime:
            host_io["idle_inputs"] = tuple(
                {name: template[name] for name in phase_names if name in template and name not in constants}
                for template, phase_names in zip((prefill_template, decode_template), names, strict=True)
            )
            memory.phase = "graphs_prepared"
            memory.sample()
            print("A3_RESIDENT_GRAPHS_READY", flush=True)
            ranks = [[{} for _ in range(16)] for _ in programs]
            # Host reset images are read-only. Every rank allocates its own device caches.
            host_cache = [initial_cache for _ in range(16)]
            views = decode_cache_views(initial_cache)
            for rank in range(16):
                memory.phase = "weight_load"
                for name in names[0]:
                    if name in weights.tensor_layouts or name in outputs:
                        shape, dtype = weights.tensor_layouts[name] if name in weights.tensor_layouts else outputs[name]
                        handle = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                    else:
                        value = prefill_template[name]
                        handle = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
                    ranks[0][rank][name] = handle
                input_views = decode_inputs.bind(runtime, rank)
                for name in names[1]:
                    if name in counts:
                        continue
                    if name in DECODE_CACHE_NAMES:
                        source = ranks[0][rank][DECODE_CACHE_NAMES[name]]
                        handle = DeviceTensor(source.data_ptr, views[name].shape, source.dtype, buffer=source.buffer)
                    elif name in weights.tensor_layouts and name not in ranks[0][rank]:
                        shape, dtype = weights.tensor_layouts[name]
                        handle = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                    elif name in constants or name in ("logits", "sampled_ids"):
                        handle = ranks[0][rank][name]
                    else:
                        handle = input_views[name]
                    ranks[1][rank][name] = handle
            with ThreadPoolExecutor(max_workers=8) as uploaders:
                uploads = {
                    uploaders.submit(weights.upload, runtime, rank, ranks[0][rank] | ranks[1][rank],
                                     layers=selected_layers): rank
                    for rank in range(16)
                }
                for uploaded in as_completed(uploads):
                    uploaded.result()
                    print(f"A3_RANK_WEIGHTS_READY {uploads[uploaded]}", flush=True)
            torch.set_num_threads(16)
            memory.phase = "weights_resident"
            memory.sample()
            report["weight_load_seconds"] = time.monotonic() - load_start
            report["resident_weights_shared"] = all(ranks[0][rank][name] is ranks[1][rank][name]
                                                     for rank in range(16) for name in weights.tensor_layouts.keys() & ranks[0][rank].keys())
            report["resident_caches_shared"] = all(
                ranks[0][rank][source].data_ptr == ranks[1][rank][target].data_ptr
                for rank in range(16) for target, source in DECODE_CACHE_NAMES.items()
            )
            arguments = []
            for index in range(2):
                parameters = programs[index]._get_metadata()[0]
                if prepare_windows[index]:
                    parameters = parameters[:-1]
                stacked = {
                    name: resident_graph_argument([rank[name] for rank in ranks[index]], tuple(range(16)), param)
                    for name, param in zip(names[index], parameters, strict=True) if name not in counts
                }
                stacked.update(counts if index == 1 else {})
                arguments.append(stacked)
            reserve_times = {}
            for phase, supported in enumerate(prepare_windows):
                if not supported:
                    continue
                phase_args = [arguments[phase][name] for name in names[phase]]
                name = ("prefill", "decode")[phase]
                print("A3_RESERVE_WINDOWS_BEGIN", name, flush=True)
                before_reserve = time.monotonic()
                runtime.run(programs[phase], *phase_args, 0, config=config)
                reserve_times[name] = time.monotonic() - before_reserve
                print("A3_RESERVE_WINDOWS_END", name, flush=True)
                arguments[phase]["execute_layers"] = 1
                names[phase].append("execute_layers")
            report["window_reservation_seconds"] = reserve_times
            report["initialization_seconds"] = time.monotonic() - load_start
            report["allocator_committed_bytes_per_rank"] = [runtime.committed_device_memory(rank)
                                                            for rank in range(16)]

            host_io["prefill_dirty_groups"] = ()
            yield ResidentGraphs(runtime, programs, arguments, names, ranks, weights, engram, host_cache,
                                 rope_tables, config, host_io, decode_inputs, prefill_slab_rows)
    finally:
        torch.set_num_threads(saved_threads)


def generate(args, devices, *, manifests):
    """Reuse the same rank-local weight and cache handles across both compiled graphs."""
    checkpoint = Checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    request, replay = read_request(args, tokenizer)
    acceptance = (len(request["input_ids"]), args.capacity, args.chunk_size, args.max_new_tokens) == (8192, 8192, 8192, 8)
    if (args.validate_layers or args.reference_dir or args.profile_all_steps) and not acceptance:
        raise ValueError("Frozen layer/logit comparisons and eight-step timing reports use the 8192+8 replay")
    if args.validate_layers and args.context_capacity != 8200:
        raise ValueError("Frozen layer fixtures require history capacity 8200")
    owner = request["dp_rank"] * 4
    weights = ResidentWeights(checkpoint)
    selected_layers = [int(value) for value in args.validate_layers.split(",")] if args.validation_only else None
    engram = HostEngram(checkpoint, tokenizer)
    reference = None
    if args.reference_dir:
        policy = json.loads((args.reference_dir / "policy.json").read_text())
        if policy["checkpoint_identity"] != checkpoint.identity or policy["input_sha256"] != replay["sha256"]:
            raise ValueError("reference checkpoint or exact input replay differs")
    elif args.teacher_forced:
        raise ValueError("teacher forcing requires a frozen eight-step reference")
    report = {"status": "running", "platform": "a2a3", "topology": manifests[0]["topology"],
              "checkpoint_identity": checkpoint.identity, "weight_layout": C.WEIGHT_LAYOUT_VERSION,
              "cache_layout": C.CACHE_LAYOUT_VERSION, "replay": replay, "request_id": request["request_id"],
              "dp_rank": request["dp_rank"], "resident_weight_bytes_per_rank": weights.bytes_per_rank,
              "history_capacity": args.context_capacity, "prefill_calls": 0, "decode_calls": 0,
              "output_sequences": [], "numerical_acceptance": "pending independent W8A8 logits reference",
              "artifacts": [str(args.runtime_dir), str(args.decode_runtime_dir)],
              "teacher_forced": args.teacher_forced, "logit_checks": [], "cache_checks": [],
              "instrumented_cache_audits": args.audit_caches}
    report["initialized_weight_layers"] = selected_layers if selected_layers is not None else list(range(40))
    saved_threads = torch.get_num_threads()
    memory = HbmSampler(devices)
    memory.sample()
    memory.thread.start()
    load_start = time.monotonic()
    try:
        print("A3_PREPARE_RESIDENT_GRAPHS", flush=True)
        with open_resident_graphs(
            args, devices, weights, engram, memory, report, prefill_slab_rows=manifests[0]["moe_slab_rows"],
            selected_layers=selected_layers, load_start=load_start,
        ) as graphs:
            runtime, programs = graphs.runtime, graphs.programs
            arguments, names, ranks = graphs.arguments, graphs.names, graphs.ranks
            host_cache, rope_tables, config = graphs.host_cache, graphs.rope_tables, graphs.config
            host_io, decode_inputs = graphs.host_io, graphs.decode_inputs
            if args.validate_layers:
                from .validation import LayerValidationError, validate_layers

                layer_records = report.setdefault("layer_validation", [])
                try:
                    validate_layers(
                        runtime, programs, arguments, names, ranks, weights, engram, tokenizer, request,
                        host_cache, rope_tables, args, config, memory, audit_caches,
                        prefill_slab_rows=graphs.prefill_slab_rows, records=layer_records,
                    )
                except LayerValidationError as error:
                    if args.validation_only:
                        raise
                    report["layer_validation_error"] = str(error)
            else:
                # All Prefill input allocations still contain the initial empty template.
                host_io["prefill_dirty_groups"] = ()
            if args.validation_only:
                report.update(status="passed", acceptance_scope="representative_layers",
                              numerical_acceptance="full-model logits were not run")
                return report
            if args.reference_dir:
                reference = torch.load(args.reference_dir / "logits.pt", map_location="cpu", weights_only=True)
                if tuple(reference["logits"].shape) != (8, M.vocab_size) or len(reference["token_ids"]) != 8:
                    raise ValueError("reference must contain all eight complete-vocabulary outputs")
            replays = []
            report["request_replays"] = replays
            if args.validate_teacher_forced:
                teacher = {}
                report["teacher_forced_validation"] = teacher
                reset_caches(runtime, ranks, host_cache)
                try:
                    graphs.replay(
                        tokenizer, request, memory, teacher, options=args, reference=reference, teacher_forced=True,
                        audit=audit_caches if args.audit_caches else None, label="teacher_forced",
                    )
                except LogitValidationError as error:
                    report["teacher_forced_error"] = str(error)
            total = args.warmup_requests + args.measured_requests
            for iteration in range(total):
                if iteration or args.validate_teacher_forced:
                    reset_caches(runtime, ranks, host_cache)
                label = f"warmup_{iteration}" if iteration < args.warmup_requests else f"measured_{iteration}"
                replay_report = {}
                capture_dir = args.profile_all_steps if iteration == args.warmup_requests else None
                if capture_dir is not None:
                    report["profile_request"] = replay_report
                replays.append(replay_report)
                audit_request = args.audit_caches and (iteration < args.warmup_requests or not args.warmup_requests)
                try:
                    graphs.replay(
                        tokenizer, request, memory, replay_report, options=args,
                        reference=reference, teacher_forced=args.teacher_forced,
                        audit=audit_caches if audit_request else None, save_logits=args.save_logits, label=label,
                        profile=capture_dir is not None, profile_dir=capture_dir,
                    )
                finally:
                    report.update(replay_report)
            measured = replays[args.warmup_requests:]
            timed = [replay for replay in measured if not replay["profiled"]] or measured
            report["benchmark"] = dict(warmup_requests=args.warmup_requests,
                                       measured_requests=args.measured_requests,
                                       sample_labels=[replay["label"] for replay in timed],
                                       instrumented=any(r["profiled"] or r["instrumented_cache_audits"] for r in timed),
                                       clock="host monotonic")
            for metric in ("ttft_seconds", "tpot_seconds", "end_to_end_seconds"):
                values = torch.tensor([replay[metric] for replay in timed], dtype=torch.float64)
                report["benchmark"][metric] = dict(median=float(values.quantile(0.5)),
                                                  p95=float(values.quantile(0.95)), samples=values.tolist())
            if args.profile_last_decode:
                reset_caches(runtime, ranks, host_cache)
                profiled = {}
                report["profile_request"] = profiled
                graphs.replay(
                    tokenizer, request, memory, profiled, options=args, reference=reference, label="profile", profile=True,
                )
            memory.sample()
            if "layer_validation_error" in report:
                raise ValueError(report["layer_validation_error"])
            if "teacher_forced_error" in report:
                raise ValueError(report["teacher_forced_error"])
        if args.profile_all_steps is not None:
            report["operator_timings"] = summarize_swimlanes(args.profile_all_steps, devices, owner, convert=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        torch.set_num_threads(saved_threads)
        memory.stopped.set()
        memory.thread.join(timeout=22)
        report["hbm"] = memory.report()
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report
