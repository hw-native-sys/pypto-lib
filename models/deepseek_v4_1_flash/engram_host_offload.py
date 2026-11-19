# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
# ci: no-sim
"""Standalone Host FP8 lookup with Device scales and request-history validation."""

import argparse
from bisect import bisect_right
from contextlib import ExitStack
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time

import pypto.language as pl
import pypto.language.distributed as pld
import psutil
import torch
from pypto.backend import BackendType
from pypto.ir import DistributedCompiledProgram, DistributedConfig
from pypto.runtime import DistributedWorker, ExecutionMode, RunConfig, StackedDeviceTensor

from config import EP_SIZE, FLASH as M
from engram import EngramHistory, FETCH_TILE, HEAD_DIM, N_HASH_COLS, engram_host_lookup
from golden import validate_golden
from quant import decode_e8m0


# Dynamic lookup rows across Prefill and Decode.
ENGRAM_LOOKUPS_DYN = pl.dynamic("ENGRAM_LOOKUPS_DYN")
_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--checkpoint")
_shape_parser.add_argument("--engram-table-rows", type=int)
_shape_parser.add_argument("--engram-strategy", choices=("sliced_enp", "full_vocab_local"), default="sliced_enp")
_shape_options = _shape_parser.parse_known_args()[0]
if _shape_options.engram_table_rows is None:
    _shape_options.engram_table_rows = max(M.engram_num_embeddings) if _shape_options.checkpoint else 4096
if _shape_options.engram_table_rows <= 0:
    _shape_parser.error("--engram-table-rows must be positive")
ENGRAM_ROWS = (_shape_options.engram_table_rows + EP_SIZE - 1) // EP_SIZE * EP_SIZE
ENGRAM_FULL_VOCAB = _shape_options.engram_strategy == "full_vocab_local"
ENGRAM_SHARD_ROWS = ENGRAM_ROWS if ENGRAM_FULL_VOCAB else ENGRAM_ROWS // EP_SIZE


def write_forward_engram_layout(directory):
    """Record compile-time Host-window layout beside the integrated forward artifact."""
    layout = {
        "abi_version": 1, "ranks": EP_SIZE, "rows_per_layer": ENGRAM_ROWS,
        "strategy": "full_vocab_local" if ENGRAM_FULL_VOCAB else "sliced_enp",
    }
    path = Path(directory) / "engram_layout.json"
    path.write_text(json.dumps(layout, indent=2) + "\n")


def validate_forward_engram_layout(directory, ranks, rows, strategy):
    """Reject artifacts compiled for different Host-window ownership or table extents."""
    path = Path(directory) / "engram_layout.json"
    expected = {"abi_version": 1, "ranks": ranks, "rows_per_layer": rows, "strategy": strategy}
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError("forward Engram layout is missing or mismatched; recompile with matching --ep/--engram-table-rows/--engram-strategy")


def build_host_lookup(ranks, rows_per_layer, full_vocab):
    """Allocate two Host layer payloads and two Device staging tensors per rank."""
    shard_rows = rows_per_layer if full_vocab else rows_per_layer // ranks

    @pl.jit
    def lookup(
        ids: pl.Tensor[[ENGRAM_LOOKUPS_DYN, 1], pl.INT32],
        scales: pl.Tensor[[2 * rows_per_layer, HEAD_DIM // 32], pl.UINT8],
        output: pl.Out[pl.Tensor[[ENGRAM_LOOKUPS_DYN, HEAD_DIM], pl.BF16]],
        table: pld.DistributedTensor[[2 * shard_rows, HEAD_DIM], pl.UINT8],
    ):
        ids.bind_dynamic(0, ENGRAM_LOOKUPS_DYN)
        output.bind_dynamic(0, ENGRAM_LOOKUPS_DYN)
        staging_first = pl.create_tensor([FETCH_TILE, HEAD_DIM], dtype=pl.UINT8)
        staging_second = pl.create_tensor([FETCH_TILE, HEAD_DIM], dtype=pl.UINT8)
        output = engram_host_lookup(ids, table, staging_first, staging_second, scales, output, full_vocab, shard_rows)
        return output

    @pl.jit.host
    def host_lookup(
        ids: pl.Tensor[[ranks, ENGRAM_LOOKUPS_DYN, 1], pl.INT32],
        scales: pl.Tensor[[ranks, 2 * rows_per_layer, HEAD_DIM // 32], pl.UINT8],
        output: pl.Out[pl.Tensor[[ranks, ENGRAM_LOOKUPS_DYN, HEAD_DIM], pl.BF16]],
    ):
        ids.bind_dynamic(1, ENGRAM_LOOKUPS_DYN)
        output.bind_dynamic(1, ENGRAM_LOOKUPS_DYN)
        payload = pld.alloc_window_buffer(
            [2 * shard_rows, HEAD_DIM], dtype=pl.UINT8, memory_source="host", init="engram_payload",
        )
        for rank in pl.range(ranks):
            table = pld.window(payload, [2 * shard_rows, HEAD_DIM], dtype=pl.UINT8)
            lookup(ids[rank], scales[rank], output[rank], table, device=rank)
        return output

    return host_lookup


class EngramPayloadSource(io.RawIOBase):
    """Expose checkpoint slices and zero padding as one seekable Host-window source."""

    def __init__(self, segments):
        super().__init__()
        self._files = ExitStack()
        self._segments = []
        self._ends = []
        self._position = 0
        total = 0
        try:
            for path, offset, count in segments:
                if offset < 0 or count < 0:
                    raise ValueError("Engram source ranges must be nonnegative")
                if count:
                    file = None if path is None else self._files.enter_context(Path(path).open("rb"))
                    self._segments.append((file, offset, count))
                    total += count
                    self._ends.append(total)
        except BaseException:
            self.close()
            raise
        self._size = total

    def readable(self):
        return not self.closed

    def seekable(self):
        return not self.closed

    def tell(self):
        if self.closed:
            raise ValueError("Engram source is closed")
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        if self.closed:
            raise ValueError("Engram source is closed")
        if whence == io.SEEK_CUR:
            offset += self._position
        elif whence == io.SEEK_END:
            offset += self._size
        elif whence != io.SEEK_SET:
            raise ValueError("invalid Engram seek origin")
        if offset < 0:
            raise ValueError("negative Engram source position")
        self._position = offset
        return offset

    def readinto(self, buffer):
        if self.closed:
            raise ValueError("Engram source is closed")
        destination = memoryview(buffer).cast("B")
        copied = 0
        while copied < len(destination) and self._position < self._size:
            segment = bisect_right(self._ends, self._position)
            start = 0 if segment == 0 else self._ends[segment - 1]
            file, offset, length = self._segments[segment]
            local = self._position - start
            count = min(length - local, len(destination) - copied)
            target = destination[copied:copied + count]
            if file is None:
                target[:] = bytes(count)
            else:
                file.seek(offset + local)
                count = file.readinto(target)
                if not count:
                    raise EOFError("truncated Engram checkpoint payload")
            copied += count
            self._position += count
        return copied

    def close(self):
        self._files.close()
        super().close()


class EngramHostTables:
    """Own streamed Host payload sources and complete Device-resident scale replicas."""

    def __init__(self, checkpoint, ranks, strategy, *, num_embeddings=M.engram_num_embeddings, program_count=1):
        if ranks not in (4, 8) or strategy not in ("sliced_enp", "full_vocab_local"):
            raise ValueError("Engram offload requires ENP4/ENP8 or Full-Vocab on four/eight ranks")
        if program_count < 1:
            raise ValueError("Engram Host-window program count must be positive")
        self.ranks = ranks
        self.strategy = strategy
        self.payload_spans = []
        self.scale_spans = []
        if len(num_embeddings) != len(M.engram_layer_ids) or any(rows <= 0 for rows in num_embeddings):
            raise ValueError("Engram table sizes must contain one positive row count per layer")
        for slot, layer in enumerate(M.engram_layer_ids):
            prefix = f"layers.{layer}.engram.embed."
            payload = checkpoint.tensor_span(prefix + "weight")
            scale = checkpoint.tensor_span(prefix + "scale")
            if payload[2] != (num_embeddings[slot], HEAD_DIM) or payload[3] != torch.float8_e4m3fn:
                raise ValueError(f"unsupported Engram payload layout: {prefix}weight")
            if scale[2] != (payload[2][0], HEAD_DIM // 32) or scale[3] != torch.float8_e8m0fnu:
                raise ValueError(f"unsupported Engram scale layout: {prefix}scale")
            self.payload_spans.append(payload)
            self.scale_spans.append(scale)
        self.rows_per_layer = (max(span[2][0] for span in self.payload_spans) + ranks - 1) // ranks * ranks
        full_vocab = strategy == "full_vocab_local"
        self.shard_rows = self.rows_per_layer if full_vocab else self.rows_per_layer // ranks
        self.program_count = program_count
        self.host_payload_bytes_per_program = ranks * 2 * self.shard_rows * HEAD_DIM
        self.host_payload_bytes = program_count * self.host_payload_bytes_per_program
        self.device_scale_bytes_per_rank = 2 * self.rows_per_layer * (HEAD_DIM // 32)
        self.upload_buffer = torch.empty(64 * 1024**2, dtype=torch.uint8).share_memory_()
        self.sources = []
        self.device_scales = None

    def __enter__(self):
        available = psutil.virtual_memory().available
        if self.host_payload_bytes > available:
            raise MemoryError(
                f"{self.strategy} requires {self.host_payload_bytes} Host payload bytes, "
                f"only {available} are available; shared Full-Vocab L3 binding is not implemented"
            )
        try:
            for rank in range(self.ranks):
                row = 0 if self.strategy == "full_vocab_local" else rank * self.shard_rows
                segments = []
                for path, offset, shape, _dtype in self.payload_spans:
                    count = max(0, min(self.shard_rows, shape[0] - row))
                    segments.append((path, offset + row * HEAD_DIM, count * HEAD_DIM))
                    segments.append((None, 0, (self.shard_rows - count) * HEAD_DIM))
                self.sources.append(EngramPayloadSource(segments))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exception):
        for source in self.sources:
            source.close()
        self.sources.clear()

    def upload_scales(self, runtime):
        """Upload each native scale byte once to every rank using bounded shared scratch."""
        shape = (2 * self.rows_per_layer, HEAD_DIM // 32)
        shards = [runtime.alloc_tensor(shape, torch.uint8, worker_id=rank) for rank in range(self.ranks)]
        target = memoryview(self.upload_buffer.numpy()).cast("B")
        for layer, (path, offset, source_shape, _dtype) in enumerate(self.scale_spans):
            live = source_shape[0] * (HEAD_DIM // 32)
            layer_bytes = self.rows_per_layer * (HEAD_DIM // 32)
            with path.open("rb") as file:
                file.seek(offset)
                for begin in range(0, layer_bytes, len(target)):
                    count = min(len(target), layer_bytes - begin)
                    valid = max(0, min(count, live - begin))
                    if valid:
                        read = file.readinto(target[:valid])
                        if read != valid:
                            raise EOFError("truncated Engram scale table")
                    if valid < count:
                        target[valid:count] = bytes(count - valid)
                    for rank, shard in enumerate(shards):
                        offset = layer * layer_bytes + begin
                        runtime.copy_to(
                            shard.data_ptr, self.upload_buffer.data_ptr(), count, dst_offset=offset, worker_id=rank,
                        )
        stack_shape = (self.ranks, *shape)
        self.device_scales = StackedDeviceTensor(shards, stack_shape, tuple(range(self.ranks)))
        return self.device_scales


def prepare_model_lookup(directory, ranks, strategy):
    """Compile the released table extents without reading or allocating table payloads."""
    if ranks not in (4, 8):
        raise ValueError("model Engram offload supports four or eight ranks")
    directory.mkdir(parents=True, exist_ok=True)
    rows = (max(M.engram_num_embeddings) + ranks - 1) // ranks * ranks
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=list(range(ranks)), num_sub_workers=0),
    )
    program = build_host_lookup(ranks, rows, strategy == "full_vocab_local").compile(config=config)
    from pypto.runtime.distributed_runner import _assemble_chip_callables
    _assemble_chip_callables(program)
    manifest = {
        "directory": str(program.output_dir), "ranks": ranks, "rows_per_layer": rows,
        "strategy": strategy, "dynamic_lookups": True, "source_sha256": source_identity(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def golden_host_lookup(table, scales, ids):
    """Gather native FP8 rows and group-32 scales with zero inactive rows."""
    rows = table.shape[1]
    lookups = ids.shape[1]
    layer = torch.arange(lookups) // (lookups // 2)
    selection = ids[..., 0].long()
    indices = selection.clamp_min(0) + layer * rows
    payload = table.flatten(0, 1)[indices].view(torch.float8_e4m3fn).float()
    codes = scales.flatten(0, 1)[indices]
    values = payload * decode_e8m0(codes).repeat_interleave(32, dim=-1)
    values.masked_fill_((selection < 0).unsqueeze(-1), 0)
    return values.to(torch.bfloat16)


def build_history_fixture():
    """Use small prime buckets and token aliases with the production hash code."""
    primes = []
    candidate = 2
    while len(primes) < 2 * N_HASH_COLS:
        if all(candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1)):
            primes.append(candidate)
        candidate += 1
    generator = torch.Generator().manual_seed(10007)
    multipliers = torch.randint(0, (1 << 62) // 64, (2, M.engram_max_ngram_size), generator=generator)
    multipliers = multipliers * 2 + 1
    return EngramHistory(torch.arange(64) // 2, 1, torch.tensor(primes).reshape(2, N_HASH_COLS), multipliers)


def source_identity():
    """Fingerprint the lookup entry and its shared compute implementation."""
    paths = [Path(__file__), Path(__file__).with_name("engram.py")]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def prepare_case(directory, ranks, tokens, strategy, table_rows=None):
    """Freeze table bytes, changing histories, CPU goldens and compiled artifacts."""
    if ranks not in (4, 8) or tokens < 1:
        raise ValueError("validation requires four or eight ranks and a positive token count")
    directory.mkdir(parents=True, exist_ok=True)
    history = build_history_fixture()
    actual_rows = history.primes.sum(-1)
    requested_rows = int(actual_rows.max()) if table_rows is None else table_rows
    if requested_rows < int(actual_rows.max()):
        raise ValueError("--table-rows must cover the validation hash buckets")
    rows = (requested_rows + ranks - 1) // ranks * ranks
    lookups = 2 * tokens * N_HASH_COLS
    raw = torch.arange(2 * rows * HEAD_DIM, dtype=torch.int64).reshape(2, rows, HEAD_DIM)
    values = ((raw * 17 + raw // HEAD_DIM * 13) % 127 - 63).float() / 8
    table = values.to(torch.float8_e4m3fn).view(torch.uint8)
    row_ids = torch.arange(2 * rows).reshape(2, rows)
    for column in range(3):
        table[:, :, column] = ((row_ids // (127 ** column)) % 127).to(torch.uint8)
    scales = (torch.arange(2 * rows * (HEAD_DIM // 32)) % 17 + 119).to(torch.uint8)
    scales = scales.reshape(2, rows, HEAD_DIM // 32)
    prefix = [torch.empty(0, dtype=torch.int64) for _ in range(ranks)]
    rounds = []
    for step in range(3):
        ids = torch.full((ranks, lookups, 1), -1, dtype=torch.int32)
        for rank in range(ranks):
            active = tokens if tokens > 3 else max(0, tokens - rank)
            current = (torch.arange(active) * 11 + rank * 13 + step * 7) % 64
            start = prefix[rank].numel() if step < 2 else prefix[rank].numel() // 2
            hashes = history.hash_ids(current, start, request_id=rank)
            reference = build_history_fixture()
            merged = torch.cat((prefix[rank][:start], current))
            expected_hashes = reference.hash_ids(merged)[:, start:]
            if not torch.equal(hashes, expected_hashes):
                raise RuntimeError("chunked/rollback history differs from one-shot hashing")
            prefix[rank] = merged
            shaped = ids[rank].reshape(2, tokens, N_HASH_COLS)
            shaped[:, :active] = hashes.to(torch.int32)
        expected = golden_host_lookup(table, scales, ids)
        name = f"round-{step}.pt"
        torch.save({"ids": ids, "expected": expected}, directory / name)
        rounds.append(name)
    ids = torch.empty((ranks, lookups, 1), dtype=torch.int32)
    boundaries = [0, rows - 1, *[i * (rows // ranks) for i in range(ranks)]]
    boundaries.extend(i * (rows // ranks) - 1 for i in range(1, ranks))
    choices = torch.tensor([*boundaries, -1, -1], dtype=torch.int32)
    for rank in range(ranks):
        ids[rank, :, 0] = choices[(torch.arange(lookups) + rank) % choices.numel()]
    torch.save({"ids": ids, "expected": golden_host_lookup(table, scales, ids)}, directory / "boundaries.pt")
    rounds.append("boundaries.pt")
    generator = torch.Generator().manual_seed(1830)
    mixed = torch.randint(rows, (ranks, lookups, 1), generator=generator, dtype=torch.int32)
    modes = ["mixed"] if strategy == "full_vocab_local" else ["local", "remote", "mixed"]
    for mode in modes:
        ids = mixed.clone()
        owner = ids[..., 0] // (rows // ranks)
        local = owner == torch.arange(ranks)[:, None]
        if mode == "local":
            ids[..., 0].masked_fill_(~local, -1)
        elif mode == "remote":
            ids[..., 0].masked_fill_(local, -1)
        record = {"ids": ids, "expected": golden_host_lookup(table, scales, ids)}
        torch.save(record, directory / f"benchmark-{mode}.pt")
    full_vocab = strategy == "full_vocab_local"
    sources = [table.flatten(0, 1)] * ranks if full_vocab else [
        part.contiguous().flatten(0, 1) for part in table.chunk(ranks, dim=1)
    ]
    fixture = {"sources": sources, "scales": scales.flatten(0, 1).repeat(ranks, 1, 1)}
    torch.save(fixture, directory / "fixture.pt")
    ids = torch.zeros((ranks, lookups, 1), dtype=torch.int32)
    outputs = torch.empty((ranks, lookups, HEAD_DIM), dtype=torch.bfloat16)
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=list(range(ranks)), num_sub_workers=0),
    )
    program = build_host_lookup(ranks, rows, full_vocab).compile(ids, fixture["scales"], outputs, config=config)
    from pypto.runtime.distributed_runner import _assemble_chip_callables
    _assemble_chip_callables(program)
    manifest = {
        "directory": str(program.output_dir), "ranks": ranks, "tokens": tokens,
        "strategy": strategy, "rows_per_layer": rows, "actual_rows_per_layer": actual_rows.tolist(),
        "dynamic_lookups": True,
        "benchmark_modes": modes,
        "rounds": rounds, "source_sha256": source_identity(),
        "host_physical_storage": "independent per-rank windows", "scale_storage": "Device-resident replicas",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(directory), **manifest}), flush=True)


def validate_case(directory, devices, swimlane=0):
    """Load frozen cases and reuse Device scales and retained Host windows."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source_sha256"] != source_identity():
        raise ValueError("source changed since preparation; regenerate this artifact")
    if len(devices) != manifest["ranks"] or len(set(devices)) != len(devices):
        raise ValueError("allocated devices must match the prepared rank count")
    (directory / "results.json").unlink(missing_ok=True)
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, enable_chip_swimlane=swimlane,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
    )
    program = DistributedCompiledProgram.from_dir(manifest["directory"], distributed_config=config.distributed_config)
    fixture = torch.load(directory / "fixture.pt", weights_only=True)
    lookups = 2 * manifest["tokens"] * N_HASH_COLS
    outputs = torch.empty((len(devices), lookups, HEAD_DIM), dtype=torch.bfloat16).share_memory_()
    ids = torch.empty((len(devices), lookups, 1), dtype=torch.int32).share_memory_()
    results = []
    with DistributedWorker(
        program, config=config, persistent=True, reset_persistent_windows=False,
        window_sources={"engram_payload": fixture["sources"]},
    ) as worker:
        if worker.dc.device_ids != devices:
            raise RuntimeError("runtime device binding differs from allocation")
        device_scales = worker.alloc_stacked_tensor(fixture["scales"])
        for name in manifest["rounds"]:
            frozen = torch.load(directory / name, weights_only=True)
            ids.copy_(frozen["ids"])
            outputs.fill_(float("nan"))
            worker(ids, device_scales, outputs)
            validate_golden({"output": outputs}, {"output": frozen["expected"]}, atol=0, rtol=0)
            record = {"case": name, "status": "PASS"}
            results.append(record)
            print(json.dumps(record), flush=True)
    result = {
        "status": "PASS", "allocated_devices": devices, "strategy": manifest["strategy"],
        "tokens": manifest["tokens"], "cases": results, "payload_bytes_per_rank": lookups * HEAD_DIM,
        "staging_bytes_per_rank": 2 * FETCH_TILE * HEAD_DIM, "device_scale_bytes_per_rank": fixture["scales"][0].numel(),
        "host_physical_storage": manifest["host_physical_storage"], "source_sha256": source_identity(),
    }
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def payload_counts(ids, rows_per_layer, strategy):
    """Count useful row bytes on each path, excluding inactive rows and protocol traffic."""
    ranks = ids.shape[0]
    selection = ids[..., 0]
    active = selection >= 0
    owner = selection // (rows_per_layer // ranks)
    local = active if strategy == "full_vocab_local" else active & (owner == torch.arange(ranks)[:, None])
    return [
        {"rank": rank, "local_bytes": int(local[rank].sum()) * HEAD_DIM,
         "remote_bytes": int((active[rank] & ~local[rank]).sum()) * HEAD_DIM}
        for rank in range(ranks)
    ]


def sample_summary(samples):
    """Report nearest-rank P99 and the sample median in microseconds."""
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    return {"p50_us": statistics.median(samples), "p99_us": sorted(samples)[math.ceil(0.99 * len(samples)) - 1]}


def host_memory_snapshot():
    """Sum process-tree proportional memory without repeatedly counting shared pages."""
    parent = psutil.Process()
    processes = [parent, *parent.children(recursive=True)]
    pss = 0
    for process in processes:
        try:
            pss += process.memory_full_info().pss
        except psutil.NoSuchProcess:
            continue
    return {"process_tree_pss_bytes": pss, "system_available_bytes": psutil.virtual_memory().available}


def benchmark_case(directory, devices, rounds, warmup):
    """Measure retained lookup and matched local/remote subsets with resident Device outputs."""
    from pypto.runtime.bench import _capture_fd_stderr, _parse_stats_from_strace
    from pypto.runtime.log_config import configure_log, current_level

    if rounds < 100 or warmup < 1:
        raise ValueError("P99 benchmarking requires at least 100 samples and one warmup per path")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source_sha256"] != source_identity():
        raise ValueError("source changed since preparation; regenerate this artifact")
    if len(devices) != manifest["ranks"] or len(set(devices)) != len(devices):
        raise ValueError("allocated devices must match the prepared rank count")
    (directory / "benchmark-results.json").unlink(missing_ok=True)
    cfg = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
    )
    program = DistributedCompiledProgram.from_dir(manifest["directory"], distributed_config=cfg.distributed_config)
    fixture = torch.load(directory / "fixture.pt", weights_only=True)
    cases = [(mode, torch.load(directory / f"benchmark-{mode}.pt", weights_only=True)) for mode in manifest["benchmark_modes"]]
    for _mode, case in cases:
        case["ids"].share_memory_()
    shape = tuple(cases[0][1]["expected"].shape)
    output = torch.empty(shape, dtype=torch.bfloat16)
    trace_path = directory / "benchmark.strace"
    before = host_memory_snapshot()
    snapshots, host_samples = {}, {}
    previous_level = current_level()
    configure_log("timing")
    try:
        with _capture_fd_stderr(trace_path):
            with program.prepare(
                cfg, persistent=True, reset_persistent_windows=False,
                window_sources={"engram_payload": fixture["sources"]},
            ) as runtime:
                device_scales = runtime.alloc_stacked_tensor(fixture["scales"])
                shards = [runtime.alloc_tensor(shape[1:], torch.bfloat16, worker_id=rank) for rank in range(len(devices))]
                resident = StackedDeviceTensor(shards, shape, tuple(range(len(devices))))
                for mode, case in cases:
                    samples = []
                    for iteration in range(warmup + rounds):
                        start = time.perf_counter_ns()
                        runtime.run(program, case["ids"], device_scales, resident, config=cfg)
                        samples.append((time.perf_counter_ns() - start) / 1000)
                        if iteration in (0, warmup + rounds - 1):
                            runtime.copy_stacked_from(resident, output)
                            validate_golden({"output": output}, {"output": case["expected"]}, atol=0, rtol=0)
                    host_samples[mode] = samples[warmup:]
                    snapshots[mode] = host_memory_snapshot()
    except BaseException:
        if trace_path.exists():
            print(trace_path.read_text(), file=sys.stderr)
        raise
    finally:
        configure_log(previous_level)
    count = len(cases) * (warmup + rounds)
    stats = _parse_stats_from_strace(trace_path.read_text(), rounds=count, warmup=0, distributed=True)
    per_rank = stats.per_rank("effective")
    if len(per_rank) != len(devices) or any(len(values) != count for values in per_rank.values()):
        raise RuntimeError("incomplete device timing; retain raw trace and do not publish partial percentiles")
    pid_devices = {}
    for line in trace_path.read_text().splitlines():
        if "[chip_process pid=" in line and " dev=" in line:
            pid = int(line.split("[chip_process pid=", 1)[1].split()[0])
            device = int(line.split(" dev=", 1)[1].split("]", 1)[0])
            pid_devices[pid] = device
    results = {}
    for index, (mode, case) in enumerate(cases):
        start = index * (warmup + rounds) + warmup
        samples = {pid: values[start:start + rounds] for pid, values in per_rank.items()}
        payload = payload_counts(case["ids"], manifest["rows_per_layer"], manifest["strategy"])
        rank_results = []
        for pid, values in samples.items():
            device = pid_devices.get(pid)
            if device not in devices:
                raise RuntimeError("device timing PID cannot be mapped to an allocated device")
            rank = devices.index(device)
            useful = payload[rank]["local_bytes"] + payload[rank]["remote_bytes"]
            rank_results.append({"rank": rank, "device": device, "samples_us": values, **sample_summary(values),
                                 **payload[rank], "effective_GB_s": useful / statistics.median(values) / 1000})
        group = [max(values[iteration] for values in samples.values()) for iteration in range(rounds)]
        results[mode] = {
            "ranks": rank_results, "group_max_effective_samples_us": group, **sample_summary(group),
            "fastest_rank_median_us": min(row["p50_us"] for row in rank_results),
            "host_submit_to_complete_samples_us": host_samples[mode], "memory": snapshots[mode],
        }
    result = {
        "status": "PASS", "allocated_devices": devices, "rounds": rounds, "warmup": warmup,
        "strategy": manifest["strategy"], "tokens": manifest["tokens"], "rows_per_layer": manifest["rows_per_layer"],
        "source_sha256": source_identity(), "paths": results, "memory_before": before,
        "registered_payload_bytes": len(devices) * fixture["sources"][0].numel(),
        "staging_bytes_per_rank": 2 * FETCH_TILE * HEAD_DIM,
        "device_scale_bytes_per_rank": fixture["scales"][0].numel(),
        "timing_boundary": "Device effective lookup window: orchestration, GET completion, MTE, dequantization and stores; setup and explicit output copy-back excluded",
        "cache_policy": "fixed random IDs, retained warm tables, no flush or deduplication",
        "contention_boundary": "local/remote isolation masks the complementary mixed-case rows; masked rows still execute zero stores; mixed execution permits pipelined overlap",
        "physical_host_bandwidth": None, "ub_interconnect_traffic": None, "host_device_link_utilization": None,
        "unavailable_metrics": "physical counters and per-path concurrent completion require platform profiling; useful bytes/time is effective bandwidth",
        "end_to_end_model_latency": None, "full_model_validation": False,
    }
    if "local" in results:
        isolated = results["local"]["p50_us"] + results["remote"]["p50_us"]
        result["mixed_over_sum_isolated"] = results["mixed"]["p50_us"] / isolated
    (directory / "benchmark-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"benchmark": str(directory / "benchmark-results.json"), "status": "PASS"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a5",), default="a5")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--model-tables", action="store_true", help="compile released table extents without loading weights")
    parser.add_argument("--ranks", type=int, choices=(4, 8), default=4)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--table-rows", type=int, help="padded rows per layer for synthetic table-size sweeps")
    parser.add_argument("--benchmark", action="store_true", help="explicit retained lookup benchmark with raw P50/P99 samples")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--enable-chip-swimlane", type=int, choices=range(5), default=0)
    parser.add_argument("--strategy", choices=("sliced_enp", "full_vocab_local"), default="sliced_enp")
    parser.add_argument("--artifact", "--runtime-dir", type=Path, default=Path("build_output/engram-host-offload"))
    parser.add_argument("-d", "--device", "--devices", dest="devices", help="queue-assigned comma-separated devices")
    args = parser.parse_args()
    if args.sim:
        parser.error("Host URMA lookup requires A5 hardware; simulation is unsupported")
    if args.model_tables and not args.compile_only:
        parser.error("--model-tables requires --compile-only; execution belongs to the checkpoint forward")
    if args.compile_only:
        if args.model_tables:
            prepare_model_lookup(args.artifact, args.ranks, args.strategy)
        else:
            prepare_case(args.artifact, args.ranks, args.tokens, args.strategy, args.table_rows)
    else:
        if not args.devices:
            parser.error("--devices is required for execution; prepare first with --compile-only")
        if not (args.artifact / "manifest.json").is_file():
            parser.error("artifact is missing; run --compile-only outside the device allocation first")
        devices = [int(device) for device in args.devices.split(",")]
        if args.benchmark:
            if args.enable_chip_swimlane:
                parser.error("capture a separate correctness swimlane; benchmark timing must not include dep-gen")
            benchmark_case(args.artifact, devices, args.rounds, args.warmup)
        else:
            validate_case(args.artifact, devices, args.enable_chip_swimlane)
