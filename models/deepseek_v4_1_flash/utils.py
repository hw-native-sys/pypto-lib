# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared Host utilities, metadata lowering, and validation helpers."""

import argparse
from bisect import bisect_right
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path

import pypto.language as pl
import torch
from pypto.runtime import StackedDeviceTensor

import config as C
from config import BLOCK_SIZE, EP_SIZE, FLASH as M
from engram import HEAD_DIM


def parse_decode_startpos(value):
    """Parse one absolute starting position per request."""
    import argparse

    try:
        positions = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("start-pos must be a comma-separated list of integers") from error
    if not positions:
        raise argparse.ArgumentTypeError("start-pos must contain at least one position")
    max_start = M.max_position_embeddings - 1
    if any(position < 0 or position > max_start for position in positions):
        raise argparse.ArgumentTypeError(f"positions must be in [0, {max_start}]")
    return positions


def parse_decode_query_lens(value):
    """Parse positive query lengths in request order."""
    import argparse

    try:
        lengths = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("query-lens must be a comma-separated list of integers") from error
    if any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("query-lens must contain positive lengths")
    return lengths


def configure_decode_workload(args, parser):
    """Resolve query lengths, packed token count, and absolute position bounds."""
    if args.startpos is None:
        if args.query_lens is not None:
            parser.error("--query-lens requires --start-pos")
        return
    args.requests = len(args.startpos)
    if args.query_lens is None:
        args.query_lens = [C.DECODE_SEQ] * args.requests
    if len(args.query_lens) != args.requests:
        parser.error("--query-lens must contain one length per --start-pos value")
    args.tokens = sum(args.query_lens)
    ends = [pos + length for pos, length in zip(args.startpos, args.query_lens)]
    if max(ends) > M.max_position_embeddings:
        parser.error("a query block exceeds max_position_embeddings")


def parse_prefill_groups(value):
    """Parse comma-separated request values, with semicolons separating DP groups."""
    import argparse

    try:
        return [[int(item) for item in group.split(",")] for group in value.split(";")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integers and semicolons between DP groups") from error


def prefill_index_validation_cases(args, ratio):
    """Exercise score-dependent TopK in the default C1A/C2A full validation."""
    import copy
    import os
    import sys

    # CI supplies only the platform and device. A requested workload, capture,
    # saved fixture, or benchmark must run exactly the selected invocation.
    explicit = any(arg.startswith("--") and arg.partition("=")[0] not in ("--platform", "--device")
                   for arg in sys.argv[1:])
    benchmark = os.environ.get("PYPTO_BENCH", "").strip() not in ("", "0", "false", "False")
    if explicit or benchmark:
        return [args]

    scored = copy.copy(args)
    scored.query_lens = [[8]]
    scored.start_pos = [[ratio * (M.index_topk + 32)]]
    return [args, scored]


def prefill_batch_layout(args):
    """Resolve independent per-DP request lengths and absolute starting positions."""
    lengths = getattr(args, "query_lens", None)
    counts = getattr(args, "dp_tokens", None)
    if lengths is None:
        counts = counts if counts is not None else [args.tokens] * args.dp
        requests = getattr(args, "requests", 1)
        if requests < 1 or len(counts) != args.dp:
            raise ValueError("requests must be positive and dp-tokens must contain one count per DP")
        lengths = [[count // requests + (index < count % requests) for index in range(requests)] for count in counts]
    elif len(lengths) == 1:
        lengths = lengths * args.dp
    if len(lengths) != args.dp or any(not group for group in lengths):
        raise ValueError("query-lens must provide one nonempty request list per DP, or one list to broadcast")
    if counts is not None and [sum(group) for group in lengths] != list(counts):
        raise ValueError("dp-tokens must equal the sum of query-lens in each DP")
    if any(n < 0 for group in lengths for n in group) or any(sum(group) > args.tokens for group in lengths):
        raise ValueError("request lengths must be nonnegative and fit the per-DP token capacity")
    if C.CP_SIZE > 1 and any(len(group) != 1 for group in lengths):
        raise ValueError("CP > 1 requires one prefill request per DP group")
    starts = getattr(args, "start_pos", None)
    if starts is None:
        starts = [[127 if getattr(args, "case", "normal") == "prefix" else 0] * len(group) for group in lengths]
    elif len(starts) == 1:
        starts = starts * args.dp
    if len(starts) != args.dp or any(len(p) != len(q) for p, q in zip(starts, lengths)):
        raise ValueError("start-pos must match each DP group's request list")
    if any(p < 0 or p + n > M.max_position_embeddings for ps, ns in zip(starts, lengths) for p, n in zip(ps, ns)):
        raise ValueError("each request must fit max_position_embeddings")
    return lengths, starts


def prefill_fixture_lengths(lengths, tokens):
    """Append physical padding to the final request in a packed fixture."""
    padded = list(lengths)
    padded[-1] += tokens - sum(padded)
    return padded


def prefill_window_pages(lengths, starts):
    """Count the paged SWA fixture's retained and current rows, including its spare page."""
    return 1 + sum((p + n + 128) // 128 - max(0, p - 127) // 128 + 1 for p, n in zip(starts, lengths))


def packed_query_lengths(tokens, requests, query_lens=None):
    """Use explicit request lengths or a balanced packed token split."""
    if query_lens is None:
        return [tokens // requests + (index < tokens % requests) for index in range(requests)]
    if len(query_lens) != requests or sum(query_lens) != tokens or any(length < 0 for length in query_lens):
        raise ValueError("query_lens must contain one nonnegative length per request and sum to tokens")
    return list(query_lens)


def paged_slots(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int = BLOCK_SIZE,
    logical_divisor: int = 1,
    publish_only_complete: bool = False,
    *, rolling: bool = False,
) -> torch.Tensor:
    """Map logical token positions to flattened physical cache rows."""
    if positions.ndim != 1 or request_ids.shape != positions.shape or block_table.ndim != 2:
        raise ValueError("positions/request_ids must be matching vectors and block_table must be a matrix")
    if storage_block_size <= 0 or logical_divisor <= 0:
        raise ValueError("storage_block_size and logical_divisor must be positive")
    if any(value.dtype not in (torch.int32, torch.int64) for value in (positions, request_ids, block_table)):
        raise ValueError("positions, request_ids and block_table must contain integer indices")
    publish = torch.ones_like(positions, dtype=torch.bool)
    if publish_only_complete and logical_divisor > 1:
        publish = (positions + 1).remainder(logical_divisor) == 0
    slots = torch.full_like(positions, -1, dtype=torch.int64)
    logical = torch.div(positions[publish], logical_divisor, rounding_mode="floor")
    logical_block = torch.div(logical, storage_block_size, rounding_mode="floor")
    if rolling:
        logical_block = logical_block.remainder(block_table.shape[1])
    offset = logical.remainder(storage_block_size)
    active_requests = request_ids[publish]
    if bool(((active_requests < 0) | (active_requests >= block_table.shape[0])).any()):
        raise ValueError("published request id is outside the block table")
    if bool(((logical_block < 0) | (logical_block >= block_table.shape[1])).any()):
        raise ValueError("block table does not cover a required logical page")
    physical = block_table[request_ids[publish].to(torch.long), logical_block.to(torch.long)]
    if bool((physical < 0).any()):
        raise ValueError("required logical page has no physical allocation")
    slots[publish] = physical.to(torch.int64) * storage_block_size + offset.to(torch.int64)
    return slots


def window_metadata(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    window: int = M.sliding_window,
    *, rolling: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged write slots plus causal indices for the visible sliding window."""
    slots = paged_slots(positions, request_ids, block_table, rolling=rolling)
    lens = torch.minimum(positions + 1, torch.full_like(positions, window)).to(torch.int32)
    offsets = torch.arange(window, device=positions.device)
    starts = positions - lens.to(positions.dtype) + 1
    visible = starts.unsqueeze(-1) + offsets
    valid = offsets.unsqueeze(0) < lens.unsqueeze(-1)
    indices = torch.full_like(visible, -1, dtype=torch.int64)
    visible_requests = request_ids.unsqueeze(-1).expand_as(visible)
    indices[valid] = paged_slots(visible[valid], visible_requests[valid], block_table, rolling=rolling)
    return slots, indices.to(torch.int32), lens


def compact_window_table(prefixes, lengths):
    """Map each request's visible sliding-window pages and retain one unmapped control page."""
    columns = max((prefix + length - 1) // BLOCK_SIZE + 1 for prefix, length in zip(prefixes, lengths))
    table = torch.full((len(prefixes), columns), -1, dtype=torch.int32)
    next_page = 0
    for request, (prefix, length) in enumerate(zip(prefixes, lengths)):
        first = max(0, prefix - M.sliding_window + 1) // BLOCK_SIZE
        end = (prefix + length - 1) // BLOCK_SIZE + 1
        count = end - first
        table[request, first:end] = torch.arange(next_page, next_page + count, dtype=torch.int32)
        next_page += count
    return table, next_page + 1



def prefill_head_tail_rows(query_lengths, cp_size):
    """Return original packed row IDs per PCP rank using request-local DualChunkSwap."""
    if cp_size < 1 or any(length < 0 for length in query_lengths):
        raise ValueError("CP size must be positive and scheduled lengths nonnegative")
    rows = [[] for _ in range(cp_size)]
    offset = 0
    for length in query_lengths:
        width = (length + 2 * cp_size - 1) // (2 * cp_size)
        for rank in range(cp_size):
            for chunk in (rank, 2 * cp_size - 1 - rank):
                begin = min(chunk * width, length)
                end = min(begin + width, length)
                rows[rank].extend(range(offset + begin, offset + end))
        offset += length
    return [torch.tensor(rank_rows, dtype=torch.int32) for rank_rows in rows]


# Dynamic lookup rows across Prefill and Decode.
ENGRAM_LOOKUPS_DYN = pl.dynamic("ENGRAM_LOOKUPS_DYN")
_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--checkpoint")
_shape_parser.add_argument("--engram-table-rows", type=int)
_shape_options = _shape_parser.parse_known_args()[0]
if _shape_options.engram_table_rows is None:
    _shape_options.engram_table_rows = max(M.engram_num_embeddings) if _shape_options.checkpoint else 4096
if _shape_options.engram_table_rows <= 0:
    _shape_parser.error("--engram-table-rows must be positive")
ENGRAM_ROWS = (_shape_options.engram_table_rows + EP_SIZE - 1) // EP_SIZE * EP_SIZE
ENGRAM_SHARD_ROWS = ENGRAM_ROWS // EP_SIZE


def write_forward_engram_layout(directory):
    """Record compile-time Host-window layout beside the integrated forward artifact."""
    layout = {
        "abi_version": 1, "ranks": EP_SIZE, "rows_per_layer": ENGRAM_ROWS,
    }
    path = Path(directory) / "engram_layout.json"
    path.write_text(json.dumps(layout, indent=2) + "\n")


def validate_forward_engram_layout(directory, ranks, rows):
    """Reject artifacts compiled for different Host-window ownership or table extents."""
    path = Path(directory) / "engram_layout.json"
    expected = {"abi_version": 1, "ranks": ranks, "rows_per_layer": rows}
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError("forward Engram layout is missing or mismatched; recompile with matching --ep and --engram-table-rows")


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
    """Own streamed ENP Host shards and complete Device-resident scale replicas."""

    def __init__(self, checkpoint, ranks, *, num_embeddings=M.engram_num_embeddings, program_count=1):
        if ranks not in (2, 4, 8):
            raise ValueError("Engram ENP offload requires two, four, or eight ranks")
        if program_count < 1:
            raise ValueError("Engram Host-window program count must be positive")
        self.ranks = ranks
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
        self.shard_rows = self.rows_per_layer // ranks
        self.program_count = program_count
        self.host_payload_bytes_per_program = ranks * 2 * self.shard_rows * HEAD_DIM
        self.host_payload_bytes = program_count * self.host_payload_bytes_per_program
        self.device_scale_bytes_per_rank = 2 * self.rows_per_layer * (HEAD_DIM // 32)
        self.upload_buffer = torch.empty(64 * 1024**2, dtype=torch.uint8).share_memory_()
        self.sources = []
        self.device_scales = None

    def __enter__(self):
        try:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            available = None
        if available is not None and self.host_payload_bytes > available:
            raise MemoryError(
                f"ENP requires {self.host_payload_bytes} Host payload bytes, "
                f"only {available} are available"
            )
        try:
            for rank in range(self.ranks):
                row = rank * self.shard_rows
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
