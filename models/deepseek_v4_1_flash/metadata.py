# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch metadata lowering for packed prefill and continuous-batch decode."""

from dataclasses import dataclass
import json
import math
import struct
from pathlib import Path
from numbers import Integral
from typing import Any, Mapping, Sequence

import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import BLOCK_SIZE, FLASH, TP_SIZE
from models.deepseek_v4_1_flash.rope_tables import precompute_rope_tables


@dataclass(frozen=True)
class ForwardMetadata:
    """Canonical token-major metadata consumed by all layer kernels."""

    query_start_loc: torch.Tensor
    query_lens: torch.Tensor
    logit_row_indices: torch.Tensor
    token_to_req_indices: torch.Tensor
    moe_token_owners: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_lens: torch.Tensor
    new_kv_seq_lens: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lens: torch.Tensor
    compressed_slots: Mapping[int, torch.Tensor]
    index_slots: Mapping[int, torch.Tensor]
    state_block_tables: Mapping[int, torch.Tensor]
    compressed_seq_lens: Mapping[int, torch.Tensor]
    compressed_seq_remainders: Mapping[int, torch.Tensor]
    compressed_lens: Mapping[int, torch.Tensor]
    compressor_output_start_loc: Mapping[int, torch.Tensor]
    compressor_source_token_indices: Mapping[int, torch.Tensor]
    compressor_position_ids: Mapping[int, torch.Tensor]
    compressed_rope_position_ids: Mapping[int, torch.Tensor]


def request_ids_from_starts(query_start_loc: torch.Tensor) -> torch.Tensor:
    """Expand packed cumulative query lengths into one request id per token."""
    lengths = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=lengths.device), lengths).to(
        torch.int32
    )


def paged_slots(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int = BLOCK_SIZE,
    logical_divisor: int = 1,
    publish_only_complete: bool = False,
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


def compressor_metadata(
    query_start_loc: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    compression_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ragged compressor output starts, source-token rows, and group-first positions."""
    query_lens = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    old_groups = torch.div(kv_seq_lens.to(torch.int64), compression_ratio, rounding_mode="floor")
    new_groups = torch.div(kv_seq_lens.to(torch.int64) + query_lens, compression_ratio, rounding_mode="floor")
    output_lens = new_groups - old_groups
    output_starts = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=query_start_loc.device), output_lens.cumsum(0))
    )
    source_rows: list[torch.Tensor] = []
    output_positions: list[torch.Tensor] = []
    for request in range(query_lens.numel()):
        groups = torch.arange(old_groups[request], new_groups[request], device=query_start_loc.device)
        completed_positions = (groups + 1) * compression_ratio - 1
        local_rows = completed_positions - kv_seq_lens[request].to(torch.int64)
        source_rows.append(query_start_loc[request].to(torch.int64) + local_rows)
        output_positions.append(groups * compression_ratio)
    empty = torch.empty(0, dtype=torch.int64, device=query_start_loc.device)
    return (
        output_starts.to(torch.int32),
        torch.cat(source_rows).to(torch.int32) if source_rows else empty.to(torch.int32),
        torch.cat(output_positions).to(torch.int32) if output_positions else empty.to(torch.int32),
    )


def window_metadata(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    window: int = FLASH.sliding_window,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged write slots plus causal indices for the visible sliding window."""
    slots = paged_slots(positions, request_ids, block_table)
    lens = torch.minimum(positions + 1, torch.full_like(positions, window)).to(torch.int32)
    offsets = torch.arange(window, device=positions.device)
    starts = positions - lens.to(positions.dtype) + 1
    visible = starts.unsqueeze(-1) + offsets
    valid = offsets.unsqueeze(0) < lens.unsqueeze(-1)
    indices = torch.full_like(visible, -1, dtype=torch.int64)
    visible_requests = request_ids.unsqueeze(-1).expand_as(visible)
    indices[valid] = paged_slots(visible[valid], visible_requests[valid], block_table)
    return slots, indices.to(torch.int32), lens


def build_forward_metadata(
    query_start_loc: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    window_block_table: torch.Tensor,
    compressed_block_tables: Mapping[int, torch.Tensor],
    state_block_tables: Mapping[int, torch.Tensor],
) -> ForwardMetadata:
    """Lower engine inputs for packed prefill or one-token-per-request decode."""
    if query_start_loc.ndim != 1 or kv_seq_lens.ndim != 1:
        raise ValueError("query_start_loc and kv_seq_lens must be one-dimensional")
    if query_start_loc.numel() != kv_seq_lens.numel() + 1:
        raise ValueError("query_start_loc must contain one more element than kv_seq_lens")
    query_lens = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    if bool((query_lens < 0).any()) or int(query_start_loc[0]) != 0:
        raise ValueError("query_start_loc must be nondecreasing and start at zero")
    if bool((kv_seq_lens < 0).any()):
        raise ValueError("kv_seq_lens must be non-negative")
    tables_by_ratio: dict[int, torch.Tensor] = {}
    for source in FLASH.kv_source_layer_ids:
        if source not in compressed_block_tables:
            raise ValueError(f"missing compressed block table for source layer {source}")
        ratio = FLASH.compress_ratios[source]
        previous = tables_by_ratio.setdefault(ratio, compressed_block_tables[source])
        if not torch.equal(previous, compressed_block_tables[source]):
            raise ValueError(f"compression ratio {ratio} sources must share one compressed block table")
    request_ids = request_ids_from_starts(query_start_loc)
    logit_rows = torch.where(query_lens > 0, query_start_loc[1:].to(torch.int64) - 1, -1).to(torch.int32)
    local_offsets = torch.arange(request_ids.numel(), device=request_ids.device)
    local_offsets -= query_start_loc[request_ids.to(torch.long)].to(local_offsets.dtype)
    positions = kv_seq_lens[request_ids.to(torch.long)].to(torch.int64) + local_offsets
    window_slots, window_indices, window_lens = window_metadata(positions, request_ids, window_block_table)
    compressed_slots: dict[int, torch.Tensor] = {}
    index_slots: dict[int, torch.Tensor] = {}
    state_tables: dict[int, torch.Tensor] = {}
    compressed_seq_lens: dict[int, torch.Tensor] = {}
    compressed_seq_remainders: dict[int, torch.Tensor] = {}
    compressed_lens: dict[int, torch.Tensor] = {}
    compressor_output_starts: dict[int, torch.Tensor] = {}
    compressor_source_rows: dict[int, torch.Tensor] = {}
    compressor_positions: dict[int, torch.Tensor] = {}
    compressed_rope_positions: dict[int, torch.Tensor] = {}
    new_kv_seq_lens = kv_seq_lens.to(torch.int64) + query_lens
    for source in FLASH.kv_source_layer_ids:
        ratio = FLASH.compress_ratios[source]
        storage_rows = BLOCK_SIZE
        table = compressed_block_tables[source]
        if table.ndim != 2 or table.shape[0] != query_lens.numel():
            raise ValueError("compressed block tables must have one row per request")
        # Sparse attention can read the entire compressed history of an active request.
        for request in range(query_lens.numel()):
            if int(query_lens[request]) == 0:
                continue
            visible_rows = int(new_kv_seq_lens[request]) // ratio
            pages = (visible_rows + storage_rows - 1) // storage_rows
            if pages > table.shape[1] or bool((table[request, :pages] < 0).any()):
                raise ValueError(f"source layer {source} has missing visible compressed pages")
        compressed_slots[source] = paged_slots(
            positions, request_ids, compressed_block_tables[source], storage_rows, ratio, True
        )
        index_slots[source] = paged_slots(
            positions, request_ids, compressed_block_tables[source], storage_rows, ratio, True
        )
        compressed_seq_lens[source] = torch.div(new_kv_seq_lens, ratio, rounding_mode="floor").to(torch.int32)
        compressed_seq_remainders[source] = new_kv_seq_lens.remainder(ratio).to(torch.int32)
        compressed_lens[source] = torch.div(positions + 1, ratio, rounding_mode="floor").to(torch.int32)
        output_starts, source_rows, output_positions = compressor_metadata(
            query_start_loc, kv_seq_lens, ratio
        )
        compressor_output_starts[source] = output_starts
        compressor_source_rows[source] = source_rows
        compressor_positions[source] = output_positions
        complete = (positions + 1).remainder(ratio) == 0
        compressed_rope_positions[source] = torch.where(complete, positions + 1 - ratio, -1).to(torch.int32)
        if ratio > 1:
            if source not in state_block_tables:
                raise ValueError(f"missing state block table for source layer {source}")
            table = state_block_tables[source]
            if table.shape != (kv_seq_lens.numel(), 1) or table.dtype != torch.int32:
                raise ValueError("state block tables must be INT32 [requests, 1]")
            allocated = table[table >= 0]
            if allocated.unique().numel() != allocated.numel():
                raise ValueError("live requests must own distinct state blocks")
            if bool((table < -1).any()):
                raise ValueError("inactive requests use state block -1")
            state_tables[source] = table
            inactive = table[request_ids.to(torch.long), 0] < 0
            compressed_slots[source] = compressed_slots[source].masked_fill(inactive, -1)
            index_slots[source] = index_slots[source].masked_fill(inactive, -1)
    return ForwardMetadata(
        query_start_loc=query_start_loc.to(torch.int32),
        query_lens=query_lens.to(torch.int32),
        logit_row_indices=logit_rows,
        token_to_req_indices=request_ids,
        moe_token_owners=torch.arange(
            request_ids.numel(), device=request_ids.device, dtype=torch.int32
        ).remainder(TP_SIZE),
        position_ids=positions.to(torch.int32),
        kv_seq_lens=kv_seq_lens.to(torch.int32),
        new_kv_seq_lens=new_kv_seq_lens.to(torch.int32),
        window_slots=window_slots,
        window_indices=window_indices,
        window_lens=window_lens,
        compressed_slots=compressed_slots,
        index_slots=index_slots,
        state_block_tables=state_tables,
        compressed_seq_lens=compressed_seq_lens,
        compressed_seq_remainders=compressed_seq_remainders,
        compressed_lens=compressed_lens,
        compressor_output_start_loc=compressor_output_starts,
        compressor_source_token_indices=compressor_source_rows,
        compressor_position_ids=compressor_positions,
        compressed_rope_position_ids=compressed_rope_positions,
    )


# Fresh-prompt CED replay metadata.


@dataclass(frozen=True)
class DecoderReplay:
    """Request-local positions and physical cache rows of the selected encoder tail."""

    source_rows: torch.Tensor
    query_start_loc: torch.Tensor
    query_lens: torch.Tensor
    replay_starts: torch.Tensor
    position_ids: torch.Tensor
    token_to_req_indices: torch.Tensor
    compressed_lens: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lens: torch.Tensor

    @property
    def logit_row_indices(self) -> torch.Tensor:
        """Last replay row per request, or -1 when that request has no rows."""
        return torch.where(self.query_lens > 0, self.query_start_loc[1:] - 1, -1)


def select_decoder_replay(
    metadata: ForwardMetadata,
    window_block_table: torch.Tensor,
    *,
    window: int = FLASH.sliding_window,
) -> DecoderReplay:
    """Select at most 128 final rows per request and truncate SWA to that replay."""
    starts = metadata.query_start_loc.to(torch.int64)
    if window <= 0 or starts.ndim != 1 or starts.numel() < 2 or int(starts[0]) != 0:
        raise ValueError("positive window and packed query starts are required")
    lengths = starts[1:] - starts[:-1]
    if bool((lengths < 0).any()):
        raise ValueError("query_start_loc must be nondecreasing")
    if metadata.position_ids.numel() != int(starts[-1]) or metadata.token_to_req_indices.numel() != int(starts[-1]):
        raise ValueError("encoder metadata does not cover the packed query rows")
    if metadata.kv_seq_lens.numel() != lengths.numel() or bool((metadata.kv_seq_lens != 0).any()):
        raise ValueError("fresh decoder prefill does not accept cached prefixes")
    if window_block_table.ndim != 2 or window_block_table.shape[0] != lengths.numel():
        raise ValueError("window block table must have one row per request")
    selected, positions, requests = [], [], []
    replay_lengths = torch.minimum(lengths, torch.full_like(lengths, window))
    replay_starts = lengths - replay_lengths
    for request, length in enumerate(lengths.tolist()):
        first, last = int(starts[request]), int(starts[request + 1])
        expected = torch.arange(length, device=starts.device)
        if not torch.equal(metadata.position_ids[first:last].to(torch.int64), expected):
            raise ValueError(f"request {request} has noncontiguous encoder positions")
        if not bool((metadata.token_to_req_indices[first:last] == request).all()):
            raise ValueError(f"request {request} has incorrect encoder request indices")
        count = int(replay_lengths[request])
        selected.append(torch.arange(last - count, last, device=starts.device))
        positions.append(torch.arange(length - count, length, device=starts.device))
        requests.append(torch.full((count,), request, dtype=torch.int32, device=starts.device))
    source_rows = torch.cat(selected).long()
    position_ids = torch.cat(positions).int()
    request_ids = torch.cat(requests)
    query_lens = replay_lengths.int()
    query_start_loc = torch.cat((torch.zeros(1, dtype=torch.int32, device=starts.device), query_lens.cumsum(0)))
    window_lens = position_ids - replay_starts[request_ids.long()] + 1
    offsets = torch.arange(window, device=starts.device)
    visible = position_ids[:, None] - window_lens[:, None] + 1 + offsets
    valid = offsets[None] < window_lens[:, None]
    window_indices = torch.full_like(visible, -1, dtype=torch.int32)
    window_indices[valid] = paged_slots(
        visible[valid], request_ids[:, None].expand_as(visible)[valid], window_block_table
    ).int()
    slots = paged_slots(position_ids, request_ids, window_block_table)
    return DecoderReplay(
        source_rows, query_start_loc, query_lens, replay_starts.int(), position_ids,
        request_ids, position_ids + 1, slots, window_indices, window_lens.int(),
    )


_DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
}


def decode_checkpoint_linear(weight: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Decode stored weights without activation quantization or requantizing weights.

    The returned matrix is output-major [N,K], matching the checkpoint and
    torch.nn.functional.linear. MXFP8 uses 32x32 scale blocks; MXFP4 packs the
    even input column in the low nibble and uses one scale per 32 input columns.
    """
    if weight.dtype in (torch.float32, torch.bfloat16):
        if scale is not None:
            raise ValueError("unquantized checkpoint weights must not have a scale")
        return weight.float().contiguous()
    if scale is None or scale.dtype != torch.float8_e8m0fnu:
        raise ValueError("quantized checkpoint weights require E8M0 scales")
    codes = scale.contiguous().view(torch.uint8).int()
    if bool((codes == 255).any()):
        raise ValueError("checkpoint E8M0 scales must be finite")
    factors = torch.exp2(codes.float() - 127)
    if weight.dtype == torch.float8_e4m3fn:
        if weight.ndim != 2 or tuple(weight.shape) != (scale.shape[0] * 32, scale.shape[1] * 32):
            raise ValueError("MXFP8 checkpoint weight/scale shapes disagree")
        result = weight.float() * factors.repeat_interleave(32, 0).repeat_interleave(32, 1)
    elif weight.dtype == torch.int8:
        if (
            weight.ndim != 2
            or weight.shape[0] != scale.shape[0]
            or weight.shape[1] * 2 != scale.shape[1] * 32
        ):
            raise ValueError("MXFP4 checkpoint weight/scale shapes disagree")
        packed = weight.contiguous().view(torch.uint8)
        nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
        values = magnitude[(nibbles & 7).long()]
        values = torch.where((nibbles & 8) != 0, -values, values)
        result = values * factors.repeat_interleave(32, -1)
    else:
        raise ValueError(f"unsupported checkpoint linear dtype {weight.dtype}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("decoded checkpoint linear contains non-finite values")
    return result.contiguous()


class PrefillCheckpoint:
    """Read requested original tensors and decode linear matrices as output-major FP32."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        config = json.loads((self.root / "config.json").read_text())
        text = config["text_config"]
        quant = config.get("quantization_config", {})
        if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]:
            raise ValueError("checkpoint must use the official block-32 FP8/FP4 format")
        required = {
            "hidden_size": C.D,
            "num_hidden_layers": C.FLASH.num_hidden_layers,
            "num_attention_heads": C.H,
            "head_dim": C.HEAD_DIM,
            "sliding_window": C.FLASH.sliding_window,
            "kv_source_layer_ids": list(C.FLASH.kv_source_layer_ids),
            "index_source_layer_ids": list(C.FLASH.index_source_layer_ids),
        }
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"checkpoint {name}={text.get(name)!r}, expected {expected!r}")
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers = {}

    def tensor(self, name: str) -> torch.Tensor:
        """Return an owned CPU tensor without loading the rest of its shard."""
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype = _DTYPES[entry["dtype"]]
            shape = entry["shape"]
            start, end = entry["data_offsets"]
            expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if start < 0 or end - start != expected_bytes or 8 + header_size + end > path.stat().st_size:
                raise ValueError(f"invalid safetensors range for {name}")
            file.seek(8 + header_size + start)
            payload = bytearray(file.read(expected_bytes))
        if len(payload) != expected_bytes:
            raise ValueError(f"truncated safetensors payload for {name}")
        return torch.frombuffer(payload, dtype=dtype).reshape(shape)

    def tensor_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Read selected table rows, including from the very large Engram tables."""
        if rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("table row ids must have an integer dtype")
        ids = rows.to(dtype=torch.int64, device="cpu").flatten()
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype, shape = _DTYPES[entry["dtype"]], entry["shape"]
            if len(shape) != 2 or bool(((ids < 0) | (ids >= shape[0])).any()):
                raise ValueError(f"invalid table rows for {name}")
            row_bytes = shape[1] * torch.empty((), dtype=dtype).element_size()
            start, end = entry["data_offsets"]
            if (
                start < 0
                or end - start != shape[0] * row_bytes
                or 8 + header_size + end > path.stat().st_size
            ):
                raise ValueError(f"invalid safetensors range for {name}")
            unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            payload = bytearray(unique.numel() * row_bytes)
            for index, row in enumerate(unique.tolist()):
                file.seek(8 + header_size + start + row * row_bytes)
                data = file.read(row_bytes)
                if len(data) != row_bytes:
                    raise ValueError(f"truncated safetensors row for {name}")
                payload[index * row_bytes : (index + 1) * row_bytes] = data
        if not ids.numel():
            return torch.empty(*rows.shape, shape[1], dtype=dtype)
        table = torch.frombuffer(payload, dtype=dtype).reshape(-1, shape[1])
        # CPU indexing is unavailable for FP8; index its byte representation.
        return (
            table.view(torch.uint8)
            .reshape(unique.numel(), row_bytes)[inverse]
            .contiguous()
            .view(dtype)
            .reshape(*rows.shape, shape[1])
        )


    def linear(self, stem: str) -> torch.Tensor:
        weight = self.tensor(stem + ".weight")
        scale = self.tensor(stem + ".scale") if weight.dtype in (torch.int8, torch.float8_e4m3fn) else None
        return decode_checkpoint_linear(weight, scale)


class ResidentPrefillWeights:
    """Pack the released weights into the existing TP1/EP4 operator layouts.

    Routed weights retain their FP4 nibbles; MXFP8 payloads retain their bytes.
    Scale bytes are reordered into the native MX_B_NN layout. Only projections
    whose released inference dtype is BF16 or FP32 are decoded on the host.
    Offsets count elements in their named bank. Large byte banks use 128-column
    rows so runtime shapes and slice coordinates fit their uint32 ABI.
    """

    ranks = 4
    bank_columns = 128
    local_experts = C.FLASH.n_routed_experts // ranks
    fp8_names = (
        "attn.wq_a",
        "attn.wq_b",
        "attn.wkv",
        "attn.wo_b",
        "attn.indexer.wq_b",
        "ffn.shared_experts.w1",
        "ffn.shared_experts.w3",
        "ffn.shared_experts.w2",
        "engram.wkv",
    )
    bf16_names = (
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "attn.wo_a.weight",
        "attn.compressor.wkv.weight",
        "attn.compressor.norm.weight",
        "attn.indexer.wk.weight",
        "attn.indexer.k_norm.weight",
        "attn.indexer.weights_proj.weight",
    )
    dense_names = (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "attn.attn_sink",
        "ffn.gate.weight",
        "ffn.gate.bias",
        "attn.compressor.wkv.weight",
        "attn.compressor.wgate.weight",
        "engram.q_weight",
    )
    expert_names = ("w1", "w3", "w2")

    def __init__(self, checkpoint: PrefillCheckpoint):
        self.checkpoint = checkpoint
        layers = C.FLASH.num_hidden_layers
        self.fp8_offsets = torch.full((layers, len(self.fp8_names)), -1, dtype=torch.int64)
        self.fp8_scale_offsets = torch.full_like(self.fp8_offsets, -1)
        self.bf16_offsets = torch.full((layers, len(self.bf16_names)), -1, dtype=torch.int64)
        self.dense_offsets = torch.full((layers, len(self.dense_names)), -1, dtype=torch.int64)
        self.expert_offsets = torch.empty((layers, 3), dtype=torch.int64)
        self.expert_scale_offsets = torch.empty_like(self.expert_offsets)
        self.fp8_entries, self.bf16_entries, self.dense_entries = [], [], []
        self.fp8_values = self.scale_values = self.bf16_values = self.dense_values = 0
        for layer in range(layers):
            for column, suffix in enumerate(self.fp8_names):
                stem = f"layers.{layer}.{suffix}"
                if stem + ".weight" not in checkpoint.weight_map:
                    continue
                header = self._header(stem + ".weight")
                n, k = header["shape"]
                scale = self._header(stem + ".scale")
                if (
                    header["dtype"] != "F8_E4M3"
                    or n % 32
                    or k % 64
                    or scale["dtype"] != "F8_E8M0"
                    or scale["shape"] != [n // 32, k // 32]
                ):
                    raise ValueError(f"unsupported native MXFP8 weight layout: {stem}")
                assert n * k % self.bank_columns == 0
                assert (n * k // 32) % self.bank_columns == 0
                self.fp8_offsets[layer, column] = self.fp8_values
                self.fp8_scale_offsets[layer, column] = self.scale_values
                self.fp8_entries.append((stem, self.fp8_values, self.scale_values))
                self.fp8_values += n * k
                self.scale_values += n * k // 32
            for column, suffix in enumerate(self.bf16_names):
                name = f"layers.{layer}.{suffix}"
                if name not in checkpoint.weight_map:
                    continue
                if column == 5 and C.FLASH.compress_ratios[layer] != 1:
                    continue
                header = self._header(name)
                if header["dtype"] not in ("F32", "BF16", "F8_E4M3"):
                    raise ValueError(f"unsupported BF16 inference weight: {name}")
                if header["dtype"] == "F8_E4M3" and column != 4:
                    raise ValueError(f"unexpected quantized BF16 inference weight: {name}")
                self.bf16_offsets[layer, column] = self.bf16_values
                self.bf16_entries.append((name, column, self.bf16_values))
                self.bf16_values += math.prod(header["shape"])
            for column, suffix in enumerate(self.dense_names):
                name = f"layers.{layer}.{suffix}"
                if name not in checkpoint.weight_map:
                    continue
                if column in (9, 10) and C.FLASH.compress_ratios[layer] != 2:
                    continue
                header = self._header(name)
                if header["dtype"] not in ("F32", "BF16"):
                    raise ValueError(f"unsupported FP32 inference weight: {name}")
                # Keep native matrix bases aligned after small mHC scalar vectors.
                self.dense_values = (self.dense_values + 127) // 128 * 128
                self.dense_offsets[layer, column] = self.dense_values
                self.dense_entries.append((name, column, self.dense_values))
                self.dense_values += math.prod(header["shape"])
        self.fp4_values = 0
        for layer in range(layers):
            for column in range(3):
                self.expert_offsets[layer, column] = self.fp4_values
                self.expert_scale_offsets[layer, column] = self.scale_values
                self.fp4_values += self.local_experts * C.D * C.MOE_INTER // 2
                self.scale_values += self.local_experts * C.D * C.MOE_INTER // 32
        assert (C.D * C.MOE_INTER // 2) % self.bank_columns == 0
        assert (C.D * C.MOE_INTER // 32) % self.bank_columns == 0
        for values in (self.fp4_values, self.fp8_values, self.scale_values):
            assert values % self.bank_columns == 0
            assert values // self.bank_columns < 2**32
        for offsets in (
            self.fp8_offsets, self.fp8_scale_offsets, self.expert_offsets, self.expert_scale_offsets,
        ):
            assert bool((offsets[offsets >= 0] % self.bank_columns == 0).all())
        assert self.bf16_values * 2 < 2**32
        assert self.dense_values * 4 < 2**32

    def _header(self, name: str) -> dict[str, Any]:
        checkpoint = self.checkpoint
        shard = checkpoint.weight_map[name]
        if shard not in checkpoint._headers:
            path = (checkpoint.root / shard).resolve()
            if not path.is_relative_to(checkpoint.root):
                raise ValueError(f"checkpoint shard leaves its root: {shard}")
            with path.open("rb") as file:
                size = struct.unpack("<Q", file.read(8))[0]
                if size > 64 * 1024 * 1024:
                    raise ValueError(f"safetensors header is too large: {shard}")
                checkpoint._headers[shard] = json.loads(file.read(size))
        return checkpoint._headers[shard][name]

    @property
    def tensor_layouts(self):
        """Return contiguous bank allocation shapes and exact storage dtypes."""
        return {
            "bank4": ((self.fp4_values // self.bank_columns, self.bank_columns), torch.uint8),
            "bank8": ((self.fp8_values // self.bank_columns, self.bank_columns), torch.float8_e4m3fn),
            "scales8": ((self.scale_values // self.bank_columns, self.bank_columns), torch.float8_e8m0fnu),
            "bf16": ((self.bf16_values,), torch.bfloat16),
            "dense": ((self.dense_values,), torch.float32),
        }

    @property
    def bytes_per_rank(self):
        return sum(
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in self.tensor_layouts.values()
        )

    @property
    def offset_tensors(self):
        return {
            name: getattr(self, name)
            for name in (
                "fp8_offsets",
                "fp8_scale_offsets",
                "bf16_offsets",
                "dense_offsets",
                "expert_offsets",
                "expert_scale_offsets",
            )
        }

    def upload(self, runtime, rank: int, handles):
        """Upload a rank's persistent banks with one projection staged at a time."""
        from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale, pack_mxfp4_weight_tiles

        if rank not in range(self.ranks):
            raise ValueError("resident prefill requires four contiguous expert shards")
        checkpoint = self.checkpoint

        def copy(bank, data, offset):
            data = data.contiguous()
            if bank in ("bank4", "bank8", "scales8"):
                assert int(offset) % self.bank_columns == 0
                assert data.numel() % self.bank_columns == 0
            runtime.copy_to(
                handles[bank].data_ptr,
                data.data_ptr(),
                data.numel() * data.element_size(),
                dst_offset=int(offset) * data.element_size(),
                worker_id=rank,
            )

        for stem, offset, scale_offset in self.fp8_entries:
            weight = checkpoint.tensor(stem + ".weight")
            codes = checkpoint.tensor(stem + ".scale").view(torch.uint8)
            logical_scale = codes.repeat_interleave(32, dim=0).t().contiguous()
            copy("bank8", weight.t(), offset)
            copy("scales8", pack_mx_b_scale(logical_scale).view(torch.float8_e8m0fnu), scale_offset)
        for name, column, offset in self.bf16_entries:
            if column == 4:
                value = checkpoint.linear(name.removesuffix(".weight")).to(torch.bfloat16)
            else:
                value = checkpoint.tensor(name).to(torch.bfloat16)
            if column in (5, 7, 9):
                value = value.t()
            copy("bf16", value, offset)
        for name, column, offset in self.dense_entries:
            value = checkpoint.tensor(name).float()
            if column in (9, 10):
                value = value.t()
            elif column == 11:
                value = value * checkpoint.tensor(name.replace("q_weight", "k_weight")).float()
            copy("dense", value, offset)
        for layer in range(C.FLASH.num_hidden_layers):
            for column, projection in enumerate(self.expert_names):
                k, n = (C.MOE_INTER, C.D) if projection == "w2" else (C.D, C.MOE_INTER)
                logical_scales = torch.empty((self.local_experts * (k // 32), n), dtype=torch.uint8)
                offset = int(self.expert_offsets[layer, column])
                for local in range(self.local_experts):
                    expert = rank * self.local_experts + local
                    stem = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                    weight = checkpoint.tensor(stem + ".weight")
                    scale = checkpoint.tensor(stem + ".scale")
                    if (
                        weight.dtype != torch.int8
                        or tuple(weight.shape) != (n, k // 2)
                        or scale.dtype != torch.float8_e8m0fnu
                        or tuple(scale.shape) != (n, k // 32)
                    ):
                        raise ValueError(f"unsupported native MXFP4 weight layout: {stem}")
                    packed = pack_mxfp4_weight_tiles(weight, 256, 256)
                    copy("bank4", packed, offset + local * (n * k // 2))
                    logical_scales[local * (k // 32) : (local + 1) * (k // 32)] = scale.view(torch.uint8).t()
                copy(
                    "scales8",
                    pack_mx_b_scale(logical_scales).view(torch.float8_e8m0fnu),
                    self.expert_scale_offsets[layer, column],
                )

    def lookup_rows(self, token_ids: torch.Tensor, engram_hashes: Mapping[int, torch.Tensor]):
        """Read only the prompt's embedding and Engram rows at official BF16 boundaries."""
        checkpoint = self.checkpoint
        embedding = checkpoint.tensor_rows("embed.weight", token_ids).to(torch.bfloat16)
        engram = {}
        for layer in C.FLASH.engram_layer_ids:
            stem = f"layers.{layer}.engram.embed"
            hashes = engram_hashes[layer]
            payload = checkpoint.tensor_rows(stem + ".weight", hashes)
            scale = checkpoint.tensor_rows(stem + ".scale", hashes)
            rows = payload.float() * scale.float().repeat_interleave(32, dim=-1)
            engram[layer] = rows.to(torch.bfloat16).flatten(1)
        return embedding, engram


def prepare_resident_inputs(
    checkpoint: PrefillCheckpoint,
    weights: ResidentPrefillWeights,
    token_ids: torch.Tensor | Sequence[int],
    capacity: int,
) -> dict[str, torch.Tensor]:
    """Prepare one rank's fresh prompt and causal CED replay for the native kernels."""
    import importlib.util
    import sys
    from types import SimpleNamespace

    ids = torch.as_tensor(token_ids, device="cpu")
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be a one-dimensional integer sequence")
    if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity % 16:
        raise ValueError("prefill capacity must be an integer multiple of 16")
    count = ids.numel()
    if not 1 <= count <= capacity <= C.PREFILL_MAX_TOKENS:
        raise ValueError("a nonempty prompt must fit the configured prefill capacity")
    if bool(((ids < 0) | (ids >= C.FLASH.vocab_size)).any()):
        raise ValueError("token id is outside the checkpoint vocabulary")
    ids = ids.long()
    if getattr(weights, "_hash_capacity", 0) < capacity:
        from tokenizers import Tokenizer

        spec = importlib.util.spec_from_file_location(
            "pypto_v41_checkpoint_engram", checkpoint.root / "inference/engram.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("checkpoint does not contain the official Engram implementation")
        official = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = official
        spec.loader.exec_module(official)
        backend = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))

        class TokenizerView:
            backend_tokenizer = backend

            def __len__(self):
                return backend.get_vocab_size(with_added_tokens=True)

        args = SimpleNamespace(**json.loads((checkpoint.root / "inference/config.json").read_text()))
        args.max_batch_size, args.max_seq_len = 1, capacity
        weights._hash_state = official.NgramHashState(
            args, official.EngramLayout.from_args(args), TokenizerView()
        )
        weights._hash_capacity = capacity
    hashes = weights._hash_state(ids.unsqueeze(0), 0)[0]
    engram_hashes = {
        layer: hashes[:, index] for index, layer in enumerate(C.FLASH.engram_layer_ids)
    }
    embedding, lookups = weights.lookup_rows(ids, engram_hashes)

    data_pages = (capacity + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    pages = data_pages + 1
    table = torch.arange(data_pages, dtype=torch.int32).reshape(1, -1)
    compressed_tables = {source: table for source in C.FLASH.kv_source_layer_ids}
    state_table = torch.zeros((1, 1), dtype=torch.int32)
    state_tables = {
        source: state_table
        for source in C.FLASH.kv_source_layer_ids
        if C.FLASH.compress_ratios[source] == 2
    }
    encoder = build_forward_metadata(
        torch.tensor([0, count], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        table,
        compressed_tables,
        state_tables,
    )
    decoder = select_decoder_replay(encoder, table)
    decoder_count = decoder.source_rows.numel()
    c2_source = next(source for source in C.FLASH.kv_source_layer_ids if C.FLASH.compress_ratios[source] == 2)
    publisher_source = C.FLASH.num_hidden_layers // 2
    plain_cos, plain_sin = precompute_rope_tables(count, False)
    compressed_cos, compressed_sin = precompute_rope_tables(count, True)

    def pad(value, fill=0):
        result = torch.full((capacity, *value.shape[1:]), fill, dtype=value.dtype)
        result[: value.shape[0]] = value
        return result

    def rope_rows(cosine, sine, positions):
        cos = torch.ones((capacity, C.ROPE_DIM // 2), dtype=torch.float32)
        sin = torch.zeros_like(cos)
        active = positions >= 0
        rows = torch.arange(positions.numel())[active]
        cos[rows] = cosine[positions[active].long()]
        sin[rows] = sine[positions[active].long()]
        return cos, sin

    encoder_plain = rope_rows(plain_cos, plain_sin, encoder.position_ids)
    encoder_compressed = rope_rows(compressed_cos, compressed_sin, encoder.position_ids)
    decoder_compressed = rope_rows(compressed_cos, compressed_sin, decoder.position_ids)
    pair_cos, pair_sin = rope_rows(
        compressed_cos, compressed_sin, encoder.compressed_rope_position_ids[c2_source]
    )
    initial_pre = torch.zeros((capacity, C.HC_MULT), dtype=torch.float32)
    initial_pre[:, 0] = 1.0
    index_tables = torch.full((2, 1, pages), -1, dtype=torch.int32)
    index_tables[:, :, :data_pages] = table
    logit_rows = torch.full((16,), -1, dtype=torch.int32)
    logit_rows[0] = decoder_count - 1
    window_shape = (C.FLASH.num_hidden_layers, pages, C.BLOCK_SIZE, 1)
    source_shape = (len(C.FLASH.kv_source_layer_ids), pages, C.BLOCK_SIZE, 1)
    values = {
        "embedding": pad(embedding),
        "embedding_ids": torch.arange(capacity, dtype=torch.int64),
        "initial_pre": initial_pre,
        "identity_rows": pad(torch.arange(count, dtype=torch.int32), -1),
        "engram_lookup": torch.stack([pad(lookups[layer]) for layer in C.FLASH.engram_layer_ids]),
        "counts": torch.tensor([count, decoder_count], dtype=torch.int32),
        "replay_rows": pad(decoder.source_rows.int(), -1),
        "position_ids": torch.stack((pad(encoder.position_ids, -1), pad(decoder.position_ids, -1))),
        "rope_cos": torch.stack((encoder_plain[0], encoder_compressed[0], decoder_compressed[0])),
        "rope_sin": torch.stack((encoder_plain[1], encoder_compressed[1], decoder_compressed[1])),
        "compressed_cos": pair_cos,
        "compressed_sin": pair_sin,
        "publisher_cos": encoder_compressed[0],
        "publisher_sin": encoder_compressed[1],
        "window_slots": torch.stack((pad(encoder.window_slots, -1), pad(decoder.window_slots, -1))),
        "window_indices": torch.stack((pad(encoder.window_indices, -1), pad(decoder.window_indices, -1))),
        "compressed_slots": pad(encoder.compressed_slots[c2_source], -1),
        "publisher_slots": pad(encoder.compressed_slots[publisher_source], -1),
        "token_requests": torch.stack((pad(encoder.token_to_req_indices, -1), pad(decoder.token_to_req_indices, -1))),
        "compressed_lens": torch.stack((pad(encoder.compressed_lens[c2_source]), pad(decoder.compressed_lens))),
        "query_starts": torch.stack((encoder.query_start_loc, decoder.query_start_loc.int())),
        "index_block_tables": index_tables,
        "state_block_table": state_table,
        "window_cache": torch.zeros((*window_shape, C.HEAD_DIM), dtype=torch.uint8).view(torch.float8_e4m3fn),
        "window_scales": torch.full(
            (*window_shape, C.HEAD_DIM // C.WINDOW_CACHE_GROUP), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "compressed_cache": torch.zeros((*source_shape, C.HEAD_DIM // 2), dtype=torch.uint8),
        "compressed_scales": torch.full(
            (*source_shape, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP), 0x38, dtype=torch.uint8
        ).view(torch.float8_e4m3fn),
        "index_cache": torch.zeros((*source_shape, C.INDEX_DIM // 2), dtype=torch.uint8),
        "index_scales": torch.full(
            (*source_shape, C.INDEX_DIM // C.INDEX_CACHE_GROUP), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "state_cache": torch.zeros((len(state_tables), 2, C.STATE_CAPACITY, C.STATE_WIDTH), dtype=torch.float32),
        "logit_rows": logit_rows,
    }
    if not hasattr(weights, "_output_weights"):
        head = checkpoint.tensor("head.weight")
        if head.dtype != torch.bfloat16 or tuple(head.shape) != (C.FLASH.vocab_size, C.D):
            raise ValueError("the native LM head requires the released BF16 checkpoint matrix")
        weights._output_weights = {
            "head": head,
            "final_norm": checkpoint.tensor("norm.weight").to(torch.bfloat16),
        }
    values.update(weights._output_weights)
    values.update(weights.offset_tensors)
    return values
