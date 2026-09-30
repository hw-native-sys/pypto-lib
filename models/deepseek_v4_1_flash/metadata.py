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
from numbers import Integral
from typing import Any, Literal, Mapping, Sequence

import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import (
    AttentionMode, BLOCK_SIZE, DeepSeekV41LayerConfig, FLASH, TP_SIZE,
)
from models.deepseek_v4_1_flash.golden import (
    PrefillAttentionCaches, PrefillAttentionMetadata, hc_post, hc_pre,
    prefill_attention_reference, prefill_engram_reference, prefill_hc_mixes,
    prefill_moe_reference, prefill_publish_decoder_reference, prefill_rms_norm,
    prefill_round_activation,
)
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


# Fresh-prompt CED metadata and independent reference state.

@dataclass(frozen=True)
class PrefillLayerPlan:
    """The checkpoint layer and its encoder/decoder stage."""

    layer: DeepSeekV41LayerConfig
    stage: Literal["encoder", "decoder"]

    @property
    def layer_id(self) -> int:
        return self.layer.layer_id


def resolve_prefill_layer_plan(layer_id: int) -> PrefillLayerPlan:
    """Resolve CED semantics without selecting a device implementation."""
    layer = FLASH.layer_config(layer_id)
    encoder = layer_id < FLASH.num_hidden_layers // 2
    modes = (AttentionMode.SWA, AttentionMode.FULL, AttentionMode.REUSE) if encoder else (
        AttentionMode.FULL, AttentionMode.REINDEX, AttentionMode.REUSE
    )
    ratios = (0, 2) if encoder else (1,)
    if layer.mode not in modes or layer.compression_ratio not in ratios:
        raise ValueError(f"unsupported CED prefill configuration at layer {layer_id}")
    return PrefillLayerPlan(layer, "encoder" if encoder else "decoder")


@dataclass(frozen=True)
class PrefillState:
    """Expanded mHC residual and delayed pre-mix in FP32 transport buffers."""

    x_hc: torch.Tensor
    pre_mix: torch.Tensor

    def __post_init__(self) -> None:
        if self.x_hc.ndim not in (3, 4) or self.pre_mix.ndim != self.x_hc.ndim - 1:
            raise ValueError("x_hc and pre_mix must have token-major or rank/token-major shapes")
        if self.x_hc.shape[:-1] != self.pre_mix.shape:
            raise ValueError("x_hc and pre_mix must share rank, token, and mHC dimensions")
        if self.x_hc.shape[-2:] != (FLASH.hc_mult, FLASH.hidden_size):
            raise ValueError("x_hc does not have the configured mHC and hidden dimensions")
        if self.x_hc.dtype != torch.float32 or self.pre_mix.dtype != torch.float32:
            raise ValueError("mHC residual and delayed pre-mix transport buffers must be FP32")

    @property
    def num_tokens(self) -> int:
        return self.x_hc.shape[-3]

    def take_rows(self, rows: torch.Tensor) -> "PrefillState":
        """Gather identical packed rows from the residual and its delayed mix."""
        return PrefillState(
            self.x_hc.index_select(-3, rows.to(self.x_hc.device)),
            self.pre_mix.index_select(-2, rows.to(self.pre_mix.device)),
        )


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


@dataclass(frozen=True)
class PrefillLayerInputs:
    plan: PrefillLayerPlan
    mode: str
    ratio: int
    weights: dict[str, torch.Tensor]
    metadata: PrefillAttentionMetadata
    caches: PrefillAttentionCaches
    compressed_indices: torch.Tensor | None
    candidate_mask: torch.Tensor | None
    previous_rows: torch.Tensor
    source_rows: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return self.metadata.rope_cos.shape[0]


@dataclass(frozen=True)
class PrefillPublisherInputs:
    weights: dict[str, torch.Tensor]
    metadata: PrefillAttentionMetadata
    caches: PrefillAttentionCaches


class PrefillContext:
    """Own one fresh request batch through encoder, publication, and decoder.

    All lengths are positive and their packed total fits capacity <= 4096.
    The rectangular request layout uses at most 32 physical pages per layer,
    excluding the sentinel, so dense index-dot storage remains bounded.
    Every context allocates its own cache storage for a fresh prompt batch.
    """

    def __init__(
        self,
        checkpoint: Any,
        lengths: Sequence[int],
        capacity: int | None = None,
        *,
        precision: str = "fp32",
    ):
        if not lengths or any(not isinstance(n, Integral) or isinstance(n, bool) or n < 1 for n in lengths):
            raise ValueError("fresh prefill needs a nonempty sequence of positive integer lengths")
        self.lengths = tuple(int(n) for n in lengths)
        self.num_tokens = sum(self.lengths)
        if capacity is None:
            capacity = self.num_tokens
        if not isinstance(capacity, Integral) or isinstance(capacity, bool):
            raise ValueError("prefill capacity must be an integer")
        if not self.num_tokens <= capacity <= C.PREFILL_MAX_TOKENS:
            raise ValueError(f"fresh packed rows must fit capacity <= {C.PREFILL_MAX_TOKENS}")
        requests, maximum = len(self.lengths), max(self.lengths)
        pages = (maximum + 127) // 128
        # Index queries currently score every physical cache row. Bound the
        # rectangular layout before allocating metadata or any persistent cache:
        # 4096 * 32 heads * (32 + 1 sentinel) * 128 rows * 4 bytes = 2.0625 GiB.
        if requests * pages > 32:
            raise ValueError(
                "fresh FP32 prefill requires requests * ceil(max(lengths) / 128) <= 32 physical pages"
            )
        if precision not in ("fp32", "official"):
            raise ValueError("precision must be 'fp32' or 'official'")
        self.precision = precision
        self.checkpoint = checkpoint
        self.capacity = int(capacity)
        self.window_table = torch.arange(requests * pages, dtype=torch.int32).reshape(requests, pages)
        self.global_tables = {}
        for source in C.FLASH.kv_source_layer_ids:
            pages = max(1, (maximum // C.FLASH.compress_ratios[source] + 127) // 128)
            self.global_tables[source] = torch.arange(requests * pages, dtype=torch.int32).reshape(
                requests, pages
            )
        self.state_tables = {
            source: torch.arange(requests, dtype=torch.int32).reshape(-1, 1)
            for source in C.FLASH.kv_source_layer_ids
            if source < C.FLASH.num_hidden_layers // 2
        }
        starts = torch.tensor([0, *torch.tensor(self.lengths).cumsum(0).tolist()], dtype=torch.int32)
        self.metadata = build_forward_metadata(
            starts,
            torch.zeros(requests, dtype=torch.int32),
            self.window_table,
            self.global_tables,
            self.state_tables,
        )
        self.replay = select_decoder_replay(self.metadata, self.window_table)
        self.decoder_num_tokens = self.replay.source_rows.numel()
        self.encoder_rows = torch.arange(self.num_tokens, dtype=torch.int64)
        self.decoder_rows = self.replay.source_rows
        self.rope = {compressed: precompute_rope_tables(maximum, compressed) for compressed in (False, True)}
        # Each extra physical page is an untouched sentinel.
        self.window = {
            layer: torch.zeros(self.window_table.numel() + 1, 128, C.HEAD_DIM)
            for layer in range(C.FLASH.num_hidden_layers)
        }
        self.global_cache = {
            source: torch.zeros(table.numel() + 1, 128, C.HEAD_DIM)
            for source, table in self.global_tables.items()
        }
        self.index_cache = {
            source: torch.zeros(table.numel() + 1, 128, C.INDEX_DIM)
            for source, table in self.global_tables.items()
        }
        self.state_cache = {
            source: torch.zeros(requests + 1, C.STATE_CAPACITY, 2 * C.HEAD_DIM)
            for source in self.state_tables
        }
        self.last_topk: dict[int, torch.Tensor] = {}
        self.candidates: torch.Tensor | None = None
        self.last_routes: torch.Tensor | None = None
        self.decoder_published = False

    def initial_state(self, token_sequences: Sequence[torch.Tensor | Sequence[int]]) -> PrefillState:
        """Expand original embedding rows and use the official initial one-hot mix."""
        if len(token_sequences) != len(self.lengths):
            raise ValueError("token sequences must have one span per request")
        sequences = []
        for expected, sequence in zip(self.lengths, token_sequences):
            value = torch.as_tensor(sequence, device="cpu")
            if (
                value.ndim != 1
                or value.numel() != expected
                or value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            ):
                raise ValueError("token ids must be integer vectors matching the configured lengths")
            sequences.append(value.long())
        input_ids = torch.cat(sequences)
        embedding = prefill_round_activation(self.checkpoint.tensor_rows("embed.weight", input_ids).float(), self.precision)
        expanded = embedding.unsqueeze(1).repeat(1, C.HC_MULT, 1)
        pre_mix = torch.zeros(self.num_tokens, C.HC_MULT, dtype=torch.float32)
        pre_mix[:, 0] = 1
        return PrefillState(expanded, pre_mix)

    def layer_inputs(
        self,
        layer_id: int,
        *,
        weights: dict[str, torch.Tensor] | None = None,
        load_weights: bool = True,
    ) -> PrefillLayerInputs:
        """Resolve fresh encoder/full-tail decoder metadata without model arithmetic."""
        plan = resolve_prefill_layer_plan(layer_id)
        encoder = plan.stage == "encoder"
        mode, ratio = plan.layer.mode.value.lower(), plan.layer.compression_ratio
        source = plan.layer.kv_source_layer_id
        positions = self.metadata.position_ids if encoder else self.replay.position_ids
        requests = self.metadata.token_to_req_indices if encoder else self.replay.token_to_req_indices
        cosine, sine = self.rope[bool(ratio)]
        request_lengths = torch.tensor(self.lengths, dtype=torch.int32)[requests.long()]
        global_widths = (request_lengths // ratio).clamp(max=C.INDEX_TOPK) if ratio else torch.zeros_like(request_lengths)
        values = dict(
            rope_cos=cosine[positions.long()],
            rope_sin=sine[positions.long()],
            window_slots=self.metadata.window_slots if encoder else self.replay.window_slots,
            window_indices=self.metadata.window_indices if encoder else self.replay.window_indices,
            request_ids=requests,
            query_start_loc=self.metadata.query_start_loc if encoder else self.replay.query_start_loc,
            position_ids=positions,
            attention_extents=torch.stack((request_lengths.clamp(max=C.FLASH.sliding_window), global_widths), -1),
        )
        previous_rows = torch.full_like(positions, -1, dtype=torch.int32)
        if ratio:
            compressed_positions = (
                self.metadata.compressed_rope_position_ids[source].clamp_min(0) if encoder else positions
            )
            values.update(
                compressed_lens=self.metadata.compressed_lens[source]
                if encoder
                else self.replay.compressed_lens,
                index_block_table=self.global_tables[source],
                compressed_slots=(
                    self.metadata.compressed_slots[source]
                    if encoder
                    else torch.full_like(positions, -1, dtype=torch.int64)
                ),
                compressed_rope_cos=cosine[compressed_positions.long()],
                compressed_rope_sin=sine[compressed_positions.long()],
                state_block_table=self.state_tables.get(source),
            )
            if ratio == 2:
                complete = positions.remainder(2) == 1
                previous_rows[complete] = torch.arange(positions.numel(), dtype=torch.int32)[complete] - 1
        caches = PrefillAttentionCaches(
            self.window[layer_id],
            self.global_cache.get(source),
            self.index_cache.get(source),
            self.state_cache.get(source),
        )
        if weights is None:
            weights = self.checkpoint.attention(layer_id) if load_weights else {}
        return PrefillLayerInputs(
            plan,
            mode,
            ratio,
            weights,
            PrefillAttentionMetadata(**values),
            caches,
            self.last_topk.get(source),
            self.candidates if mode == "reindex" else None,
            previous_rows,
            self.encoder_rows if encoder else self.decoder_rows,
        )

    def update_attention(self, layer_id: int, result: Any) -> None:
        """Advance owned caches from a device or reference result, without diagnostics."""
        plan = resolve_prefill_layer_plan(layer_id)
        source = plan.layer.kv_source_layer_id
        caches = result.caches
        self.window[layer_id] = caches.window
        if plan.layer.compression_ratio:
            if caches.compressed is None or caches.index is None:
                raise ValueError("compressed layers must return their global and index caches")
            self.global_cache[source], self.index_cache[source] = caches.compressed, caches.index
            topk = getattr(result, "topk_indices", None)
            if topk is not None:
                self.last_topk[source] = topk
            if caches.state is not None:
                self.state_cache[source] = caches.state
            if plan.stage == "decoder" and plan.layer.is_kv_source:
                self.candidates = getattr(result, "candidate_mask", None)

    def publisher_inputs(self, *, weights=None, load_weights=True) -> PrefillPublisherInputs:
        """Layer20 global publication always sees every newly encoded prompt row."""
        layer_id = C.FLASH.num_hidden_layers // 2
        cosine, sine = self.rope[True]
        positions = self.metadata.position_ids.long()
        metadata = PrefillAttentionMetadata(
            cosine[positions],
            sine[positions],
            self.metadata.window_slots,
            self.metadata.window_indices,
            request_ids=self.metadata.token_to_req_indices,
            compressed_slots=self.metadata.compressed_slots[layer_id],
            compressed_rope_cos=cosine[positions],
            compressed_rope_sin=sine[positions],
            query_start_loc=self.metadata.query_start_loc,
            position_ids=self.metadata.position_ids,
        )
        if weights is None:
            weights = self.checkpoint.publisher(layer_id) if load_weights else {}
        return PrefillPublisherInputs(
            weights,
            metadata,
            PrefillAttentionCaches(
                self.window[layer_id], self.global_cache[layer_id], self.index_cache[layer_id]
            ),
        )

    def update_publisher(self, caches: PrefillAttentionCaches) -> None:
        layer_id = C.FLASH.num_hidden_layers // 2
        if caches.compressed is None or caches.index is None:
            raise ValueError("decoder publisher must return both global and index caches")
        self.global_cache[layer_id], self.index_cache[layer_id] = caches.compressed, caches.index
        self.decoder_published = True

    def take_decoder_rows(self, x_hc, pre_mix):
        """Gather both encoder streams with exactly the accepted CED replay map."""
        state = PrefillState(x_hc, pre_mix)
        if state.num_tokens != self.num_tokens:
            raise ValueError("decoder handoff must contain every configured encoder row")
        result = state.take_rows(self.decoder_rows)
        return result.x_hc, result.pre_mix

    def attention(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        """Run the independent CPU attention reference and advance its private state."""
        if layer_id >= C.FLASH.num_hidden_layers // 2 and not self.decoder_published:
            raise ValueError("publish final encoder rows before decoder attention")
        values = self.layer_inputs(layer_id)
        if x.shape != (values.num_tokens, C.D) or x.dtype != torch.float32:
            raise ValueError("attention input must be normalized FP32 rows for this stage")
        result = prefill_attention_reference(
            x,
            values.weights,
            values.metadata,
            values.caches,
            mode=values.mode,
            ratio=values.ratio,
            compressed_indices=values.compressed_indices,
            candidate_mask=values.candidate_mask,
            precision=self.precision,
        )
        self.update_attention(layer_id, result)
        return result.output

    def attention_block(self, layer_id: int, x_hc, pre_mix):
        """CPU delayed-HC attention sublayer; caller applies Engram and the FFN."""
        stem = f"layers.{layer_id}."
        pre, post, comb = prefill_hc_mixes(
            x_hc,
            *[
                self.checkpoint.tensor(stem + name).float()
                for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base")
            ],
        )
        x = prefill_rms_norm(
            prefill_round_activation(hc_pre(x_hc, pre_mix), self.precision),
            self.checkpoint.tensor(stem + "attn_norm.weight").float(), precision=self.precision,
        )
        return prefill_round_activation(hc_post(self.attention(layer_id, x), x_hc, post, comb), self.precision), pre

    def publish_decoder(self, x_hc, pre_mix):
        """CPU reference for global publication from the full final encoder state."""
        if PrefillState(x_hc, pre_mix).num_tokens != self.num_tokens:
            raise ValueError("decoder publication requires the full encoder state")
        layer_id = C.FLASH.num_hidden_layers // 2
        x = prefill_rms_norm(
            prefill_round_activation(hc_pre(x_hc, pre_mix), self.precision),
            self.checkpoint.tensor(f"layers.{layer_id}.attn_norm.weight").float(),
            precision=self.precision,
        )
        values = self.publisher_inputs()
        caches = prefill_publish_decoder_reference(
            x, values.weights, values.metadata, values.caches, precision=self.precision
        )
        self.update_publisher(caches)
        return caches

    def block_reference(self, layer_id: int, state: PrefillState, *, hash_ids=None) -> PrefillState:
        """Complete CPU block: optional Engram, delayed-HC attention, then delayed-HC MoE."""
        if state.x_hc.ndim != 3:
            raise ValueError("FP32 reference blocks consume unstacked token-major states")
        x_hc = state.x_hc
        if layer_id in C.FLASH.engram_layer_ids:
            if hash_ids is None or hash_ids.shape[0] != state.num_tokens:
                raise ValueError("Engram reference needs this layer's hash ids for every token")
            x_hc = prefill_engram_reference(
                x_hc, self.checkpoint.engram(layer_id, hash_ids), precision=self.precision
            )
        mid, attention_pre = self.attention_block(layer_id, x_hc, state.pre_mix)
        stem = f"layers.{layer_id}."
        pre, post, comb = prefill_hc_mixes(
            mid,
            *[
                self.checkpoint.tensor(stem + name).float()
                for name in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base")
            ],
        )
        x = prefill_rms_norm(
            prefill_round_activation(hc_pre(mid, attention_pre), self.precision),
            self.checkpoint.tensor(stem + "ffn_norm.weight").float(), precision=self.precision,
        )
        ffn = prefill_moe_reference(x, self.checkpoint, layer_id, precision=self.precision)
        self.last_routes = ffn["indices"]
        return PrefillState(prefill_round_activation(hc_post(ffn["output"], mid, post, comb), self.precision), pre)
