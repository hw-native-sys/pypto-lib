# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Layer schedule and full-token state for DeepSeek V4.1 causal encoder prefill.

The encoder runs layers 0–19 on every supplied token. Its final stream is retained
because decoder Full Mode projects persistent global KV from that stream before
the decoder reduces its computation to the bounded SWA replay segment.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.metadata import ForwardMetadata, paged_slots
from models.deepseek_v4_1_flash.prefill_layer_plan import PrefillLayerPlan, resolve_prefill_layer_plan


ENCODER_LAYER_IDS = tuple(range(FLASH.num_hidden_layers // 2))


@dataclass(frozen=True)
class PrefillState:
    """The expanded mHC stream and delayed pre-mix between backbone layers."""

    x_hc: torch.Tensor
    pre_mix: torch.Tensor

    def __post_init__(self) -> None:
        if self.x_hc.ndim not in (3, 4) or self.pre_mix.ndim != self.x_hc.ndim - 1:
            raise ValueError("x_hc and pre_mix must have token-major or rank/token-major shapes")
        if self.x_hc.shape[:-1] != self.pre_mix.shape:
            raise ValueError("x_hc and pre_mix must share rank, token, and mHC dimensions")
        if self.x_hc.shape[-2] != FLASH.hc_mult or self.x_hc.shape[-1] != FLASH.hidden_size:
            raise ValueError("x_hc does not have the configured mHC and hidden dimensions")
        if self.x_hc.dtype != torch.float32 or self.pre_mix.dtype != torch.float32:
            raise ValueError("mHC residual stream and delayed pre-mix must be FP32")

    @property
    def num_tokens(self) -> int:
        return self.x_hc.shape[-3]

    def take_rows(self, rows: torch.Tensor) -> "PrefillState":
        """Gather the same packed token rows from both delayed-mix streams."""
        return PrefillState(
            self.x_hc.index_select(-3, rows.to(self.x_hc.device)),
            self.pre_mix.index_select(-2, rows.to(self.pre_mix.device)),
        )


EncoderLayer = Callable[[PrefillLayerPlan, PrefillState, ForwardMetadata], PrefillState]


def run_prefill_encoder(
    state: PrefillState,
    metadata: ForwardMetadata,
    run_layer: EncoderLayer,
) -> PrefillState:
    """Run the checkpoint's first 20 complete layers over every packed input row.

    ``run_layer`` owns the device call, weights and layer-local caches. It must
    publish C2A global KV/index state on Full layers before Reuse layers read it.
    """
    if state.num_tokens != int(metadata.query_start_loc[-1]):
        raise ValueError("encoder state and packed metadata have different token counts")
    if state.num_tokens == 0:
        raise ValueError("encoder prefill needs at least one active token")
    for layer_id in ENCODER_LAYER_IDS:
        plan = resolve_prefill_layer_plan(layer_id)
        next_state = run_layer(plan, state, metadata)
        if not isinstance(next_state, PrefillState) or next_state.x_hc.shape != state.x_hc.shape:
            raise ValueError(f"encoder layer {layer_id} changed the packed stream shape")
        state = next_state
    return state


@dataclass(frozen=True)
class CachedEncoderInputTail:
    """Initial mHC rows for the last SWA window of each cached request prefix."""

    state: PrefillState
    query_start_loc: torch.Tensor

    def __post_init__(self) -> None:
        starts = self.query_start_loc
        if starts.ndim != 1 or starts.numel() < 2 or int(starts[0]) != 0:
            raise ValueError("cached encoder input starts must begin at zero")
        if bool((starts[1:] < starts[:-1]).any()) or int(starts[-1]) != self.state.num_tokens:
            raise ValueError("cached encoder input starts do not cover the tail rows")


@dataclass(frozen=True)
class EncoderReplay:
    """Packed prefix replay plus suffix; only ``is_new`` rows may publish global KV."""

    source_rows: torch.Tensor
    query_start_loc: torch.Tensor
    position_ids: torch.Tensor
    token_to_req_indices: torch.Tensor
    replay_starts: torch.Tensor
    is_new: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lens: torch.Tensor

    @property
    def new_rows(self) -> torch.Tensor:
        return torch.nonzero(self.is_new, as_tuple=False).flatten()

    @property
    def cached_rows(self) -> torch.Tensor:
        return torch.nonzero(~self.is_new, as_tuple=False).flatten()


def select_encoder_replay(
    metadata: ForwardMetadata,
    window_block_table: torch.Tensor,
    cached_input: CachedEncoderInputTail,
    *,
    window: int = FLASH.sliding_window,
) -> EncoderReplay:
    """Replay each cached prefix's last SWA window together with every new row.

    The source indices address ``cached_input.state`` followed by the new input
    state. The callback must use ``is_new`` to suppress global KV/index and
    ratio-two compressor-state writes for replayed prefix rows. SWA reads are
    truncated to ``replay_starts`` even when older window pages still exist.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    starts = metadata.query_start_loc.to(torch.int64)
    cached_starts = cached_input.query_start_loc.to(torch.int64)
    if cached_starts.numel() != starts.numel():
        raise ValueError("cached encoder input must have one span per request")
    if window_block_table.ndim != 2 or window_block_table.shape[0] != starts.numel() - 1:
        raise ValueError("window block table must have one row per request")
    lengths = starts[1:] - starts[:-1]
    if bool((lengths < 0).any()) or int(starts[0]) != 0:
        raise ValueError("query_start_loc must be packed and nondecreasing")
    if int(starts[-1]) != metadata.position_ids.numel():
        raise ValueError("metadata positions do not cover every new row")
    if metadata.kv_seq_lens.numel() != lengths.numel():
        raise ValueError("kv_seq_lens must have one value per request")
    cached_total = int(cached_starts[-1])
    source_parts = []
    position_parts = []
    request_parts = []
    new_parts = []
    replay_lengths = []
    replay_starts = []
    for request, length in enumerate(lengths.tolist()):
        prefix = int(metadata.kv_seq_lens[request])
        if prefix < 0:
            raise ValueError("kv_seq_lens must be non-negative")
        available = int(cached_starts[request + 1] - cached_starts[request])
        if available > min(prefix, window):
            raise ValueError(f"request {request} has too many cached encoder input rows")
        replay_count = min(prefix, window) if length else 0
        if available < replay_count:
            raise ValueError(f"request {request} needs the final cached encoder input window")
        first = int(cached_starts[request + 1]) - replay_count
        last = int(cached_starts[request + 1])
        new_first, new_last = int(starts[request]), int(starts[request + 1])
        source_parts.append(torch.cat((
            torch.arange(first, last, device=starts.device),
            cached_total + torch.arange(new_first, new_last, device=starts.device),
        )))
        position_parts.append(torch.arange(prefix - replay_count, prefix + length, device=starts.device))
        request_parts.append(torch.full((replay_count + length,), request, dtype=torch.int64, device=starts.device))
        new_parts.append(torch.cat((
            torch.zeros(replay_count, dtype=torch.bool, device=starts.device),
            torch.ones(length, dtype=torch.bool, device=starts.device),
        )))
        replay_lengths.append(replay_count + length)
        replay_starts.append(prefix - replay_count)
        if length:
            expected = torch.arange(prefix, prefix + length, device=starts.device)
            if not torch.equal(metadata.position_ids[new_first:new_last].to(torch.int64), expected):
                raise ValueError(f"request {request} has noncontiguous new encoder positions")
            if not bool((metadata.token_to_req_indices[new_first:new_last] == request).all()):
                raise ValueError(f"request {request} has incorrect new encoder request indices")
    source_rows = torch.cat(source_parts).to(torch.int64)
    positions = torch.cat(position_parts).to(torch.int32)
    request_ids = torch.cat(request_parts).to(torch.int32)
    is_new = torch.cat(new_parts)
    replay_starts_tensor = torch.tensor(replay_starts, dtype=torch.int32, device=starts.device)
    query_start_loc = torch.tensor([0, *torch.tensor(replay_lengths).cumsum(0).tolist()], dtype=torch.int32, device=starts.device)
    if positions.numel():
        lens = torch.minimum(positions - replay_starts_tensor[request_ids.long()] + 1,
                             torch.full_like(positions, window))
        offsets = torch.arange(window, device=positions.device)
        visible = positions[:, None] - lens[:, None] + 1 + offsets
        valid = offsets[None, :] < lens[:, None]
        indices = torch.full_like(visible, -1, dtype=torch.int32)
        indices[valid] = paged_slots(
            visible[valid], request_ids[:, None].expand_as(visible)[valid], window_block_table
        ).to(torch.int32)
        slots = paged_slots(positions, request_ids, window_block_table)
    else:
        lens = positions.clone()
        indices = torch.empty((0, window), dtype=torch.int32, device=positions.device)
        slots = positions.to(torch.int64)
    return EncoderReplay(source_rows, query_start_loc, positions, request_ids,
                         replay_starts_tensor, is_new, slots, indices, lens)


EncoderReplayLayer = Callable[
    [PrefillLayerPlan, PrefillState, EncoderReplay, ForwardMetadata], PrefillState
]


@dataclass(frozen=True)
class EncoderReplayResult:
    """New final encoder rows and replayed cached-prefix outputs for decoder handoff."""

    new_state: PrefillState
    cached_state: PrefillState
    cached_query_start_loc: torch.Tensor
    replay: EncoderReplay


def run_prefill_encoder_replay(
    new_state: PrefillState,
    metadata: ForwardMetadata,
    window_block_table: torch.Tensor,
    cached_input: CachedEncoderInputTail,
    run_layer: EncoderReplayLayer,
) -> EncoderReplayResult:
    """Run layers 0–19 on bounded cached input rows and all uncached suffix rows."""
    if new_state.num_tokens != int(metadata.query_start_loc[-1]):
        raise ValueError("new encoder state and packed metadata have different token counts")
    replay = select_encoder_replay(metadata, window_block_table, cached_input)
    if not replay.new_rows.numel():
        raise ValueError("encoder replay needs at least one new token")
    if cached_input.state.x_hc.ndim != new_state.x_hc.ndim:
        raise ValueError("cached and new encoder inputs must have the same rank layout")
    combined = PrefillState(
        torch.cat((cached_input.state.x_hc, new_state.x_hc), dim=-3),
        torch.cat((cached_input.state.pre_mix, new_state.pre_mix), dim=-2),
    )
    state = combined.take_rows(replay.source_rows)
    for layer_id in ENCODER_LAYER_IDS:
        plan = resolve_prefill_layer_plan(layer_id)
        next_state = run_layer(plan, state, replay, metadata)
        if not isinstance(next_state, PrefillState) or next_state.x_hc.shape != state.x_hc.shape:
            raise ValueError(f"encoder replay layer {layer_id} changed the packed stream shape")
        state = next_state
    cached_lengths = (~replay.is_new).to(torch.int32)
    cached_counts = torch.stack([
        cached_lengths[int(replay.query_start_loc[r]):int(replay.query_start_loc[r + 1])].sum()
        for r in range(replay.query_start_loc.numel() - 1)
    ])
    cached_starts = torch.cat((torch.zeros(1, dtype=torch.int32, device=cached_counts.device),
                               cached_counts.cumsum(0)))
    return EncoderReplayResult(state.take_rows(replay.new_rows), state.take_rows(replay.cached_rows),
                               cached_starts, replay)
