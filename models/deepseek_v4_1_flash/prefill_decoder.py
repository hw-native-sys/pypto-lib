# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed token selection and SWA metadata for DeepSeek V4.1 decoder prefill."""

from dataclasses import dataclass
from typing import Callable

import torch

from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.metadata import ForwardMetadata, paged_slots
from models.deepseek_v4_1_flash.prefill_encoder import PrefillState
from models.deepseek_v4_1_flash.prefill_layer_plan import PrefillLayerPlan, resolve_prefill_layer_plan


DECODER_LAYER_IDS = tuple(range(FLASH.num_hidden_layers // 2, FLASH.num_hidden_layers))


@dataclass(frozen=True)
class CachedEncoderTail:
    """Final encoder outputs for cached prefix tails, packed in request order.

    Each request contributes the final rows of its cached prefix, ending at
    ``kv_seq_lens[request] - 1``. The state must include the delayed mHC mix
    alongside the expanded stream so decoder layer 20 can replay those rows.
    """

    state: PrefillState
    query_start_loc: torch.Tensor

    def __post_init__(self) -> None:
        starts = self.query_start_loc
        if starts.ndim != 1 or starts.numel() < 2 or int(starts[0]) != 0:
            raise ValueError("cached encoder starts must begin at zero and contain a request")
        if bool((starts[1:] < starts[:-1]).any()) or int(starts[-1]) != self.state.num_tokens:
            raise ValueError("cached encoder starts do not cover the packed prefix tails")


@dataclass(frozen=True)
class DecoderReplay:
    """Decoder rows in request order, indexed into cached tails plus new encoder rows."""

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
        """Last replay row of each nonempty request, or -1 for an empty request."""
        return torch.where(self.query_lens > 0, self.query_start_loc[1:] - 1, -1)

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        """Pack cached-prefix rows followed by new encoder rows into replay order."""
        if self.source_rows.numel() and value.shape[0] <= int(self.source_rows.max()):
            raise ValueError("encoder tensor does not cover the decoder replay rows")
        return value.index_select(0, self.source_rows.to(value.device))


def select_decoder_replay(
    metadata: ForwardMetadata,
    window_block_table: torch.Tensor,
    *,
    window: int = FLASH.sliding_window,
    cached_prefix: CachedEncoderTail | None = None,
) -> DecoderReplay:
    """Select at most the final ``window`` encoded rows of each active request.

    ``source_rows`` index a concatenation of cached encoder tails and the new
    encoder output. A resumed request supplies only the missing cached tail
    rows; absent rows are rejected before any decoder layer reads SWA state.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    starts = metadata.query_start_loc.to(torch.int64)
    if starts.ndim != 1 or starts.numel() < 2 or int(starts[0]) != 0:
        raise ValueError("query_start_loc must start at zero and contain a request")
    lengths = starts[1:] - starts[:-1]
    if bool((lengths < 0).any()):
        raise ValueError("query_start_loc must be nondecreasing")
    total = int(starts[-1])
    if metadata.position_ids.numel() != total or metadata.token_to_req_indices.numel() != total:
        raise ValueError("encoder metadata does not cover the packed query rows")
    if metadata.kv_seq_lens.numel() != lengths.numel():
        raise ValueError("kv_seq_lens must have one value per request")
    if window_block_table.ndim != 2 or window_block_table.shape[0] != lengths.numel():
        raise ValueError("window block table must have one row per request")
    if cached_prefix is not None and cached_prefix.query_start_loc.numel() != starts.numel():
        raise ValueError("cached encoder tails must have one span per request")
    cached_starts = (
        cached_prefix.query_start_loc.to(torch.int64)
        if cached_prefix is not None
        else torch.zeros_like(starts)
    )
    cached_total = int(cached_starts[-1])

    selected: list[torch.Tensor] = []
    replay_positions: list[torch.Tensor] = []
    replay_requests: list[torch.Tensor] = []
    replay_lengths: list[int] = []
    replay_starts: list[int] = []
    for request, length in enumerate(lengths.tolist()):
        prefix = int(metadata.kv_seq_lens[request])
        if prefix < 0:
            raise ValueError("kv_seq_lens must be non-negative")
        available = int(cached_starts[request + 1] - cached_starts[request])
        if available > min(prefix, window):
            raise ValueError(f"request {request} has too many cached encoder tail rows")
        count = min(prefix + length, window) if length else 0
        new_count = min(length, count)
        cached_count = count - new_count
        if available < cached_count:
            raise ValueError(
                f"request {request} needs encoder replay rows from its cached prefix"
            )
        cached_first = int(cached_starts[request + 1]) - cached_count
        new_first = int(starts[request + 1]) - new_count
        rows = torch.cat((
            torch.arange(cached_first, int(cached_starts[request + 1]), device=starts.device),
            cached_total + torch.arange(new_first, int(starts[request + 1]), device=starts.device),
        ))
        selected.append(rows)
        replay_lengths.append(count)
        replay_starts.append(prefix + length - count)
        replay_positions.append(torch.arange(prefix + length - count, prefix + length, device=starts.device))
        replay_requests.append(torch.full((count,), request, dtype=torch.int64, device=starts.device))
        if length:
            request_positions = metadata.position_ids[int(starts[request]):int(starts[request + 1])]
            expected = torch.arange(prefix, prefix + length, device=request_positions.device)
            if not torch.equal(request_positions.to(torch.int64), expected):
                raise ValueError(f"request {request} has noncontiguous encoder positions")
            request_ids = metadata.token_to_req_indices[int(starts[request]):int(starts[request + 1])]
            if not bool((request_ids == request).all()):
                raise ValueError(f"request {request} has incorrect encoder request indices")

    source_rows = torch.cat(selected).to(torch.int64)
    query_lens = torch.tensor(replay_lengths, dtype=torch.int32, device=starts.device)
    query_start_loc = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=starts.device), query_lens.cumsum(0))
    )
    replay_starts_tensor = torch.tensor(replay_starts, dtype=torch.int32, device=starts.device)
    positions = torch.cat(replay_positions).to(torch.int32)
    request_ids = torch.cat(replay_requests).to(torch.int32)
    if positions.numel():
        row_starts = replay_starts_tensor[request_ids.to(torch.long)]
        window_lens = torch.minimum(positions - row_starts + 1, torch.full_like(positions, window))
        offsets = torch.arange(window, device=positions.device)
        visible = positions.unsqueeze(1) - window_lens.unsqueeze(1) + 1 + offsets
        valid = offsets.unsqueeze(0) < window_lens.unsqueeze(1)
        window_indices = torch.full_like(visible, -1, dtype=torch.int32)
        visible_requests = request_ids.unsqueeze(1).expand_as(visible)
        window_indices[valid] = paged_slots(
            visible[valid], visible_requests[valid], window_block_table
        ).to(torch.int32)
        window_slots = paged_slots(positions, request_ids, window_block_table)
    else:
        window_lens = positions.clone()
        window_indices = torch.empty((0, window), dtype=torch.int32, device=positions.device)
        window_slots = positions.to(torch.int64)
    return DecoderReplay(
        source_rows=source_rows,
        query_start_loc=query_start_loc,
        query_lens=query_lens,
        replay_starts=replay_starts_tensor,
        position_ids=positions,
        token_to_req_indices=request_ids,
        compressed_lens=positions + 1,
        window_slots=window_slots,
        window_indices=window_indices,
        window_lens=window_lens,
    )


DecoderLayer = Callable[
    [PrefillLayerPlan, PrefillState, DecoderReplay, ForwardMetadata], PrefillState
]
GlobalPublisher = Callable[[PrefillLayerPlan, PrefillState, ForwardMetadata], None]


@dataclass(frozen=True)
class DecoderPrefillResult:
    """The replayed decoder stream and its request-local position mapping."""

    state: PrefillState
    replay: DecoderReplay


def run_prefill_decoder(
    encoder_state: PrefillState,
    encoder_metadata: ForwardMetadata,
    window_block_table: torch.Tensor,
    publish_global: GlobalPublisher,
    run_layer: DecoderLayer,
    *,
    cached_prefix: CachedEncoderTail | None = None,
) -> DecoderPrefillResult:
    """Publish decoder global KV, then run layers 20–39 on bounded SWA replay.

    Full Mode's publisher sees every newly encoded row. The layer executor sees
    only the final window of each request, including cached encoder tail rows
    when the new suffix is short. It reads the published global cache without
    re-publishing it from its own replayed hidden stream. Reindex and Reuse
    layers use that source cache and their own index selections.
    """
    if encoder_state.num_tokens != int(encoder_metadata.query_start_loc[-1]):
        raise ValueError("final encoder state and packed metadata have different token counts")
    replay = select_decoder_replay(encoder_metadata, window_block_table, cached_prefix=cached_prefix)
    if not replay.source_rows.numel():
        raise ValueError("decoder prefill needs at least one active token")
    if cached_prefix is not None:
        if cached_prefix.state.x_hc.ndim != encoder_state.x_hc.ndim:
            raise ValueError("cached and new encoder states must have the same rank layout")
        combined = PrefillState(
            torch.cat((cached_prefix.state.x_hc, encoder_state.x_hc), dim=-3),
            torch.cat((cached_prefix.state.pre_mix, encoder_state.pre_mix), dim=-2),
        )
    else:
        combined = encoder_state
    for layer_id in DECODER_LAYER_IDS:
        plan = resolve_prefill_layer_plan(layer_id)
        if plan.global_publisher_module is not None:
            publish_global(plan, encoder_state, encoder_metadata)
    state = combined.take_rows(replay.source_rows)
    for layer_id in DECODER_LAYER_IDS:
        plan = resolve_prefill_layer_plan(layer_id)
        next_state = run_layer(plan, state, replay, encoder_metadata)
        if not isinstance(next_state, PrefillState) or next_state.x_hc.shape != state.x_hc.shape:
            raise ValueError(f"decoder layer {layer_id} changed the replay stream shape")
        state = next_state
    return DecoderPrefillResult(state, replay)
