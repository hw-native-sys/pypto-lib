# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Request-owned state for V4.1 DSpark bootstrap and recurrent decode."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import (
    BLOCK_SIZE,
    DECODE_ROWS_PER_REQUEST,
    DSPARK_DRAFT_LAYERS,
    DSPARK_QUERY_WIDTH,
    EP_SIZE,
    FLASH,
    MAX_BATCH_PER_DP,
    TABLE_DYN,
    T_DYN,
    TP_SIZE,
)


BATCH = MAX_BATCH_PER_DP
ROWS = DECODE_ROWS_PER_REQUEST
STATE_VALID = 0
STATE_GENERATION = 1
STATE_ANCHOR = 2
STATE_COMMITTED = 3
STATE_DRAFT_COUNT = 4
STATE_LIMIT = 5
STATE_META_WIDTH = 6
STATE_TOKEN_WIDTH = ROWS
PREFILL_SAMPLE_ROWS = 16
SAMPLED_STRIDE = 8
ROPE_HALF = FLASH.qk_rope_head_dim // 2
ROPE_ROWS_DYN = pl.dynamic("V41_DSPARK_ROPE_ROWS_DYN")

if TP_SIZE != 1:
    raise ValueError("DSpark request state currently requires TP1")


def golden_bootstrap(
    slot_ids: torch.Tensor,
    generations: torch.Tensor,
    prompt_lengths: torch.Tensor,
    position_limits: torch.Tensor,
    sampled_rows: torch.Tensor,
    sampled_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    state_tokens: torch.Tensor,
    state_meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish one terminal prefill token and its five speculative successors."""
    tokens = state_tokens.clone()
    meta = state_meta.clone()
    for request in range(slot_ids.numel()):
        slot = int(slot_ids[request])
        row = int(sampled_rows[request])
        if slot < 0 or slot >= meta.shape[0] or row < 0 or row >= sampled_ids.shape[0]:
            continue
        generation = int(generations[request])
        prompt = int(prompt_lengths[request])
        limit = int(position_limits[request])
        if prompt <= 0 or prompt >= limit:
            continue
        if int(meta[slot, STATE_GENERATION]) != generation or int(meta[slot, STATE_VALID]) != 0:
            continue
        tokens[slot, 0] = sampled_ids[row, 0]
        meta[slot, STATE_VALID] = 1
        meta[slot, STATE_ANCHOR] = prompt
        meta[slot, STATE_COMMITTED] = prompt
        meta[slot, STATE_LIMIT] = limit
        count = DSPARK_QUERY_WIDTH if prompt + DSPARK_QUERY_WIDTH < limit else 0
        meta[slot, STATE_DRAFT_COUNT] = count
        if count:
            tokens[slot, 1:ROWS] = draft_ids[request]
    return tokens, meta


def golden_bootstrap_inputs(tensors, _params):
    """Keep the sampled anchor while disabling draft queries beyond the limit."""
    tokens = tensors["next_prefill_tokens"]
    first = tensors["first_query_positions"]
    tokens.zero_()
    first.fill_(-1)
    for request, slot in enumerate(tensors["slot_ids"].tolist()):
        row = int(tensors["sampled_rows"][request])
        prompt = int(tensors["prompt_lengths"][request])
        limit = int(tensors["position_limits"][request])
        if not (0 <= slot < BATCH and 0 <= row < PREFILL_SAMPLE_ROWS and 0 < prompt < limit):
            continue
        meta = tensors["state_meta"][slot]
        if int(meta[STATE_VALID]) != 0 or int(meta[STATE_GENERATION]) != int(tensors["generations"][request]):
            continue
        tokens[request] = tensors["sampled_ids"][row, 0]
        if prompt + DSPARK_QUERY_WIDTH < limit:
            first[request] = prompt


@pl.jit
def bootstrap_inputs_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[BATCH], pl.INT32],
    position_limits: pl.Tensor[[BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], pl.INT32],
    state_meta: pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32],
    next_prefill_tokens: pl.Out[pl.Tensor[[BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_inputs"):
        for request in pl.range(BATCH):
            pl.write(next_prefill_tokens, [request], pl.cast(0, pl.INT64))
            pl.write(first_query_positions, [request], pl.cast(-1, pl.INT32))
            slot = pl.read(slot_ids, [request])
            row = pl.read(sampled_rows, [request])
            prompt = pl.read(prompt_lengths, [request])
            limit = pl.read(position_limits, [request])
            if slot >= 0 and slot < BATCH and row >= 0 and row < PREFILL_SAMPLE_ROWS:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(generations, [request])
                if valid == 0 and generation == expected and prompt > 0 and prompt < limit:
                    token = pl.read(sampled_ids, [row, 0])
                    pl.write(next_prefill_tokens, [request], pl.cast(token, pl.INT64))
                    if prompt + DSPARK_QUERY_WIDTH < limit:
                        pl.write(first_query_positions, [request], prompt)
    return next_prefill_tokens, first_query_positions


@pl.jit.host
def l3_bootstrap_inputs(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    position_limits: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[EP_SIZE, PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], pl.INT32],
    state_meta: pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32],
    next_prefill_tokens: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        bootstrap_inputs_rank(
            slot_ids[rank], generations[rank], prompt_lengths[rank], position_limits[rank], sampled_rows[rank],
            sampled_ids[rank], state_meta[rank], next_prefill_tokens[rank], first_query_positions[rank],
            device=rank,
        )


@pl.jit
def publish_bootstrap_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[BATCH], pl.INT32],
    position_limits: pl.Tensor[[BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], pl.INT32],
    draft_ids: pl.Tensor[[BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_publish"):
        for request in pl.range(BATCH):
            slot = pl.read(slot_ids, [request])
            row = pl.read(sampled_rows, [request])
            prompt = pl.read(prompt_lengths, [request])
            limit = pl.read(position_limits, [request])
            if slot >= 0 and slot < BATCH and row >= 0 and row < PREFILL_SAMPLE_ROWS:
                if prompt > 0 and prompt < limit:
                    generation = pl.read(state_meta, [slot, STATE_GENERATION])
                    expected = pl.read(generations, [request])
                    valid = pl.read(state_meta, [slot, STATE_VALID])
                    if generation == expected and valid == 0:
                        token = pl.read(sampled_ids, [row, 0])
                        pl.write(state_tokens, [slot, 0], pl.cast(token, pl.INT64))
                        pl.write(state_meta, [slot, STATE_ANCHOR], prompt)
                        pl.write(state_meta, [slot, STATE_COMMITTED], prompt)
                        pl.write(state_meta, [slot, STATE_LIMIT], limit)
                        draft_count = pl.cast(0, pl.INT32)
                        if prompt + DSPARK_QUERY_WIDTH < limit:
                            draft_count = pl.cast(DSPARK_QUERY_WIDTH, pl.INT32)
                            for offset in pl.range(DSPARK_QUERY_WIDTH):
                                draft = pl.read(draft_ids, [request, offset])
                                pl.write(state_tokens, [slot, offset + 1], pl.cast(draft, pl.INT64))
                        pl.write(state_meta, [slot, STATE_DRAFT_COUNT], draft_count)
                        pl.write(state_meta, [slot, STATE_VALID], pl.cast(1, pl.INT32))
    return state_tokens, state_meta


@pl.jit.host
def l3_publish_bootstrap(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    position_limits: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[EP_SIZE, PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], pl.INT32],
    draft_ids: pl.Tensor[[EP_SIZE, BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        publish_bootstrap_rank(
            slot_ids[rank], generations[rank], prompt_lengths[rank], position_limits[rank],
            sampled_rows[rank], sampled_ids[rank], draft_ids[rank], state_tokens[rank], state_meta[rank],
            device=rank,
        )


def golden_prepare_target(
    slot_ids: torch.Tensor,
    generations: torch.Tensor,
    state_tokens: torch.Tensor,
    state_meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve one six-row verification slab per request."""
    ids = torch.zeros((slot_ids.numel(), ROWS), dtype=torch.int64, device=slot_ids.device)
    positions = torch.zeros_like(ids, dtype=torch.int32)
    widths = torch.zeros_like(slot_ids, dtype=torch.int32)
    for request in range(slot_ids.numel()):
        slot = int(slot_ids[request])
        if slot < 0 or slot >= state_meta.shape[0]:
            continue
        if int(state_meta[slot, STATE_VALID]) != 1 or int(state_meta[slot, STATE_GENERATION]) != int(generations[request]):
            continue
        anchor = int(state_meta[slot, STATE_ANCHOR])
        limit = int(state_meta[slot, STATE_LIMIT])
        if anchor >= limit:
            continue
        drafts = min(max(int(state_meta[slot, STATE_DRAFT_COUNT]), 0), DSPARK_QUERY_WIDTH)
        width = drafts + 1 if anchor + drafts < limit else 1
        widths[request] = width
        ids[request, :width] = state_tokens[slot, :width]
        positions[request] = torch.arange(anchor, anchor + ROWS, device=slot_ids.device).clamp_max(limit - 1)
    return ids, positions, widths


@pl.jit
def prepare_target_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    state_tokens: pl.Tensor[[BATCH, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT64]],
    position_ids: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_prepare_target"):
        for request in pl.range(BATCH):
            width = pl.cast(0, pl.INT32)
            anchor = pl.cast(0, pl.INT32)
            limit = pl.cast(1, pl.INT32)
            slot = pl.read(slot_ids, [request])
            if slot >= 0 and slot < BATCH:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(generations, [request])
                anchor = pl.read(state_meta, [slot, STATE_ANCHOR])
                limit = pl.read(state_meta, [slot, STATE_LIMIT])
                if valid == 1 and generation == expected and anchor < limit:
                    count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                    count = pl.max(pl.min(count, DSPARK_QUERY_WIDTH), 0)
                    width = pl.cast(1, pl.INT32)
                    if anchor + count < limit:
                        width = pl.cast(count + 1, pl.INT32)
            pl.write(active_widths, [request], width)
            for offset in pl.range(ROWS):
                token = pl.cast(0, pl.INT64)
                position = pl.cast(0, pl.INT32)
                if width > 0:
                    position = pl.cast(pl.min(anchor + offset, limit - 1), pl.INT32)
                    if offset < width:
                        token = pl.read(state_tokens, [slot, offset])
                pl.write(input_ids, [request, offset], token)
                pl.write(position_ids, [request, offset], position)
    return input_ids, position_ids, active_widths


@pl.jit.host
def l3_prepare_target(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    state_tokens: pl.Tensor[[EP_SIZE, BATCH, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT64]],
    position_ids: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        prepare_target_rank(
            slot_ids[rank], generations[rank], state_tokens[rank], state_meta[rank],
            input_ids[rank], position_ids[rank], active_widths[rank], device=rank,
        )


def golden_accept_target(
    slot_ids: torch.Tensor,
    generations: torch.Tensor,
    active_widths: torch.Tensor,
    sampled_ids: torch.Tensor,
    state_tokens: torch.Tensor,
    state_meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept the longest matching draft prefix and retain the next target token."""
    tokens = state_tokens.clone()
    meta = state_meta.clone()
    accepted_ids = torch.full((slot_ids.numel(), ROWS), -1, dtype=torch.int32, device=slot_ids.device)
    accepted_counts = torch.zeros_like(slot_ids, dtype=torch.int32)
    for request in range(slot_ids.numel()):
        slot = int(slot_ids[request])
        width = int(active_widths[request])
        if slot < 0 or slot >= meta.shape[0] or width < 1 or width > ROWS:
            continue
        if int(meta[slot, STATE_VALID]) != 1 or int(meta[slot, STATE_GENERATION]) != int(generations[request]):
            continue
        count = min(max(int(meta[slot, STATE_DRAFT_COUNT]), 0), width - 1)
        matched = 0
        for offset in range(count):
            if int(tokens[slot, offset + 1]) != int(sampled_ids[request * ROWS + offset, 0]):
                break
            matched += 1
        accepted = matched + 1
        accepted_counts[request] = accepted
        accepted_ids[request, :accepted] = sampled_ids[request * ROWS : request * ROWS + accepted, 0]
        tokens[slot, 0] = sampled_ids[request * ROWS + matched, 0]
        meta[slot, STATE_ANCHOR] += accepted
        meta[slot, STATE_COMMITTED] += accepted
        meta[slot, STATE_DRAFT_COUNT] = 0
    return tokens, meta, accepted_ids, accepted_counts


@pl.jit
def accept_target_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    active_widths: pl.Tensor[[BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[BATCH * ROWS, SAMPLED_STRIDE], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
    context_valid: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT32]],
    context_positions: pl.Out[pl.Tensor[[BATCH, ROWS], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[BATCH], pl.INT64]],
    anchor_ids: pl.Out[pl.Tensor[[BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_accept_target"):
        for request in pl.range(BATCH):
            accepted = pl.cast(0, pl.INT32)
            anchor = pl.cast(0, pl.INT32)
            can_draft = pl.cast(0, pl.INT32)
            next_token = pl.cast(0, pl.INT64)
            slot = pl.read(slot_ids, [request])
            width = pl.read(active_widths, [request])
            if slot >= 0 and slot < BATCH and width > 0 and width <= ROWS:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(generations, [request])
                if valid == 1 and generation == expected:
                    anchor = pl.read(state_meta, [slot, STATE_ANCHOR])
                    limit = pl.read(state_meta, [slot, STATE_LIMIT])
                    count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                    count = pl.max(pl.min(count, width - 1), 0)
                    matched = pl.cast(0, pl.INT32)
                    still_matching = pl.cast(1, pl.INT32)
                    for offset in pl.range(DSPARK_QUERY_WIDTH):
                        if offset < count and still_matching == 1:
                            draft = pl.read(state_tokens, [slot, offset + 1])
                            predicted = pl.read(sampled_ids, [request * ROWS + offset, 0])
                            if draft == pl.cast(predicted, pl.INT64):
                                matched = pl.cast(offset + 1, pl.INT32)
                            else:
                                still_matching = pl.cast(0, pl.INT32)
                    accepted = pl.cast(matched + 1, pl.INT32)
                    next_token = pl.cast(pl.read(sampled_ids, [request * ROWS + matched, 0]), pl.INT64)
                    pl.write(state_tokens, [slot, 0], next_token)
                    pl.write(state_meta, [slot, STATE_ANCHOR], pl.cast(anchor + accepted, pl.INT32))
                    committed = pl.read(state_meta, [slot, STATE_COMMITTED])
                    pl.write(state_meta, [slot, STATE_COMMITTED], pl.cast(committed + accepted, pl.INT32))
                    pl.write(state_meta, [slot, STATE_DRAFT_COUNT], pl.cast(0, pl.INT32))
                    if anchor + accepted + DSPARK_QUERY_WIDTH < limit:
                        can_draft = pl.cast(1, pl.INT32)
            pl.write(accepted_counts, [request], accepted)
            pl.write(num_sampled, [request], pl.cast(accepted * can_draft, pl.INT32))
            pl.write(last_sampled, [request], next_token)
            # Publish the exact token selected by target acceptance.  This is
            # the handoff consumed by the first Markov step.
            pl.write(anchor_ids, [request], next_token)
            pl.write(first_query_positions, [request], pl.cast((anchor + accepted) * can_draft, pl.INT32))
            for offset in pl.range(ROWS):
                token = pl.cast(-1, pl.INT32)
                valid = pl.cast(0, pl.INT32)
                position = pl.cast(0, pl.INT32)
                if offset < accepted:
                    token = pl.read(sampled_ids, [request * ROWS + offset, 0])
                    position = pl.cast(anchor + offset, pl.INT32)
                    valid = can_draft
                pl.write(accepted_ids, [request, offset], token)
                pl.write(context_valid, [request, offset], valid)
                pl.write(context_positions, [request, offset], position)
    return state_tokens, state_meta, accepted_ids, accepted_counts


@pl.jit.host
def l3_accept_target(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    active_widths: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[EP_SIZE, BATCH * ROWS, SAMPLED_STRIDE], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
    context_valid: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32]],
    context_positions: pl.Out[pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT64]],
    anchor_ids: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        accept_target_rank(
            slot_ids[rank], generations[rank], active_widths[rank], sampled_ids[rank],
            state_tokens[rank], state_meta[rank], accepted_ids[rank], accepted_counts[rank],
            context_valid[rank], context_positions[rank], num_sampled[rank],
            last_sampled[rank], anchor_ids[rank], first_query_positions[rank], device=rank,
        )


def golden_commit_drafts(
    slot_ids: torch.Tensor,
    generations: torch.Tensor,
    draft_ids: torch.Tensor,
    state_tokens: torch.Tensor,
    state_meta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish Markov drafts only after the preceding target window was consumed."""
    tokens = state_tokens.clone()
    meta = state_meta.clone()
    committed = torch.zeros_like(slot_ids, dtype=torch.int32)
    for request in range(slot_ids.numel()):
        slot = int(slot_ids[request])
        if slot < 0 or slot >= meta.shape[0]:
            continue
        if int(meta[slot, STATE_VALID]) != 1 or int(meta[slot, STATE_GENERATION]) != int(generations[request]):
            continue
        if int(meta[slot, STATE_DRAFT_COUNT]) != 0:
            continue
        if int(meta[slot, STATE_ANCHOR]) + DSPARK_QUERY_WIDTH >= int(meta[slot, STATE_LIMIT]):
            continue
        tokens[slot, 1:ROWS] = draft_ids[request]
        meta[slot, STATE_DRAFT_COUNT] = DSPARK_QUERY_WIDTH
        committed[request] = 1
    return tokens, meta, committed


@pl.jit
def commit_drafts_rank(
    slot_ids: pl.Tensor[[BATCH], pl.INT32],
    generations: pl.Tensor[[BATCH], pl.INT32],
    draft_ids: pl.Tensor[[BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[BATCH, STATE_META_WIDTH], pl.INT32]],
    committed: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_commit_drafts"):
        for request in pl.range(BATCH):
            published = pl.cast(0, pl.INT32)
            slot = pl.read(slot_ids, [request])
            if slot >= 0 and slot < BATCH:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(generations, [request])
                count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                anchor = pl.read(state_meta, [slot, STATE_ANCHOR])
                limit = pl.read(state_meta, [slot, STATE_LIMIT])
                if valid == 1 and generation == expected and count == 0:
                    if anchor + DSPARK_QUERY_WIDTH < limit:
                        for offset in pl.range(DSPARK_QUERY_WIDTH):
                            draft = pl.read(draft_ids, [request, offset])
                            pl.write(state_tokens, [slot, offset + 1], pl.cast(draft, pl.INT64))
                        pl.write(state_meta, [slot, STATE_DRAFT_COUNT], pl.cast(DSPARK_QUERY_WIDTH, pl.INT32))
                        published = pl.cast(1, pl.INT32)
            pl.write(committed, [request], published)
    return state_tokens, state_meta, committed


@pl.jit.host
def l3_commit_drafts(
    slot_ids: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    draft_ids: pl.Tensor[[EP_SIZE, BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, BATCH, STATE_META_WIDTH], pl.INT32]],
    committed: pl.Out[pl.Tensor[[EP_SIZE, BATCH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        commit_drafts_rank(
            slot_ids[rank], generations[rank], draft_ids[rank], state_tokens[rank],
            state_meta[rank], committed[rank], device=rank,
        )


@pl.jit
def prepare_drafter_geometry_rank(
    context_valid: pl.Tensor[[BATCH, ROWS], pl.INT32],
    context_positions: pl.Tensor[[BATCH, ROWS], pl.INT32],
    first_query_positions: pl.Tensor[[BATCH], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, BATCH, TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, BATCH * ROWS], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[BATCH * ROWS, ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[BATCH * ROWS, ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
):
    block_tables.bind_dynamic(2, TABLE_DYN)
    rope_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    rope_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    table_blocks = pl.tensor.dim(block_tables, 2)
    rope_rows = pl.tensor.dim(rope_cos_table, 0)
    for _core in pl.spmd(1, name_hint="dspark_drafter_geometry"):
        for request in pl.range(BATCH):
            first = pl.read(first_query_positions, [request])
            for offset in pl.range(ROWS):
                row = request * ROWS + offset
                position = pl.read(context_positions, [request, offset])
                active = pl.read(context_valid, [request, offset]) == 1
                active = active and position >= 0 and position < rope_rows
                for col in pl.range(ROPE_HALF):
                    cos = pl.cast(0.0, pl.FP32)
                    sin = pl.cast(0.0, pl.FP32)
                    if active:
                        cos = pl.read(rope_cos_table, [position, col])
                        sin = pl.read(rope_sin_table, [position, col])
                    pl.write(context_rope_cos, [row, col], cos)
                    pl.write(context_rope_sin, [row, col], sin)
                for layer in pl.range(DSPARK_DRAFT_LAYERS):
                    slot = pl.cast(-1, pl.INT64)
                    if active and position // BLOCK_SIZE < table_blocks:
                        physical = pl.read(block_tables, [layer, request, position // BLOCK_SIZE])
                        if physical >= 0:
                            slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                    pl.write(context_slots, [layer, row], slot)
            for offset in pl.range(DSPARK_QUERY_WIDTH):
                row = request * DSPARK_QUERY_WIDTH + offset
                query_position = pl.cast(first + offset, pl.INT32)
                active = first > 0 and query_position < rope_rows
                for col in pl.range(ROPE_HALF):
                    cos = pl.cast(0.0, pl.FP32)
                    sin = pl.cast(0.0, pl.FP32)
                    if active:
                        cos = pl.read(rope_cos_table, [query_position, col])
                        sin = pl.read(rope_sin_table, [query_position, col])
                    pl.write(query_rope_cos, [row, col], cos)
                    pl.write(query_rope_sin, [row, col], sin)
    return context_slots, context_rope_cos, context_rope_sin, query_rope_cos, query_rope_sin


@pl.jit.host
def l3_prepare_drafter_geometry(
    context_valid: pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32],
    context_positions: pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32],
    first_query_positions: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    block_tables: pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, BATCH, TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[EP_SIZE, ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[EP_SIZE, ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, BATCH * ROWS], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, BATCH * ROWS, ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, BATCH * ROWS, ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
):
    block_tables.bind_dynamic(3, TABLE_DYN)
    rope_cos_table.bind_dynamic(1, ROPE_ROWS_DYN)
    rope_sin_table.bind_dynamic(1, ROPE_ROWS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_drafter_geometry_rank(
            context_valid[rank], context_positions[rank], first_query_positions[rank],
            block_tables[rank], rope_cos_table[rank], rope_sin_table[rank],
            context_slots[rank], context_rope_cos[rank], context_rope_sin[rank],
            query_rope_cos[rank], query_rope_sin[rank], device=rank,
        )


@pl.jit
def prepare_prefill_drafter_geometry_rank(
    positions: pl.Tensor[[T_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    first_query_positions: pl.Tensor[[BATCH], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, BATCH, TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[T_DYN, ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[T_DYN, ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
):
    positions.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    block_tables.bind_dynamic(2, TABLE_DYN)
    rope_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    rope_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    context_slots.bind_dynamic(1, T_DYN)
    context_rope_cos.bind_dynamic(0, T_DYN)
    context_rope_sin.bind_dynamic(0, T_DYN)
    tokens = pl.tensor.dim(positions, 0)
    table_blocks = pl.tensor.dim(block_tables, 2)
    rope_rows = pl.tensor.dim(rope_cos_table, 0)
    for _core in pl.spmd(1, name_hint="dspark_prefill_geometry"):
        active_tokens = pl.read(num_tokens, [0])
        for row in pl.range(tokens):
            position = pl.read(positions, [row])
            request = pl.read(request_ids, [row])
            active = row < active_tokens and request >= 0 and request < BATCH
            active = active and position >= 0 and position < rope_rows
            for col in pl.range(ROPE_HALF):
                cos = pl.cast(0.0, pl.FP32)
                sin = pl.cast(0.0, pl.FP32)
                if active:
                    cos = pl.read(rope_cos_table, [position, col])
                    sin = pl.read(rope_sin_table, [position, col])
                pl.write(context_rope_cos, [row, col], cos)
                pl.write(context_rope_sin, [row, col], sin)
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                slot = pl.cast(-1, pl.INT64)
                if active and position // BLOCK_SIZE < table_blocks:
                    physical = pl.read(block_tables, [layer, request, position // BLOCK_SIZE])
                    if physical >= 0:
                        slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                pl.write(context_slots, [layer, row], slot)
        for request in pl.range(BATCH):
            first = pl.read(first_query_positions, [request])
            for offset in pl.range(DSPARK_QUERY_WIDTH):
                row = request * DSPARK_QUERY_WIDTH + offset
                query_position = pl.cast(first + offset, pl.INT32)
                active = first > 0 and query_position < rope_rows
                for col in pl.range(ROPE_HALF):
                    cos = pl.cast(0.0, pl.FP32)
                    sin = pl.cast(0.0, pl.FP32)
                    if active:
                        cos = pl.read(rope_cos_table, [query_position, col])
                        sin = pl.read(rope_sin_table, [query_position, col])
                    pl.write(query_rope_cos, [row, col], cos)
                    pl.write(query_rope_sin, [row, col], sin)
    return context_slots, context_rope_cos, context_rope_sin, query_rope_cos, query_rope_sin


@pl.jit.host
def l3_prepare_prefill_drafter_geometry(
    positions: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    request_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    num_tokens: pl.Tensor[[EP_SIZE, 1], pl.INT32],
    first_query_positions: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    block_tables: pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, BATCH, TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[EP_SIZE, ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[EP_SIZE, ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
):
    positions.bind_dynamic(1, T_DYN)
    request_ids.bind_dynamic(1, T_DYN)
    block_tables.bind_dynamic(3, TABLE_DYN)
    rope_cos_table.bind_dynamic(1, ROPE_ROWS_DYN)
    rope_sin_table.bind_dynamic(1, ROPE_ROWS_DYN)
    context_slots.bind_dynamic(2, T_DYN)
    context_rope_cos.bind_dynamic(1, T_DYN)
    context_rope_sin.bind_dynamic(1, T_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_prefill_drafter_geometry_rank(
            positions[rank], request_ids[rank], num_tokens[rank], first_query_positions[rank],
            block_tables[rank], rope_cos_table[rank], rope_sin_table[rank],
            context_slots[rank], context_rope_cos[rank], context_rope_sin[rank],
            query_rope_cos[rank], query_rope_sin[rank], device=rank,
        )


if __name__ == "__main__":
    import argparse

    from golden import TensorSpec, run

    parser = argparse.ArgumentParser(description="V4.1 DSpark bootstrap context-boundary validation")
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    slots = torch.arange(BATCH, dtype=torch.int32)
    sampled_rows = slots.clone()
    prompts = torch.tensor([1, 122, 123, 125, 127, 128, 0, 125, 125, 125, 125, 125], dtype=torch.int32)
    meta = torch.zeros(BATCH, STATE_META_WIDTH, dtype=torch.int32)
    meta[:, STATE_GENERATION] = 7
    meta[7, STATE_GENERATION] = 6
    meta[8, STATE_VALID] = 1
    slots[9] = -1
    sampled_rows[10] = -1
    sampled_rows[11] = PREFILL_SAMPLE_ROWS
    sampled_ids = torch.zeros(PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE, dtype=torch.int32)
    sampled_ids[:, 0] = torch.arange(100, 100 + PREFILL_SAMPLE_ROWS)
    specs = [
        TensorSpec("slot_ids", [BATCH], torch.int32, init_value=slots),
        TensorSpec("generations", [BATCH], torch.int32, init_value=7),
        TensorSpec("prompt_lengths", [BATCH], torch.int32, init_value=prompts),
        TensorSpec("position_limits", [BATCH], torch.int32, init_value=128),
        TensorSpec("sampled_rows", [BATCH], torch.int32, init_value=sampled_rows),
        TensorSpec("sampled_ids", [PREFILL_SAMPLE_ROWS, SAMPLED_STRIDE], torch.int32, init_value=sampled_ids),
        TensorSpec("state_meta", [BATCH, STATE_META_WIDTH], torch.int32, init_value=meta),
        TensorSpec("next_prefill_tokens", [BATCH], torch.int64),
        TensorSpec("first_query_positions", [BATCH], torch.int32),
    ]
    result = run(
        fn=bootstrap_inputs_rank, specs=specs,
        golden_fn=lambda tensors: golden_bootstrap_inputs(tensors, {}), rtol=0.0, atol=0.0,
        config={"backend_type": "a5", "device_id": args.device, "execution_mode": args.platform == "a5sim"},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
