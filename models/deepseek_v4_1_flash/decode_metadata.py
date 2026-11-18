# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare, publish, accept, and commit V4.1 DSpark decode state."""

import pypto.language as pl
import torch

from config import BLOCK_SIZE, C2A_STORAGE_ROWS, DECODE_BATCH, DECODE_SEQ
from config import DSPARK_QUERY_WIDTH, DSPARK_DRAFT_LAYERS, FLASH, TABLE_DYN
from lm_head import SAMPLED_IDS_PAD
from input_pack import VOCAB_DYN, pack_x_hc


# Dynamic shape variables.
ROPE_ROWS_DYN = pl.dynamic("DECODE_DSPARK_ROPE_ROWS_DYN")
WINDOW_TABLE_DYN = pl.dynamic("DSPARK_WINDOW_TABLE_DYN")
C2A_TABLE_DYN = pl.dynamic("DECODE_C2A_TABLE_DYN")
C1A_TABLE_DYN = pl.dynamic("DECODE_C1A_TABLE_DYN")

# model config
ROWS = DECODE_SEQ
STATE_VALID = 0
STATE_GENERATION = 1
STATE_ANCHOR = 2
STATE_COMMITTED = 3
STATE_DRAFT_COUNT = 4
STATE_LIMIT = 5
STATE_META_WIDTH = 6
STATE_TOKEN_WIDTH = ROWS
ROPE_HALF = FLASH.qk_rope_head_dim // 2
TARGET_ROWS = DECODE_BATCH * ROWS
D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
WINDOW = FLASH.sliding_window
MAIN_IN = DSPARK_DRAFT_LAYERS * D
QUERY_ROWS = DECODE_BATCH * DSPARK_QUERY_WIDTH

# tiling
PADDED_ROWS = (TARGET_ROWS + 15) // 16 * 16


@pl.jit.inline(auto_scope=False)
def decode_prepare(
    slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    state_tokens: pl.Tensor[[DECODE_BATCH, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[DECODE_BATCH, STATE_META_WIDTH], pl.INT32],
    active_widths: pl.Tensor[[DECODE_BATCH], pl.INT32],
    embedding_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    window_table: pl.Tensor[[DECODE_BATCH, WINDOW_TABLE_DYN], pl.INT32],
    c2a_table: pl.Tensor[[DECODE_BATCH, C2A_TABLE_DYN], pl.INT32],
    c1a_table: pl.Tensor[[DECODE_BATCH, C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Tensor[[PADDED_ROWS, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[PADDED_ROWS, HC_MULT], pl.FP32],
    position_ids: pl.Tensor[[PADDED_ROWS], pl.INT32],
    request_ids: pl.Tensor[[PADDED_ROWS], pl.INT32],
    window_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    window_indices: pl.Tensor[[PADDED_ROWS, WINDOW], pl.INT32],
    c2a_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    c1a_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    query_starts: pl.Tensor[[DECODE_BATCH + 1], pl.INT32],
    logit_row_indices: pl.Tensor[[PADDED_ROWS], pl.INT32],
):
    embedding_weight.bind_dynamic(0, VOCAB_DYN)
    window_table.bind_dynamic(1, WINDOW_TABLE_DYN)
    c2a_table.bind_dynamic(1, C2A_TABLE_DYN)
    c1a_table.bind_dynamic(1, C1A_TABLE_DYN)
    padded_ids = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    table_pages = pl.tensor.dim(window_table, 1)
    c2a_pages = pl.tensor.dim(c2a_table, 1)
    c1a_pages = pl.tensor.dim(c1a_table, 1)
    for _core in pl.spmd(1, name_hint="decode_prepare"):
        for request in pl.range(DECODE_BATCH + 1):
            pl.write(query_starts, [request], pl.cast(request * ROWS, pl.INT32))
        for row in pl.range(PADDED_ROWS):
            request = row // ROWS
            offset = row % ROWS
            active = row < TARGET_ROWS
            token = pl.cast(0, pl.INT64)
            position = pl.cast(-1, pl.INT32)
            mapped_request = pl.cast(-1, pl.INT32)
            if active:
                width = pl.cast(0, pl.INT32)
                anchor = pl.cast(0, pl.INT32)
                slot = pl.read(slot_ids, [request])
                if slot >= 0 and slot < DECODE_BATCH:
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
                if offset == 0:
                    pl.write(active_widths, [request], width)
                if offset < width:
                    token = pl.read(state_tokens, [slot, offset])
                    position = pl.cast(anchor + offset, pl.INT32)
                    mapped_request = pl.cast(request, pl.INT32)
            logit_row = pl.cast(-1, pl.INT32)
            if mapped_request >= 0:
                logit_row = pl.cast(row, pl.INT32)
            pl.write(logit_row_indices, [row], logit_row)
            pl.write(padded_ids, [row], token)
            pl.write(position_ids, [row], position)
            pl.write(request_ids, [row], mapped_request)
            window_slot = pl.cast(-1, pl.INT64)
            c2a_slot = pl.cast(-1, pl.INT64)
            c1a_slot = pl.cast(-1, pl.INT64)
            if mapped_request >= 0 and position >= 0:
                page = position // BLOCK_SIZE
                if table_pages > 0:
                    physical = pl.read(window_table, [request, page % table_pages])
                    if physical >= 0:
                        window_slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                if page < c1a_pages:
                    physical = pl.read(c1a_table, [request, page])
                    if physical >= 0:
                        c1a_slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                compressed = position // 2
                if (position + 1) % 2 == 0 and compressed // C2A_STORAGE_ROWS < c2a_pages:
                    physical = pl.read(c2a_table, [request, compressed // C2A_STORAGE_ROWS])
                    if physical >= 0:
                        c2a_slot = pl.cast(physical * C2A_STORAGE_ROWS + compressed % C2A_STORAGE_ROWS, pl.INT64)
            pl.write(window_slots, [row], window_slot)
            pl.write(c2a_slots, [row], c2a_slot)
            pl.write(c1a_slots, [row], c1a_slot)
            for column in pl.range(WINDOW):
                visible = pl.cast(-1, pl.INT32)
                visible_length = pl.min(position + 1, WINDOW)
                past = position - visible_length + 1 + column
                if mapped_request >= 0 and column < visible_length:
                    past_page = past // BLOCK_SIZE
                    if table_pages > 0:
                        physical = pl.read(window_table, [request, past_page % table_pages])
                        if physical >= 0:
                            visible = pl.cast(physical * BLOCK_SIZE + past % BLOCK_SIZE, pl.INT32)
                pl.write(window_indices, [row, column], visible)
        for row in pl.range(PADDED_ROWS):
            for lane in pl.range(HC_MULT):
                value = pl.cast(0.0, pl.FP32)
                if lane == 0:
                    value = pl.cast(1.0, pl.FP32)
                pl.write(pre_mix, [row, lane], value)
    pack_x_hc(padded_ids, embedding_weight, x_hc, hidden_tile=512)
    return x_hc, pre_mix, active_widths


@pl.jit.inline(auto_scope=False)
def decode_accept(
    slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    active_widths: pl.Tensor[[DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[DECODE_BATCH * ROWS, SAMPLED_IDS_PAD], pl.INT32],
    state_tokens: pl.Tensor[[DECODE_BATCH, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[DECODE_BATCH, STATE_META_WIDTH], pl.INT32],
    accepted_ids: pl.Tensor[[DECODE_BATCH, ROWS], pl.INT32],
    accepted_counts: pl.Tensor[[DECODE_BATCH], pl.INT32],
    num_sampled: pl.Tensor[[DECODE_BATCH], pl.INT32],
    last_sampled: pl.Tensor[[DECODE_BATCH], pl.INT64],
    anchor_ids: pl.Tensor[[DECODE_BATCH], pl.INT64],
    first_query_positions: pl.Tensor[[DECODE_BATCH], pl.INT32],
    target_hidden_padded: pl.Tensor[[PADDED_ROWS, MAIN_IN], pl.BF16],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, DECODE_BATCH, TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    context_slots: pl.Tensor[[DSPARK_DRAFT_LAYERS, TARGET_ROWS], pl.INT64],
    context_rope_cos: pl.Tensor[[TARGET_ROWS, ROPE_HALF], pl.FP32],
    context_rope_sin: pl.Tensor[[TARGET_ROWS, ROPE_HALF], pl.FP32],
    query_rope_cos: pl.Tensor[[QUERY_ROWS, ROPE_HALF], pl.FP32],
    query_rope_sin: pl.Tensor[[QUERY_ROWS, ROPE_HALF], pl.FP32],
    markov_row_indices: pl.Tensor,
):
    block_tables.bind_dynamic(2, TABLE_DYN)
    rope_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    rope_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    table_blocks = pl.tensor.dim(block_tables, 2)
    rope_rows = pl.tensor.dim(rope_cos_table, 0)
    markov_capacity = pl.tensor.dim(markov_row_indices, 0)
    for _core in pl.spmd(1, name_hint="decode_accept"):
        for row in pl.range(markov_capacity):
            source = pl.cast(-1, pl.INT32)
            if row < QUERY_ROWS:
                source = pl.cast(row, pl.INT32)
            pl.write(markov_row_indices, [row], source)
        for request in pl.range(DECODE_BATCH):
            accepted = pl.cast(0, pl.INT32)
            anchor = pl.cast(0, pl.INT32)
            can_draft = pl.cast(0, pl.INT32)
            next_token = pl.cast(0, pl.INT64)
            slot = pl.read(slot_ids, [request])
            width = pl.read(active_widths, [request])
            if slot >= 0 and slot < DECODE_BATCH and width > 0 and width <= ROWS:
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
            pl.write(anchor_ids, [request], next_token)
            first = pl.cast((anchor + accepted) * can_draft, pl.INT32)
            pl.write(first_query_positions, [request], first)
            for offset in pl.range(ROWS):
                token = pl.cast(-1, pl.INT32)
                valid = pl.cast(0, pl.INT32)
                position = pl.cast(0, pl.INT32)
                if offset < accepted:
                    token = pl.read(sampled_ids, [request * ROWS + offset, 0])
                    position = pl.cast(anchor + offset, pl.INT32)
                    valid = can_draft
                pl.write(accepted_ids, [request, offset], token)
                row = request * ROWS + offset
                active = valid == 1 and position >= 0 and position < rope_rows
                for col in pl.range(ROPE_HALF):
                    cos = pl.cast(0.0, pl.FP32)
                    sin = pl.cast(0.0, pl.FP32)
                    if active:
                        cos = pl.read(rope_cos_table, [position, col])
                        sin = pl.read(rope_sin_table, [position, col])
                    pl.write(context_rope_cos, [row, col], cos)
                    pl.write(context_rope_sin, [row, col], sin)
                for layer in pl.range(DSPARK_DRAFT_LAYERS):
                    context_slot = pl.cast(-1, pl.INT64)
                    if active and table_blocks > 0:
                        page = position // BLOCK_SIZE % table_blocks
                        physical = pl.read(block_tables, [layer, request, page])
                        if physical >= 0:
                            context_slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                    pl.write(context_slots, [layer, row], context_slot)
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
    return target_hidden_padded[:TARGET_ROWS, :]


@pl.jit.inline(auto_scope=False)
def decode_publish(
    local_x_hc: pl.Tensor[[PADDED_ROWS, HC_MULT, D], pl.FP32],
    local_pre_mix: pl.Tensor[[PADDED_ROWS, HC_MULT], pl.FP32],
    local_target_hidden: pl.Tensor[[PADDED_ROWS, MAIN_IN], pl.BF16],
    local_position_ids: pl.Tensor[[PADDED_ROWS], pl.INT32],
    local_request_ids: pl.Tensor[[PADDED_ROWS], pl.INT32],
    local_window_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    local_window_indices: pl.Tensor[[PADDED_ROWS, WINDOW], pl.INT32],
    local_c2a_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    local_c1a_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    local_query_starts: pl.Tensor[[DECODE_BATCH + 1], pl.INT32],
    x_hc: pl.Tensor[[PADDED_ROWS, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[PADDED_ROWS, HC_MULT], pl.FP32],
    target_hidden: pl.Tensor[[PADDED_ROWS, MAIN_IN], pl.BF16],
    position_ids: pl.Tensor[[PADDED_ROWS], pl.INT32],
    request_ids: pl.Tensor[[PADDED_ROWS], pl.INT32],
    window_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    window_indices: pl.Tensor[[PADDED_ROWS, WINDOW], pl.INT32],
    c2a_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    c1a_slots: pl.Tensor[[PADDED_ROWS], pl.INT64],
    query_starts: pl.Tensor[[DECODE_BATCH + 1], pl.INT32],
):
    """Publish target backbone hidden states and decode metadata."""
    published_x = pl.reshape(x_hc, [PADDED_ROWS, HC_MULT * D])
    source_x = pl.reshape(local_x_hc, [PADDED_ROWS, HC_MULT * D])
    for row in pl.spmd(PADDED_ROWS, name_hint="decode_publish"):
        hidden_tile = pl.load(source_x, [row, 0], [1, HC_MULT * D])
        pl.store(hidden_tile, [row, 0], published_x)
        target_tile = pl.load(local_target_hidden, [row, 0], [1, MAIN_IN])
        pl.store(target_tile, [row, 0], target_hidden)
        indices_tile = pl.load(local_window_indices, [row, 0], [1, WINDOW])
        pl.store(indices_tile, [row, 0], window_indices)
        if row == 0:
            for metadata_row in pl.range(PADDED_ROWS):
                for stream in pl.range(HC_MULT):
                    value = pl.read(local_pre_mix, [metadata_row, stream])
                    pl.write(pre_mix, [metadata_row, stream], value)
                pl.write(position_ids, [metadata_row], pl.read(local_position_ids, [metadata_row]))
                pl.write(request_ids, [metadata_row], pl.read(local_request_ids, [metadata_row]))
                pl.write(window_slots, [metadata_row], pl.read(local_window_slots, [metadata_row]))
                pl.write(c2a_slots, [metadata_row], pl.read(local_c2a_slots, [metadata_row]))
                pl.write(c1a_slots, [metadata_row], pl.read(local_c1a_slots, [metadata_row]))
            for request in pl.range(DECODE_BATCH + 1):
                pl.write(query_starts, [request], pl.read(local_query_starts, [request]))


@pl.jit.inline(auto_scope=False)
def commit_drafts(
    slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    draft_ids: pl.Tensor[[DECODE_BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[DECODE_BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[DECODE_BATCH, STATE_META_WIDTH], pl.INT32]],
    committed: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_commit_drafts"):
        for request in pl.range(DECODE_BATCH):
            published = pl.cast(0, pl.INT32)
            slot = pl.read(slot_ids, [request])
            if slot >= 0 and slot < DECODE_BATCH:
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
