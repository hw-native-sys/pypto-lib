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
PREPARE_STARTS = DECODE_BATCH + 1
REQUEST_TILE = (DECODE_BATCH + 7) // 8 * 8
ACCEPT_ROW_WORK = TARGET_ROWS + QUERY_ROWS


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
    """Build target-row metadata with 64 AIV workers, then pack embeddings."""
    embedding_weight.bind_dynamic(0, VOCAB_DYN)
    window_table.bind_dynamic(1, WINDOW_TABLE_DYN)
    c2a_table.bind_dynamic(1, C2A_TABLE_DYN)
    c1a_table.bind_dynamic(1, C1A_TABLE_DYN)
    padded_ids = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    table_pages = pl.tensor.dim(window_table, 1)
    c2a_pages = pl.tensor.dim(c2a_table, 1)
    c1a_pages = pl.tensor.dim(c1a_table, 1)
    flat_query_starts = pl.reshape(query_starts, [1, PREPARE_STARTS])
    flat_active_widths = pl.reshape(active_widths, [1, DECODE_BATCH])
    flat_logit_row_indices = pl.reshape(logit_row_indices, [1, PADDED_ROWS])
    flat_padded_ids = pl.reshape(padded_ids, [1, PADDED_ROWS])
    flat_position_ids = pl.reshape(position_ids, [1, PADDED_ROWS])
    flat_request_ids = pl.reshape(request_ids, [1, PADDED_ROWS])
    flat_window_slots = pl.reshape(window_slots, [1, PADDED_ROWS])
    flat_c2a_slots = pl.reshape(c2a_slots, [1, PADDED_ROWS])
    flat_c1a_slots = pl.reshape(c1a_slots, [1, PADDED_ROWS])
    for _core in pl.spmd(64, name_hint='decode_prepare'):
        for request in pl.range(_core, DECODE_BATCH + 1, 64):
            element_query_starts = pl.tile.create([1, 8], pl.INT32)
            pl.tile.write(element_query_starts, [0, 0], pl.cast(request * ROWS, pl.INT32))
            pl.store(pl.set_validshape(element_query_starts, 1, 1), [0, request], flat_query_starts)
        for row in pl.range(_core, PADDED_ROWS, 64):
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
                    if valid == 1 and generation == expected and (anchor < limit):
                        count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                        count = pl.max(pl.min(count, DSPARK_QUERY_WIDTH), 0)
                        width = pl.cast(1, pl.INT32)
                        if anchor + count < limit:
                            width = pl.cast(count + 1, pl.INT32)
                if offset == 0:
                    element_active_widths = pl.tile.create([1, 8], pl.INT32)
                    pl.tile.write(element_active_widths, [0, 0], width)
                    pl.store(pl.set_validshape(element_active_widths, 1, 1), [0, request], flat_active_widths)
                if offset < width:
                    token = pl.read(state_tokens, [slot, offset])
                    position = pl.cast(anchor + offset, pl.INT32)
                    mapped_request = pl.cast(request, pl.INT32)
            logit_row = pl.cast(-1, pl.INT32)
            if mapped_request >= 0:
                logit_row = pl.cast(row, pl.INT32)
            element_logit_row_indices = pl.tile.create([1, 8], pl.INT32)
            pl.tile.write(element_logit_row_indices, [0, 0], logit_row)
            pl.store(pl.set_validshape(element_logit_row_indices, 1, 1), [0, row], flat_logit_row_indices)
            element_padded_ids = pl.tile.create([1, 4], pl.INT64)
            pl.tile.write(element_padded_ids, [0, 0], token)
            pl.store(pl.set_validshape(element_padded_ids, 1, 1), [0, row], flat_padded_ids)
            element_position_ids = pl.tile.create([1, 8], pl.INT32)
            pl.tile.write(element_position_ids, [0, 0], position)
            pl.store(pl.set_validshape(element_position_ids, 1, 1), [0, row], flat_position_ids)
            element_request_ids = pl.tile.create([1, 8], pl.INT32)
            pl.tile.write(element_request_ids, [0, 0], mapped_request)
            pl.store(pl.set_validshape(element_request_ids, 1, 1), [0, row], flat_request_ids)
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
            element_window_slots = pl.tile.create([1, 4], pl.INT64)
            pl.tile.write(element_window_slots, [0, 0], window_slot)
            pl.store(pl.set_validshape(element_window_slots, 1, 1), [0, row], flat_window_slots)
            element_c2a_slots = pl.tile.create([1, 4], pl.INT64)
            pl.tile.write(element_c2a_slots, [0, 0], c2a_slot)
            pl.store(pl.set_validshape(element_c2a_slots, 1, 1), [0, row], flat_c2a_slots)
            element_c1a_slots = pl.tile.create([1, 4], pl.INT64)
            pl.tile.write(element_c1a_slots, [0, 0], c1a_slot)
            pl.store(pl.set_validshape(element_c1a_slots, 1, 1), [0, row], flat_c1a_slots)
            indices = pl.full([1, WINDOW], dtype=pl.INT32, value=-1)
            if mapped_request >= 0 and position >= 0:
                columns = pl.arange(0, [1, WINDOW], dtype=pl.INT32)
                visible_length = pl.min(position + 1, WINDOW)
                prefix = position - visible_length + 1
                positions = pl.add(columns, pl.cast(prefix, pl.INT32))
                for page in pl.range(prefix // BLOCK_SIZE, position // BLOCK_SIZE + 1):
                    if table_pages > 0:
                        physical = pl.read(window_table, [request, page % table_pages])
                        if physical >= 0:
                            lower = pl.cmp(positions, pl.cast(page * BLOCK_SIZE, pl.INT32), cmp_type=5)
                            upper = pl.cmp(positions, pl.cast((page + 1) * BLOCK_SIZE, pl.INT32), cmp_type=2)
                            page_mask = pl.mul(lower, upper)
                            visible = pl.cmp(columns, pl.cast(visible_length, pl.INT32), cmp_type=2)
                            mask = pl.mul(page_mask, visible)
                            slots = pl.add(positions, pl.cast((physical - page) * BLOCK_SIZE, pl.INT32))
                            delta = pl.sub(slots, indices)
                            indices = pl.add(indices, pl.mul(mask, delta))
            window_indices[row:row + 1, :] = indices
        for row in pl.range(_core, PADDED_ROWS, 64):
            for lane in pl.range(HC_MULT):
                value = pl.cast(0.0, pl.FP32)
                if lane == 0:
                    value = pl.cast(1.0, pl.FP32)
                element_pre_mix = pl.tile.create([1, 8], pl.FP32)
                pl.tile.write(element_pre_mix, [0, 0], value)
                pl.store(pl.set_validshape(element_pre_mix, 1, 1), [row, lane], pre_mix)
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
    """Accept proposals by state-slot owner, then publish context and query rows in parallel."""
    block_tables.bind_dynamic(2, TABLE_DYN)
    rope_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    rope_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    flat_slot_ids = pl.reshape(slot_ids, [1, DECODE_BATCH])
    flat_generations = pl.reshape(generations, [1, DECODE_BATCH])
    flat_widths = pl.reshape(active_widths, [1, DECODE_BATCH])
    original_anchors = pl.create_tensor([1, DECODE_BATCH], dtype=pl.INT32)
    table_blocks = pl.tensor.dim(block_tables, 2)
    rope_rows = pl.tensor.dim(rope_cos_table, 0)
    markov_capacity = pl.tensor.dim(markov_row_indices, 0)
    flat_markov_row_indices = pl.reshape(markov_row_indices, [1, markov_capacity])
    flat_accepted_counts = pl.reshape(accepted_counts, [1, DECODE_BATCH])
    flat_num_sampled = pl.reshape(num_sampled, [1, DECODE_BATCH])
    flat_last_sampled = pl.reshape(last_sampled, [1, DECODE_BATCH])
    flat_anchor_ids = pl.reshape(anchor_ids, [1, DECODE_BATCH])
    flat_first_query_positions = pl.reshape(first_query_positions, [1, DECODE_BATCH])
    for _core in pl.spmd(DECODE_BATCH, name_hint='accept_state'):
        cached_slots = pl.load(flat_slot_ids, [0, 0], [1, REQUEST_TILE], [1, DECODE_BATCH])
        cached_generations = pl.load(flat_generations, [0, 0], [1, REQUEST_TILE], [1, DECODE_BATCH])
        cached_widths = pl.load(flat_widths, [0, 0], [1, REQUEST_TILE], [1, DECODE_BATCH])
        if _core == 0:
            for start in pl.range(0, markov_capacity, 64):
                columns = pl.arange(pl.cast(start, pl.INT32), [1, 64], dtype=pl.INT32)
                mask = pl.cmp(columns, pl.cast(QUERY_ROWS, pl.INT32), cmp_type=2)
                incremented = pl.add(columns, pl.cast(1, pl.INT32))
                active_indices = pl.mul(mask, incremented)
                ids = pl.sub(active_indices, pl.cast(1, pl.INT32))
                markov_valid = pl.min(64, markov_capacity - start)
                flat_markov_row_indices[:, start:start + markov_valid] = ids[:, :markov_valid]
        for request in pl.range(DECODE_BATCH):
            owner = pl.cast(request, pl.INT32)
            owned_slot = pl.cast(0, pl.INT32)
            owned_slot = pl.read(cached_slots, [0, request])
            if owned_slot >= 0 and owned_slot < DECODE_BATCH:
                owner = owned_slot
            if owner % DECODE_BATCH == _core:
                accepted = pl.cast(0, pl.INT32)
                anchor = pl.cast(0, pl.INT32)
                can_draft = pl.cast(0, pl.INT32)
                next_token = pl.cast(0, pl.INT64)
                slot = pl.cast(0, pl.INT32)
                slot = owned_slot
                width = pl.cast(0, pl.INT32)
                width = pl.read(cached_widths, [0, request])
                if slot >= 0 and slot < DECODE_BATCH and (width > 0) and (width <= ROWS):
                    meta_row = pl.load(state_meta, [slot, 0], [1, 8], [1, STATE_META_WIDTH])
                    token_row = pl.load(state_tokens, [slot, 0], [1, 8], [1, STATE_TOKEN_WIDTH])
                    valid = pl.read(meta_row, [0, STATE_VALID])
                    generation = pl.read(meta_row, [0, STATE_GENERATION])
                    expected = pl.cast(0, pl.INT32)
                    expected = pl.read(cached_generations, [0, request])
                    if valid == 1 and generation == expected:
                        anchor = pl.read(meta_row, [0, STATE_ANCHOR])
                        limit = pl.read(meta_row, [0, STATE_LIMIT])
                        count = pl.read(meta_row, [0, STATE_DRAFT_COUNT])
                        count = pl.max(pl.min(count, width - 1), 0)
                        matched = pl.cast(0, pl.INT32)
                        still_matching = pl.cast(1, pl.INT32)
                        for offset in pl.range(DSPARK_QUERY_WIDTH):
                            if offset < count and still_matching == 1:
                                draft = pl.read(token_row, [0, offset + 1])
                                predicted = pl.read(sampled_ids, [request * ROWS + offset, 0])
                                if draft == pl.cast(predicted, pl.INT64):
                                    matched = pl.cast(offset + 1, pl.INT32)
                                else:
                                    still_matching = pl.cast(0, pl.INT32)
                        accepted = pl.cast(matched + 1, pl.INT32)
                        next_token = pl.cast(pl.read(sampled_ids, [request * ROWS + matched, 0]), pl.INT64)
                        element_state_tokens = pl.tile.create([1, 4], pl.INT64)
                        pl.tile.write(element_state_tokens, [0, 0], next_token)
                        pl.store(pl.set_validshape(element_state_tokens, 1, 1), [slot, 0], state_tokens)
                        element_state_meta = pl.tile.create([1, 8], pl.INT32)
                        pl.tile.write(element_state_meta, [0, 0], pl.cast(anchor + accepted, pl.INT32))
                        pl.store(pl.set_validshape(element_state_meta, 1, 1), [slot, STATE_ANCHOR], state_meta)
                        committed = pl.read(meta_row, [0, STATE_COMMITTED])
                        element_state_meta = pl.tile.create([1, 8], pl.INT32)
                        pl.tile.write(element_state_meta, [0, 0], pl.cast(committed + accepted, pl.INT32))
                        pl.store(pl.set_validshape(element_state_meta, 1, 1), [slot, STATE_COMMITTED], state_meta)
                        element_state_meta = pl.tile.create([1, 8], pl.INT32)
                        pl.tile.write(element_state_meta, [0, 0], pl.cast(0, pl.INT32))
                        pl.store(pl.set_validshape(element_state_meta, 1, 1), [slot, STATE_DRAFT_COUNT], state_meta)
                        if anchor + accepted + DSPARK_QUERY_WIDTH < limit:
                            can_draft = pl.cast(1, pl.INT32)
                element_accepted_counts = pl.tile.create([1, 8], pl.INT32)
                pl.tile.write(element_accepted_counts, [0, 0], accepted)
                pl.store(pl.set_validshape(element_accepted_counts, 1, 1), [0, request], flat_accepted_counts)
                element_num_sampled = pl.tile.create([1, 8], pl.INT32)
                pl.tile.write(element_num_sampled, [0, 0], pl.cast(accepted * can_draft, pl.INT32))
                pl.store(pl.set_validshape(element_num_sampled, 1, 1), [0, request], flat_num_sampled)
                element_last_sampled = pl.tile.create([1, 4], pl.INT64)
                pl.tile.write(element_last_sampled, [0, 0], next_token)
                pl.store(pl.set_validshape(element_last_sampled, 1, 1), [0, request], flat_last_sampled)
                element_anchor_ids = pl.tile.create([1, 4], pl.INT64)
                pl.tile.write(element_anchor_ids, [0, 0], next_token)
                pl.store(pl.set_validshape(element_anchor_ids, 1, 1), [0, request], flat_anchor_ids)
                first = pl.cast((anchor + accepted) * can_draft, pl.INT32)
                element_first_query_positions = pl.tile.create([1, 8], pl.INT32)
                pl.tile.write(element_first_query_positions, [0, 0], first)
                pl.store(pl.set_validshape(element_first_query_positions, 1, 1), [0, request], flat_first_query_positions)
                anchor_tile = pl.tile.create([1, 8], pl.INT32)
                pl.tile.write(anchor_tile, [0, 0], anchor)
                pl.store(pl.set_validshape(anchor_tile, 1, 1), [0, request], original_anchors)
    for core in pl.spmd(64, name_hint='accept_rows'):
        counts_tile = pl.load(flat_accepted_counts, [0, 0], [1, REQUEST_TILE], [1, DECODE_BATCH])
        sampled_tile = pl.load(flat_num_sampled, [0, 0], [1, REQUEST_TILE], [1, DECODE_BATCH])
        anchors_tile = pl.load(original_anchors, [0, 0], [1, REQUEST_TILE], [1, DECODE_BATCH])
        first_tile = pl.load(flat_first_query_positions, [0, 0], [1, REQUEST_TILE], [1, DECODE_BATCH])
        for work in pl.range(core, ACCEPT_ROW_WORK, 64):
            if work < TARGET_ROWS:
                request = work // ROWS
                offset = work % ROWS
                accepted = pl.read(counts_tile, [0, request])
                sampled = pl.read(sampled_tile, [0, request])
                anchor = pl.read(anchors_tile, [0, request])
                token = pl.cast(-1, pl.INT32)
                position = pl.cast(0, pl.INT32)
                if offset < accepted:
                    token = pl.read(sampled_ids, [work, 0])
                    position = pl.cast(anchor + offset, pl.INT32)
                ids_tile = pl.tile.create([1, 8], pl.INT32)
                pl.tile.write(ids_tile, [0, 0], token)
                pl.store(pl.set_validshape(ids_tile, 1, 1), [request, offset], accepted_ids)
                active = offset < accepted and sampled > 0 and (position >= 0) and (position < rope_rows)
                cos = pl.full([1, ROPE_HALF], dtype=pl.FP32, value=0.0)
                sin = pl.full([1, ROPE_HALF], dtype=pl.FP32, value=0.0)
                if active:
                    cos = rope_cos_table[position:position + 1, :]
                    sin = rope_sin_table[position:position + 1, :]
                context_rope_cos[work:work + 1, :] = cos
                context_rope_sin[work:work + 1, :] = sin
                for layer in pl.range(DSPARK_DRAFT_LAYERS):
                    context_slot = pl.cast(-1, pl.INT64)
                    if active and table_blocks > 0:
                        page = position // BLOCK_SIZE % table_blocks
                        physical = pl.read(block_tables, [layer, request, page])
                        if physical >= 0:
                            context_slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                    slot_tile = pl.tile.create([1, 4], pl.INT64)
                    pl.tile.write(slot_tile, [0, 0], context_slot)
                    pl.store(pl.set_validshape(slot_tile, 1, 1), [layer, work], context_slots)
            else:
                row = work - TARGET_ROWS
                request = row // DSPARK_QUERY_WIDTH
                first = pl.read(first_tile, [0, request])
                position = pl.cast(first + row % DSPARK_QUERY_WIDTH, pl.INT32)
                active = first > 0 and position < rope_rows
                cos = pl.full([1, ROPE_HALF], dtype=pl.FP32, value=0.0)
                sin = pl.full([1, ROPE_HALF], dtype=pl.FP32, value=0.0)
                if active:
                    cos = rope_cos_table[position:position + 1, :]
                    sin = rope_sin_table[position:position + 1, :]
                query_rope_cos[row:row + 1, :] = cos
                query_rope_sin[row:row + 1, :] = sin
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
