# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Persistent device state for the DeepSeek-V4 DSpark speculative loop.

The state slot is stable for a request's lifetime.  Serving descriptors map a
transient dense decode row to that slot and carry an allocation generation, so
an early-prepared command cannot consume a slot that has since been recycled.
"""

import pypto.language as pl
import pypto.language.distributed as pld

from config import DECODE_BATCH, DECODE_SEQ, EP as N_RANKS, FLASH as M, TP


LOCAL_BATCH = DECODE_BATCH // TP
S = DECODE_SEQ
T = LOCAL_BATCH * S
# One replica per TP rank, indexed by the stable group-local drafter lease.
# Replication lets a request move between dense owner rows after batch
# compaction; the fused wrapper publishes each updated row to its three peers.
STATE_CAPACITY = DECODE_BATCH
MAIN_HIDDEN_DIM = 3 * M.hidden_size
DSPARK_QUERY_WIDTH = S - 1
MAX_POSITION = M.max_position_embeddings
COMMIT_B_DYN = pl.dynamic("DSPARK_STATE_COMMIT_B_DYN")

STATE_VALID = 0
STATE_GENERATION = 1
STATE_ANCHOR_POSITION = 2
STATE_COMMITTED_COUNT = 3
STATE_DRAFT_COUNT = 4
STATE_META_WIDTH = 5

STATE_CURRENT_TOKEN = 0
STATE_FIRST_DRAFT = 1
STATE_TOKEN_WIDTH = 1 + DSPARK_QUERY_WIDTH

assert S == STATE_TOKEN_WIDTH
assert S == DSPARK_QUERY_WIDTH + 1


@pl.jit.inline(auto_scope=False)
def prepare_target_from_device_state(
    state_slot_ids: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_tokens: pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.InOut[pl.Tensor[[T], pl.INT64]],
    position_ids: pl.InOut[pl.Tensor[[T], pl.INT32]],
    csa_kv_seq_lens: pl.InOut[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
    hca_kv_seq_lens: pl.InOut[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
):
    """Late-bind target tokens and positions from persistent request slots."""
    for core in pl.spmd(1, name_hint="dspark_state_prepare"):
        for request in pl.range(core, LOCAL_BATCH):
            pl.write(active_widths, [request], pl.cast(0, pl.INT32))
            slot_raw = pl.read(state_slot_ids, [request])
            slot = pl.cast(
                pl.max(pl.min(slot_raw, STATE_CAPACITY - 1), 0),
                pl.INDEX,
            )
            if slot_raw >= 0 and slot_raw < STATE_CAPACITY:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    anchor = pl.read(state_meta, [slot, STATE_ANCHOR_POSITION])
                    draft_count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                    row = request * S
                    for offset in pl.range(S):
                        token = pl.read(state_tokens, [slot, offset])
                        pl.write(input_ids, [row + offset], token)
                        pl.write(
                            position_ids,
                            [row + offset],
                            pl.cast(anchor + offset, pl.INT32),
                        )
                    active_width = pl.cast(draft_count + 1, pl.INT32)
                    pl.write(
                        csa_kv_seq_lens,
                        [request],
                        pl.cast(anchor + active_width, pl.INT32),
                    )
                    pl.write(
                        hca_kv_seq_lens,
                        [request],
                        pl.cast(anchor + active_width, pl.INT32),
                    )
                    pl.write(
                        active_widths,
                        [request],
                        active_width,
                    )
    return input_ids, position_ids, csa_kv_seq_lens, hca_kv_seq_lens, active_widths


@pl.jit.inline(auto_scope=False)
def accept_target_into_device_state(
    state_slot_ids: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    sampled_row_offsets: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    hidden_row_offsets: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32]],
    sampled_ids: pl.Tensor[[T, S], pl.INT32],
    target_hidden: pl.Tensor[[T, MAIN_HIDDEN_DIM], pl.BF16],
    accepted_token_ids: pl.Out[pl.Tensor[[LOCAL_BATCH, S], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
    drafter_target_hidden: pl.Out[pl.Tensor[[T, MAIN_HIDDEN_DIM], pl.BF16]],
    drafter_context_positions: pl.Out[pl.Tensor[[T], pl.INT32]],
    drafter_context_valid: pl.Out[pl.Tensor[[T], pl.INT32]],
    drafter_last_sampled: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT64]],
    drafter_anchor_positions: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
    drafter_row_offsets: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
    drafter_ready: pl.Out[pl.Tensor[[1], pl.INT32]],
):
    """Accept the longest matching prefix and prepare the next drafter inputs."""
    for core in pl.spmd(1, name_hint="dspark_state_accept"):
        next_drafter_row = pl.cast(0, pl.INT32)
        for request in pl.range(core, LOCAL_BATCH):
            pl.write(accepted_counts, [request], pl.cast(0, pl.INT32))
            pl.write(drafter_last_sampled, [request], pl.cast(0, pl.INT64))
            pl.write(drafter_anchor_positions, [request], pl.cast(0, pl.INT32))
            pl.write(drafter_row_offsets, [request], pl.cast(-1, pl.INT32))
            for offset in pl.range(S):
                pl.write(accepted_token_ids, [request, offset], pl.cast(-1, pl.INT32))
                context_row = request * S + offset
                pl.write(drafter_context_positions, [context_row], pl.cast(0, pl.INT32))
                pl.write(drafter_context_valid, [context_row], pl.cast(0, pl.INT32))
            slot_raw = pl.read(state_slot_ids, [request])
            sampled_row_raw = pl.read(sampled_row_offsets, [request])
            slot = pl.cast(
                pl.max(pl.min(slot_raw, STATE_CAPACITY - 1), 0),
                pl.INDEX,
            )
            if slot_raw >= 0 and slot_raw < STATE_CAPACITY and sampled_row_raw >= 0:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    sampled_row = pl.cast(sampled_row_raw, pl.INDEX)
                    draft_count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                    matched = pl.cast(0, pl.INT32)
                    still_matching = pl.cast(1, pl.INT32)
                    for draft_offset in pl.range(DSPARK_QUERY_WIDTH):
                        draft = pl.read(
                            state_tokens,
                            [slot, STATE_FIRST_DRAFT + draft_offset],
                        )
                        predicted = pl.cast(
                            pl.read(sampled_ids, [sampled_row + draft_offset, 0]),
                            pl.INT64,
                        )
                        if (
                            draft_offset < draft_count
                            and still_matching == 1
                            and draft == predicted
                        ):
                            matched = pl.cast(draft_offset + 1, pl.INT32)
                        else:
                            still_matching = pl.cast(0, pl.INT32)
                    accepted = pl.cast(matched + 1, pl.INT32)
                    next_token = pl.cast(
                        pl.read(sampled_ids, [sampled_row + matched, 0]),
                        pl.INT64,
                    )
                    old_anchor = pl.read(state_meta, [slot, STATE_ANCHOR_POSITION])
                    committed = pl.read(state_meta, [slot, STATE_COMMITTED_COUNT])
                    pl.write(accepted_counts, [request], accepted)
                    pl.write(drafter_last_sampled, [request], next_token)
                    pl.write(
                        drafter_anchor_positions,
                        [request],
                        pl.cast(old_anchor + accepted - 1, pl.INT32),
                    )
                    for offset in pl.range(S):
                        if offset < accepted:
                            accepted_token = pl.read(
                                sampled_ids,
                                [sampled_row + offset, 0],
                            )
                            pl.write(
                                accepted_token_ids,
                                [request, offset],
                                accepted_token,
                            )
                            context_row = request * S + offset
                            pl.write(
                                drafter_context_positions,
                                [context_row],
                                pl.cast(old_anchor + offset, pl.INT32),
                            )
                            pl.write(
                                drafter_context_valid,
                                [context_row],
                                pl.cast(1, pl.INT32),
                            )
                    pl.write(state_tokens, [slot, STATE_CURRENT_TOKEN], next_token)
                    pl.write(
                        state_meta,
                        [slot, STATE_ANCHOR_POSITION],
                        pl.cast(old_anchor + accepted, pl.INT32),
                    )
                    pl.write(
                        state_meta,
                        [slot, STATE_COMMITTED_COUNT],
                        pl.cast(committed + accepted, pl.INT32),
                    )
                    # Old drafts are consumed by this verification.  Publish a
                    # compact drafter destination only when another full K=7
                    # query window fits; commit_drafts restores draft_count.
                    pl.write(
                        state_meta,
                        [slot, STATE_DRAFT_COUNT],
                        pl.cast(0, pl.INT32),
                    )
                    next_anchor = pl.cast(old_anchor + accepted - 1, pl.INT32)
                    if next_anchor + DSPARK_QUERY_WIDTH < MAX_POSITION:
                        pl.write(
                            drafter_row_offsets,
                            [request],
                            pl.cast(next_drafter_row * S, pl.INT32),
                        )
                        next_drafter_row = pl.cast(next_drafter_row + 1, pl.INT32)

    with pl.spmd(T, name_hint="dspark_state_pack_hidden") as pack_hidden_tid:
        token = pl.tile.get_block_idx()
        request = token // S
        offset = token % S
        accepted = pl.read(accepted_counts, [request])
        destination_base = pl.read(drafter_row_offsets, [request])
        if destination_base >= 0 and offset < accepted:
            source_base = pl.read(hidden_row_offsets, [request])
            source = pl.cast(source_base + offset, pl.INDEX)
            destination = pl.cast(destination_base + offset, pl.INDEX)
            drafter_target_hidden[destination : destination + 1, 0:MAIN_HIDDEN_DIM] = (
                target_hidden[source : source + 1, 0:MAIN_HIDDEN_DIM]
            )
        elif destination_base >= 0:
            destination = pl.cast(destination_base + offset, pl.INDEX)
            drafter_target_hidden[destination : destination + 1, 0:MAIN_HIDDEN_DIM] = pl.full(
                [1, MAIN_HIDDEN_DIM],
                dtype=pl.BF16,
                value=0.0,
            )
    with pl.spmd(
        1,
        name_hint="dspark_state_publish_drafter_ready",
        deps=[pack_hidden_tid],
    ):
        publish_core = pl.tile.get_block_idx()
        pl.write(drafter_ready, [publish_core], pl.cast(0, pl.INT32))
    return (
        state_tokens,
        state_meta,
        accepted_token_ids,
        accepted_counts,
        drafter_target_hidden,
        drafter_context_positions,
        drafter_context_valid,
        drafter_last_sampled,
        drafter_anchor_positions,
        drafter_row_offsets,
        drafter_ready,
    )


@pl.jit.inline(auto_scope=False)
def commit_drafts_to_device_state(
    state_slot_ids: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32]],
    draft_token_ids: pl.Tensor[[COMMIT_B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
):
    """Commit the Markov output as the next target verification window."""
    draft_token_ids.bind_dynamic(0, COMMIT_B_DYN)
    batch = pl.tensor.dim(draft_token_ids, 0)
    for core in pl.spmd(1, name_hint="dspark_state_commit_drafts"):
        for request in pl.range(core, batch):
            slot_raw = pl.read(state_slot_ids, [request])
            slot = pl.cast(
                pl.max(pl.min(slot_raw, STATE_CAPACITY - 1), 0),
                pl.INDEX,
            )
            if slot_raw >= 0 and slot_raw < STATE_CAPACITY:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    for offset in pl.range(DSPARK_QUERY_WIDTH):
                        token = pl.cast(
                            pl.read(draft_token_ids, [request, offset]),
                            pl.INT64,
                        )
                        pl.write(
                            state_tokens,
                            [slot, STATE_FIRST_DRAFT + offset],
                            token,
                        )
                    pl.write(
                        state_meta,
                        [slot, STATE_DRAFT_COUNT],
                        pl.cast(DSPARK_QUERY_WIDTH, pl.INT32),
                    )
    return state_tokens, state_meta


@pl.jit
def commit_drafts_to_device_state_rank(
    state_slot_ids: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32]],
    draft_token_ids: pl.Tensor[[COMMIT_B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
):
    """Rank-local entry used when the seed commit is dispatched standalone."""
    return commit_drafts_to_device_state(
        state_slot_ids,
        state_generations,
        state_tokens,
        state_meta,
        draft_token_ids,
    )


@pl.jit.host
def l3_prepare_target_from_device_state(
    state_slot_ids: pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32],
    state_tokens: pl.Tensor[
        [N_RANKS, STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64
    ],
    state_meta: pl.Tensor[
        [N_RANKS, STATE_CAPACITY, STATE_META_WIDTH], pl.INT32
    ],
    input_ids: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    position_ids: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT32]],
    csa_kv_seq_lens: pl.InOut[
        pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32]
    ],
    hca_kv_seq_lens: pl.InOut[
        pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32]
    ],
    active_widths: pl.Out[
        pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32]
    ],
):
    for rank in pl.range(pld.world_size()):
        prepare_target_from_device_state(
            state_slot_ids[rank],
            state_generations[rank],
            state_tokens[rank],
            state_meta[rank],
            input_ids[rank],
            position_ids[rank],
            csa_kv_seq_lens[rank],
            hca_kv_seq_lens[rank],
            active_widths[rank],
            device=rank,
        )


@pl.jit.host
def l3_accept_target_into_device_state(
    state_slot_ids: pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32],
    sampled_row_offsets: pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32],
    hidden_row_offsets: pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[
        pl.Tensor[[N_RANKS, STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]
    ],
    state_meta: pl.InOut[
        pl.Tensor[[N_RANKS, STATE_CAPACITY, STATE_META_WIDTH], pl.INT32]
    ],
    sampled_ids: pl.Tensor[[N_RANKS, T, S], pl.INT32],
    target_hidden: pl.Tensor[
        [N_RANKS, T, MAIN_HIDDEN_DIM], pl.BF16
    ],
    accepted_token_ids: pl.Out[
        pl.Tensor[[N_RANKS, LOCAL_BATCH, S], pl.INT32]
    ],
    accepted_counts: pl.Out[
        pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32]
    ],
    drafter_target_hidden: pl.Out[
        pl.Tensor[[N_RANKS, T, MAIN_HIDDEN_DIM], pl.BF16]
    ],
    drafter_context_positions: pl.Out[
        pl.Tensor[[N_RANKS, T], pl.INT32]
    ],
    drafter_context_valid: pl.Out[
        pl.Tensor[[N_RANKS, T], pl.INT32]
    ],
    drafter_last_sampled: pl.Out[
        pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT64]
    ],
    drafter_anchor_positions: pl.Out[
        pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32]
    ],
    drafter_row_offsets: pl.Out[
        pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32]
    ],
):
    drafter_ready = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        accept_target_into_device_state(
            state_slot_ids[rank],
            state_generations[rank],
            sampled_row_offsets[rank],
            hidden_row_offsets[rank],
            state_tokens[rank],
            state_meta[rank],
            sampled_ids[rank],
            target_hidden[rank],
            accepted_token_ids[rank],
            accepted_counts[rank],
            drafter_target_hidden[rank],
            drafter_context_positions[rank],
            drafter_context_valid[rank],
            drafter_last_sampled[rank],
            drafter_anchor_positions[rank],
            drafter_row_offsets[rank],
            drafter_ready[rank],
            device=rank,
        )


@pl.jit.host
def l3_commit_drafts_to_device_state(
    state_slot_ids: pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[N_RANKS, LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[
        pl.Tensor[[N_RANKS, STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]
    ],
    state_meta: pl.InOut[pl.Tensor[
        [N_RANKS, STATE_CAPACITY, STATE_META_WIDTH], pl.INT32
    ]],
    draft_token_ids: pl.Tensor[
        [N_RANKS, LOCAL_BATCH, DSPARK_QUERY_WIDTH], pl.INT32
    ],
):
    for rank in pl.range(pld.world_size()):
        commit_drafts_to_device_state_rank(
            state_slot_ids[rank],
            state_generations[rank],
            state_tokens[rank],
            state_meta[rank],
            draft_token_ids[rank],
            device=rank,
        )
