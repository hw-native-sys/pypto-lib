# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 DSpark initialization and shared request state."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    DECODE_SEQ,
    DSPARK_DRAFT_LAYERS,
    DSPARK_QUERY_WIDTH,
    EP_SIZE,
    FLASH,
    DECODE_BATCH,
    TABLE_DYN,
    T_DYN,
)


from decode_metadata import (
    STATE_VALID, STATE_GENERATION, STATE_ANCHOR, STATE_COMMITTED, STATE_DRAFT_COUNT, STATE_LIMIT,
    STATE_META_WIDTH, STATE_TOKEN_WIDTH,
)
from lm_head import SAMPLED_IDS_PAD


StageRunner = Callable[[Any, Mapping[str, Any]], None]


@dataclass
class DSparkState:
    """Caller-owned request slots; generations protect recycled slots."""

    slot_ids: Any
    generations: Any
    tokens: Any
    meta: Any


def _run_stage(runner: StageRunner, program: Any, inputs: Mapping[str, Any]) -> None:
    required = set(program.param_names)
    supplied = set(inputs)
    if required != supplied:
        raise ValueError(
            f"{program._func.__name__} inputs mismatch: "
            f"missing={sorted(required - supplied)}, extra={sorted(supplied - required)}"
        )
    runner(program, inputs)


ROWS = DECODE_SEQ
PREFILL_SAMPLE_ROWS = 16
ROPE_HALF = FLASH.qk_rope_head_dim // 2
ROPE_ROWS_DYN = pl.dynamic("DSPARK_ROPE_ROWS_DYN")



@pl.jit
def bootstrap_inputs_rank(
    slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[DECODE_BATCH], pl.INT32],
    position_limits: pl.Tensor[[DECODE_BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[PREFILL_SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    state_meta: pl.Tensor[[DECODE_BATCH, STATE_META_WIDTH], pl.INT32],
    next_prefill_tokens: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_inputs"):
        for request in pl.range(DECODE_BATCH):
            pl.write(next_prefill_tokens, [request], pl.cast(0, pl.INT64))
            pl.write(first_query_positions, [request], pl.cast(-1, pl.INT32))
            slot = pl.read(slot_ids, [request])
            row = pl.read(sampled_rows, [request])
            prompt = pl.read(prompt_lengths, [request])
            limit = pl.read(position_limits, [request])
            if slot >= 0 and slot < DECODE_BATCH and row >= 0 and row < PREFILL_SAMPLE_ROWS:
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
    slot_ids: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    position_limits: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[EP_SIZE, PREFILL_SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    state_meta: pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_META_WIDTH], pl.INT32],
    next_prefill_tokens: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        bootstrap_inputs_rank(
            slot_ids[rank], generations[rank], prompt_lengths[rank], position_limits[rank], sampled_rows[rank],
            sampled_ids[rank], state_meta[rank], next_prefill_tokens[rank], first_query_positions[rank],
            device=rank,
        )


@pl.jit
def publish_bootstrap_rank(
    slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[DECODE_BATCH], pl.INT32],
    position_limits: pl.Tensor[[DECODE_BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[PREFILL_SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    draft_ids: pl.Tensor[[DECODE_BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[DECODE_BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[DECODE_BATCH, STATE_META_WIDTH], pl.INT32]],
):
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_publish"):
        for request in pl.range(DECODE_BATCH):
            slot = pl.read(slot_ids, [request])
            row = pl.read(sampled_rows, [request])
            prompt = pl.read(prompt_lengths, [request])
            limit = pl.read(position_limits, [request])
            if slot >= 0 and slot < DECODE_BATCH and row >= 0 and row < PREFILL_SAMPLE_ROWS:
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
    slot_ids: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    prompt_lengths: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    position_limits: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    sampled_rows: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Tensor[[EP_SIZE, PREFILL_SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    draft_ids: pl.Tensor[[EP_SIZE, DECODE_BATCH, DSPARK_QUERY_WIDTH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_META_WIDTH], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        publish_bootstrap_rank(
            slot_ids[rank], generations[rank], prompt_lengths[rank], position_limits[rank],
            sampled_rows[rank], sampled_ids[rank], draft_ids[rank], state_tokens[rank], state_meta[rank],
            device=rank,
        )


@pl.jit
def prepare_prefill_drafter_geometry_rank(
    positions: pl.Tensor[[T_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    first_query_positions: pl.Tensor[[DECODE_BATCH], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, DECODE_BATCH, TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[T_DYN, ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[T_DYN, ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[DECODE_BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[DECODE_BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
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
            active = row < active_tokens and request >= 0 and request < DECODE_BATCH
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
                if active and table_blocks > 0:
                    page = position // BLOCK_SIZE % table_blocks
                    physical = pl.read(block_tables, [layer, request, page])
                    if physical >= 0:
                        slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                pl.write(context_slots, [layer, row], slot)
        for request in pl.range(DECODE_BATCH):
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
    first_query_positions: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    block_tables: pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, DECODE_BATCH, TABLE_DYN], pl.INT32],
    rope_cos_table: pl.Tensor[[EP_SIZE, ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    rope_sin_table: pl.Tensor[[EP_SIZE, ROPE_ROWS_DYN, ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * DSPARK_QUERY_WIDTH, ROPE_HALF], pl.FP32]],
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


def dspark_init(
    runner: StageRunner,
    state: DSparkState,
    workspace: Mapping[str, Any],
    draft_model: Mapping[str, Any],
    markov_model: Mapping[str, Any],
    cache_tables: Mapping[str, Any],
    rope_tables: Mapping[str, Any],
    *,
    moe_epoch: int,
    logits_epoch: int,
) -> None:
    """Publish the first draft block from completed prefill target taps."""
    from dspark_drafter import l3_dspark_drafter
    from markov import l3_markov_output

    _run_stage(runner, l3_bootstrap_inputs, dict(
        slot_ids=state.slot_ids, generations=state.generations,
        prompt_lengths=workspace["prompt_lengths"], position_limits=workspace["position_limits"],
        sampled_rows=workspace["sampled_rows"], sampled_ids=workspace["prefill_sampled_ids"],
        state_meta=state.meta, next_prefill_tokens=workspace["next_prefill_tokens"],
        first_query_positions=workspace["first_query_positions"],
    ))
    _run_stage(runner, l3_prepare_prefill_drafter_geometry, dict(
        positions=workspace["prefill_positions"], request_ids=workspace["prefill_request_ids"],
        num_tokens=workspace["prefill_counts"], first_query_positions=workspace["first_query_positions"],
        block_tables=cache_tables["draft"], rope_cos_table=rope_tables["draft_cos"],
        rope_sin_table=rope_tables["draft_sin"], context_slots=workspace["context_slots"],
        context_rope_cos=workspace["context_rope_cos"], context_rope_sin=workspace["context_rope_sin"],
        query_rope_cos=workspace["query_rope_cos"], query_rope_sin=workspace["query_rope_sin"],
    ))
    draft_args = dict(draft_model)
    draft_args.update(
        target_hidden=workspace["prefill_target_hidden"], num_sampled=workspace["zero_sampled"],
        last_sampled=workspace["zero_last_sampled"], next_prefill_tokens=workspace["next_prefill_tokens"],
        anchor_ids=workspace["next_prefill_tokens"],
        first_query_positions=workspace["first_query_positions"], block_tables=cache_tables["draft"],
        context_slots=workspace["context_slots"], context_rope_cos=workspace["context_rope_cos"],
        context_rope_sin=workspace["context_rope_sin"], query_rope_cos=workspace["query_rope_cos"],
        query_rope_sin=workspace["query_rope_sin"],
        draft_x_hc=workspace["draft_x_hc"], draft_pre_mix=workspace["draft_pre_mix"],
        moe_epoch=moe_epoch,
    )
    _run_stage(runner, l3_dspark_drafter, draft_args)
    markov_args = dict(markov_model)
    markov_args.update(
        draft_x_hc=workspace["draft_x_hc"], draft_pre_mix=workspace["draft_pre_mix"],
        head_hidden=workspace["draft_head_hidden"], num_sampled=workspace["zero_sampled"],
        last_sampled=workspace["zero_last_sampled"], next_prefill_tokens=workspace["next_prefill_tokens"],
        anchor_ids=workspace["next_prefill_tokens"],
        draft_ids=workspace["draft_ids"], confidences=workspace["confidences"], logits_epoch=logits_epoch,
    )
    _run_stage(runner, l3_markov_output, markov_args)
    _run_stage(runner, l3_publish_bootstrap, dict(
        slot_ids=state.slot_ids, generations=state.generations,
        prompt_lengths=workspace["prompt_lengths"], position_limits=workspace["position_limits"],
        sampled_rows=workspace["sampled_rows"], sampled_ids=workspace["prefill_sampled_ids"],
        draft_ids=workspace["draft_ids"], state_tokens=state.tokens, state_meta=state.meta,
    ))
