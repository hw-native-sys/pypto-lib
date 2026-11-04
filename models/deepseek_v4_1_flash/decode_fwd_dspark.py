# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ordered V4.1 DSpark bootstrap and recurrent decode stages for TP1/DP=EP."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    DECODE_ROWS_PER_REQUEST,
    DSPARK_DRAFT_LAYERS,
    EP_SIZE,
    FLASH,
    MAX_BATCH_PER_DP,
    TP_SIZE,
)
from decode_fwd import FWD_C1A_TABLE_DYN, FWD_C2A_TABLE_DYN
from decode_fwd import l3_decode_fwd
from dspark_drafter import l3_dspark_drafter
from dspark_markov import l3_markov_sample_from_hidden
from dspark_state import (
    l3_accept_target,
    l3_bootstrap_inputs,
    l3_commit_drafts,
    l3_prepare_drafter_geometry,
    l3_prepare_prefill_drafter_geometry,
    l3_prepare_target,
    l3_publish_bootstrap,
)
from dspark_target_sample import CHUNKS, l3_merge_target_chunks, l3_sample_target_chunk
from input_pack import VOCAB_DYN, pack_x_hc


BATCH = MAX_BATCH_PER_DP
ROWS = DECODE_ROWS_PER_REQUEST
TARGET_ROWS = BATCH * ROWS
PADDED_ROWS = (TARGET_ROWS + 15) // 16 * 16
D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
MAIN_IN = DSPARK_DRAFT_LAYERS * D
WINDOW = FLASH.sliding_window
WINDOW_TABLE_DYN = pl.dynamic("V41_DSPARK_WINDOW_TABLE_DYN")

if TP_SIZE != 1:
    raise ValueError("V4.1 DSpark recurrent entry currently requires TP1")


@pl.jit
def prepare_target_decode_rank(
    input_ids: pl.Tensor[[BATCH, ROWS], pl.INT64],
    positions: pl.Tensor[[BATCH, ROWS], pl.INT32],
    active_widths: pl.Tensor[[BATCH], pl.INT32],
    embedding_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    window_table: pl.Tensor[[BATCH, WINDOW_TABLE_DYN], pl.INT32],
    c2a_table: pl.Tensor[[BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_table: pl.Tensor[[BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[BATCH + 1], pl.INT32]],
):
    embedding_weight.bind_dynamic(0, VOCAB_DYN)
    window_table.bind_dynamic(1, WINDOW_TABLE_DYN)
    c2a_table.bind_dynamic(1, FWD_C2A_TABLE_DYN)
    c1a_table.bind_dynamic(1, FWD_C1A_TABLE_DYN)
    padded_ids = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    table_pages = pl.tensor.dim(window_table, 1)
    c2a_pages = pl.tensor.dim(c2a_table, 1)
    c1a_pages = pl.tensor.dim(c1a_table, 1)
    for _core in pl.spmd(1, name_hint="dspark_target_decode_metadata"):
        for request in pl.range(BATCH + 1):
            pl.write(query_starts, [request], pl.cast(request * ROWS, pl.INT32))
        for row in pl.range(PADDED_ROWS):
            request = row // ROWS
            offset = row % ROWS
            active = row < TARGET_ROWS
            token = pl.cast(0, pl.INT64)
            position = pl.cast(-1, pl.INT32)
            mapped_request = pl.cast(-1, pl.INT32)
            if active:
                if offset < pl.read(active_widths, [request]):
                    token = pl.read(input_ids, [request, offset])
                    position = pl.read(positions, [request, offset])
                    mapped_request = pl.cast(request, pl.INT32)
            pl.write(padded_ids, [row], token)
            pl.write(position_ids, [row], position)
            pl.write(request_ids, [row], mapped_request)
            window_slot = pl.cast(-1, pl.INT64)
            c2a_slot = pl.cast(-1, pl.INT64)
            c1a_slot = pl.cast(-1, pl.INT64)
            if mapped_request >= 0 and position >= 0:
                page = position // BLOCK_SIZE
                if page < table_pages:
                    physical = pl.read(window_table, [request, page])
                    if physical >= 0:
                        window_slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                if page < c1a_pages:
                    physical = pl.read(c1a_table, [request, page])
                    if physical >= 0:
                        c1a_slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                compressed = position // 2
                if (position + 1) % 2 == 0 and compressed // BLOCK_SIZE < c2a_pages:
                    physical = pl.read(c2a_table, [request, compressed // BLOCK_SIZE])
                    if physical >= 0:
                        c2a_slot = pl.cast(physical * BLOCK_SIZE + compressed % BLOCK_SIZE, pl.INT64)
            pl.write(window_slots, [row], window_slot)
            pl.write(c2a_slots, [row], c2a_slot)
            pl.write(c1a_slots, [row], c1a_slot)
            for column in pl.range(WINDOW):
                visible = pl.cast(-1, pl.INT32)
                visible_length = pl.min(position + 1, WINDOW)
                past = position - visible_length + 1 + column
                if mapped_request >= 0 and column < visible_length:
                    past_page = past // BLOCK_SIZE
                    if past_page < table_pages:
                        physical = pl.read(window_table, [request, past_page])
                        if physical >= 0:
                            visible = pl.cast(physical * BLOCK_SIZE + past % BLOCK_SIZE, pl.INT32)
                pl.write(window_indices, [row, column], visible)
        for row in pl.range(PADDED_ROWS):
            for lane in pl.range(HC_MULT):
                pl.write(pre_mix, [row, lane], pl.cast(1.0, pl.FP32))
    pack_x_hc(padded_ids, embedding_weight, x_hc)
    return x_hc, pre_mix


@pl.jit.host
def l3_prepare_target_decode(
    input_ids: pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT64],
    positions: pl.Tensor[[EP_SIZE, BATCH, ROWS], pl.INT32],
    active_widths: pl.Tensor[[EP_SIZE, BATCH], pl.INT32],
    embedding_weight: pl.Tensor[[EP_SIZE, VOCAB_DYN, D], pl.BF16],
    window_table: pl.Tensor[[EP_SIZE, BATCH, WINDOW_TABLE_DYN], pl.INT32],
    c2a_table: pl.Tensor[[EP_SIZE, BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_table: pl.Tensor[[EP_SIZE, BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[EP_SIZE, BATCH + 1], pl.INT32]],
):
    embedding_weight.bind_dynamic(1, VOCAB_DYN)
    window_table.bind_dynamic(2, WINDOW_TABLE_DYN)
    c2a_table.bind_dynamic(2, FWD_C2A_TABLE_DYN)
    c1a_table.bind_dynamic(2, FWD_C1A_TABLE_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_target_decode_rank(
            input_ids[rank], positions[rank], active_widths[rank], embedding_weight[rank],
            window_table[rank], c2a_table[rank], c1a_table[rank], x_hc[rank], pre_mix[rank],
            position_ids[rank], request_ids[rank], window_slots[rank], window_indices[rank],
            c2a_slots[rank], c1a_slots[rank], query_starts[rank], device=rank,
        )


@pl.jit
def copy_target_hidden_rank(
    padded: pl.Tensor[[PADDED_ROWS, MAIN_IN], pl.BF16],
    compact: pl.Out[pl.Tensor[[TARGET_ROWS, MAIN_IN], pl.BF16]],
):
    for row in pl.spmd(TARGET_ROWS, name_hint="dspark_compact_target_hidden"):
        compact[row : row + 1, :] = padded[row : row + 1, :]
    return compact


@pl.jit.host
def l3_copy_target_hidden(
    padded: pl.Tensor[[EP_SIZE, PADDED_ROWS, MAIN_IN], pl.BF16],
    compact: pl.Out[pl.Tensor[[EP_SIZE, TARGET_ROWS, MAIN_IN], pl.BF16]],
):
    for rank in pl.range(pld.world_size()):
        copy_target_hidden_rank(padded[rank], compact[rank], device=rank)


def golden_prepare_target_decode(values: Mapping[str, Any]) -> None:
    """Reference the base decode paging ABI for one local DP rank."""
    import torch

    from utils import paged_slots, window_metadata

    ids = values["input_ids"].reshape(-1).clone()
    positions = values["positions"].reshape(-1)
    widths = values["active_widths"]
    requests = torch.arange(BATCH, dtype=torch.int32).repeat_interleave(ROWS)
    offsets = torch.arange(ROWS).repeat(BATCH)
    active = offsets < widths.repeat_interleave(ROWS)
    requests[~active] = -1
    active_positions = positions[active]
    active_requests = requests[active]
    ids[~active] = 0
    values["position_ids"].fill_(-1)
    values["position_ids"][:TARGET_ROWS][active] = active_positions
    values["request_ids"].fill_(-1)
    values["request_ids"][:TARGET_ROWS] = requests
    values["window_slots"].fill_(-1)
    values["window_indices"].fill_(-1)
    values["c2a_slots"].fill_(-1)
    values["c1a_slots"].fill_(-1)
    if bool(active.any()):
        window_slots, window_indices, _ = window_metadata(
            active_positions, active_requests, values["window_table"], WINDOW,
        )
        values["window_slots"][:TARGET_ROWS][active] = window_slots
        values["window_indices"][:TARGET_ROWS][active] = window_indices
        values["c2a_slots"][:TARGET_ROWS][active] = paged_slots(
            active_positions, active_requests, values["c2a_table"], BLOCK_SIZE, 2, True,
        )
        values["c1a_slots"][:TARGET_ROWS][active] = paged_slots(
            active_positions, active_requests, values["c1a_table"], BLOCK_SIZE, 1, True,
        )
    values["query_starts"].copy_(torch.arange(BATCH + 1, dtype=torch.int32) * ROWS)
    values["pre_mix"].fill_(1)
    padded = torch.zeros(PADDED_ROWS, dtype=torch.int64)
    padded[:TARGET_ROWS] = ids
    embedded = values["embedding_weight"].index_select(0, padded).float()
    values["x_hc"].copy_(embedded[:, None, :].expand(-1, HC_MULT, -1))


def run_prepare_target_decode_case(*, platform: str = "a5", devices: tuple[int, ...] = (0,),
                                   execution_mode: bool = False, compile_only: bool = False):
    """Exercise six-row, partial, and idle requests on one rank or all EP ranks."""
    import torch

    from golden import TensorSpec, run
    from pypto.ir import DistributedConfig

    widths = torch.tensor([6, 6, 1, 0] + [0] * (BATCH - 4), dtype=torch.int32)
    ids = torch.arange(TARGET_ROWS, dtype=torch.int64).reshape(BATCH, ROWS) % 32
    positions = torch.zeros((BATCH, ROWS), dtype=torch.int32)
    positions[0] = torch.arange(125, 125 + ROWS, dtype=torch.int32)
    positions[1] = torch.arange(5, 5 + ROWS, dtype=torch.int32)
    positions[2, 0] = 127
    tables = torch.arange(BATCH * 2, dtype=torch.int32).reshape(BATCH, 2)
    c2a_table = torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1)
    c1a_table = (tables + BATCH * 2).contiguous()
    embedding = (torch.arange(32 * D).reshape(32, D).remainder(257) / 64).to(torch.bfloat16)
    specs = [
        TensorSpec("input_ids", [BATCH, ROWS], torch.int64, init_value=ids),
        TensorSpec("positions", [BATCH, ROWS], torch.int32, init_value=positions),
        TensorSpec("active_widths", [BATCH], torch.int32, init_value=widths),
        TensorSpec("embedding_weight", [32, D], torch.bfloat16, init_value=embedding),
        TensorSpec("window_table", [BATCH, 2], torch.int32, init_value=tables),
        TensorSpec("c2a_table", [BATCH, 1], torch.int32, init_value=c2a_table),
        TensorSpec("c1a_table", [BATCH, 2], torch.int32, init_value=c1a_table),
        TensorSpec("x_hc", [PADDED_ROWS, HC_MULT, D], torch.float32),
        TensorSpec("pre_mix", [PADDED_ROWS, HC_MULT], torch.float32),
        TensorSpec("position_ids", [PADDED_ROWS], torch.int32),
        TensorSpec("request_ids", [PADDED_ROWS], torch.int32),
        TensorSpec("window_slots", [PADDED_ROWS], torch.int64),
        TensorSpec("window_indices", [PADDED_ROWS, WINDOW], torch.int32),
        TensorSpec("c2a_slots", [PADDED_ROWS], torch.int64),
        TensorSpec("c1a_slots", [PADDED_ROWS], torch.int64),
        TensorSpec("query_starts", [BATCH + 1], torch.int32),
    ]
    if len(devices) == 1:
        fn = prepare_target_decode_rank
        golden_fn = golden_prepare_target_decode
        config = {"backend_type": platform, "execution_mode": execution_mode, "device_id": devices[0]}
    elif len(devices) == EP_SIZE:
        fn = l3_prepare_target_decode
        rank_specs = specs
        specs = []
        for spec in rank_specs:
            initial = spec.init_value
            if isinstance(initial, torch.Tensor):
                initial = initial.unsqueeze(0).expand(EP_SIZE, *initial.shape).clone()
            specs.append(TensorSpec(spec.name, [EP_SIZE, *spec.shape], spec.dtype, init_value=initial))

        def golden_fn(values):
            for rank in range(EP_SIZE):
                golden_prepare_target_decode({name: value[rank] for name, value in values.items()})

        config = {
            "backend_type": platform,
            "execution_mode": execution_mode,
            "distributed_config": DistributedConfig(device_ids=list(devices), num_sub_workers=0),
        }
    else:
        raise ValueError(f"provide one device or EP_SIZE={EP_SIZE} devices, got {devices}")
    return run(fn=fn, specs=specs, golden_fn=golden_fn, config=config, compile_only=compile_only)


def run_copy_target_hidden_case(*, devices: tuple[int, ...], execution_mode: bool = False,
                                compile_only: bool = False):
    """Check that the three target-layer taps retain request-major row order."""
    import torch

    from golden import TensorSpec, run
    from pypto.ir import DistributedConfig

    if len(devices) != EP_SIZE:
        raise ValueError(f"copy case requires EP_SIZE={EP_SIZE} devices, got {devices}")
    padded = torch.arange(EP_SIZE * PADDED_ROWS * MAIN_IN, dtype=torch.int32)
    padded = padded.reshape(EP_SIZE, PADDED_ROWS, MAIN_IN).remainder(1024).to(torch.bfloat16)

    def golden(values):
        values["compact"].copy_(values["padded"][:, :TARGET_ROWS])

    return run(
        fn=l3_copy_target_hidden,
        specs=[
            TensorSpec("padded", [EP_SIZE, PADDED_ROWS, MAIN_IN], torch.bfloat16, init_value=padded),
            TensorSpec("compact", [EP_SIZE, TARGET_ROWS, MAIN_IN], torch.bfloat16),
        ],
        golden_fn=golden,
        config={
            "backend_type": "a5",
            "execution_mode": execution_mode,
            "distributed_config": DistributedConfig(device_ids=list(devices), num_sub_workers=0),
        },
        compile_only=compile_only,
    )


StageRunner = Callable[[Any, Mapping[str, Any]], None]


@dataclass
class DSparkState:
    """Caller-owned request slots; generations protect recycled slots."""

    slot_ids: Any
    generations: Any
    tokens: Any
    meta: Any


def _run(runner: StageRunner, program: Any, inputs: Mapping[str, Any]) -> None:
    required = set(program.param_names)
    supplied = set(inputs)
    if required != supplied:
        raise ValueError(
            f"{program._func.__name__} inputs mismatch: "
            f"missing={sorted(required - supplied)}, extra={sorted(supplied - required)}"
        )
    runner(program, inputs)


def decode_fwd_dspark(
    runner: StageRunner,
    state: DSparkState,
    workspace: Mapping[str, Any],
    target_model: Mapping[str, Any],
    draft_model: Mapping[str, Any],
    markov_model: Mapping[str, Any],
    cache_tables: Mapping[str, Any],
    rope_tables: Mapping[str, Any],
    *,
    attention_epoch: int,
    moe_epoch: int,
    logits_epoch: int,
) -> None:
    """Execute one six-row verification and publish five drafts for the next call.

    The runner executes each completed L3 program before the next stage. Its
    tensors stay resident across calls; the caller owns weights, caches, and
    request-slot lifetimes. The lib owns the ordering and all state transitions.
    """
    _run(runner, l3_prepare_target, dict(
        slot_ids=state.slot_ids, generations=state.generations, state_tokens=state.tokens,
        state_meta=state.meta, input_ids=workspace["input_ids"],
        position_ids=workspace["target_positions"], active_widths=workspace["active_widths"],
    ))
    _run(runner, l3_prepare_target_decode, dict(
        input_ids=workspace["input_ids"], positions=workspace["target_positions"],
        active_widths=workspace["active_widths"], embedding_weight=target_model["embedding_weight"],
        window_table=cache_tables["target_window"], c2a_table=cache_tables["target_c2a"],
        c1a_table=cache_tables["target_c1a"], x_hc=workspace["x_hc"], pre_mix=workspace["pre_mix"],
        position_ids=workspace["position_ids"], request_ids=workspace["request_ids"],
        window_slots=workspace["window_slots"], window_indices=workspace["window_indices"],
        c2a_slots=workspace["c2a_slots"], c1a_slots=workspace["c1a_slots"],
        query_starts=workspace["query_starts"],
    ))
    decode_args = dict(target_model)
    decode_args.pop("embedding_weight")
    decode_args.update(
        x_hc=workspace["x_hc"], pre_mix=workspace["pre_mix"],
        dspark_target_hidden=workspace["target_hidden_padded"], position_ids=workspace["position_ids"],
        token_to_req_indices=workspace["request_ids"], window_slots=workspace["window_slots"],
        window_indices=workspace["window_indices"], c2a_compressed_slots=workspace["c2a_slots"],
        c1a_compressed_slots=workspace["c1a_slots"], query_start_loc=workspace["query_starts"],
        logit_row_indices=target_model["logit_row_indices"],
    )
    _run(runner, l3_decode_fwd, decode_args)
    for chunk in range(CHUNKS):
        _run(runner, l3_sample_target_chunk, dict(
            x_hc=workspace["x_hc"], pre_mix=workspace["pre_mix"],
            final_norm_weight=target_model["final_norm_weight"],
            lm_head_weight=target_model["lm_head_weight"], active_widths=workspace["active_widths"],
            sampled_ids=workspace[f"chunk_{chunk}"], chunk=chunk,
        ))
    _run(runner, l3_merge_target_chunks, {
        **{f"chunk_{chunk}": workspace[f"chunk_{chunk}"] for chunk in range(CHUNKS)},
        "sampled_ids": workspace["target_sampled_ids"],
    })
    _run(runner, l3_accept_target, dict(
        slot_ids=state.slot_ids, generations=state.generations, active_widths=workspace["active_widths"],
        sampled_ids=workspace["target_sampled_ids"], state_tokens=state.tokens, state_meta=state.meta,
        accepted_ids=workspace["accepted_ids"], accepted_counts=workspace["accepted_counts"],
        context_valid=workspace["context_valid"], context_positions=workspace["context_positions"],
        num_sampled=workspace["num_sampled"], last_sampled=workspace["last_sampled"],
        anchor_ids=workspace["anchor_ids"],
        first_query_positions=workspace["first_query_positions"],
    ))
    _run(runner, l3_copy_target_hidden, dict(
        padded=workspace["target_hidden_padded"], compact=workspace["target_hidden"],
    ))
    _run(runner, l3_prepare_drafter_geometry, dict(
        context_valid=workspace["context_valid"], context_positions=workspace["context_positions"],
        first_query_positions=workspace["first_query_positions"], block_tables=cache_tables["draft"],
        rope_cos_table=rope_tables["draft_cos"], rope_sin_table=rope_tables["draft_sin"],
        context_slots=workspace["context_slots"], context_rope_cos=workspace["context_rope_cos"],
        context_rope_sin=workspace["context_rope_sin"], query_rope_cos=workspace["query_rope_cos"],
        query_rope_sin=workspace["query_rope_sin"],
    ))
    draft_args = dict(draft_model)
    draft_args.update(
        target_hidden=workspace["target_hidden"], num_sampled=workspace["num_sampled"],
        last_sampled=workspace["last_sampled"], next_prefill_tokens=workspace["zero_prefill_tokens"],
        anchor_ids=workspace["anchor_ids"],
        first_query_positions=workspace["first_query_positions"], block_tables=cache_tables["draft"],
        context_slots=workspace["context_slots"], context_rope_cos=workspace["context_rope_cos"],
        context_rope_sin=workspace["context_rope_sin"], query_rope_cos=workspace["query_rope_cos"],
        query_rope_sin=workspace["query_rope_sin"], head_hidden=workspace["draft_head_hidden"],
        attention_epoch=attention_epoch, moe_epoch=moe_epoch,
    )
    _run(runner, l3_dspark_drafter, draft_args)
    markov_args = dict(markov_model)
    markov_args.update(
        head_hidden=workspace["draft_head_hidden"], num_sampled=workspace["num_sampled"],
        last_sampled=workspace["last_sampled"], next_prefill_tokens=workspace["zero_prefill_tokens"],
        anchor_ids=workspace["anchor_ids"],
        draft_ids=workspace["draft_ids"], confidences=workspace["confidences"], logits_epoch=logits_epoch,
    )
    _run(runner, l3_markov_sample_from_hidden, markov_args)
    _run(runner, l3_commit_drafts, dict(
        slot_ids=state.slot_ids, generations=state.generations, draft_ids=workspace["draft_ids"],
        state_tokens=state.tokens, state_meta=state.meta, committed=workspace["committed"],
    ))


def prefill_bootstrap_dspark(
    runner: StageRunner,
    state: DSparkState,
    workspace: Mapping[str, Any],
    draft_model: Mapping[str, Any],
    markov_model: Mapping[str, Any],
    cache_tables: Mapping[str, Any],
    rope_tables: Mapping[str, Any],
    *,
    attention_epoch: int,
    moe_epoch: int,
    logits_epoch: int,
) -> None:
    """Publish the first draft block from completed prefill target taps."""
    _run(runner, l3_bootstrap_inputs, dict(
        slot_ids=state.slot_ids, generations=state.generations,
        prompt_lengths=workspace["prompt_lengths"], position_limits=workspace["position_limits"],
        sampled_rows=workspace["sampled_rows"], sampled_ids=workspace["prefill_sampled_ids"],
        state_meta=state.meta, next_prefill_tokens=workspace["next_prefill_tokens"],
        first_query_positions=workspace["first_query_positions"],
    ))
    _run(runner, l3_prepare_prefill_drafter_geometry, dict(
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
        query_rope_sin=workspace["query_rope_sin"], head_hidden=workspace["draft_head_hidden"],
        attention_epoch=attention_epoch, moe_epoch=moe_epoch,
    )
    _run(runner, l3_dspark_drafter, draft_args)
    markov_args = dict(markov_model)
    markov_args.update(
        head_hidden=workspace["draft_head_hidden"], num_sampled=workspace["zero_sampled"],
        last_sampled=workspace["zero_last_sampled"], next_prefill_tokens=workspace["next_prefill_tokens"],
        anchor_ids=workspace["next_prefill_tokens"],
        draft_ids=workspace["draft_ids"], confidences=workspace["confidences"], logits_epoch=logits_epoch,
    )
    _run(runner, l3_markov_sample_from_hidden, markov_args)
    _run(runner, l3_publish_bootstrap, dict(
        slot_ids=state.slot_ids, generations=state.generations,
        prompt_lengths=workspace["prompt_lengths"], position_limits=workspace["position_limits"],
        sampled_rows=workspace["sampled_rows"], sampled_ids=workspace["prefill_sampled_ids"],
        draft_ids=workspace["draft_ids"], state_tokens=state.tokens, state_meta=state.meta,
    ))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1])
    parser.add_argument("--ep", type=int, default=EP_SIZE)
    parser.add_argument("--dp", type=int, default=EP_SIZE)
    parser.add_argument("-d", "--device", default="0")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if (args.tp, args.ep, args.dp) != (TP_SIZE, EP_SIZE, EP_SIZE):
        parser.error("--tp/--ep/--dp must match the import-time TP1/DP=EP configuration")
    devices = tuple(int(value) for value in args.device.split(","))
    result = run_prepare_target_decode_case(
        platform=args.platform, devices=devices, execution_mode=args.execution_mode,
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(str(result))
    if len(devices) == EP_SIZE:
        result = run_copy_target_hidden_case(
            devices=devices, execution_mode=args.execution_mode, compile_only=args.compile_only,
        )
        if not result.passed:
            raise SystemExit(str(result))
