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
from decode_fwd import FWD_C1A_TABLE_DYN, FWD_C2A_TABLE_DYN, N_LOCAL_EXPERTS, WEIGHT_BANK_SIZE
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


def run_decode_chain_case(devices: tuple[int, ...], *, compile_only: bool = False,
                          manifest_path: str | None = None,
                          persistent_chain: bool = False) -> None:
    """Compare two deterministic target-to-draft rounds with Torch.

    The second round is intentional: it consumes the draft block committed by
    the first round, so a passing first round alone cannot hide a broken state
    handoff.
    """
    import inspect
    import json
    import math
    from pathlib import Path

    import torch

    from golden import normalize_config
    from pypto.ir import DistributedConfig
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

    import decode_fwd as target
    import dspark_drafter as drafter
    from dspark_markov import golden_markov_from_state
    from dspark_state import (
        STATE_ANCHOR, STATE_COMMITTED, STATE_DRAFT_COUNT, STATE_GENERATION,
        STATE_LIMIT, STATE_VALID, golden_accept_target, golden_commit_drafts,
        golden_prepare_target,
    )

    if len(devices) != EP_SIZE:
        raise ValueError(f"chain case requires EP_SIZE={EP_SIZE} devices")
    if target.WEIGHT_BANK_SIZE != 1:
        raise ValueError("chain case requires --weight-bank-size 1")
    extents = {
        "V41_B_DYN": BATCH,
        "V41_T_DYN": TARGET_ROWS,
        "V41_DSPARK_CONTEXT_T_DYN": TARGET_ROWS,
        "V41_LOCAL_T_DYN": PADDED_ROWS,
        "V41_DSPARK_QUERY_T_DYN": BATCH * (ROWS - 1),
        "V41_TABLE_DYN": 1,
        "V41_ORI_BLOCKS_DYN": BATCH,
        "V41_DSPARK_ROPE_ROWS_DYN": 64,
        "V41_DSPARK_WINDOW_TABLE_DYN": 1,
        "V41_FWD_C2A_TABLE_DYN": 1,
        "V41_FWD_C1A_TABLE_DYN": 2,
        "PACK_X_HC_VOCAB_DYN": FLASH.vocab_size,
    }

    def tensor_spec(program, name):
        annotation = inspect.signature(program._func).parameters[name].annotation
        shape = [dim if isinstance(dim, int) else extents[dim.name] for dim in annotation.shape]
        return shape, target._torch_dtype(annotation.dtype)

    def make_tensor(program, name):
        shape, dtype = tensor_spec(program, name)
        return torch.zeros(shape, dtype=dtype)

    target_specs = target.build_tensor_specs()
    target_bytes = sum(math.prod(spec.shape) * torch.empty((), dtype=spec.dtype).element_size()
                       for spec in target_specs)
    draft_names = [name for name in l3_dspark_drafter.param_names
                   if hasattr(inspect.signature(l3_dspark_drafter._func).parameters[name].annotation, "shape")]
    markov_names = ("final_norm_weight", "lm_head_weight", "markov_w1", "markov_w2", "confidence_weight")

    def program_bytes(program, names):
        return sum(math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                   for shape, dtype in (tensor_spec(program, name) for name in names))

    fixture_bytes = target_bytes + program_bytes(l3_dspark_drafter, draft_names)
    fixture_bytes += program_bytes(l3_markov_sample_from_hidden, markov_names)
    if target_bytes > 16 * 1024**3 or fixture_bytes > 64 * 1024**3:
        raise MemoryError(f"chain fixture exceeds budget: target={target_bytes / 1024**3:.2f} GiB, "
                          f"total={fixture_bytes / 1024**3:.2f} GiB")
    print(f"[chain] fixture: target={target_bytes / 1024**3:.2f} GiB, "
          f"total={fixture_bytes / 1024**3:.2f} GiB, one weight bank", flush=True)
    target_model = {spec.name: spec.create_tensor() for spec in target_specs}
    target_model["embedding_weight"] = torch.zeros(EP_SIZE, 32, D, dtype=torch.bfloat16)
    # Make the target sampler publish token 1 as the first accepted anchor.
    # The target fixture is otherwise zero-weighted; this small nonzero path
    # keeps the anchor handoff deterministic without changing model code.
    target_model["embedding_weight"][:, 0, :32] = 0.25
    # The recurrent round starts from the accepted anchor (token 1).  Give it
    # the same embedding contribution so the target fixture remains stable
    # across the target -> accept -> draft handoff.
    target_model["embedding_weight"][:, 1, :32] = 0.25
    target_model["lm_head_weight"][:, 1, :32] = 1.0
    target_model["attention_num_tokens"].fill_(ROWS)
    target_model["moe_num_tokens"].fill_(ROWS)
    target_model["logit_row_indices"].fill_(-1)
    target_model["logit_row_indices"][:, 0] = ROWS - 1
    draft_model = {}
    for name in draft_names:
        draft_model[name] = make_tensor(l3_dspark_drafter, name)
    draft_model["main_norm_weight"].fill_(1)
    draft_model["embedding_weight"][:, 1, :32] = 0.25
    draft_model["embedding_weight"][:, drafter.DSPARK_NOISE_TOKEN_ID, :32] = 0.5
    for name in ("context_rope_cos", "query_rope_cos", "window_cache_scale"):
        draft_model[name].fill_(1)

    markov_model = {name: make_tensor(l3_markov_sample_from_hidden, name) for name in markov_names}
    markov_model["final_norm_weight"].fill_(1)
    markov_model["markov_w2"].fill_(-16)
    # The target sampler supplies anchor 1.  Alternate the two token rows so
    # the five Markov drafts are unambiguously [0, 1, 0, 1, 0].
    markov_model["markov_w1"][:, 0, 0] = 16
    markov_model["markov_w1"][:, 1, 1] = 16
    markov_model["markov_w2"][:, 1, 0] = 16
    markov_model["markov_w2"][:, 0, 1] = 16

    state = DSparkState(
        slot_ids=make_tensor(l3_prepare_target, "slot_ids"),
        generations=make_tensor(l3_prepare_target, "generations"),
        tokens=make_tensor(l3_prepare_target, "state_tokens"),
        meta=make_tensor(l3_prepare_target, "state_meta"),
    )
    state.slot_ids.fill_(-1)
    state.slot_ids[:, 0] = 0
    state.generations[:, 0] = 1
    state.meta[:, 0, STATE_VALID] = 1
    state.meta[:, 0, STATE_GENERATION] = 1
    state.meta[:, 0, STATE_ANCHOR] = 0
    state.meta[:, 0, STATE_COMMITTED] = 0
    state.meta[:, 0, STATE_DRAFT_COUNT] = ROWS - 1
    state.meta[:, 0, STATE_LIMIT] = 64
    initial_tokens = state.tokens.clone()
    initial_meta = state.meta.clone()

    workspace = {}
    sources = (
        (l3_prepare_target, ("input_ids", "position_ids", "active_widths")),
        (l3_prepare_target_decode, ("x_hc", "pre_mix", "position_ids", "request_ids",
                                    "window_slots", "window_indices", "c2a_slots", "c1a_slots",
                                    "query_starts")),
        (l3_copy_target_hidden, ("compact",)),
        (l3_merge_target_chunks, ("sampled_ids",)),
        (l3_accept_target, ("accepted_ids", "accepted_counts", "context_valid",
                            "context_positions", "num_sampled", "last_sampled",
                            "anchor_ids",
                            "first_query_positions")),
        (l3_prepare_drafter_geometry, ("context_slots", "context_rope_cos",
                                       "context_rope_sin", "query_rope_cos", "query_rope_sin")),
        (l3_commit_drafts, ("committed",)),
    )
    for program, names in sources:
        for name in names:
            key = {"position_ids": "target_positions", "compact": "target_hidden",
                   "sampled_ids": "target_sampled_ids"}.get(name, name)
            if key == "target_positions" and program is l3_prepare_target_decode:
                key = "position_ids"
            workspace[key] = make_tensor(program, name)
    workspace["target_hidden_padded"] = target_model["dspark_target_hidden"]
    workspace["draft_head_hidden"] = draft_model["head_hidden"]
    workspace["draft_ids"] = make_tensor(l3_markov_sample_from_hidden, "draft_ids")
    workspace["confidences"] = make_tensor(l3_markov_sample_from_hidden, "confidences")
    workspace["zero_prefill_tokens"] = make_tensor(l3_markov_sample_from_hidden, "next_prefill_tokens")
    workspace["anchor_ids"] = make_tensor(l3_accept_target, "anchor_ids")
    for chunk in range(CHUNKS):
        workspace[f"chunk_{chunk}"] = make_tensor(l3_sample_target_chunk, "sampled_ids")

    window_table = torch.arange(BATCH, dtype=torch.int32).reshape(1, BATCH, 1).expand(EP_SIZE, -1, -1).clone()
    draft_table = window_table[:, None].expand(-1, DSPARK_DRAFT_LAYERS, -1, -1).clone()
    cache_tables = {
        "target_window": window_table,
        "target_c2a": target_model["c2a_index_block_table"],
        "target_c1a": target_model["c1a_index_block_table"],
        "draft": draft_table,
    }
    rope_tables = {
        "draft_cos": torch.ones(EP_SIZE, 64, FLASH.qk_rope_head_dim // 2),
        "draft_sin": torch.zeros(EP_SIZE, 64, FLASH.qk_rope_head_dim // 2),
    }
    config = normalize_config({
        "backend_type": "a5",
        "execution_mode": False,
        "distributed_config": DistributedConfig(device_ids=list(devices), num_sub_workers=0),
        "ring_heap": (2 * 1024**3, 2 * 1024**3, 1024**3, 512 * 1024**2),
        "ring_dep_pool": 131072,
    })
    manifest_file = Path(manifest_path) if manifest_path else None
    artifacts = json.loads(manifest_file.read_text()) if manifest_file and manifest_file.exists() else {}
    compiled_programs = {}
    persistent_inputs = {}
    active_runner = None
    persistent_worker = None

    def refresh_state_handoff(name):
        """Republish accept outputs so the next L3 program cannot reuse stale state."""
        if name != "l3_accept_target":
            return
        for key in ("num_sampled", "last_sampled"):
            refreshed = workspace[key].clone()
            if isinstance(refreshed, torch.Tensor) and not refreshed.is_shared():
                refreshed.share_memory_()
            workspace[key] = refreshed

    def compile_stage(program, inputs):
        name = program._func.__name__
        parameters = inspect.signature(program._func).parameters
        scalars = {param: pl.RUNTIME for param in program.param_names
                   if not hasattr(parameters[param].annotation, "shape")}
        ordered = [inputs[param] if param not in scalars else
                   torch.tensor(inputs[param], dtype=target._torch_dtype(parameters[param].annotation.dtype))
                   for param in program.param_names]
        persistent_inputs[name] = ordered
        if name not in compiled_programs:
            if name in artifacts:
                compiled = DistributedCompiledProgram.from_dir(
                    artifacts[name], backend_type=config.backend_type,
                    execution_mode=config.execution_mode, distributed_config=config.distributed_config,
                )
            else:
                compiled = program.compile(config=config, **scalars) if scalars else program.compile(*ordered, config=config)
                if manifest_file:
                    artifacts[name] = str(compiled.output_dir)
                    manifest_file.write_text(json.dumps(artifacts, indent=2) + "\n")
            compiled_programs[name] = compiled
        compiled = compiled_programs[name]
        return compiled, ordered

    def runner(program, inputs):
        name = program._func.__name__
        compiled, ordered = compile_stage(program, inputs)
        if compile_only:
            print(f"[chain] {name} COMPILED", flush=True)
            return
        compiled(*ordered, config=config)
        refresh_state_handoff(name)
        print(f"[chain] {name} PASS", flush=True)

    if persistent_chain:
        # Compile every stage before preparing the worker. PyPTO's reusable L3
        # handle must register all HOST programs up front so their device
        # contexts and communication domains share one lifecycle.
        def compile_runner(program, inputs):
            name = program._func.__name__
            compile_stage(program, inputs)
            print(f"[chain] {name} COMPILED", flush=True)

        decode_fwd_dspark(
            compile_runner, state, workspace, target_model, draft_model, markov_model, cache_tables, rope_tables,
            attention_epoch=1, moe_epoch=1, logits_epoch=1,
        )
        if compile_only:
            return
        for tensors in persistent_inputs.values():
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor) and not tensor.is_shared():
                    tensor.share_memory_()
        primary = next(iter(compiled_programs.values()))
        extras = [program for program in compiled_programs.values() if program is not primary]
        persistent_worker = primary.prepare(
            config=config,
            extra_compiled=extras,
            persistent=True,
            reset_persistent_windows=False,
        )

        def persistent_runner(program, inputs):
            name = program._func.__name__
            compiled, ordered = compile_stage(program, inputs)
            persistent_worker.run(compiled, *ordered, config=config)
            refresh_state_handoff(name)
            print(f"[chain] {name} PASS", flush=True)

        active_runner = persistent_runner
        decode_fwd_dspark(
            active_runner, state, workspace, target_model, draft_model,
            markov_model, cache_tables, rope_tables,
            attention_epoch=1, moe_epoch=1, logits_epoch=1,
        )
    else:
        active_runner = runner
        decode_fwd_dspark(
            active_runner, state, workspace, target_model, draft_model, markov_model, cache_tables, rope_tables,
            attention_epoch=1, moe_epoch=1, logits_epoch=1,
        )
    if compile_only:
        return

    reference = dict(draft_model)
    reference.update(
        target_hidden=workspace["target_hidden"], num_sampled=workspace["num_sampled"],
        last_sampled=workspace["last_sampled"], next_prefill_tokens=workspace["zero_prefill_tokens"],
        anchor_ids=workspace["anchor_ids"],
        first_query_positions=workspace["first_query_positions"], block_tables=cache_tables["draft"],
        context_slots=workspace["context_slots"],
        head_hidden=torch.zeros_like(workspace["draft_head_hidden"]),
        window_cache=draft_model["window_cache"].clone(),
        window_cache_scale=draft_model["window_cache_scale"].clone(),
    )
    drafter.golden_dspark_drafter(reference)

    for rank in range(EP_SIZE):
        expected_ids, expected_positions, expected_widths = golden_prepare_target(
            state.slot_ids[rank], state.generations[rank], initial_tokens[rank], initial_meta[rank],
        )
        if not torch.equal(workspace["input_ids"][rank], expected_ids):
            raise AssertionError(f"rank {rank}: target input IDs mismatch")
        if not torch.equal(workspace["target_positions"][rank], expected_positions):
            raise AssertionError(f"rank {rank}: target positions mismatch")
        if not torch.equal(workspace["active_widths"][rank], expected_widths):
            raise AssertionError(f"rank {rank}: active widths mismatch")
        if not torch.equal(workspace["target_hidden"][rank],
                           workspace["target_hidden_padded"][rank, :TARGET_ROWS]):
            raise AssertionError(f"rank {rank}: target hidden handoff mismatch")
        expected_target_sampled = torch.zeros_like(workspace["target_sampled_ids"][rank])
        for request, width in enumerate(expected_widths.tolist()):
            if width > 0:
                expected_target_sampled[request * ROWS:request * ROWS + width, 0] = 1
        if not torch.equal(workspace["target_sampled_ids"][rank], expected_target_sampled):
            raise AssertionError(
                f"rank {rank}: target sampled IDs mismatch "
                f"actual={workspace['target_sampled_ids'][rank, :, 0].tolist()} "
                f"expected={expected_target_sampled[:, 0].tolist()}"
            )
        expected_tokens, expected_meta, accepted_ids, accepted_counts = golden_accept_target(
            state.slot_ids[rank], state.generations[rank], expected_widths,
            workspace["target_sampled_ids"][rank], initial_tokens[rank], initial_meta[rank],
        )
        if not torch.equal(workspace["accepted_ids"][rank], accepted_ids):
            raise AssertionError(f"rank {rank}: accepted IDs mismatch")
        if not torch.equal(workspace["accepted_counts"][rank], accepted_counts):
            raise AssertionError(f"rank {rank}: accepted counts mismatch")
        if not torch.equal(workspace["anchor_ids"][rank, :1], torch.ones(1, dtype=torch.int64)):
            raise AssertionError(
                f"rank {rank}: target anchor handoff mismatch "
                f"actual={workspace['anchor_ids'][rank, :1].tolist()} expected=[1]"
            )
        actual_hidden = workspace["draft_head_hidden"][rank].float()
        expected_hidden = reference["head_hidden"][rank].float()
        hidden_error = (actual_hidden - expected_hidden).abs()
        if not torch.allclose(actual_hidden, expected_hidden, rtol=0.03, atol=0.03):
            index = tuple(int(value) for value in torch.unravel_index(hidden_error.argmax(), hidden_error.shape))
            actual_rows = actual_hidden[:3, :, 0].tolist()
            expected_rows = expected_hidden[:3, :, 0].tolist()
            raise AssertionError(f"rank {rank}: drafter hidden mismatch at {index}: "
                                 f"actual={actual_hidden[index].item():.6g}, "
                                 f"expected={expected_hidden[index].item():.6g}, "
                                 f"max_abs={hidden_error.max().item():.6g}, "
                                 f"actual_rows={actual_rows}, expected_rows={expected_rows}")
        base_logits = torch.zeros(1, ROWS - 1, FLASH.vocab_size)
        expected_active, expected_confidence = golden_markov_from_state(
            base_logits, workspace["num_sampled"][rank, :1], workspace["last_sampled"][rank, :1],
            workspace["zero_prefill_tokens"][rank, :1],
            markov_model["markov_w1"][rank], markov_model["markov_w2"][rank],
            reference["head_hidden"][rank, :1], markov_model["confidence_weight"][rank],
        )
        expected_pattern = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32)
        if not torch.equal(expected_active[0].int(), expected_pattern):
            raise AssertionError(
                f"rank {rank}: invalid Markov fixture for anchor 1: "
                f"golden={expected_active[0].tolist()} expected={expected_pattern.tolist()}"
            )
        expected_drafts = torch.zeros(BATCH, ROWS - 1, dtype=torch.int32)
        expected_drafts[0] = expected_active[0].int()
        if not torch.equal(workspace["draft_ids"][rank, 0], expected_drafts[0]):
            raise AssertionError(
                f"rank {rank}: Markov draft IDs mismatch "
                f"actual={workspace['draft_ids'][rank, 0].tolist()} "
                f"expected={expected_drafts[0].tolist()} "
                f"num={workspace['num_sampled'][rank, :1].tolist()} "
                f"last={workspace['last_sampled'][rank, :1].tolist()} "
                f"next={workspace['zero_prefill_tokens'][rank, :1].tolist()}"
            )
        if not torch.allclose(workspace["confidences"][rank, 0], expected_confidence[0],
                              rtol=1e-3, atol=1e-3):
            raise AssertionError(f"rank {rank}: Markov confidence mismatch")
        expected_tokens, expected_meta, expected_committed = golden_commit_drafts(
            state.slot_ids[rank], state.generations[rank], expected_drafts, expected_tokens, expected_meta,
        )
        for actual, expected, label in (
            (state.tokens[rank], expected_tokens, "state tokens"),
            (state.meta[rank], expected_meta, "state metadata"),
            (workspace["committed"][rank], expected_committed, "commit flags"),
        ):
            if not torch.equal(actual, expected):
                raise AssertionError(f"rank {rank}: {label} mismatch")
    # Re-enter the same target-to-draft path with the state published above.
    # This checks that the accepted anchor and five drafts become the next
    # target slab, rather than merely checking one isolated invocation.
    recurrent_tokens = state.tokens.clone()
    recurrent_meta = state.meta.clone()
    # Persistent windows retain one epoch for every draft chunk and layer.
    draft_epoch = 1
    if persistent_chain:
        draft_chunks = (BATCH + drafter.DSPARK_CHUNK_REQUESTS - 1) // drafter.DSPARK_CHUNK_REQUESTS
        draft_epoch += draft_chunks * DSPARK_DRAFT_LAYERS
    decode_fwd_dspark(
        active_runner, state, workspace, target_model, draft_model, markov_model, cache_tables, rope_tables,
        attention_epoch=draft_epoch, moe_epoch=draft_epoch, logits_epoch=2,
    )
    if persistent_worker is not None:
        persistent_worker.close()
        persistent_worker = None
    for rank in range(EP_SIZE):
        expected_ids, expected_positions, expected_widths = golden_prepare_target(
            state.slot_ids[rank], state.generations[rank], recurrent_tokens[rank], recurrent_meta[rank],
        )
        if not torch.equal(workspace["input_ids"][rank], expected_ids):
            raise AssertionError(f"rank {rank}: recurrent target input IDs mismatch")
        if not torch.equal(workspace["target_positions"][rank], expected_positions):
            raise AssertionError(f"rank {rank}: recurrent target positions mismatch")
        if not torch.equal(workspace["active_widths"][rank], expected_widths):
            raise AssertionError(f"rank {rank}: recurrent active widths mismatch")
        expected_target_sampled = torch.zeros_like(workspace["target_sampled_ids"][rank])
        for request, width in enumerate(expected_widths.tolist()):
            if width > 0:
                expected_target_sampled[request * ROWS:request * ROWS + width, 0] = 1
        if not torch.equal(workspace["target_sampled_ids"][rank], expected_target_sampled):
            raise AssertionError(
                f"rank {rank}: recurrent target sampled IDs mismatch "
                f"actual={workspace['target_sampled_ids'][rank, :, 0].tolist()} "
                f"expected={expected_target_sampled[:, 0].tolist()} "
                f"widths={expected_widths.tolist()} "
                f"input={workspace['input_ids'][rank, :, 0].tolist()}"
            )
        expected_tokens, expected_meta, accepted_ids, accepted_counts = golden_accept_target(
            state.slot_ids[rank], state.generations[rank], expected_widths,
            workspace["target_sampled_ids"][rank], recurrent_tokens[rank], recurrent_meta[rank],
        )
        expected_drafts = torch.zeros(BATCH, ROWS - 1, dtype=torch.int32)
        expected_drafts[0] = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32)
        expected_tokens, expected_meta, expected_committed = golden_commit_drafts(
            state.slot_ids[rank], state.generations[rank], expected_drafts, expected_tokens, expected_meta,
        )
        for actual, expected, label in (
            (workspace["accepted_ids"][rank], accepted_ids, "recurrent accepted IDs"),
            (workspace["accepted_counts"][rank], accepted_counts, "recurrent accepted counts"),
            (state.tokens[rank], expected_tokens, "recurrent state tokens"),
            (state.meta[rank], expected_meta, "recurrent state metadata"),
            (workspace["committed"][rank], expected_committed, "recurrent commit flags"),
        ):
            if not torch.equal(actual, expected):
                raise AssertionError(f"rank {rank}: {label} mismatch")
        if not torch.equal(workspace["anchor_ids"][rank, :1], torch.ones(1, dtype=torch.int64)):
            raise AssertionError(f"rank {rank}: recurrent anchor handoff mismatch")
    print("[chain] target decode -> drafter -> Markov -> state numerical PASS (2 rounds)", flush=True)


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
    parser.add_argument("--validate-chain", action="store_true")
    parser.add_argument("--compile-chain-only", action="store_true")
    parser.add_argument("--chain-manifest")
    parser.add_argument("--persistent-chain", action="store_true",
                        help="run all chain stages on one reusable L3 worker")
    parser.add_argument("--weight-bank-size", type=int, default=WEIGHT_BANK_SIZE, choices=[1, FLASH.num_hidden_layers])
    parser.add_argument("--experts-per-rank", type=int, default=N_LOCAL_EXPERTS)
    args = parser.parse_args()
    if (args.tp, args.ep, args.dp) != (TP_SIZE, EP_SIZE, EP_SIZE):
        parser.error("--tp/--ep/--dp must match the import-time TP1/DP=EP configuration")
    if args.experts_per_rank != N_LOCAL_EXPERTS:
        parser.error("--experts-per-rank must match the import-time configuration")
    devices = tuple(int(value) for value in args.device.split(","))
    if args.validate_chain:
        if args.compile_only or args.execution_mode:
            parser.error("--validate-chain does not accept --compile-only or --sim")
        if args.compile_chain_only and not args.chain_manifest:
            parser.error("--compile-chain-only requires --chain-manifest")
        if args.weight_bank_size != 1:
            parser.error("--validate-chain requires --weight-bank-size 1")
        run_decode_chain_case(devices, compile_only=args.compile_chain_only,
                              manifest_path=args.chain_manifest,
                              persistent_chain=args.persistent_chain)
        raise SystemExit(0)
    if args.compile_chain_only or args.chain_manifest:
        parser.error("--compile-chain-only and --chain-manifest require --validate-chain")
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
