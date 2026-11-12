# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""Ordered V4.1 DSpark bootstrap and recurrent decode stages for DP=EP."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pypto.language as pl
import pypto.language.distributed as pld

if __name__ == "__main__":
    import sys

    # Select the bounded smoke shapes before decode_fwd reads its import-time options.
    sys.argv[1:1] = ["--weight-bank-size", "1", "--experts-per-rank", "16"]

from config import (
    BLOCK_SIZE,
    DECODE_SEQ,
    DSPARK_DRAFT_LAYERS,
    EP_SIZE,
    FLASH,
    DECODE_BATCH,
    LM_HEAD_TP_SIZE, SUPPORTED_LM_HEAD_TP_SIZES, WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES,
)
from quant import unpack_quantized_cache
from decode_fwd import FWD_C1A_TABLE_DYN, FWD_C2A_TABLE_DYN
from dspark_drafter import l3_dspark_drafter
from dspark_markov import l3_markov_sample_from_hidden
from dspark_state import (
    l3_bootstrap_inputs,
    l3_prepare_prefill_drafter_geometry,
    l3_publish_bootstrap,
)
import decode_fwd as target_fwd
import dspark_drafter as draft
import dspark_markov as markov
import dspark_state as state_ops
import dspark_target_sample as sampling
from dspark_state import accept_target, commit_drafts, prepare_drafter_geometry, prepare_target
from decode_fwd import decode_fwd
from dspark_target_sample import sample_target
from dspark_drafter import dspark_drafter
from dspark_markov import markov_sample_from_hidden
from input_pack import VOCAB_DYN, pack_x_hc


ROWS = DECODE_SEQ
TARGET_ROWS = DECODE_BATCH * ROWS
PADDED_ROWS = (TARGET_ROWS + 15) // 16 * 16
D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
MAIN_IN = DSPARK_DRAFT_LAYERS * D
WINDOW = FLASH.sliding_window
WINDOW_TABLE_DYN = pl.dynamic("DSPARK_WINDOW_TABLE_DYN")



def _prepare_target_decode_rank(
    input_ids: pl.Tensor[[DECODE_BATCH, ROWS], pl.INT64],
    positions: pl.Tensor[[DECODE_BATCH, ROWS], pl.INT32],
    active_widths: pl.Tensor[[DECODE_BATCH], pl.INT32],
    embedding_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    window_table: pl.Tensor[[DECODE_BATCH, WINDOW_TABLE_DYN], pl.INT32],
    c2a_table: pl.Tensor[[DECODE_BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_table: pl.Tensor[[DECODE_BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[DECODE_BATCH + 1], pl.INT32]],
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
                value = pl.cast(0.0, pl.FP32)
                if lane == 0:
                    value = pl.cast(1.0, pl.FP32)
                pl.write(pre_mix, [row, lane], value)
    pack_x_hc(padded_ids, embedding_weight, x_hc, hidden_tile=512)
    return x_hc, pre_mix

prepare_target_decode = pl.jit.inline(auto_scope=False)(_prepare_target_decode_rank)
prepare_target_decode_rank = pl.jit(auto_scope=False)(_prepare_target_decode_rank)


@pl.jit.host
def l3_prepare_target_decode(
    input_ids: pl.Tensor[[EP_SIZE, DECODE_BATCH, ROWS], pl.INT64],
    positions: pl.Tensor[[EP_SIZE, DECODE_BATCH, ROWS], pl.INT32],
    active_widths: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    embedding_weight: pl.Tensor[[EP_SIZE, VOCAB_DYN, D], pl.BF16],
    window_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, WINDOW_TABLE_DYN], pl.INT32],
    c2a_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH + 1], pl.INT32]],
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


def _copy_target_hidden_rank(
    padded: pl.Tensor[[PADDED_ROWS, MAIN_IN], pl.BF16],
    compact: pl.Out[pl.Tensor[[TARGET_ROWS, MAIN_IN], pl.BF16]],
):
    for row in pl.spmd(TARGET_ROWS, name_hint="dspark_compact_target_hidden"):
        compact[row : row + 1, :] = padded[row : row + 1, :]
    return compact

copy_target_hidden = pl.jit.inline(auto_scope=False)(_copy_target_hidden_rank)
copy_target_hidden_rank = pl.jit(auto_scope=False)(_copy_target_hidden_rank)


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
    requests = torch.arange(DECODE_BATCH, dtype=torch.int32).repeat_interleave(ROWS)
    offsets = torch.arange(ROWS).repeat(DECODE_BATCH)
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
    values["query_starts"].copy_(torch.arange(DECODE_BATCH + 1, dtype=torch.int32) * ROWS)
    values["pre_mix"].zero_()
    values["pre_mix"][:, 0] = 1.0
    padded = torch.zeros(PADDED_ROWS, dtype=torch.int64)
    padded[:TARGET_ROWS] = ids
    embedded = values["embedding_weight"].index_select(0, padded).float()
    values["x_hc"].copy_(embedded[:, None, :].expand(-1, HC_MULT, -1))


def _compare_target_initial_mix(actual, expected, *, actual_outputs, expected_outputs, **kwargs):
    """Check identity coefficients and preservation of the active embeddings."""
    from golden import ratio_allclose
    from hc import golden_hc_collapse

    compare = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    context = dict(actual_outputs=actual_outputs, expected_outputs=expected_outputs, **kwargs)
    ok, detail = compare(actual, expected, **context)
    if not ok:
        return ok, detail
    hidden = actual_outputs["x_hc"].reshape(-1, HC_MULT, D)
    collapsed = golden_hc_collapse(hidden, actual.reshape(-1, HC_MULT))
    embedding = expected_outputs["x_hc"].reshape(-1, HC_MULT, D)[:, 0, :]
    active = expected_outputs["request_ids"].reshape(-1) >= 0
    return compare(collapsed[active], embedding[active], **context)


def run_prepare_target_decode_case(*, platform: str = "a5", devices: tuple[int, ...] = (0,),
                                   execution_mode: bool = False, compile_only: bool = False):
    """Exercise six-row, partial, and idle requests on one rank or all EP ranks."""
    import torch

    from golden import TensorSpec, run
    from pypto.ir import DistributedConfig

    widths = torch.tensor([6, 6, 1, 0] + [0] * (DECODE_BATCH - 4), dtype=torch.int32)
    ids = torch.arange(TARGET_ROWS, dtype=torch.int64).reshape(DECODE_BATCH, ROWS) % 32
    positions = torch.zeros((DECODE_BATCH, ROWS), dtype=torch.int32)
    positions[0] = torch.arange(125, 125 + ROWS, dtype=torch.int32)
    positions[1] = torch.arange(5, 5 + ROWS, dtype=torch.int32)
    positions[2, 0] = 127
    tables = torch.arange(DECODE_BATCH * 2, dtype=torch.int32).reshape(DECODE_BATCH, 2)
    c2a_table = torch.arange(DECODE_BATCH, dtype=torch.int32).reshape(DECODE_BATCH, 1)
    c1a_table = (tables + DECODE_BATCH * 2).contiguous()
    embedding = (torch.arange(32 * D).reshape(32, D).remainder(257) / 64).to(torch.bfloat16)
    specs = [
        TensorSpec("input_ids", [DECODE_BATCH, ROWS], torch.int64, init_value=ids),
        TensorSpec("positions", [DECODE_BATCH, ROWS], torch.int32, init_value=positions),
        TensorSpec("active_widths", [DECODE_BATCH], torch.int32, init_value=widths),
        TensorSpec("embedding_weight", [32, D], torch.bfloat16, init_value=embedding),
        TensorSpec("window_table", [DECODE_BATCH, 2], torch.int32, init_value=tables),
        TensorSpec("c2a_table", [DECODE_BATCH, 1], torch.int32, init_value=c2a_table),
        TensorSpec("c1a_table", [DECODE_BATCH, 2], torch.int32, init_value=c1a_table),
        TensorSpec("x_hc", [PADDED_ROWS, HC_MULT, D], torch.float32),
        TensorSpec("pre_mix", [PADDED_ROWS, HC_MULT], torch.float32),
        TensorSpec("position_ids", [PADDED_ROWS], torch.int32),
        TensorSpec("request_ids", [PADDED_ROWS], torch.int32),
        TensorSpec("window_slots", [PADDED_ROWS], torch.int64),
        TensorSpec("window_indices", [PADDED_ROWS, WINDOW], torch.int32),
        TensorSpec("c2a_slots", [PADDED_ROWS], torch.int64),
        TensorSpec("c1a_slots", [PADDED_ROWS], torch.int64),
        TensorSpec("query_starts", [DECODE_BATCH + 1], torch.int32),
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
    return run(
        fn=fn, specs=specs, golden_fn=golden_fn, config=config, compile_only=compile_only,
        compare_fn={"pre_mix": _compare_target_initial_mix},
    )


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


STATE_OPS_STATE_TOKEN_WIDTH = state_ops.STATE_TOKEN_WIDTH
STATE_OPS_STATE_META_WIDTH = state_ops.STATE_META_WIDTH
STATE_OPS_ROWS = state_ops.ROWS
TARGET_FWD_T_DYN = target_fwd.T_DYN
TARGET_FWD_ENGRAM_K = target_fwd.ENGRAM_K
TARGET_FWD_KV_OUT = target_fwd.KV_OUT
TARGET_FWD_DSPARK_MAIN_IN = target_fwd.DSPARK_MAIN_IN
TARGET_FWD_WEIGHT_BANK_SIZE = target_fwd.WEIGHT_BANK_SIZE
TARGET_FWD_MIX_HC = target_fwd.MIX_HC
TARGET_FWD_HC_DIM = target_fwd.HC_DIM
TARGET_FWD_Q_LORA = target_fwd.Q_LORA
TARGET_FWD_MX_GROUP = target_fwd.MX_GROUP
TARGET_FWD_LOCAL_H = target_fwd.LOCAL_H
TARGET_FWD_HEAD_DIM = target_fwd.HEAD_DIM
TARGET_FWD_LOCAL_O_GROUPS = target_fwd.LOCAL_O_GROUPS
TARGET_FWD_O_GROUP_IN = target_fwd.O_GROUP_IN
TARGET_FWD_O_LORA = target_fwd.O_LORA
TARGET_FWD_LOCAL_O_WIDTH = target_fwd.LOCAL_O_WIDTH
TARGET_FWD_ROPE_ROWS_DYN = target_fwd.ROPE_ROWS_DYN
TARGET_FWD_ROPE_DIM = target_fwd.ROPE_DIM
TARGET_FWD_FWD_ORI_BLOCKS_DYN = target_fwd.FWD_ORI_BLOCKS_DYN
TARGET_FWD_FWD_CMP_BLOCKS_DYN = target_fwd.FWD_CMP_BLOCKS_DYN
TARGET_FWD_FWD_INDEX_BLOCKS_DYN = target_fwd.FWD_INDEX_BLOCKS_DYN
TARGET_FWD_INDEX_DIM = target_fwd.INDEX_DIM
TARGET_FWD_INDEX_CACHE_GROUP = target_fwd.INDEX_CACHE_GROUP
TARGET_FWD_B_DYN = target_fwd.B_DYN
TARGET_FWD_FWD_C2A_TABLE_DYN = target_fwd.FWD_C2A_TABLE_DYN
TARGET_FWD_FWD_C1A_TABLE_DYN = target_fwd.FWD_C1A_TABLE_DYN
TARGET_FWD_C2A_SOURCE_COUNT = target_fwd.C2A_SOURCE_COUNT
TARGET_FWD_FWD_STATE_BLOCKS_DYN = target_fwd.FWD_STATE_BLOCKS_DYN
TARGET_FWD_STATE_CAPACITY = target_fwd.STATE_CAPACITY
TARGET_FWD_STATE_WIDTH = target_fwd.STATE_WIDTH
TARGET_FWD_KV_SOURCE_COUNT = target_fwd.KV_SOURCE_COUNT
TARGET_FWD_INDEX_SOURCE_COUNT = target_fwd.INDEX_SOURCE_COUNT
TARGET_FWD_INDEX_H = target_fwd.INDEX_H
TARGET_FWD_N_EXPERTS = target_fwd.N_EXPERTS
TARGET_FWD_N_LOCAL_EXPERTS = target_fwd.N_LOCAL_EXPERTS
TARGET_FWD_MX_W1_PACKED_ROWS = target_fwd.MX_W1_PACKED_ROWS
TARGET_FWD_MX_PACKED_LANE_COLS = target_fwd.MX_PACKED_LANE_COLS
TARGET_FWD_MOE_INTER = target_fwd.MOE_INTER
TARGET_FWD_MX_W2_PACKED_ROWS = target_fwd.MX_W2_PACKED_ROWS
TARGET_FWD_MX_W3_PACKED_ROWS = target_fwd.MX_W3_PACKED_ROWS
TARGET_FWD_VOCAB_PER_TP = target_fwd.VOCAB_PER_TP
TARGET_FWD_MAX_LOGIT_ROWS = target_fwd.MAX_LOGIT_ROWS
TARGET_FWD_VOCAB = target_fwd.VOCAB
TARGET_FWD_SAMPLED_IDS_PAD = target_fwd.SAMPLED_IDS_PAD
SAMPLING_ROWS = sampling.ROWS
SAMPLING_SAMPLED_IDS_PAD = sampling.SAMPLED_IDS_PAD
STATE_OPS_TABLE_DYN = state_ops.TABLE_DYN
STATE_OPS_ROPE_ROWS_DYN = state_ops.ROPE_ROWS_DYN
STATE_OPS_ROPE_HALF = state_ops.ROPE_HALF
STATE_OPS_DSPARK_QUERY_WIDTH = state_ops.DSPARK_QUERY_WIDTH
DRAFT_EP_SIZE = draft.C.EP_SIZE
DRAFT_MAIN_IN = draft.MAIN_IN
DRAFT_B_DYN = draft.B_DYN
DRAFT_VOCAB = draft.VOCAB
DRAFT_MIX_HC = draft.MIX_HC
DRAFT_HC_DIM = draft.HC_DIM
DRAFT_Q_LORA = draft.Q_LORA
DRAFT_MX_GROUP = draft.MX_GROUP
DRAFT_LOCAL_H = draft.LOCAL_H
DRAFT_HEAD_DIM = draft.HEAD_DIM
DRAFT_LOCAL_O_GROUPS = draft.LOCAL_O_GROUPS
DRAFT_O_GROUP_IN = draft.O_GROUP_IN
DRAFT_O_LORA = draft.O_LORA
DRAFT_LOCAL_O_WIDTH = draft.LOCAL_O_WIDTH
DRAFT_ORI_BLOCKS_DYN = draft.C.ORI_BLOCKS_DYN
DRAFT_N_EXPERTS = draft.N_EXPERTS
DRAFT_N_LOCAL_EXPERTS = draft.N_LOCAL_EXPERTS
DRAFT_MX_W1_PACKED_ROWS = draft.MX_W1_PACKED_ROWS
DRAFT_MX_PACKED_LANE_COLS = draft.MX_PACKED_LANE_COLS
DRAFT_MOE_INTER = draft.MOE_INTER
DRAFT_MX_W2_PACKED_ROWS = draft.MX_W2_PACKED_ROWS
DRAFT_MX_W3_PACKED_ROWS = draft.MX_W3_PACKED_ROWS
DRAFT_DSPARK_QUERY_WIDTH = draft.DSPARK_QUERY_WIDTH
MARKOV_VOCAB_PER_TP = markov.VOCAB_PER_TP
MARKOV_VOCAB = markov.VOCAB
MARKOV_MARKOV_RANK = markov.MARKOV_RANK
MARKOV_B_DYN = markov.B_DYN
MARKOV_DSPARK_QUERY_WIDTH = markov.DSPARK_QUERY_WIDTH
TARGET_FWD_RECV_MAX = target_fwd.RECV_MAX
TARGET_FWD_AUX_WIDTH = target_fwd.AUX_WIDTH
TARGET_FWD_ROUTE_WIDTH = target_fwd.ROUTE_WIDTH
TARGET_FWD_SIGNAL_PAD = target_fwd.SIGNAL_PAD
TARGET_FWD_MOE_TOKENS = target_fwd.MOE_TOKENS
TARGET_FWD_TOPK = target_fwd.TOPK
TARGET_FWD_GROUP_LOGIT_ROWS = target_fwd.GROUP_LOGIT_ROWS
DRAFT_RECV_MAX = draft.RECV_MAX
DRAFT_AUX_WIDTH = draft.AUX_WIDTH
DRAFT_ROUTE_WIDTH = draft.ROUTE_WIDTH
DRAFT_SIGNAL_PAD = draft.SIGNAL_PAD
DRAFT_MOE_TOKENS = draft.MOE_TOKENS
DRAFT_TOPK = draft.TOPK
MARKOV_GROUP_LOGIT_ROWS = markov.GROUP_LOGIT_ROWS
MARKOV_MAX_LOGIT_ROWS = markov.MAX_LOGIT_ROWS


@pl.jit(auto_scope=False)
def l2_decode_fwd_dspark(
    slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[DECODE_BATCH, STATE_OPS_STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[DECODE_BATCH, STATE_OPS_STATE_META_WIDTH], pl.INT32]],
    input_ids: pl.Out[pl.Tensor[[DECODE_BATCH, STATE_OPS_ROWS], pl.INT64]],
    target_positions: pl.Out[pl.Tensor[[DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    target_embedding_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    target_window_table: pl.Tensor[[DECODE_BATCH, WINDOW_TABLE_DYN], pl.INT32],
    target_c2a_table: pl.Tensor[[DECODE_BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    target_c1a_table: pl.Tensor[[DECODE_BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[DECODE_BATCH + 1], pl.INT32]],
    engram_lookup: pl.Tensor[[2, PADDED_ROWS, TARGET_FWD_ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, TARGET_FWD_ENGRAM_K, TARGET_FWD_KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (TARGET_FWD_ENGRAM_K // 32), TARGET_FWD_KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    target_hidden_padded: pl.Out[pl.Tensor[[PADDED_ROWS, TARGET_FWD_DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_Q_LORA, TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN, TARGET_FWD_O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN // 32, TARGET_FWD_O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_LOCAL_O_WIDTH // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    window_cache_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, TARGET_FWD_INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, TARGET_FWD_INDEX_DIM // TARGET_FWD_INDEX_CACHE_GROUP], pl.FP8E8M0]],
    c2a_index_block_table: pl.Tensor[[TARGET_FWD_B_DYN, TARGET_FWD_FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[TARGET_FWD_B_DYN, TARGET_FWD_FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, TARGET_FWD_HEAD_DIM], pl.BF16, pl.NZ],
    state_block_table: pl.Tensor[[TARGET_FWD_B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_STATE_BLOCKS_DYN, TARGET_FWD_STATE_CAPACITY, TARGET_FWD_STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[TARGET_FWD_KV_SOURCE_COUNT * TARGET_FWD_HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_HEAD_DIM, TARGET_FWD_INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT * TARGET_FWD_INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_Q_LORA, TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT, D, TARGET_FWD_INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W1_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W2_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W3_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, TARGET_FWD_VOCAB_PER_TP], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[TARGET_FWD_MAX_LOGIT_ROWS], pl.INT32],
    target_logits: pl.Out[pl.Tensor[[TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], pl.FP32]],
    target_backbone_sampled_ids: pl.Out[pl.Tensor[[TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_SAMPLED_IDS_PAD], pl.INT32]],
    target_sampled_ids: pl.Out[pl.Tensor[[SAMPLING_ROWS, SAMPLING_SAMPLED_IDS_PAD], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    context_valid: pl.Out[pl.Tensor[[DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    context_positions: pl.Out[pl.Tensor[[DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT64]],
    anchor_ids: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    target_hidden: pl.Out[pl.Tensor[[TARGET_ROWS, MAIN_IN], pl.BF16]],
    draft_table: pl.Tensor[[DSPARK_DRAFT_LAYERS, DECODE_BATCH, STATE_OPS_TABLE_DYN], pl.INT32],
    draft_cos: pl.Tensor[[STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    draft_sin: pl.Tensor[[STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, DECODE_BATCH * STATE_OPS_ROWS], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    draft_main_proj_weight: pl.Tensor[[DRAFT_MAIN_IN, D], pl.BF16],
    draft_main_norm_weight: pl.Tensor[[D], pl.BF16],
    zero_prefill_tokens: pl.Tensor[[DRAFT_B_DYN], pl.INT64],
    draft_embedding_weight: pl.Tensor[[DRAFT_VOCAB, D], pl.BF16],
    draft_hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_attn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, DRAFT_Q_LORA], pl.FP8E4M3FN],
    draft_wq_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_Q_LORA], pl.FP8E8M0],
    draft_q_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA], pl.BF16],
    draft_wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA // DRAFT_MX_GROUP, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wkv_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_kv_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_HEAD_DIM], pl.BF16],
    draft_attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_H], pl.FP32],
    draft_wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN, DRAFT_O_LORA], pl.FP8E4M3FN],
    draft_wo_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN // DRAFT_MX_GROUP, DRAFT_O_LORA], pl.FP8E8M0],
    draft_wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    draft_wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_window_cache: pl.InOut[pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    draft_hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_ffn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_gate_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS, D], pl.BF16],
    draft_correction_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS], pl.FP32],
    draft_routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W1_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W2_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_MOE_INTER // DRAFT_MX_GROUP), D], pl.FP8E8M0],
    draft_routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W3_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, D], pl.FP8E4M3FN],
    draft_shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_head_hidden: pl.Out[pl.Tensor[[DRAFT_B_DYN, DRAFT_DSPARK_QUERY_WIDTH, D], pl.BF16]],
    moe_epoch: pl.Scalar[pl.INT32],
    markov_final_norm_weight: pl.Tensor[[D], pl.BF16],
    markov_lm_head_weight: pl.Tensor[[D, MARKOV_VOCAB_PER_TP], pl.BF16, pl.NZ],
    markov_markov_w1: pl.Tensor[[MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_markov_w2: pl.Tensor[[MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_confidence_weight: pl.Tensor[[1, D + MARKOV_MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
    committed: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    target_fwd_recv_meta: pld.DistributedTensor[[EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS], pl.INT32],
    target_fwd_recv_x: pld.DistributedTensor[[TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, D], pl.INT8],
    target_fwd_recv_scale: pld.DistributedTensor[[TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, D // TARGET_FWD_MX_GROUP], pl.UINT8],
    target_fwd_recv_weights: pld.DistributedTensor[[TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_AUX_WIDTH], pl.FP32],
    target_fwd_recv_routes: pld.DistributedTensor[[TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_ROUTE_WIDTH], pl.INT32],
    target_fwd_data_arrived: pld.DistributedTensor[[EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], pl.INT32],
    target_fwd_routed_output: pld.DistributedTensor[[TARGET_FWD_MOE_TOKENS * TARGET_FWD_TOPK, D], pl.BF16],
    target_fwd_combine_arrived: pld.DistributedTensor[[EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], pl.INT32],
    target_fwd_head_hidden: pld.DistributedTensor[[TARGET_FWD_GROUP_LOGIT_ROWS, D], pl.BF16],
    target_fwd_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    target_fwd_head_logits: pld.DistributedTensor[[TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], pl.FP32],
    target_fwd_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    draft_recv_meta: pld.DistributedTensor[[EP_SIZE, DRAFT_N_LOCAL_EXPERTS], pl.INT32],
    draft_recv_x: pld.DistributedTensor[[DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, D], pl.INT8],
    draft_recv_scale: pld.DistributedTensor[[DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, D // DRAFT_MX_GROUP], pl.UINT8],
    draft_recv_weights: pld.DistributedTensor[[DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_AUX_WIDTH], pl.FP32],
    draft_recv_routes: pld.DistributedTensor[[DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], pl.INT32],
    draft_data_arrived: pld.DistributedTensor[[EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], pl.INT32],
    draft_routed_output: pld.DistributedTensor[[DRAFT_MOE_TOKENS * DRAFT_TOPK, D], pl.BF16],
    draft_combine_arrived: pld.DistributedTensor[[EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], pl.INT32],
    markov_hidden_window: pld.DistributedTensor[[MARKOV_GROUP_LOGIT_ROWS, D], pl.BF16],
    markov_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    markov_logits_window: pld.DistributedTensor[[MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], pl.FP32],
    markov_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    sampling_hidden_window: pld.DistributedTensor[[TARGET_FWD_GROUP_LOGIT_ROWS, D], pl.BF16],
    sampling_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    sampling_logits_window: pld.DistributedTensor[[TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], pl.FP32],
    sampling_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Verify one rank's draft block and publish its next proposals."""
    local_x_hc = pl.create_tensor([PADDED_ROWS, HC_MULT, D], dtype=pl.FP32)
    local_pre_mix = pl.create_tensor([PADDED_ROWS, HC_MULT], dtype=pl.FP32)
    local_position_ids = pl.create_tensor([PADDED_ROWS], dtype=pl.INT32)
    local_request_ids = pl.create_tensor([PADDED_ROWS], dtype=pl.INT32)
    local_window_slots = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    local_window_indices = pl.create_tensor([PADDED_ROWS, WINDOW], dtype=pl.INT32)
    local_c2a_slots = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    local_c1a_slots = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    local_query_starts = pl.create_tensor([DECODE_BATCH + 1], dtype=pl.INT32)
    local_target_hidden_padded = pl.create_tensor([PADDED_ROWS, TARGET_FWD_DSPARK_MAIN_IN], dtype=pl.BF16)
    target_embedding_weight.bind_dynamic(0, VOCAB_DYN)
    target_window_table.bind_dynamic(1, WINDOW_TABLE_DYN)
    target_c2a_table.bind_dynamic(1, FWD_C2A_TABLE_DYN)
    target_c1a_table.bind_dynamic(1, FWD_C1A_TABLE_DYN)
    freqs_cos.bind_dynamic(1, TARGET_FWD_ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(1, TARGET_FWD_ROPE_ROWS_DYN)
    window_cache_pool.bind_dynamic(0, TARGET_FWD_FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(0, TARGET_FWD_FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(0, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(0, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    c2a_index_block_table.bind_dynamic(0, TARGET_FWD_B_DYN)
    c2a_index_block_table.bind_dynamic(1, TARGET_FWD_FWD_C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(0, TARGET_FWD_B_DYN)
    c1a_index_block_table.bind_dynamic(1, TARGET_FWD_FWD_C1A_TABLE_DYN)
    state_block_table.bind_dynamic(0, TARGET_FWD_B_DYN)
    state_cache_pool.bind_dynamic(0, TARGET_FWD_FWD_STATE_BLOCKS_DYN)
    draft_table.bind_dynamic(2, STATE_OPS_TABLE_DYN)
    draft_cos.bind_dynamic(0, STATE_OPS_ROPE_ROWS_DYN)
    draft_sin.bind_dynamic(0, STATE_OPS_ROPE_ROWS_DYN)
    zero_prefill_tokens.bind_dynamic(0, DRAFT_B_DYN)
    draft_window_cache.bind_dynamic(1, DRAFT_ORI_BLOCKS_DYN)
    draft_head_hidden.bind_dynamic(0, DRAFT_B_DYN)
    draft_ids.bind_dynamic(0, MARKOV_B_DYN)
    confidences.bind_dynamic(0, MARKOV_B_DYN)
    prepare_target(
        slot_ids, generations, state_tokens, state_meta,
        input_ids, target_positions, active_widths,
    )
    prepare_target_decode(
        input_ids, target_positions, active_widths, target_embedding_weight,
        target_window_table, target_c2a_table, target_c1a_table, local_x_hc,
        local_pre_mix, local_position_ids, local_request_ids, local_window_slots,
        local_window_indices, local_c2a_slots, local_c1a_slots, local_query_starts,
    )
    decode_fwd(
        engram_lookup, engram_wkv, engram_wkv_scale, engram_weight,
        local_x_hc, local_pre_mix, local_target_hidden_padded, hc_attn_fn,
        hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a,
        q_norm_weight, wq_b, wkv, kv_norm_weight,
        attn_sink, wo_a, wo_a_scale, wo_b,
        freqs_cos, freqs_sin, local_window_slots, local_window_indices,
        window_cache_pool, compressed_cache_pool, local_request_ids, local_position_ids,
        index_cache_pool, index_cache_scale_pool, c2a_index_block_table, c1a_index_block_table,
        c2a_compressor_wkv, c2a_compressor_wgate, c1a_compressor_wkv, local_query_starts,
        state_block_table, state_cache_pool, compressor_norm_weight, local_c2a_slots,
        local_c1a_slots, index_wk, index_norm_weight, index_wq_b,
        index_weights_proj, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight, gate_weight, correction_bias, routed_w1,
        routed_w2, routed_w3, shared_w1, shared_w2,
        shared_w3, target_fwd_recv_meta, target_fwd_recv_x, target_fwd_recv_scale,
        target_fwd_recv_weights, target_fwd_recv_routes, target_fwd_data_arrived, target_fwd_routed_output,
        target_fwd_combine_arrived, moe_num_tokens, attention_num_tokens, rank,
        wq_a_scale, wq_b_scale, wkv_scale, wo_b_scale,
        index_wq_b_scale, routed_w1_scale, routed_w2_scale, routed_w3_scale,
        shared_w1_scale, shared_w2_scale, shared_w3_scale, final_norm_weight,
        lm_head_weight, logit_row_indices, target_logits, target_backbone_sampled_ids,
        target_fwd_head_hidden, target_fwd_head_hidden_done, target_fwd_head_logits, target_fwd_head_logits_done,
    )
    sample_hidden = pl.reshape(local_x_hc, [PADDED_ROWS, HC_MULT, D])
    sample_pre_mix = pl.reshape(local_pre_mix, [PADDED_ROWS, HC_MULT])
    sample_collapsed = pl.create_tensor([PADDED_ROWS, D], dtype=pl.BF16)
    sample_normalized = pl.create_tensor([PADDED_ROWS, D], dtype=pl.BF16)
    sample_target(
        sample_hidden, sample_pre_mix, final_norm_weight, lm_head_weight,
        active_widths, target_sampled_ids, sampling_hidden_window, sampling_hidden_done,
        sampling_logits_window, sampling_logits_done, rank, sample_collapsed,
        sample_normalized,
    )
    accept_target(
        slot_ids, generations, active_widths, target_sampled_ids,
        state_tokens, state_meta, accepted_ids, accepted_counts,
        context_valid, context_positions, num_sampled, last_sampled,
        anchor_ids, first_query_positions,
    )
    copy_target_hidden(local_target_hidden_padded, target_hidden)
    prepare_drafter_geometry(
        context_valid, context_positions, first_query_positions, draft_table,
        draft_cos, draft_sin, context_slots, context_rope_cos,
        context_rope_sin, query_rope_cos, query_rope_sin,
    )
    dspark_drafter(
        target_hidden, draft_main_proj_weight, draft_main_norm_weight, num_sampled,
        last_sampled, zero_prefill_tokens, anchor_ids, first_query_positions,
        draft_table, context_slots, context_rope_cos, context_rope_sin,
        query_rope_cos, query_rope_sin, draft_embedding_weight, draft_hc_attn_fn,
        draft_hc_attn_scale, draft_hc_attn_base, draft_attn_norm_weight, draft_wq_a,
        draft_wq_a_scale, draft_q_norm_weight, draft_wq_b, draft_wq_b_scale,
        draft_wkv, draft_wkv_scale, draft_kv_norm_weight, draft_attn_sink,
        draft_wo_a, draft_wo_a_scale, draft_wo_b, draft_wo_b_scale,
        draft_window_cache, draft_hc_ffn_fn, draft_hc_ffn_scale, draft_hc_ffn_base,
        draft_ffn_norm_weight, draft_gate_weight, draft_correction_bias, draft_routed_w1,
        draft_routed_w1_scale, draft_routed_w2, draft_routed_w2_scale, draft_routed_w3,
        draft_routed_w3_scale, draft_shared_w1, draft_shared_w1_scale, draft_shared_w2,
        draft_shared_w2_scale, draft_shared_w3, draft_shared_w3_scale, draft_head_hidden,
        draft_recv_meta, draft_recv_x, draft_recv_scale, draft_recv_weights,
        draft_recv_routes, draft_data_arrived, draft_routed_output, draft_combine_arrived,
        rank, moe_epoch,
    )
    markov_sample_from_hidden(
        draft_head_hidden, markov_final_norm_weight, markov_lm_head_weight, num_sampled,
        last_sampled, zero_prefill_tokens, anchor_ids, markov_markov_w1,
        markov_markov_w2, markov_confidence_weight, draft_ids, confidences,
        markov_hidden_window, markov_hidden_done, markov_logits_window, markov_logits_done,
        rank, logits_epoch,
    )
    commit_drafts(
        slot_ids, generations, draft_ids, state_tokens,
        state_meta, committed,
    )
    published_x = pl.reshape(x_hc, [PADDED_ROWS, HC_MULT * D])
    source_x = pl.reshape(local_x_hc, [PADDED_ROWS, HC_MULT * D])
    for row in pl.spmd(PADDED_ROWS, name_hint="dspark_publish_target_hidden"):
        hidden_tile = pl.load(source_x, [row, 0], [1, HC_MULT * D])
        pl.store(hidden_tile, [row, 0], published_x)
        target_tile = pl.load(local_target_hidden_padded, [row, 0], [1, TARGET_FWD_DSPARK_MAIN_IN])
        pl.store(target_tile, [row, 0], target_hidden_padded)
        indices_tile = pl.load(local_window_indices, [row, 0], [1, WINDOW])
        pl.store(indices_tile, [row, 0], window_indices)
    for _core in pl.spmd(1, name_hint="dspark_publish_target_metadata"):
        for row in pl.range(PADDED_ROWS):
            for stream in pl.range(HC_MULT):
                pl.write(pre_mix, [row, stream], pl.read(local_pre_mix, [row, stream]))
            pl.write(position_ids, [row], pl.read(local_position_ids, [row]))
            pl.write(request_ids, [row], pl.read(local_request_ids, [row]))
            pl.write(window_slots, [row], pl.read(local_window_slots, [row]))
            pl.write(c2a_slots, [row], pl.read(local_c2a_slots, [row]))
            pl.write(c1a_slots, [row], pl.read(local_c1a_slots, [row]))
        for request in pl.range(DECODE_BATCH + 1):
            pl.write(query_starts, [request], pl.read(local_query_starts, [request]))



@pl.jit.host
def l3_decode_fwd_dspark(
    slot_ids: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_STATE_META_WIDTH], pl.INT32]],
    input_ids: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_ROWS], pl.INT64]],
    target_positions: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    target_embedding_weight: pl.Tensor[[EP_SIZE, VOCAB_DYN, D], pl.BF16],
    target_window_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, WINDOW_TABLE_DYN], pl.INT32],
    target_c2a_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    target_c1a_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH + 1], pl.INT32]],
    engram_lookup: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_T_DYN, TARGET_FWD_ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ENGRAM_K, TARGET_FWD_KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[EP_SIZE, 2 * (TARGET_FWD_ENGRAM_K // 32), TARGET_FWD_KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[EP_SIZE, 2, HC_MULT, D], pl.FP32],
    target_hidden_padded: pl.Out[pl.Tensor[[EP_SIZE, TARGET_FWD_T_DYN, TARGET_FWD_DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_Q_LORA, TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN, TARGET_FWD_O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN // 32, TARGET_FWD_O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_LOCAL_O_WIDTH // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    window_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, TARGET_FWD_INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, TARGET_FWD_INDEX_DIM // TARGET_FWD_INDEX_CACHE_GROUP], pl.FP8E8M0]],
    c2a_index_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, TARGET_FWD_FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, TARGET_FWD_FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[EP_SIZE, TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[EP_SIZE, TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[EP_SIZE, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    state_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_STATE_BLOCKS_DYN, TARGET_FWD_STATE_CAPACITY, TARGET_FWD_STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_KV_SOURCE_COUNT * TARGET_FWD_HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_HEAD_DIM, TARGET_FWD_INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT * TARGET_FWD_INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_Q_LORA, TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, D, TARGET_FWD_INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W1_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W2_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W3_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[EP_SIZE, D, TARGET_FWD_VOCAB_PER_TP], pl.BF16],
    logit_row_indices: pl.Tensor[[EP_SIZE, TARGET_FWD_MAX_LOGIT_ROWS], pl.INT32],
    target_logits: pl.Out[pl.Tensor[[EP_SIZE, TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], pl.FP32]],
    target_backbone_sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_SAMPLED_IDS_PAD], pl.INT32]],
    target_sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, SAMPLING_ROWS, SAMPLING_SAMPLED_IDS_PAD], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    context_valid: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    context_positions: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT64]],
    anchor_ids: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    target_hidden: pl.Out[pl.Tensor[[EP_SIZE, TARGET_ROWS, MAIN_IN], pl.BF16]],
    draft_table: pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, DECODE_BATCH, STATE_OPS_TABLE_DYN], pl.INT32],
    draft_cos: pl.Tensor[[EP_SIZE, STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    draft_sin: pl.Tensor[[EP_SIZE, STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, DECODE_BATCH * STATE_OPS_ROWS], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    draft_main_proj_weight: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_MAIN_IN, D], pl.BF16],
    draft_main_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, D], pl.BF16],
    zero_prefill_tokens: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_B_DYN], pl.INT64],
    draft_embedding_weight: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_VOCAB, D], pl.BF16],
    draft_hc_attn_fn: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_attn_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_attn_base: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_attn_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_wq_a: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_Q_LORA], pl.FP8E4M3FN],
    draft_wq_a_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_Q_LORA], pl.FP8E8M0],
    draft_q_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA], pl.BF16],
    draft_wq_b: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wq_b_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA // DRAFT_MX_GROUP, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_wkv: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wkv_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_kv_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_HEAD_DIM], pl.BF16],
    draft_attn_sink: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_H], pl.FP32],
    draft_wo_a: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN, DRAFT_O_LORA], pl.FP8E4M3FN],
    draft_wo_a_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN // DRAFT_MX_GROUP, DRAFT_O_LORA], pl.FP8E8M0],
    draft_wo_b: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    draft_wo_b_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_window_cache: pl.InOut[pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    draft_hc_ffn_fn: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_ffn_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_ffn_base: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_ffn_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_gate_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS, D], pl.BF16],
    draft_correction_bias: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS], pl.FP32],
    draft_routed_w1: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W1_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w1_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_routed_w2: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W2_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w2_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_MOE_INTER // DRAFT_MX_GROUP), D], pl.FP8E8M0],
    draft_routed_w3: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W3_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w3_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w1: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w1_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w2: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, D], pl.FP8E4M3FN],
    draft_shared_w2_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_shared_w3: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w3_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_head_hidden: pl.Out[pl.Tensor[[DRAFT_EP_SIZE, DRAFT_B_DYN, DRAFT_DSPARK_QUERY_WIDTH, D], pl.BF16]],
    moe_epoch: pl.Scalar[pl.INT32],
    markov_final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    markov_lm_head_weight: pl.Tensor[[EP_SIZE, D, MARKOV_VOCAB_PER_TP], pl.BF16],
    markov_markov_w1: pl.Tensor[[EP_SIZE, MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_markov_w2: pl.Tensor[[EP_SIZE, MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_confidence_weight: pl.Tensor[[EP_SIZE, 1, D + MARKOV_MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
    committed: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
):
    """Run one complete recurrent DSpark L2 per rank."""
    target_embedding_weight.bind_dynamic(1, VOCAB_DYN)
    target_window_table.bind_dynamic(2, WINDOW_TABLE_DYN)
    target_c2a_table.bind_dynamic(2, FWD_C2A_TABLE_DYN)
    target_c1a_table.bind_dynamic(2, FWD_C1A_TABLE_DYN)
    engram_lookup.bind_dynamic(2, TARGET_FWD_T_DYN)
    target_hidden_padded.bind_dynamic(1, TARGET_FWD_T_DYN)
    freqs_cos.bind_dynamic(2, TARGET_FWD_ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, TARGET_FWD_ROPE_ROWS_DYN)
    window_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(1, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    c2a_index_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    c2a_index_block_table.bind_dynamic(2, TARGET_FWD_FWD_C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    c1a_index_block_table.bind_dynamic(2, TARGET_FWD_FWD_C1A_TABLE_DYN)
    state_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    state_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_STATE_BLOCKS_DYN)
    draft_table.bind_dynamic(3, STATE_OPS_TABLE_DYN)
    draft_cos.bind_dynamic(1, STATE_OPS_ROPE_ROWS_DYN)
    draft_sin.bind_dynamic(1, STATE_OPS_ROPE_ROWS_DYN)
    zero_prefill_tokens.bind_dynamic(1, DRAFT_B_DYN)
    draft_window_cache.bind_dynamic(2, DRAFT_ORI_BLOCKS_DYN)
    draft_head_hidden.bind_dynamic(1, DRAFT_B_DYN)
    draft_ids.bind_dynamic(1, MARKOV_B_DYN)
    confidences.bind_dynamic(1, MARKOV_B_DYN)
    target_fwd_recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS], dtype=pl.INT32)
    target_fwd_recv_x_buffer = pld.alloc_window_buffer([TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, D], dtype=pl.INT8)
    target_fwd_recv_scale_buffer = pld.alloc_window_buffer([TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, D // TARGET_FWD_MX_GROUP], dtype=pl.UINT8)
    target_fwd_recv_weights_buffer = pld.alloc_window_buffer([TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_AUX_WIDTH], dtype=pl.FP32)
    target_fwd_recv_routes_buffer = pld.alloc_window_buffer([TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_ROUTE_WIDTH], dtype=pl.INT32)
    target_fwd_data_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
    target_fwd_routed_output_buffer = pld.alloc_window_buffer([TARGET_FWD_MOE_TOKENS * TARGET_FWD_TOPK, D], dtype=pl.BF16)
    target_fwd_combine_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
    target_fwd_head_hidden_buffer = pld.alloc_window_buffer([TARGET_FWD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    target_fwd_head_hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    target_fwd_head_logits_buffer = pld.alloc_window_buffer([TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], dtype=pl.FP32)
    target_fwd_head_logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    sample_hidden_buffer = pld.alloc_window_buffer([TARGET_FWD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    sample_hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    sample_logits_buffer = pld.alloc_window_buffer([TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], dtype=pl.FP32)
    sample_logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    draft_recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS], dtype=pl.INT32)
    draft_recv_x_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, D], dtype=pl.INT8)
    draft_recv_scale_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, D // DRAFT_MX_GROUP], dtype=pl.UINT8)
    draft_recv_weights_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_AUX_WIDTH], dtype=pl.FP32)
    draft_recv_routes_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], dtype=pl.INT32)
    draft_data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
    draft_routed_output_buf = pld.alloc_window_buffer([DRAFT_MOE_TOKENS * DRAFT_TOPK, D], dtype=pl.BF16)
    draft_combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
    markov_hidden_window_buf = pld.alloc_window_buffer([MARKOV_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    markov_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    markov_logits_window_buf = pld.alloc_window_buffer([MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.FP32)
    markov_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        target_fwd_recv_meta = pld.window(target_fwd_recv_meta_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS], dtype=pl.INT32)
        target_fwd_recv_x = pld.window(target_fwd_recv_x_buffer, [TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, D], dtype=pl.INT8)
        target_fwd_recv_scale = pld.window(target_fwd_recv_scale_buffer, [TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, D // TARGET_FWD_MX_GROUP], dtype=pl.UINT8)
        target_fwd_recv_weights = pld.window(target_fwd_recv_weights_buffer, [TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_AUX_WIDTH], dtype=pl.FP32)
        target_fwd_recv_routes = pld.window(target_fwd_recv_routes_buffer, [TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_ROUTE_WIDTH], dtype=pl.INT32)
        target_fwd_data_arrived = pld.window(target_fwd_data_arrived_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
        target_fwd_routed_output = pld.window(target_fwd_routed_output_buffer, [TARGET_FWD_MOE_TOKENS * TARGET_FWD_TOPK, D], dtype=pl.BF16)
        target_fwd_combine_arrived = pld.window(target_fwd_combine_arrived_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
        target_fwd_head_hidden = pld.window(target_fwd_head_hidden_buffer, [TARGET_FWD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        target_fwd_head_hidden_done = pld.window(target_fwd_head_hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        target_fwd_head_logits = pld.window(target_fwd_head_logits_buffer, [TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], dtype=pl.FP32)
        target_fwd_head_logits_done = pld.window(target_fwd_head_logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        sampling_hidden_window = pld.window(sample_hidden_buffer, [TARGET_FWD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        sampling_hidden_done = pld.window(sample_hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        sampling_logits_window = pld.window(sample_logits_buffer, [TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], dtype=pl.FP32)
        sampling_logits_done = pld.window(sample_logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        draft_recv_meta = pld.window(draft_recv_meta_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS], dtype=pl.INT32)
        draft_recv_x = pld.window(draft_recv_x_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, D], dtype=pl.INT8)
        draft_recv_scale = pld.window(draft_recv_scale_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, D // DRAFT_MX_GROUP], dtype=pl.UINT8)
        draft_recv_weights = pld.window(draft_recv_weights_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_AUX_WIDTH], dtype=pl.FP32)
        draft_recv_routes = pld.window(draft_recv_routes_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], dtype=pl.INT32)
        draft_data_arrived = pld.window(draft_data_arrived_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
        draft_routed_output = pld.window(draft_routed_output_buf, [DRAFT_MOE_TOKENS * DRAFT_TOPK, D], dtype=pl.BF16)
        draft_combine_arrived = pld.window(draft_combine_arrived_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
        markov_hidden_window = pld.window(markov_hidden_window_buf, [MARKOV_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        markov_hidden_done = pld.window(markov_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        markov_logits_window = pld.window(markov_logits_window_buf, [MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.FP32)
        markov_logits_done = pld.window(markov_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_decode_fwd_dspark(
            slot_ids[rank], generations[rank], state_tokens[rank], state_meta[rank],
            input_ids[rank], target_positions[rank], active_widths[rank], target_embedding_weight[rank],
            target_window_table[rank], target_c2a_table[rank], target_c1a_table[rank], x_hc[rank],
            pre_mix[rank], position_ids[rank], request_ids[rank], window_slots[rank],
            window_indices[rank], c2a_slots[rank], c1a_slots[rank], query_starts[rank],
            engram_lookup[rank], engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            target_hidden_padded[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank], wkv[rank], wkv_scale[rank],
            kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank],
            wo_b[rank], wo_b_scale[rank], freqs_cos[rank], freqs_sin[rank],
            window_cache_pool[rank], compressed_cache_pool[rank], index_cache_pool[rank], index_cache_scale_pool[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank], c2a_compressor_wkv[rank], c2a_compressor_wgate[rank],
            c1a_compressor_wkv[rank], state_block_table[rank], state_cache_pool[rank], compressor_norm_weight[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale[rank],
            index_weights_proj[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank], routed_w1[rank],
            routed_w1_scale[rank], routed_w2[rank], routed_w2_scale[rank], routed_w3[rank],
            routed_w3_scale[rank], shared_w1[rank], shared_w1_scale[rank], shared_w2[rank],
            shared_w2_scale[rank], shared_w3[rank], shared_w3_scale[rank], moe_num_tokens,
            attention_num_tokens, final_norm_weight[rank], lm_head_weight[rank], logit_row_indices[rank],
            target_logits[rank], target_backbone_sampled_ids[rank], target_sampled_ids[rank], accepted_ids[rank],
            accepted_counts[rank], context_valid[rank], context_positions[rank], num_sampled[rank],
            last_sampled[rank], anchor_ids[rank], first_query_positions[rank], target_hidden[rank],
            draft_table[rank], draft_cos[rank], draft_sin[rank], context_slots[rank],
            context_rope_cos[rank], context_rope_sin[rank], query_rope_cos[rank], query_rope_sin[rank],
            draft_main_proj_weight[rank], draft_main_norm_weight[rank], zero_prefill_tokens[rank], draft_embedding_weight[rank],
            draft_hc_attn_fn[rank], draft_hc_attn_scale[rank], draft_hc_attn_base[rank], draft_attn_norm_weight[rank],
            draft_wq_a[rank], draft_wq_a_scale[rank], draft_q_norm_weight[rank], draft_wq_b[rank],
            draft_wq_b_scale[rank], draft_wkv[rank], draft_wkv_scale[rank], draft_kv_norm_weight[rank],
            draft_attn_sink[rank], draft_wo_a[rank], draft_wo_a_scale[rank], draft_wo_b[rank],
            draft_wo_b_scale[rank], draft_window_cache[rank], draft_hc_ffn_fn[rank], draft_hc_ffn_scale[rank],
            draft_hc_ffn_base[rank], draft_ffn_norm_weight[rank], draft_gate_weight[rank], draft_correction_bias[rank],
            draft_routed_w1[rank], draft_routed_w1_scale[rank], draft_routed_w2[rank], draft_routed_w2_scale[rank],
            draft_routed_w3[rank], draft_routed_w3_scale[rank], draft_shared_w1[rank], draft_shared_w1_scale[rank],
            draft_shared_w2[rank], draft_shared_w2_scale[rank], draft_shared_w3[rank], draft_shared_w3_scale[rank],
            draft_head_hidden[rank], moe_epoch, markov_final_norm_weight[rank], markov_lm_head_weight[rank],
            markov_markov_w1[rank], markov_markov_w2[rank], markov_confidence_weight[rank], draft_ids[rank],
            confidences[rank], logits_epoch, committed[rank], target_fwd_recv_meta,
            target_fwd_recv_x, target_fwd_recv_scale, target_fwd_recv_weights, target_fwd_recv_routes,
            target_fwd_data_arrived, target_fwd_routed_output, target_fwd_combine_arrived, target_fwd_head_hidden,
            target_fwd_head_hidden_done, target_fwd_head_logits, target_fwd_head_logits_done, draft_recv_meta,
            draft_recv_x, draft_recv_scale, draft_recv_weights, draft_recv_routes,
            draft_data_arrived, draft_routed_output, draft_combine_arrived, markov_hidden_window,
            markov_hidden_done, markov_logits_window, markov_logits_done, sampling_hidden_window,
            sampling_hidden_done, sampling_logits_window, sampling_logits_done, rank,
            device=rank,
        )


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
    moe_epoch: int,
    logits_epoch: int,
) -> None:
    """Bind resident tensors for one recurrent decode dispatch."""
    arguments = {
        **target_model,
        **{f"draft_{name}": tensor for name, tensor in draft_model.items()},
        **{f"markov_{name}": tensor for name, tensor in markov_model.items()},
        **workspace,
        "slot_ids": state.slot_ids,
        "generations": state.generations,
        "state_tokens": state.tokens,
        "state_meta": state.meta,
        "target_embedding_weight": target_model["embedding_weight"],
        "target_logits": target_model["logits"],
        "target_backbone_sampled_ids": target_model["sampled_ids"],
        "target_window_table": cache_tables["target_window"],
        "target_c2a_table": cache_tables["target_c2a"],
        "target_c1a_table": cache_tables["target_c1a"],
        "draft_table": cache_tables["draft"],
        "draft_cos": rope_tables["draft_cos"],
        "draft_sin": rope_tables["draft_sin"],
        "moe_epoch": moe_epoch,
        "logits_epoch": logits_epoch,
    }
    _run(runner, l3_decode_fwd_dspark, {
        name: arguments[name] for name in l3_decode_fwd_dspark.param_names
    })


def prefill_bootstrap_dspark(
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
        moe_epoch=moe_epoch,
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


def run_recurrent_decode_case(*, devices: tuple[int, ...], compile_only: bool = False) -> None:
    """Validate rejection and full acceptance across two resident decode rounds."""
    import inspect
    import torch

    from golden import TensorSpec, normalize_config
    from pypto.ir import DistributedConfig
    from utils import upload_forward_inputs

    limit = 64
    extents = {
        VOCAB_DYN.name: 32,
        WINDOW_TABLE_DYN.name: 1,
        state_ops.TABLE_DYN.name: 1,
        DRAFT_ORI_BLOCKS_DYN.name: DECODE_BATCH,
        TARGET_FWD_ROPE_ROWS_DYN.name: limit,
        STATE_OPS_ROPE_ROWS_DYN.name: limit,
    }
    entry = l3_decode_fwd_dspark
    target_specs = {s.name: s for s in target_fwd.build_tensor_specs()}
    table = torch.arange(DECODE_BATCH, dtype=torch.int32).reshape(1, DECODE_BATCH, 1).repeat(EP_SIZE, 1, 1)
    target_cos, target_sin = target_fwd.prepare_forward_rope_tables(limit)
    initial = {
        "slot_ids": -1,
        "target_window_table": table,
        "target_c2a_table": target_specs["c2a_index_block_table"].init_value,
        "target_c1a_table": target_specs["c1a_index_block_table"].init_value,
        "draft_table": table[:, None].repeat(1, DSPARK_DRAFT_LAYERS, 1, 1),
        "freqs_cos": target_cos.unsqueeze(0).repeat(EP_SIZE, 1, 1, 1),
        "freqs_sin": target_sin.unsqueeze(0).repeat(EP_SIZE, 1, 1, 1),
        "draft_cos": 1.0,
        "draft_main_norm_weight": 1.0,
        "markov_final_norm_weight": 1.0,
        "logit_row_indices": -1,
        "attention_num_tokens": ROWS,
        "moe_num_tokens": ROWS,
    }
    aliases = {
        "markov_lm_head_weight": "lm_head_weight",
    }
    specs = {}
    scalars = {}
    for name, param in inspect.signature(entry._func).parameters.items():
        ann = param.annotation
        if not hasattr(ann, "shape"):
            scalars[name] = 1
            continue
        if name in aliases:
            continue
        shape = [
            extents[d.name] if hasattr(d, "name") and d.name in extents else target_fwd._fixture_extent(d)
            for d in ann.shape
        ]
        dtype = target_fwd._torch_dtype(ann.dtype)
        base = target_specs.get(name)
        value = initial.get(name, base.init_value if base is not None else None)
        if name == "draft_window_cache":
            value = torch.zeros(shape, dtype=torch.uint8)
            _, factors = unpack_quantized_cache(value, False)
            factors.fill_(1.0)
        if value is None and dtype == torch.float8_e8m0fnu:
            value = 1.0
        specs[name] = TensorSpec(name, shape, dtype, init_value=value)

    for name in ("slot_ids", "generations", "state_meta", "state_tokens"):
        specs[name].init_value = specs[name].create_tensor()
    specs["slot_ids"].init_value[:, 0] = 0
    specs["generations"].init_value[:, 0] = 1
    meta = specs["state_meta"].init_value[:, 0]
    meta[:, state_ops.STATE_VALID] = 1
    meta[:, state_ops.STATE_GENERATION] = 1
    meta[:, state_ops.STATE_ANCHOR] = 2
    meta[:, state_ops.STATE_COMMITTED] = 2
    meta[:, state_ops.STATE_DRAFT_COUNT] = ROWS - 1
    meta[:, state_ops.STATE_LIMIT] = limit
    specs["state_tokens"].init_value[:, 0, 1] = 1

    cfg = normalize_config(dict(
        backend_type="a5", execution_mode=False,
        distributed_config=DistributedConfig(device_ids=list(devices), num_sub_workers=0),
        ring_heap=(2 * 1024**3, 2 * 1024**3, 1024**3, 512 * 1024**2), ring_dep_pool=131072,
    ))
    compiled = entry.compile(config=cfg, **dict.fromkeys(scalars, pl.RUNTIME))
    if compile_only:
        return

    replicated = ("moe_num_tokens", "attention_num_tokens")
    host_inputs = {name: specs[name].create_tensor().share_memory_() for name in replicated}
    workspace_names = (
        "x_hc", "pre_mix", "position_ids", "request_ids", "window_slots",
        "window_indices", "c2a_slots", "c1a_slots", "query_starts", "target_hidden_padded",
    )
    output_names = ("accepted_counts", "committed", "state_meta", "draft_ids", *workspace_names)
    outputs = {name: specs[name].create_tensor().share_memory_() for name in output_names}
    upload = torch.empty(16 * 1024**2, dtype=torch.uint8).share_memory_()
    inherited = [upload, *host_inputs.values(), *outputs.values()]
    with compiled.prepare(cfg, persistent=True, reset_persistent_windows=False,
                          inherited_host_tensors=inherited) as runtime:
        params = dict(zip(entry.param_names, compiled._get_metadata()[0], strict=True))
        resident = dict(zip(specs, upload_forward_inputs(
            runtime, list(specs.values()), [params[name] for name in specs], EP_SIZE, host_inputs, upload,
        ), strict=True))
        resident.update((name, resident[source]) for name, source in aliases.items())
        resident.update(scalars)
        anchor = 2
        for round_index, accepted in enumerate((1, ROWS)):
            resident["moe_epoch"] = 1 + round_index * DSPARK_DRAFT_LAYERS
            resident["logits_epoch"] = 1 + round_index
            for name in workspace_names:
                fill = float("nan") if specs[name].dtype.is_floating_point else -777
                for rank, handle in enumerate(resident[name].shards):
                    upload[:handle.nbytes].view(specs[name].dtype).fill_(fill)
                    runtime.copy_to(handle.data_ptr, upload.data_ptr(), handle.nbytes, worker_id=rank)
            runtime(*(resident[name] for name in entry.param_names), config=cfg)
            for name, tensor in outputs.items():
                runtime.copy_stacked_from(resident[name], tensor)
            for name in workspace_names:
                actual = outputs[name]
                if actual.dtype.is_floating_point:
                    if not torch.isfinite(actual).all().item():
                        raise RuntimeError(f"unpublished DSpark workspace {name} in round {round_index}")
                elif torch.any(actual == -777).item():
                    raise RuntimeError(f"unpublished DSpark metadata {name} in round {round_index}")
            expected_positions = torch.full_like(outputs["position_ids"], -1)
            expected_positions[:, :ROWS] = torch.arange(anchor, anchor + ROWS, dtype=torch.int32)
            if not torch.equal(outputs["position_ids"], expected_positions):
                raise RuntimeError(f"unexpected DSpark target positions in round {round_index}")
            expected_requests = torch.full_like(outputs["request_ids"], -1)
            expected_requests[:, :ROWS] = 0
            if not torch.equal(outputs["request_ids"], expected_requests):
                raise RuntimeError(f"unexpected DSpark request mapping in round {round_index}")
            expected_starts = torch.arange(DECODE_BATCH + 1, dtype=torch.int32) * ROWS
            if not torch.all(outputs["query_starts"] == expected_starts).item():
                raise RuntimeError(f"unexpected DSpark query starts in round {round_index}")
            # The initial proposal 1 is rejected; the next five proposals are 0.
            expected = torch.zeros(EP_SIZE, DECODE_BATCH, dtype=torch.int32)
            expected[:, 0] = accepted
            if not torch.equal(outputs["accepted_counts"], expected):
                raise RuntimeError(f"unexpected DSpark acceptance in round {round_index}")
            expected[:, 0] = 1
            if not torch.equal(outputs["committed"], expected):
                raise RuntimeError(f"unexpected DSpark publication in round {round_index}")
            anchor += accepted
            state = outputs["state_meta"][:, 0]
            for column, value in (
                (state_ops.STATE_ANCHOR, anchor),
                (state_ops.STATE_COMMITTED, anchor),
                (state_ops.STATE_DRAFT_COUNT, ROWS - 1),
                (state_ops.STATE_GENERATION, 1),
                (state_ops.STATE_VALID, 1),
            ):
                if not torch.all(state[:, column] == value):
                    raise RuntimeError(f"unexpected DSpark state column {column} in round {round_index}")
            if torch.count_nonzero(outputs["draft_ids"][:, 0]).item():
                raise RuntimeError(f"unexpected DSpark proposals in round {round_index}")
    print("[DSpark] Fused recurrent decode smoke passed.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--lm-head-tp", type=int, default=LM_HEAD_TP_SIZE, choices=SUPPORTED_LM_HEAD_TP_SIZES)
    parser.add_argument("--ep", type=int, default=EP_SIZE)
    parser.add_argument("--dp", type=int, default=EP_SIZE)
    parser.add_argument("--weight-bank-size", type=int, choices=[1], default=1)
    parser.add_argument("--experts-per-rank", type=int, choices=[16], default=16)
    parser.add_argument("--case", choices=["decode", "metadata"], default="decode")
    parser.add_argument("-d", "--device", default=",".join(map(str, range(EP_SIZE))))
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if (args.ep, args.dp) != (EP_SIZE, EP_SIZE):
        parser.error("--ep/--dp must match the import-time DP=EP configuration")
    devices = tuple(int(value) for value in args.device.split(","))
    if args.case == "decode":
        if args.execution_mode:
            parser.error("fused recurrent decode smoke requires A5 hardware")
        if len(devices) != EP_SIZE or len(set(devices)) != EP_SIZE or min(devices) < 0:
            parser.error(f"need {EP_SIZE} distinct non-negative device IDs")
        run_recurrent_decode_case(devices=devices, compile_only=args.compile_only)
        raise SystemExit(0)
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
