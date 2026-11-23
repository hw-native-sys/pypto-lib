# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill MoE execution geometry, routing metadata, and peer exchanges."""

from .config import FLASH as M, _parse_int_argv
import pypto.language as pl
import pypto.language.distributed as pld


# Model configuration.
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
N_EXPERTS = M.n_routed_experts
N_BANK = N_EXPERTS + 1
SHARED_EID = N_EXPERTS
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size
N_RANKS = _parse_int_argv("--tp", 8)
if N_RANKS != 8:
    raise ValueError("Prefill MoE requires TP8")
MOE_INTER = M.moe_intermediate_size // N_RANKS
SWIGLU_LIMIT = M.swiglu_limit

# Prefill plan.
T = _parse_int_argv("--prefill-moe-wave-tokens", 2048)
if T not in (8, 32, 64, 128, 1024, 2048, 8192):
    raise ValueError("Prefill MoE wave tokens must be 8, 32, 64, 128, 1024, 2048 or 8192")
if MOE_INTER != 256:
    raise ValueError("Prefill MoE requires TP8 intermediate shards of 256 columns")

PREFILL_MOE_OWNER_REDUCE = _parse_int_argv("--prefill-moe-owner-reduce", 0)
if PREFILL_MOE_OWNER_REDUCE not in (0, 1):
    raise ValueError("Prefill owner reduction must be zero or one")

PREFILL_MOE_EXPERT_AFFINITY = _parse_int_argv("--prefill-moe-expert-affinity", 0)
if PREFILL_MOE_EXPERT_AFFINITY not in (0, 1):
    raise ValueError("Prefill expert affinity must be zero or one")
PREFILL_MOE_BATCHED_GATHER = _parse_int_argv("--prefill-moe-batched-gather", 0)
if PREFILL_MOE_BATCHED_GATHER not in (0, 1):
    raise ValueError("Prefill batched gather must be zero or one")
PREFILL_MOE_STAGE_VIEW = _parse_int_argv("--prefill-moe-stage-view", 0)
if PREFILL_MOE_STAGE_VIEW not in (0, 1):
    raise ValueError("Prefill MoE stage view must be zero or one")
PREFILL_MOE_DIRECT_POST = _parse_int_argv("--prefill-moe-direct-post", 0)
if PREFILL_MOE_DIRECT_POST not in (0, 1):
    raise ValueError("Prefill direct post must be zero or one")
if PREFILL_MOE_DIRECT_POST and not PREFILL_MOE_OWNER_REDUCE:
    raise ValueError("Prefill direct post requires owner reduction")
PREFILL_MOE_FUSED_NORM_QUANT = _parse_int_argv("--prefill-moe-fused-norm-quant", 0)
if PREFILL_MOE_FUSED_NORM_QUANT not in (0, 1):
    raise ValueError("Prefill fused norm quant must be zero or one")
PREFILL_MOE_COMBINE_ROWS = _parse_int_argv("--prefill-moe-combine-rows", 1)
if PREFILL_MOE_COMBINE_ROWS not in (1, 4, 8):
    raise ValueError("Prefill combine rows must be 1, 4 or 8")
PREFILL_MOE_QUANT_GRID = _parse_int_argv("--prefill-moe-quant-grid", 0)
if PREFILL_MOE_QUANT_GRID not in (0, 1):
    raise ValueError("Quantization grid must be zero or one")
if PREFILL_MOE_QUANT_GRID and PREFILL_MOE_FUSED_NORM_QUANT:
    raise ValueError("Quantization grid requires separate original quantization")

PREFILL_MOE_OWNER_PREPARE = _parse_int_argv("--prefill-moe-owner-prepare", 0)
if PREFILL_MOE_OWNER_PREPARE not in (0, 1):
    raise ValueError("Owner preparation must be zero or one")
if PREFILL_MOE_OWNER_PREPARE and (T < 128 or T % 128):
    raise ValueError("Owner preparation requires waves divisible by128")

PREFILL_MOE_GATE_COPY_ROWS = _parse_int_argv("--prefill-moe-gate-copy-rows", 16)
if PREFILL_MOE_GATE_COPY_ROWS not in (16, 64, 128):
    raise ValueError("Prefill gate copy rows must be16,64 or128")

PREFILL_MOE_SHARE_GATE = _parse_int_argv("--prefill-moe-share-gate", 0)
if PREFILL_MOE_SHARE_GATE not in (0, 1):
    raise ValueError("Prefill gate sharing must be zero or one")
if PREFILL_MOE_OWNER_PREPARE and (PREFILL_MOE_SHARE_GATE or not PREFILL_MOE_QUANT_GRID or PREFILL_MOE_FUSED_NORM_QUANT):
    raise ValueError("Owner preparation requires unshared gate and grouped original quantization")

PREFILL_MOE_LOCAL_ROUTE_STATE = _parse_int_argv("--prefill-moe-local-route-state", 0)
if PREFILL_MOE_LOCAL_ROUTE_STATE not in (0, 1):
    raise ValueError("Prefill local route state must be zero or one")
EXPERT_IDS_PAD = (N_BANK + 7) // 8 * 8

# Expert and transport tiling.
RECV_TILE = 16
GATHER_TOKEN_TILE = 8
SHARED_GATHER_T_TILE = 16
RECV_MAX = RECV_TILE
IDX_PAD = 8
N_ROUTES = T * TOPK
N_EXPERTS_LIVE = min(N_EXPERTS, N_ROUTES)
N_SLOTS = N_EXPERTS_LIVE + (N_ROUTES - N_EXPERTS_LIVE) // RECV_TILE
SH_SLOT = N_SLOTS
SHARED_TILES = (T + RECV_TILE - 1) // RECV_TILE
N_SLOTS_B = N_SLOTS + SHARED_TILES
WORK_SLOTS_PAD = (N_SLOTS_B + IDX_PAD - 1) // IDX_PAD * IDX_PAD
T_PAD = (T + RECV_TILE - 1) // RECV_TILE * RECV_TILE
OWNER_ROWS = T // N_RANKS
OWNER_REDUCE_T_TILE = min(8, OWNER_ROWS)
OWNER_COPY_T_TILE = min(4, OWNER_ROWS)
RESULT_WINDOW_ROWS = 2 * T_PAD
REDUCE_LANE_ROWS = T_PAD if PREFILL_MOE_OWNER_REDUCE else N_RANKS * T_PAD
REDUCE_WINDOW_ROWS = 2 * REDUCE_LANE_ROWS
COPY_T_TILE = 4
COPY_D_TILE = 2048
REDUCE_T_TILE = 8
REDUCE_D_TILE = 512

GATE_WINDOW_ROWS = 2 * T_PAD

OWNER_WINDOW_ROWS = 2 * T_PAD
PREP_OWNER_ROWS = T // N_RANKS

PREFILL_MOE_PARALLEL_ROUTE = _parse_int_argv("--prefill-moe-parallel-route", 0)
if PREFILL_MOE_PARALLEL_ROUTE not in (0, 1):
    raise ValueError("Parallel routing must be zero or one")


GATE_COPY_TILE = min(PREFILL_MOE_GATE_COPY_ROWS, T_PAD)
GATE_COLUMNS = N_EXPERTS // N_RANKS


@pl.jit.inline(auto_scope=False)
def exchange_gate_columns(
    logits: pl.Tensor[[T_PAD, N_EXPERTS], pl.FP32],
    window: pld.DistributedTensor[[GATE_WINDOW_ROWS, N_EXPERTS], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    active: pl.Scalar[pl.INDEX],
    computed: pl.Scalar[pl.TASK_ID],
    prior_copy: pl.Scalar[pl.TASK_ID],
):
    lane_base = pl.cast((epoch - 1) % 2, pl.INDEX) * T_PAD
    column = pl.cast(rank, pl.INDEX) * GATE_COLUMNS
    tasks = pl.array.create(3, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(N_RANKS, name_hint="prefill_gate_publish", deps=[computed, prior_copy]) as published:
            peer = pl.tile.get_block_idx()
            pld.tensor.put(dst=window, peer=peer, src=logits, dst_offsets=[lane_base, column], src_offsets=[0, column], shape=[active, GATE_COLUMNS], chunk_rows=GATE_COPY_TILE, chunk_cols=GATE_COLUMNS)
            pld.system.notify(signal, peer, [rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        tasks[0] = published
    with pl.manual_scope():
        published_scalar = tasks[0]
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_gate_defer", deps=[published_scalar]) as waited:
            for source in pl.range(N_RANKS):
                pld.system.defer_wait(signal, [source, 0], epoch, cmp=pld.WaitCmp.Ge)
        tasks[1] = waited
    with pl.manual_scope():
        waited_scalar = tasks[1]
        blocks = (active + GATE_COPY_TILE - 1) // GATE_COPY_TILE
        with pl.spmd(blocks, name_hint="prefill_gate_collect", deps=[waited_scalar]) as copied:
            start = pl.tile.get_block_idx() * GATE_COPY_TILE
            rows = pl.min(GATE_COPY_TILE, active - start)
            value = pl.load(window, [lane_base + start, 0], [GATE_COPY_TILE, N_EXPERTS])
            valid = pl.set_validshape(value, rows, N_EXPERTS)
            pl.store(valid, [start, 0], logits)
        tasks[2] = copied
    return tasks[2]


@pl.jit.inline(auto_scope=False)
def retire_gate_reservation(
    signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    unused: pl.Scalar[pl.INT32],
    completed: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(N_RANKS, name_hint="prefill_gate_epoch_padding", deps=[completed]) as retired:
        peer = pl.tile.get_block_idx()
        pld.system.notify(signal, peer, [rank, 0], value=unused, op=pld.NotifyOp.AtomicAdd)
    return retired


OWNER_COMB_D_TILE = HC_MULT * HC_MULT
OWNER_META_T_TILE = min(64, PREP_OWNER_ROWS)
OWNER_SCALE_D_TILE = min(256, PREP_OWNER_ROWS)
OWNER_POST_D_TILE = min(256, PREP_OWNER_ROWS * HC_MULT)


@pl.jit.inline(auto_scope=False)
def exchange_preparation(
    source_norm: pl.Tensor[[PREP_OWNER_ROWS, D], pl.INT8],
    source_scale: pl.Tensor[[PREP_OWNER_ROWS, 1], pl.FP32],
    source_post: pl.Tensor[[PREP_OWNER_ROWS, HC_MULT], pl.FP32],
    source_comb: pl.Tensor[[PREP_OWNER_ROWS, HC_MULT * HC_MULT], pl.FP32],
    source_indices: pl.Tensor[[PREP_OWNER_ROWS, 16], pl.INT32],
    source_weights: pl.Tensor[[PREP_OWNER_ROWS, 16], pl.FP32],
    norm: pl.Tensor[[T, D], pl.INT8],
    scale: pl.Tensor[[T, 1], pl.FP32],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32],
    indices: pl.Tensor[[T, 16], pl.INT32],
    weights: pl.Tensor[[T, 16], pl.FP32],
    norm_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, D], pl.INT8],
    scale_window: pld.DistributedTensor[[1, OWNER_WINDOW_ROWS], pl.FP32],
    post_window: pld.DistributedTensor[[1, OWNER_WINDOW_ROWS * HC_MULT], pl.FP32],
    comb_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, HC_MULT * HC_MULT], pl.FP32],
    indices_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, 16], pl.INT32],
    weights_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, 16], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    fields_ready: pl.Scalar[pl.TASK_ID],
    previous_copy: pl.Scalar[pl.TASK_ID],
):
    lane_base = pl.cast((epoch - 1) % 2, pl.INDEX) * T_PAD
    owner_base = pl.cast(rank, pl.INDEX) * PREP_OWNER_ROWS
    source_scale_flat = pl.reshape(source_scale, [1, PREP_OWNER_ROWS])
    source_post_flat = pl.reshape(source_post, [1, PREP_OWNER_ROWS * HC_MULT])
    scale_flat = pl.reshape(scale, [1, T])
    post_flat = pl.reshape(post, [1, T * HC_MULT])
    tasks = pl.array.create(3, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(N_RANKS, name_hint="prefill_prepare_owner_publish", deps=[fields_ready, previous_copy]) as published:
            peer = pl.tile.get_block_idx()
            row = lane_base + owner_base
            pld.tensor.put(dst=norm_window, peer=peer, src=source_norm, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[PREP_OWNER_ROWS, D], chunk_rows=16, chunk_cols=D)
            pld.tensor.put(dst=scale_window, peer=peer, src=source_scale_flat, dst_offsets=[0, row], src_offsets=[0, 0], shape=[1, PREP_OWNER_ROWS], chunk_rows=1, chunk_cols=OWNER_SCALE_D_TILE)
            pld.tensor.put(dst=post_window, peer=peer, src=source_post_flat, dst_offsets=[0, row * HC_MULT], src_offsets=[0, 0], shape=[1, PREP_OWNER_ROWS * HC_MULT], chunk_rows=1, chunk_cols=OWNER_POST_D_TILE)
            pld.tensor.put(dst=comb_window, peer=peer, src=source_comb, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[PREP_OWNER_ROWS, HC_MULT * HC_MULT], chunk_rows=OWNER_META_T_TILE, chunk_cols=OWNER_COMB_D_TILE)
            pld.tensor.put(dst=indices_window, peer=peer, src=source_indices, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[PREP_OWNER_ROWS, 16], chunk_rows=OWNER_META_T_TILE, chunk_cols=16)
            pld.tensor.put(dst=weights_window, peer=peer, src=source_weights, dst_offsets=[row, 0], src_offsets=[0, 0], shape=[PREP_OWNER_ROWS, 16], chunk_rows=OWNER_META_T_TILE, chunk_cols=16)
            pld.system.notify(signal, peer, [rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        tasks[0] = published
    with pl.manual_scope():
        published_scalar = tasks[0]
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_prepare_owner_defer", deps=[published_scalar]) as waited:
            for source in pl.range(N_RANKS):
                pld.system.defer_wait(signal, [source, 0], epoch, cmp=pld.WaitCmp.Ge)
        tasks[1] = waited
    with pl.manual_scope():
        waited_scalar = tasks[1]
        with pl.spmd(T // 16, name_hint="prefill_prepare_owner_collect", deps=[waited_scalar]) as copied:
            start = pl.tile.get_block_idx() * 16
            norm_tile = pl.load(norm_window, [lane_base + start, 0], [16, D])
            pl.store(norm_tile, [start, 0], norm)
            scale_tile = pl.load(scale_window, [0, lane_base + start], [1, 16])
            pl.store(scale_tile, [0, start], scale_flat)
            post_tile = pl.load(post_window, [0, (lane_base + start) * HC_MULT], [1, 16 * HC_MULT])
            pl.store(post_tile, [0, start * HC_MULT], post_flat)
            comb_tile = pl.load(comb_window, [lane_base + start, 0], [16, HC_MULT * HC_MULT])
            pl.store(comb_tile, [start, 0], comb)
            indices_tile = pl.load(indices_window, [lane_base + start, 0], [16, 16])
            pl.store(indices_tile, [start, 0], indices)
            weights_tile = pl.load(weights_window, [lane_base + start, 0], [16, 16])
            pl.store(weights_tile, [start, 0], weights)
        tasks[2] = copied
    return tasks[2]


ROUTE_BLOCK = 384
TOKEN_BLOCK = ROUTE_BLOCK // TOPK
ROUTE_BLOCKS = (N_ROUTES + ROUTE_BLOCK - 1) // ROUTE_BLOCK
ROUTE_PAD = ROUTE_BLOCKS * ROUTE_BLOCK
EXPERT_TILE = 16
SLOT_TILE = 16


@pl.jit.inline(auto_scope=False)
def parallel_route_metadata(
    indices: pl.Tensor[[T, 16], pl.INT32],
    weights: pl.Tensor[[T, 16], pl.FP32],
    scales: pl.Tensor[[T, 1], pl.FP32],
    recv_scale: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_weight: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    active_tokens: pl.Scalar[pl.INDEX],
    initialized: pl.Scalar[pl.TASK_ID],
    prepared: pl.Scalar[pl.TASK_ID],
):
    histogram = pl.create_tensor([ROUTE_BLOCKS, N_EXPERTS], dtype=pl.INT32)
    prefix = pl.create_tensor([ROUTE_BLOCKS, N_EXPERTS], dtype=pl.INT32)
    totals = pl.create_tensor([1, N_EXPERTS], dtype=pl.INT32)
    local_order = pl.create_tensor([1, ROUTE_PAD], dtype=pl.INT32)
    global_order = pl.create_tensor([1, ROUTE_PAD], dtype=pl.INT32)
    start_counts = pl.create_tensor([ROUTE_BLOCKS, 16], dtype=pl.INT32)
    block_bases = pl.create_tensor([ROUTE_BLOCKS, 16], dtype=pl.INT32)
    expert_bases = pl.create_tensor([1, N_EXPERTS], dtype=pl.INT32)
    total_slots = pl.create_tensor([16], dtype=pl.INT32)
    tile_slots = pl.create_tensor([N_SLOTS, 16], dtype=pl.INT32)
    slot_info = pl.create_tensor([N_SLOTS, 16], dtype=pl.INT32)
    slot_routes = pl.create_tensor([N_SLOTS * RECV_MAX, 16], dtype=pl.INT32)
    active_routes = active_tokens * TOPK

    with pl.spmd(ROUTE_BLOCKS, name_hint="route_parallel_histogram", deps=[initialized, prepared]) as histogram_done:
        p0_block = pl.tile.get_block_idx()
        p0_route_begin = p0_block * ROUTE_BLOCK
        p0_token_begin = p0_block * TOKEN_BLOCK
        p0_valid_tokens = pl.min(TOKEN_BLOCK, T - p0_token_begin)
        p0_ids = pl.slice(indices, [TOKEN_BLOCK, 16], [p0_token_begin, 0], valid_shape=[p0_valid_tokens, 16])
        p0_counts = pl.tile.full([1, N_EXPERTS], dtype=pl.INT32, value=0)
        p0_orders = pl.tile.full([1, ROUTE_BLOCK], dtype=pl.INT32, value=0)
        p0_route_end = pl.min(p0_route_begin + ROUTE_BLOCK, active_routes)
        for p0_route in pl.range(p0_route_begin, p0_route_end):
            p0_token_in_block = (p0_route - p0_route_begin) // TOPK
            p0_expert = pl.cast(pl.read(p0_ids, [p0_token_in_block, p0_route % TOPK]), pl.INDEX)
            p0_ordinal = pl.read(p0_counts, [0, p0_expert])
            pl.write(p0_orders, [0, p0_route - p0_route_begin], p0_ordinal)
            pl.write(p0_counts, [0, p0_expert], pl.cast(p0_ordinal + 1, pl.INT32))
        pl.store(p0_counts, [p0_block, 0], histogram)
        pl.store(p0_orders, [0, p0_route_begin], local_order)

    with pl.spmd(N_EXPERTS // EXPERT_TILE, name_hint="route_parallel_expert_prefix", deps=[histogram_done]) as prefix_done:
        p1_expert_begin = pl.tile.get_block_idx() * EXPERT_TILE
        p1_cumulative = pl.tile.full([1, EXPERT_TILE], dtype=pl.INT32, value=0)
        p1_prefixes = pl.tile.full([ROUTE_BLOCKS, EXPERT_TILE], dtype=pl.INT32, value=0)
        for p1_block_index in pl.range(ROUTE_BLOCKS):
            for p1_expert_index in pl.range(EXPERT_TILE):
                pl.write(p1_prefixes, [p1_block_index, p1_expert_index], pl.read(p1_cumulative, [0, p1_expert_index]))
            p1_group_count = pl.load(histogram, [p1_block_index, p1_expert_begin], [1, EXPERT_TILE])
            p1_cumulative = pl.add(p1_cumulative, p1_group_count)
        pl.store(p1_prefixes, [0, p1_expert_begin], prefix)
        pl.store(p1_cumulative, [0, p1_expert_begin], totals)

    with pl.spmd(ROUTE_BLOCKS, name_hint="route_parallel_ordinals", deps=[prefix_done]) as ordinals_done:
        p2_block = pl.tile.get_block_idx()
        p2_begin = p2_block * ROUTE_BLOCK
        p2_token_begin = p2_block * TOKEN_BLOCK
        p2_valid_tokens = pl.min(TOKEN_BLOCK, T - p2_token_begin)
        p2_ids = pl.slice(indices, [TOKEN_BLOCK, 16], [p2_token_begin, 0], valid_shape=[p2_valid_tokens, 16])
        p2_offsets = pl.load(prefix, [p2_block, 0], [1, N_EXPERTS])
        p2_local = pl.load(local_order, [0, p2_begin], [1, ROUTE_BLOCK])
        p2_output = pl.tile.full([1, ROUTE_BLOCK], dtype=pl.INT32, value=0)
        p2_starts = pl.const(0, pl.INT32)
        p2_end = pl.min(p2_begin + ROUTE_BLOCK, active_routes)
        for p2_route in pl.range(p2_begin, p2_end):
            p2_expert = pl.cast(pl.read(p2_ids, [(p2_route - p2_begin) // TOPK, p2_route % TOPK]), pl.INDEX)
            p2_ordinal = pl.read(p2_local, [0, p2_route - p2_begin]) + pl.read(p2_offsets, [0, p2_expert])
            pl.write(p2_output, [0, p2_route - p2_begin], pl.cast(p2_ordinal, pl.INT32))
            if p2_ordinal % RECV_MAX == 0:
                p2_starts = p2_starts + pl.const(1, pl.INT32)
        pl.store(p2_output, [0, p2_begin], global_order)
        pl.write(start_counts, [p2_block, 0], p2_starts)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_parallel_bases", deps=[ordinals_done]) as bases_done:
        p3_count_tile = pl.load(totals, [0, 0], [1, N_EXPERTS])
        p3_tile_bases = pl.tile.full([1, N_EXPERTS], dtype=pl.INT32, value=0)
        p3_expert_total = pl.const(0, pl.INT32)
        for p3_expert in pl.range(N_EXPERTS):
            pl.write(p3_tile_bases, [0, p3_expert], p3_expert_total)
            p3_count = pl.read(p3_count_tile, [0, p3_expert])
            p3_expert_total = pl.cast(p3_expert_total + (p3_count + RECV_MAX - 1) // RECV_MAX, pl.INT32)
        pl.store(p3_tile_bases, [0, 0], expert_bases)
        p3_running = pl.const(0, pl.INT32)
        for p3_block_index in pl.range(ROUTE_BLOCKS):
            pl.write(block_bases, [p3_block_index, 0], p3_running)
            p3_running = p3_running + pl.read(start_counts, [p3_block_index, 0])
        pl.write(total_slots, [0], p3_running)

    with pl.spmd(ROUTE_BLOCKS, name_hint="route_parallel_slot_starts", deps=[bases_done]) as slots_done:
        p4_block = pl.tile.get_block_idx()
        p4_begin = p4_block * ROUTE_BLOCK
        p4_token_begin = p4_block * TOKEN_BLOCK
        p4_valid_tokens = pl.min(TOKEN_BLOCK, T - p4_token_begin)
        p4_ids = pl.slice(indices, [TOKEN_BLOCK, 16], [p4_token_begin, 0], valid_shape=[p4_valid_tokens, 16])
        p4_order = pl.load(global_order, [0, p4_begin], [1, ROUTE_BLOCK])
        p4_keys = pl.load(expert_bases, [0, 0], [1, N_EXPERTS])
        p4_slot = pl.read(block_bases, [p4_block, 0])
        p4_end = pl.min(p4_begin + ROUTE_BLOCK, active_routes)
        for p4_route in pl.range(p4_begin, p4_end):
            p4_ordinal = pl.read(p4_order, [0, p4_route - p4_begin])
            if p4_ordinal % RECV_MAX == 0:
                p4_expert = pl.cast(pl.read(p4_ids, [(p4_route - p4_begin) // TOPK, p4_route % TOPK]), pl.INDEX)
                p4_tile = p4_ordinal // RECV_MAX
                p4_key = pl.cast(pl.read(p4_keys, [0, p4_expert]) + p4_tile, pl.INDEX)
                pl.write(tile_slots, [p4_key, 0], p4_slot)
                p4_physical_slot = pl.cast(p4_slot, pl.INDEX)
                pl.write(slot_info, [p4_physical_slot, 0], pl.cast(p4_expert, pl.INT32))
                pl.write(slot_info, [p4_physical_slot, 1], pl.cast(p4_tile, pl.INT32))
                p4_slot = p4_slot + pl.const(1, pl.INT32)

    with pl.spmd(ROUTE_BLOCKS, name_hint="route_parallel_scatter", deps=[slots_done]) as scattered:
        p5_block = pl.tile.get_block_idx()
        p5_begin = p5_block * ROUTE_BLOCK
        p5_token_begin = p5_block * TOKEN_BLOCK
        p5_valid_tokens = pl.min(TOKEN_BLOCK, T - p5_token_begin)
        p5_ids = pl.slice(indices, [TOKEN_BLOCK, 16], [p5_token_begin, 0], valid_shape=[p5_valid_tokens, 16])
        p5_order = pl.load(global_order, [0, p5_begin], [1, ROUTE_BLOCK])
        p5_keys = pl.load(expert_bases, [0, 0], [1, N_EXPERTS])
        p5_route_rows = pl.tile.full([ROUTE_BLOCK, IDX_PAD], dtype=pl.INT32, value=0)
        p5_end = pl.min(p5_begin + ROUTE_BLOCK, active_routes)
        for p5_route in pl.range(p5_begin, p5_end):
            p5_expert = pl.cast(pl.read(p5_ids, [(p5_route - p5_begin) // TOPK, p5_route % TOPK]), pl.INDEX)
            p5_ordinal = pl.read(p5_order, [0, p5_route - p5_begin])
            p5_key = pl.cast(pl.read(p5_keys, [0, p5_expert]) + p5_ordinal // RECV_MAX, pl.INDEX)
            p5_slot = pl.read(tile_slots, [p5_key, 0])
            p5_destination = pl.cast(p5_slot * RECV_MAX + p5_ordinal % RECV_MAX, pl.INDEX)
            pl.write(p5_route_rows, [p5_route - p5_begin, 0], pl.cast(p5_destination, pl.INT32))
            pl.write(slot_routes, [p5_destination, 0], pl.cast(p5_route, pl.INT32))
        p5_physical_rows = pl.min(ROUTE_BLOCK, N_ROUTES - p5_begin)
        p5_valid_routes = pl.set_validshape(p5_route_rows, p5_physical_rows, IDX_PAD)
        pl.store(p5_valid_routes, [p5_begin, 0], route_slot_row)

    with pl.spmd((N_SLOTS_B + SLOT_TILE - 1) // SLOT_TILE, name_hint="route_parallel_materialize", deps=[scattered]) as materialized:
        p6_first = pl.tile.get_block_idx() * SLOT_TILE
        p6_valid_slots = pl.min(SLOT_TILE, N_SLOTS_B - p6_first)
        p6_routed_slots = pl.cast(pl.read(total_slots, [0]), pl.INDEX)
        p6_count_tile = pl.load(totals, [0, 0], [1, N_EXPERTS])
        p6_expert_tile = pl.tile.full([SLOT_TILE, IDX_PAD], dtype=pl.INT32, value=0)
        p6_scale_tile = pl.tile.full([SLOT_TILE, RECV_MAX], dtype=pl.FP32, value=0.0)
        p6_weight_tile = pl.tile.full([SLOT_TILE, RECV_MAX], dtype=pl.FP32, value=0.0)
        for p6_local_slot in pl.range(p6_valid_slots):
            p6_slot = p6_first + p6_local_slot
            if p6_slot < p6_routed_slots:
                p6_expert = pl.cast(pl.read(slot_info, [p6_slot, 0]), pl.INDEX)
                p6_ordinal_tile = pl.read(slot_info, [p6_slot, 1])
                p6_count = pl.min(RECV_MAX, pl.read(p6_count_tile, [0, p6_expert]) - p6_ordinal_tile * RECV_MAX)
                pl.write(p6_expert_tile, [p6_local_slot, 0], pl.cast(p6_expert, pl.INT32))
                for p6_row in pl.range(pl.cast(p6_count, pl.INDEX)):
                    p6_route = pl.cast(pl.read(slot_routes, [p6_slot * RECV_MAX + p6_row, 0]), pl.INDEX)
                    p6_token = p6_route // TOPK
                    p6_k = p6_route % TOPK
                    pl.write(p6_scale_tile, [p6_local_slot, p6_row], pl.read(scales, [p6_token, 0]))
                    pl.write(p6_weight_tile, [p6_local_slot, p6_row], pl.read(weights, [p6_token, p6_k]))
            elif p6_slot >= SH_SLOT:
                p6_shared = p6_slot - SH_SLOT
                p6_count = pl.min(RECV_MAX, pl.max(active_tokens - p6_shared * RECV_MAX, 0))
                for p6_column in pl.range(IDX_PAD):
                    pl.write(p6_expert_tile, [p6_local_slot, p6_column], pl.const(SHARED_EID, pl.INT32))
                for p6_row in pl.range(p6_count):
                    p6_token = p6_shared * RECV_MAX + p6_row
                    pl.write(p6_scale_tile, [p6_local_slot, p6_row], pl.read(scales, [p6_token, 0]))
                    pl.write(p6_weight_tile, [p6_local_slot, p6_row], pl.const(1.0, pl.FP32))
        p6_valid_experts = pl.set_validshape(p6_expert_tile, p6_valid_slots, IDX_PAD)
        p6_valid_scales = pl.set_validshape(p6_scale_tile, p6_valid_slots, RECV_MAX)
        p6_valid_weights = pl.set_validshape(p6_weight_tile, p6_valid_slots, RECV_MAX)
        pl.store(p6_valid_experts, [p6_first, 0], slot_expert)
        pl.store(p6_valid_scales, [p6_first, 0], recv_scale)
        pl.store(p6_valid_weights, [p6_first, 0], recv_weight)

    work_slot_flat = pl.reshape(work_slot, [1, WORK_SLOTS_PAD])
    work_rows_flat = pl.reshape(work_rows, [1, WORK_SLOTS_PAD])
    with pl.spmd((WORK_SLOTS_PAD + 15) // 16, name_hint="route_parallel_workplan", deps=[slots_done]) as planned:
        p7_begin = pl.tile.get_block_idx() * 16
        p7_count_tile = pl.load(totals, [0, 0], [1, N_EXPERTS])
        p7_routed_slots = pl.cast(pl.read(total_slots, [0]), pl.INDEX)
        p7_slots = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
        p7_rows = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
        for p7_local_plan in pl.range(pl.min(16, WORK_SLOTS_PAD - p7_begin)):
            p7_plan = p7_begin + p7_local_plan
            if p7_plan < p7_routed_slots:
                p7_expert = pl.cast(pl.read(slot_info, [p7_plan, 0]), pl.INDEX)
                p7_tile = pl.read(slot_info, [p7_plan, 1])
                p7_count = pl.min(RECV_MAX, pl.read(p7_count_tile, [0, p7_expert]) - p7_tile * RECV_MAX)
                pl.write(p7_slots, [0, p7_local_plan], pl.cast(p7_plan, pl.INT32))
                pl.write(p7_rows, [0, p7_local_plan], pl.cast(p7_count, pl.INT32))
            elif p7_plan < p7_routed_slots + SHARED_TILES:
                p7_shared = p7_plan - p7_routed_slots
                p7_count = pl.min(RECV_MAX, pl.max(active_tokens - p7_shared * RECV_MAX, 0))
                pl.write(p7_slots, [0, p7_local_plan], pl.cast(SH_SLOT + p7_shared, pl.INT32))
                pl.write(p7_rows, [0, p7_local_plan], pl.cast(p7_count, pl.INT32))
        p7_columns = pl.min(16, WORK_SLOTS_PAD - p7_begin)
        p7_valid_slots = pl.set_validshape(p7_slots, 1, p7_columns)
        p7_valid_rows = pl.set_validshape(p7_rows, 1, p7_columns)
        pl.store(p7_valid_slots, [0, p7_begin], work_slot_flat)
        pl.store(p7_valid_rows, [0, p7_begin], work_rows_flat)
    return pl.system.task_dummy(deps=[scattered, materialized, planned])
