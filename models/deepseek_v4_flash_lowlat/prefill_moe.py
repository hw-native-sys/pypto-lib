# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor-parallel prefill MoE using the low-latency decode expert shards."""

import pypto.language as pl
import pypto.language.distributed as pld

from .prefill_moe_geometry import (
    PREFILL_MOE_SHARE_GATE, GATE_WINDOW_ROWS, D, HC_DIM, HC_MULT, MIX_HC, MOE_INTER, N_BANK, N_EXPERTS, N_RANKS,
    REDUCE_WINDOW_ROWS, REDUCE_LANE_ROWS, REDUCE_D_TILE, T_PAD, T, TOPK, VOCAB,
    COPY_T_TILE, COPY_D_TILE, REDUCE_T_TILE,
    PREFILL_MOE_BATCHED_GATHER as PREFILL_MOE_BATCHED_GATHER, PREFILL_MOE_DIRECT_POST,
    PREFILL_MOE_OWNER_REDUCE, PREFILL_MOE_EXPERT_AFFINITY as PREFILL_MOE_EXPERT_AFFINITY, OWNER_ROWS, OWNER_REDUCE_T_TILE, OWNER_COPY_T_TILE, RESULT_WINDOW_ROWS,
)


from .config import _parse_int_argv
from .prefill_moe_compute import prepare_moe, moe_local
from .hc_post import hc_post, T_TILE as HC_POST_T_TILE


PREFILL_MOE_READY = _parse_int_argv("--prefill-moe-ready", 1)
PREFILL_ROWS = pl.dynamic("LOWLAT_PREFILL_MOE_ROWS")


@pl.jit.inline
def clear_prefill_moe_tp_signals(
    completion: pl.Tensor[[1], pl.INT32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    result_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Retire local counters after the final ordered TP wave completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_moe_cleanup") as retired_tid:
        completed = pl.read(completion, [0])
        if completed >= 0:
            for src in pl.range(N_RANKS):
                pl.write(reduce_signal, [src, 0], pl.const(0, pl.INT32))
                pl.write(result_signal, [src, 0], pl.const(0, pl.INT32))
    return retired_tid


@pl.jit.inline(auto_scope=False)
def _prefill_moe_legacy_publish_reduce(
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    partial_ready: pl.Scalar[pl.TASK_ID],
    previous_store: pl.Scalar[pl.TASK_ID],
):
    """Publish after local retirement, defer all peer waits, and sum rank0..7."""
    lane_base = pl.cast(((epoch - 1) % 2) * REDUCE_LANE_ROWS, pl.INDEX)
    my_row = lane_base + pl.cast(my_rank, pl.INDEX) * T_PAD
    published = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(N_RANKS, name_hint="prefill_moe_ready_publish", deps=[partial_ready, previous_store], allow_early_resolve=False) as pushed:
            peer = pl.tile.get_block_idx()
            pld.tensor.put(
                dst=reduce_window, peer=peer, src=ffn_partial,
                dst_offsets=[my_row, 0], src_offsets=[0, 0], shape=[T, D], chunk_rows=COPY_T_TILE, chunk_cols=COPY_D_TILE,
            )
            pld.system.notify(reduce_signal, peer, [my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        published[0] = pushed
    waited = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_ready_defer", deps=[published[0]], allow_early_resolve=False) as all_ready:
            for src in pl.range(N_RANKS):
                if PREFILL_MOE_READY != 0:
                    pld.system.defer_wait(reduce_signal, [src, 0], epoch, cmp=pld.WaitCmp.Ge)
                else:
                    pld.system.wait(reduce_signal, [src, 0], epoch, cmp=pld.WaitCmp.Ge)
        waited[0] = all_ready
    reduced_tasks = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd((T // REDUCE_T_TILE) * (D // REDUCE_D_TILE), name_hint="prefill_moe_ready_sum", deps=[waited[0], published[0]], allow_early_resolve=False) as summed:
            block = pl.tile.get_block_idx()
            t0 = (block // (D // REDUCE_D_TILE)) * REDUCE_T_TILE
            d0 = (block % (D // REDUCE_D_TILE)) * REDUCE_D_TILE
            acc = pl.load(reduce_window, [lane_base + t0, d0], [REDUCE_T_TILE, REDUCE_D_TILE])
            for src_rank in pl.range(1, N_RANKS):
                src_row = lane_base + src_rank * T_PAD + t0
                acc = pl.add(acc, pl.load(reduce_window, [src_row, d0], [REDUCE_T_TILE, REDUCE_D_TILE]))
            reduced = pl.cast(acc, target_type=pl.BF16, mode="rint")
            pl.store(reduced, [t0, d0], ffn_out)
        reduced_tasks[0] = summed
    return reduced_tasks[0]


@pl.jit.inline(auto_scope=False)
def _prefill_moe_owner_publish_reduce(
    partial: pl.Tensor[[T, D], pl.FP32],
    output: pl.Tensor[[T, D], pl.BF16],
    receive: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    received: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    result: pld.DistributedTensor[[RESULT_WINDOW_ROWS, D], pl.BF16],
    result_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    partial_ready: pl.Scalar[pl.TASK_ID],
    previous_store: pl.Scalar[pl.TASK_ID],
):
    """Scatter FP32 rows, sum rank0..7 once per owner, and publish BF16 rows."""
    lane = pl.cast((epoch - 1) % 2, pl.INDEX)
    receive_base = lane * T_PAD
    source_row = pl.cast(rank, pl.INDEX) * OWNER_ROWS
    tasks = pl.array.create(6, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(N_RANKS, name_hint="prefill_owner_fp32_scatter", deps=[partial_ready, previous_store], allow_early_resolve=False) as scattered:
            owner = pl.tile.get_block_idx()
            pld.tensor.put(
                dst=receive, peer=owner, src=partial,
                dst_offsets=[receive_base + source_row, 0], src_offsets=[owner * OWNER_ROWS, 0],
                shape=[OWNER_ROWS, D], chunk_rows=OWNER_COPY_T_TILE, chunk_cols=COPY_D_TILE,
            )
            pld.system.notify(received, owner, [rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        tasks[0] = scattered
    with pl.manual_scope():
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_owner_fp32_defer", deps=[tasks[0]], allow_early_resolve=False) as all_partials:
            for source in pl.range(N_RANKS):
                if PREFILL_MOE_READY != 0:
                    pld.system.defer_wait(received, [source, 0], epoch, cmp=pld.WaitCmp.Ge)
                else:
                    pld.system.wait(received, [source, 0], epoch, cmp=pld.WaitCmp.Ge)
        tasks[1] = all_partials
    owner_output = pl.create_tensor([OWNER_ROWS, D], dtype=pl.BF16)
    with pl.manual_scope():
        with pl.spmd((OWNER_ROWS // OWNER_REDUCE_T_TILE) * (D // REDUCE_D_TILE), name_hint="prefill_owner_rank_order_sum", deps=[tasks[1]], allow_early_resolve=False) as summed:
            block = pl.tile.get_block_idx()
            t0 = (block // (D // REDUCE_D_TILE)) * OWNER_REDUCE_T_TILE
            d0 = (block % (D // REDUCE_D_TILE)) * REDUCE_D_TILE
            acc = pl.load(receive, [receive_base + t0, d0], [OWNER_REDUCE_T_TILE, REDUCE_D_TILE])
            for source in pl.range(1, N_RANKS):
                row = receive_base + source * OWNER_ROWS + t0
                acc = pl.add(acc, pl.load(receive, [row, d0], [OWNER_REDUCE_T_TILE, REDUCE_D_TILE]))
            rounded = pl.cast(acc, target_type=pl.BF16, mode="rint")
            pl.store(rounded, [t0, d0], owner_output)
        tasks[2] = summed
    result_base = lane * T_PAD
    with pl.manual_scope():
        with pl.spmd(N_RANKS, name_hint="prefill_owner_bf16_publish", deps=[tasks[2]], allow_early_resolve=False) as published:
            peer = pl.tile.get_block_idx()
            pld.tensor.put(
                dst=result, peer=peer, src=owner_output,
                dst_offsets=[result_base + source_row, 0], src_offsets=[0, 0],
                shape=[OWNER_ROWS, D], chunk_rows=OWNER_COPY_T_TILE, chunk_cols=COPY_D_TILE,
            )
            pld.system.notify(result_ready, peer, [rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        tasks[3] = published
    with pl.manual_scope():
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_owner_bf16_defer", deps=[tasks[3]], allow_early_resolve=False) as all_results:
            for source in pl.range(N_RANKS):
                if PREFILL_MOE_READY != 0:
                    pld.system.defer_wait(result_ready, [source, 0], epoch, cmp=pld.WaitCmp.Ge)
                else:
                    pld.system.wait(result_ready, [source, 0], epoch, cmp=pld.WaitCmp.Ge)
        tasks[4] = all_results
    tasks[5] = tasks[4]
    if PREFILL_MOE_DIRECT_POST == 0:
        with pl.manual_scope():
            with pl.spmd((T // REDUCE_T_TILE) * (D // REDUCE_D_TILE), name_hint="prefill_owner_bf16_collect", deps=[tasks[4]], allow_early_resolve=False) as collected:
                block = pl.tile.get_block_idx()
                t0 = (block // (D // REDUCE_D_TILE)) * REDUCE_T_TILE
                d0 = (block % (D // REDUCE_D_TILE)) * REDUCE_D_TILE
                value = pl.load(result, [result_base + t0, d0], [REDUCE_T_TILE, REDUCE_D_TILE])
                pl.store(value, [t0, d0], output)
            tasks[5] = collected
    return tasks[5]


@pl.jit.inline
def _prefill_moe_direct_post(
    result: pld.DistributedTensor[[RESULT_WINDOW_ROWS, D], pl.BF16],
    residual: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32],
    result_base: pl.Scalar[pl.INDEX],
    output_base: pl.Scalar[pl.INDEX],
    active: pl.Scalar[pl.INDEX],
    reduced: pl.Scalar[pl.TASK_ID],
):
    """Apply ordered HC post from the completed BF16 lane into final token rows."""
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    output_rows = pl.tensor.dim(output, 0)
    output_flat = pl.reshape(output, [output_rows, HC_DIM])
    with pl.spmd(((T + HC_POST_T_TILE - 1) // HC_POST_T_TILE) * HC_MULT, name_hint="prefill_moe_direct_post", deps=[reduced], allow_early_resolve=True) as posted:
        block = pl.tile.get_block_idx()
        token_block = block // HC_MULT
        out_h = block % HC_MULT
        t0 = token_block * HC_POST_T_TILE
        token_end = pl.min(t0 + HC_POST_T_TILE, active)
        for t in pl.pipeline(t0, token_end, stage=2):
            post_w = pl.read(post, [t, out_h])
            source_row = result_base + t
            x_row = pl.cast(result[source_row:source_row + 1, 0:D], target_type=pl.FP32)
            y_row = pl.mul(x_row, post_w)
            for in_h in pl.pipeline(HC_MULT, stage=4):
                comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                res_d = in_h * D
                res_row = residual_flat[t:t + 1, res_d:res_d + D]
                weighted = pl.mul(res_row, comb_w)
                y_row = pl.add(y_row, weighted)
            destination_row = output_base + t
            destination_col = out_h * D
            output_flat[destination_row:destination_row + 1, destination_col:destination_col + D] = y_row
    return posted


@pl.jit.inline(auto_scope=False)
def _prefill_moe_publish_reduce(
    partial: pl.Tensor[[T, D], pl.FP32],
    output: pl.Tensor[[T, D], pl.BF16],
    receive: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    received: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    result: pld.DistributedTensor[[RESULT_WINDOW_ROWS, D], pl.BF16],
    result_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    partial_ready: pl.Scalar[pl.TASK_ID],
    previous_store: pl.Scalar[pl.TASK_ID],
):
    chosen = pl.array.create(1, pl.TASK_ID)
    if PREFILL_MOE_OWNER_REDUCE != 0:
        chosen[0] = _prefill_moe_owner_publish_reduce(
            partial, output, receive, received, result, result_ready, rank, epoch, partial_ready, previous_store,
        )
    else:
        chosen[0] = _prefill_moe_legacy_publish_reduce(
            partial, output, receive, received, rank, epoch, partial_ready, previous_store,
        )
    return chosen[0]


@pl.jit.inline(auto_scope=False)
def prefill_moe_tp(
    attn_out: pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[PREFILL_ROWS], pl.INT64],
    routed_w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_BANK, D], pl.FP32],
    x_hc: pl.Out[pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32]],
    stage_token: pl.Tensor[[1], pl.INT32],
    layer_completion: pl.Tensor[[1], pl.INT32],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    result_window: pld.DistributedTensor[[RESULT_WINDOW_ROWS, D], pl.BF16],
    result_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
    gate_window: pld.DistributedTensor[[GATE_WINDOW_ROWS, N_EXPERTS], pl.FP32],
    gate_signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    gate_epoch_base: pl.Scalar[pl.INT32],
    prior_gate: pl.Scalar[pl.TASK_ID],
):
    """Run independent prefill slabs with bounded compute and ordered TP publication."""
    attn_out.bind_dynamic(0, PREFILL_ROWS)
    input_ids.bind_dynamic(0, PREFILL_ROWS)
    x_hc.bind_dynamic(0, PREFILL_ROWS)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_ready_enter") as entered:
        pl.write(stage_token, [0], pl.read(stage_token, [0]))
    consumed = pl.array.create(2, pl.TASK_ID)
    consumed[0] = entered
    consumed[1] = entered
    stored = pl.array.create(1, pl.TASK_ID)
    stored[0] = entered
    gate_previous = pl.array.create(1, pl.TASK_ID)
    gate_previous[0] = prior_gate
    waves = (pl.cast(group_tokens, pl.INDEX) + T - 1) // T
    for wave in pl.range(waves):
        with pl.scope():
            lane = wave % 2
            base = wave * T
            active = pl.min(T, pl.cast(group_tokens, pl.INDEX) - base)
            epoch = epoch_base + pl.cast(wave, pl.INT32) + pl.const(1, pl.INT32)
            wave_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
            wave_ids = pl.create_tensor([T], dtype=pl.INT64)
            compute_ready = consumed[lane]
            if PREFILL_MOE_READY == 0:
                compute_ready = stored[0]
            with pl.spmd(T, name_hint="prefill_moe_ready_stage", deps=[compute_ready]) as staged:
                token = pl.tile.get_block_idx()
                if token < active:
                    wave_hc[token:token + 1, :, :] = attn_out[base + token:base + token + 1, :, :]
                else:
                    wave_hc[token:token + 1, :, :] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_ready_ids", deps=[compute_ready]) as ids_ready:
                for token in pl.range(T):
                    value = pl.const(0, pl.INT64)
                    if token < active:
                        value = pl.read(input_ids, [base + token])
                    pl.write(wave_ids, [token], value)
            previous_gate_scalar = gate_previous[0]
            (post_ffn, comb_ffn, recv_x, recv_scale, recv_w, work_slot, work_rows,
             slot_expert, route_slot_row, gate_finished) = prepare_moe(
                wave_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias,
                tid2eid, wave_ids, layer_id, pl.cast(active, pl.INT32),
                gate_window, gate_signal, my_rank, gate_epoch_base + epoch, previous_gate_scalar,
            )
            gate_previous[0] = gate_finished
            ffn_partial = moe_local(
                recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert, route_slot_row,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
                pl.cast(active, pl.INT32),
            )
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_ready_partial", deps=[staged, ids_ready]) as partial_ready:
                pl.write(ffn_partial, [0, 0], pl.read(ffn_partial, [0, 0]))
            ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
            previous_store = stored[0]
            reduced = _prefill_moe_publish_reduce(
                ffn_partial, ffn_out, reduce_window, reduce_signal, result_window, result_signal, my_rank, epoch, partial_ready, previous_store,
            )
            if PREFILL_MOE_DIRECT_POST != 0:
                result_base = pl.cast(((epoch - 1) % 2) * T_PAD, pl.INDEX)
                direct_done = _prefill_moe_direct_post(
                    result_window, wave_hc, post_ffn, comb_ffn, x_hc,
                    result_base, base, active, reduced,
                )
                consumed[lane] = direct_done
                stored[0] = direct_done
            else:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_ready_post_gate", deps=[reduced]) as post_ready:
                    pl.write(ffn_out, [0, 0], pl.read(ffn_out, [0, 0]))
                wave_out = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
                hc_post(ffn_out, wave_hc, post_ffn, comb_ffn, wave_out)
                with pl.spmd(T, name_hint="prefill_moe_ready_store", deps=[post_ready]) as store_done:
                    token = pl.tile.get_block_idx()
                    if token < active:
                        x_hc[base + token:base + token + 1, :, :] = wave_out[token:token + 1, :, :]
                consumed[lane] = store_done
                stored[0] = store_done
    padded = pl.tensor.dim(x_hc, 0)
    if padded > group_tokens:
        with pl.spmd(padded - group_tokens, name_hint="prefill_moe_ready_padding", deps=[stored[0]]) as padded_done:
            token = pl.cast(group_tokens, pl.INDEX) + pl.tile.get_block_idx()
            x_hc[token:token + 1, :, :] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
        stored[0] = padded_done
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_ready_complete", deps=[stored[0]]):
        pl.write(stage_token, [0], epoch_base + pl.cast(waves, pl.INT32))
        pl.write(layer_completion, [0], layer_id + pl.const(1, pl.INT32))
    return x_hc, gate_previous[0]


@pl.jit(auto_scope=False)
def l2_prefill_moe_tp(
    x: pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    ids: pl.Tensor[[PREFILL_ROWS], pl.INT64],
    w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    s1: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    s3: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    s2: pl.Tensor[[N_BANK, D], pl.FP32],
    out: pl.Out[pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32]],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    result_window: pld.DistributedTensor[[RESULT_WINDOW_ROWS, D], pl.BF16],
    result_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    gate_window: pld.DistributedTensor[[GATE_WINDOW_ROWS, N_EXPERTS], pl.FP32],
    gate_signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
):
    x.bind_dynamic(0, PREFILL_ROWS)
    ids.bind_dynamic(0, PREFILL_ROWS)
    out.bind_dynamic(0, PREFILL_ROWS)
    stage = pl.create_tensor([1], dtype=pl.INT32)
    completion = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_moe_init"):
        pl.write(stage, [0], pl.const(0, pl.INT32))
        pl.write(completion, [0], pl.const(0, pl.INT32))
    gate_seed = pl.system.task_dummy(deps=[])
    component_out, component_gate_done = prefill_moe_tp(x, hc_fn, hc_scale, hc_base, norm_w, gate_w, gate_bias,
                   tid2eid, ids, w1, s1, w3, s3, w2, s2, out, stage, completion,
                   reduce_window, reduce_signal, result_window, result_signal, layer_id, group_tokens, rank, pl.const(0, pl.INT32),
                   gate_window, gate_signal, pl.const(0, pl.INT32), gate_seed)
    if PREFILL_MOE_SHARE_GATE != 0:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_gate_component_cleanup", deps=[component_gate_done]):
            for source in pl.range(N_RANKS):
                pl.write(gate_signal, [source, 0], pl.const(0, pl.INT32))
    clear_prefill_moe_tp_signals(completion, reduce_signal, result_signal)
    return out

@pl.jit.host
def l3_prefill_moe_tp(
    x: pl.Tensor[[N_RANKS, PREFILL_ROWS, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    ids: pl.Tensor[[N_RANKS, PREFILL_ROWS], pl.INT64],
    w1: pl.Tensor[[N_RANKS, N_BANK, MOE_INTER, D], pl.INT8],
    s1: pl.Tensor[[N_RANKS, N_BANK, MOE_INTER], pl.FP32],
    w3: pl.Tensor[[N_RANKS, N_BANK, MOE_INTER, D], pl.INT8],
    s3: pl.Tensor[[N_RANKS, N_BANK, MOE_INTER], pl.FP32],
    w2: pl.Tensor[[N_RANKS, N_BANK, D, MOE_INTER], pl.INT8],
    s2: pl.Tensor[[N_RANKS, N_BANK, D], pl.FP32],
    out: pl.Out[pl.Tensor[[N_RANKS, PREFILL_ROWS, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(1, PREFILL_ROWS)
    ids.bind_dynamic(1, PREFILL_ROWS)
    out.bind_dynamic(1, PREFILL_ROWS)
    gate_window_buf = pld.alloc_window_buffer([GATE_WINDOW_ROWS, N_EXPERTS], dtype=pl.FP32)
    gate_signal_buf = pld.alloc_window_buffer([N_RANKS, 16], dtype=pl.INT32)
    rw_buf = pld.alloc_window_buffer([REDUCE_WINDOW_ROWS, D], dtype=pl.FP32)
    rs_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    result_buf = pld.alloc_window_buffer([RESULT_WINDOW_ROWS, D], dtype=pl.BF16)
    result_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        gate_window = pld.window(gate_window_buf, [GATE_WINDOW_ROWS, N_EXPERTS], dtype=pl.FP32)
        gate_signal = pld.window(gate_signal_buf, [N_RANKS, 16], dtype=pl.INT32)
        rw = pld.window(rw_buf, [REDUCE_WINDOW_ROWS, D], dtype=pl.FP32)
        rs = pld.window(rs_buf, [N_RANKS, 1], dtype=pl.INT32)
        result = pld.window(result_buf, [RESULT_WINDOW_ROWS, D], dtype=pl.BF16)
        result_ready = pld.window(result_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        l2_prefill_moe_tp(x[r], hc_fn[r], hc_scale[r], hc_base[r], norm_w[r],
            gate_w[r], gate_bias[r], tid2eid[r], ids[r], w1[r], s1[r], w3[r], s3[r],
            w2[r], s2[r], out[r], rw, rs, result, result_ready, layer_id, group_tokens, r, gate_window, gate_signal, device=r)
    return out
