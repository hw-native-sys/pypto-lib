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

from .moe import (
    D, HC_DIM, HC_MULT, MIX_HC, MOE_INTER, N_BANK, N_EXPERTS, N_RANKS,
    REDUCE_WINDOW_ROWS, REDUCE_LANE_ROWS, REDUCE_D_TILE, T_PAD, T, TOPK, VOCAB,
    moe, prepare_moe, moe_local, hc_post,
)


from .config import _parse_int_argv


PREFILL_MOE_READY = _parse_int_argv("--prefill-moe-ready", 0)
PREFILL_ROWS = pl.dynamic("LOWLAT_PREFILL_MOE_ROWS")


@pl.jit.inline
def clear_prefill_moe_tp_signals(
    completion: pl.Tensor[[1], pl.INT32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Retire local counters after the final ordered TP wave completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_moe_cleanup") as retired_tid:
        completed = pl.read(completion, [0])
        if completed >= 0:
            for src in pl.range(N_RANKS):
                pl.write(reduce_signal, [src, 0], pl.const(0, pl.INT32))
    return retired_tid


@pl.jit.inline(auto_scope=False)
def _prefill_moe_publish_reduce(
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
                dst_offsets=[my_row, 0], src_offsets=[0, 0], shape=[T, D], chunk_rows=T, chunk_cols=D,
            )
            pld.system.notify(reduce_signal, peer, [my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        published[0] = pushed
    waited = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_ready_defer", deps=[published[0]], allow_early_resolve=False) as all_ready:
            for src in pl.range(N_RANKS):
                pld.system.defer_wait(reduce_signal, [src, 0], epoch, cmp=pld.WaitCmp.Ge)
        waited[0] = all_ready
    reduced_tasks = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(D // REDUCE_D_TILE, name_hint="prefill_moe_ready_sum", deps=[waited[0], published[0]], allow_early_resolve=False) as summed:
            d0 = pl.tile.get_block_idx() * REDUCE_D_TILE
            acc = pl.load(reduce_window, [lane_base, d0], [T, REDUCE_D_TILE])
            for src_rank in pl.range(1, N_RANKS):
                src_row = lane_base + src_rank * T_PAD
                acc = pl.add(acc, pl.load(reduce_window, [src_row, d0], [T, REDUCE_D_TILE]))
            reduced = pl.cast(acc, target_type=pl.BF16, mode="rint")
            pl.store(reduced, [0, d0], ffn_out)
        reduced_tasks[0] = summed
    return reduced_tasks[0]


@pl.jit.inline(auto_scope=False)
def _prefill_moe_ready_tp(
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
    layer_id: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32]:
    """Run two bounded compute lanes ahead of ordered two-lane TP publication."""
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
    waves = (pl.cast(group_tokens, pl.INDEX) + T - 1) // T
    for wave in pl.range(waves):
        with pl.scope():
            lane = wave % 2
            base = wave * T
            active = pl.min(T, pl.cast(group_tokens, pl.INDEX) - base)
            epoch = epoch_base + pl.cast(wave, pl.INT32) + pl.const(1, pl.INT32)
            wave_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
            wave_ids = pl.create_tensor([T], dtype=pl.INT64)
            with pl.spmd(T, name_hint="prefill_moe_ready_stage", deps=[consumed[lane]]) as staged:
                token = pl.tile.get_block_idx()
                if token < active:
                    wave_hc[token:token + 1, :, :] = attn_out[base + token:base + token + 1, :, :]
                else:
                    wave_hc[token:token + 1, :, :] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_ready_ids", deps=[consumed[lane]]) as ids_ready:
                for token in pl.range(T):
                    value = pl.const(0, pl.INT64)
                    if token < active:
                        value = pl.read(input_ids, [base + token])
                    pl.write(wave_ids, [token], value)
            (post_ffn, comb_ffn, recv_x, recv_scale, recv_w, work_slot, work_rows,
             slot_expert, route_slot_row) = prepare_moe(
                wave_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias,
                tid2eid, wave_ids, layer_id, pl.cast(active, pl.INT32),
            )
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
                ffn_partial, ffn_out, reduce_window, reduce_signal, my_rank, epoch, partial_ready, previous_store,
            )
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
    return x_hc


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
    layer_id: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32]:
    """Run all prompt rows in ordered decode-sized TP waves on every rank.

    Input IDs and activations must denote the same full prompt on each rank.
    The expert bank contains all routed experts and the shared expert, with
    the intermediate dimension partitioned by the same TP rank as decode.
    """
    if PREFILL_MOE_READY != 0:
        _prefill_moe_ready_tp(
            attn_out, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
            x_hc, stage_token, layer_completion, reduce_window, reduce_signal,
            layer_id, group_tokens, my_rank, epoch_base,
        )
    else:
        attn_out.bind_dynamic(0, PREFILL_ROWS)
        input_ids.bind_dynamic(0, PREFILL_ROWS)
        x_hc.bind_dynamic(0, PREFILL_ROWS)
        waves = (pl.cast(group_tokens, pl.INDEX) + T - 1) // T
        for wave in pl.range(waves):
            with pl.scope():
                base = wave * T
                active = pl.min(T, pl.cast(group_tokens, pl.INDEX) - base)
                epoch = epoch_base + pl.cast(wave, pl.INT32) + pl.const(1, pl.INT32)
                wave_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
                wave_ids = pl.create_tensor([T], dtype=pl.INT64)
                with pl.spmd(T, name_hint="prefill_tp_moe_stage"):
                    token = pl.tile.get_block_idx()
                    ready = pl.read(stage_token, [0])
                    if ready >= 0:
                        if token < active:
                            wave_hc[token : token + 1, :, :] = attn_out[base + token : base + token + 1, :, :]
                        else:
                            wave_hc[token : token + 1, :, :] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_moe_ids"):
                    ready = pl.read(stage_token, [0])
                    if ready >= 0:
                        for token in pl.range(T):
                            value = pl.const(0, pl.INT64)
                            if token < active:
                                value = pl.read(input_ids, [base + token])
                            pl.write(wave_ids, [token], value)
                wave_out = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
                moe(wave_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                    norm_w, gate_w, gate_bias, tid2eid, wave_ids,
                    routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
                    wave_out, reduce_window, reduce_signal,
                    layer_id, pl.cast(active, pl.INT32), my_rank, epoch)
                with pl.spmd(T, name_hint="prefill_tp_moe_store") as store_tid:
                    token = pl.tile.get_block_idx()
                    if token < active:
                        x_hc[base + token : base + token + 1, :, :] = wave_out[token : token + 1, :, :]
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_moe_wave_complete", deps=[store_tid]):
                    pl.write(stage_token, [0], epoch)

        padded = pl.tensor.dim(x_hc, 0)
        if padded > group_tokens:
            with pl.spmd(padded - group_tokens, name_hint="prefill_tp_moe_padding"):
                token = pl.cast(group_tokens, pl.INDEX) + pl.tile.get_block_idx()
                x_hc[token : token + 1, :, :] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_moe_layer_complete"):
            completed = pl.read(stage_token, [0])
            if completed >= 0:
                pl.write(layer_completion, [0], layer_id + pl.const(1, pl.INT32))
    return x_hc


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
    layer_id: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, PREFILL_ROWS)
    ids.bind_dynamic(0, PREFILL_ROWS)
    out.bind_dynamic(0, PREFILL_ROWS)
    stage = pl.create_tensor([1], dtype=pl.INT32)
    completion = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_moe_init"):
        pl.write(stage, [0], pl.const(0, pl.INT32))
        pl.write(completion, [0], pl.const(0, pl.INT32))
    prefill_moe_tp(x, hc_fn, hc_scale, hc_base, norm_w, gate_w, gate_bias,
                   tid2eid, ids, w1, s1, w3, s3, w2, s2, out, stage, completion,
                   reduce_window, reduce_signal, layer_id, group_tokens, rank, pl.const(0, pl.INT32))
    clear_prefill_moe_tp_signals(completion, reduce_signal)
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
    rw_buf = pld.alloc_window_buffer([REDUCE_WINDOW_ROWS, D], dtype=pl.FP32)
    rs_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        rw = pld.window(rw_buf, [REDUCE_WINDOW_ROWS, D], dtype=pl.FP32)
        rs = pld.window(rs_buf, [N_RANKS, 1], dtype=pl.INT32)
        l2_prefill_moe_tp(x[r], hc_fn[r], hc_scale[r], hc_base[r], norm_w[r],
            gate_w[r], gate_bias[r], tid2eid[r], ids[r], w1[r], s1[r], w3[r], s3[r],
            w2[r], s2[r], out[r], rw, rs, layer_id, group_tokens, r, device=r)
    return out
