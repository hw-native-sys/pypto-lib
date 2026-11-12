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
    REDUCE_WINDOW_ROWS, T, TOPK, VOCAB, moe,
)


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
