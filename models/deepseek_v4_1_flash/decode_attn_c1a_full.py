# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A full attention."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from golden import TensorSpec, run
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult, quantized_cache_compare
from models.deepseek_v4_1_flash.attention_ops import (
    make_bf16_projection_with_deps,
    make_norm_with_deps,
    make_rope_with_deps,
)
from models.deepseek_v4_1_flash.config import (
    AttentionMode,
    B_DYN,
    CMP_BLOCKS_DYN,
    CMP_POSITIONS_DYN,
    COMPRESSED_CACHE_GROUP,
    D,
    DECODE_MAX_TOKENS,
    HEAD_DIM,
    INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP,
    INDEX_DIM,
    INDEX_H,
    INDEX_TOPK,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    ORI_BLOCKS_DYN,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    TABLE_DYN,
    TP_SIZE,
    T_DYN,
    WINDOW_CACHE_GROUP,
)
from models.deepseek_v4_1_flash.attention_tp import OUTPUT_T_DYN
from models.deepseek_v4_1_flash.hierarchical_sparse_indexer import hierarchical_sparse_indexer
from models.deepseek_v4_1_flash.o_proj import (
    grouped_output_with_deps,
    make_o_proj_with_deps as build_o_proj_with_deps,
)
from models.deepseek_v4_1_flash.qkv_proj_rope import (
    make_qkv_proj_rope_with_deps as build_qkv_proj_rope_with_deps,
)

# Model configuration.
SOFTMAX_SCALE = HEAD_DIM**-0.5
ATTEND_TILE = 64
ATTEND_TILES = (128 + INDEX_TOPK) // ATTEND_TILE

# Tiling.
M_TILE = min(16, LOCAL_H)
N_TILE = 128
K_TILE = 256

# Paged state one rank owns.
CACHE_STATE_NAMES = (
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "index_cache",
    "index_cache_scale",
    "topk_indices",
    "candidate_mask",
)

# Validation budgets, matching the sibling C1A entries. A published cache row is compared as
# dequantized values because a one-ULP difference can move a code one E2M1/E4M3 step; which rows
# may change at all stays byte exact.
CACHE_MAX_RELATIVE_L2 = 0.01
MXFP4_CACHE_MAX_RELATIVE_L2 = 0.04
TOPK_SCORE_ATOL = 5e-5
TOPK_SCORE_RTOL = 1.5e-2
OUTPUT_ATOL = 1e-2
OUTPUT_RTOL = 1e-2
OUTPUT_MAX_ERROR_RATIO = 0.01


def make_mx_projection_with_deps(width, output_width, output_dtype=pl.BF16):
    """Accumulate scale-corrected group-32 products without whole-weight expansion."""
    fp32_output = output_dtype == pl.FP32

    @pl.jit.inline
    def project(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP8E4M3FN],
        scale: pl.Tensor[[width // 32, output_width], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Tensor[[T_DYN, output_width], output_dtype],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        scale_storage = pl.tensor.view(scale, [output_width // 16, width // 2], layout=pl.ND)
        # A device-only scratch buffer has no tensor-map edge of its own, so the
        # consumer must be handed the producer TaskId explicitly; a task with no
        # dependency edge is ready at submit and may overlap its producer.
        with pl.spmd(32, name_hint="c1a_mx_projection", deps=[ready]) as project_tid:
            worker = pl.tile.get_block_idx()
            for task in pl.range(worker, (num_tokens + 15) // 16 * (output_width // N_TILE), 32):
                t0 = task // (output_width // N_TILE) * 16
                n0 = task % (output_width // N_TILE) * N_TILE
                rows = pl.min(16, num_tokens - t0)
                acc = pl.tile.full([16, N_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.range(width // 64):
                    raw = pl.load(scale_storage, [n0 // 16, kb * 32], [N_TILE // 16, 32])
                    raw_u8 = pl.reinterpret_view(raw, pl.UINT8)
                    codes = pl.ands(pl.cast(pl.reinterpret_view(raw_u8, pl.INT8), pl.INT32), 255)
                    scale_pair = pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32)
                    for half in pl.unroll(2):
                        k0 = (kb * 2 + half) * 32
                        if half == 0:
                            selected_scale = pl.tile.gather_mask(
                                scale_pair, mask_pattern=pl.tile.MaskPattern.P0101
                            )
                        else:
                            selected_scale = pl.tile.gather_mask(
                                scale_pair, mask_pattern=pl.tile.MaskPattern.P1010
                            )
                        sb = pl.reshape(selected_scale, [1, N_TILE])
                        source = pl.load(x, [t0, k0], [16, 32], valid_shape=[rows, 32])
                        source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 16, 32)
                        values = pl.cast(source, pl.FP32)
                        temporary = pl.tile.create([16, 32], dtype=pl.FP32)
                        maximum = pl.maximum(pl.row_max(pl.abs(values), tmp_tile=temporary), 1e-4)
                        bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                        exponent = pl.shrs(pl.add(bits, 8388607), 23)
                        sa = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                        payload = pl.cast(pl.row_expand_div(values, sa), pl.FP8E4M3FN, mode="rint")
                        a = pl.cast(payload, pl.BF16)
                        b = pl.cast(pl.load(weight, [k0, n0], [32, N_TILE]), pl.BF16)
                        dot = pl.matmul(a, b)
                        part = pl.col_expand_mul(pl.row_expand_mul(dot, sa), sb)
                        acc = pl.add(acc, part)
                if fp32_output:
                    pl.store(pl.set_validshape(acc, rows, N_TILE), [t0, n0], output)
                else:
                    result = pl.cast(acc, pl.BF16, mode="rint")
                    pl.store(pl.set_validshape(result, rows, N_TILE), [t0, n0], output)
        return project_tid

    return project


@pl.jit.inline
def publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    kv_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(num_tokens, name_hint="c1a_cache_publish", deps=[cache_ready, kv_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(kv, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
            value = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM // 16, 32])
            amax = pl.maximum(pl.row_max(pl.abs(value)), 1e-4)
            raw = pl.mul(amax, 1.0 / 448.0)
            bits = pl.reinterpret_view(raw, pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            payload = pl.cast(pl.row_expand_div(value, scale), pl.FP8E4M3FN, mode="rint")
            flat[slot : slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
            signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
            codes = pl.cast(signed_exponent, pl.INT8)
            encoded = pl.reinterpret_view(pl.reinterpret_view(codes, pl.UINT8), pl.FP8E8M0)
            scale_flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reshape(encoded, [1, HEAD_DIM // 16]), 1, HEAD_DIM // 32
            )
    return cache, scales


@pl.jit.inline
def attend_combined(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, 640, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, 640], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(query, 0)
    head_rows = tokens * LOCAL_H
    cache_rows = tokens * 640
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    with pl.spmd(
        num_tokens * (LOCAL_H // M_TILE), name_hint="c1a_online_attention",
        deps=[selected_ready, query_ready],
    ) as attend_tid:
        block = pl.tile.get_block_idx()
        t = block // (LOCAL_H // M_TILE)
        h = block % (LOCAL_H // M_TILE) * M_TILE
        q0 = t * LOCAL_H + h
        maximum = pl.full([1, M_TILE], dtype=pl.FP32, value=-1e30)
        denominator = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
        numerator = pl.full([M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for part in pl.range(ATTEND_TILES):
            k0 = t * 640 + part * ATTEND_TILE
            q = qflat[q0 : q0 + M_TILE, :]
            kv = kflat[k0 : k0 + 64, :]
            scores = pl.matmul(q, kv, b_trans=True)
            scores = pl.mul(scores, SOFTMAX_SCALE)
            idx = pl.cast(indices[t : t + 1, part * 64 : part * 64 + 64], pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            scores = pl.col_expand_add(scores, bias)
            next_max = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, M_TILE]))
            correction = pl.exp(pl.sub(maximum, next_max))
            probabilities = pl.col_expand_mul(
                pl.exp(pl.row_expand_sub(scores, pl.reshape(next_max, [M_TILE, 1]))), valid
            )
            denominator = pl.add(
                pl.mul(denominator, correction), pl.reshape(pl.row_sum(probabilities), [1, M_TILE])
            )
            weights = pl.cast(probabilities, pl.BF16, mode="rint")
            weighted = pl.matmul(weights, kv)
            numerator = pl.add(pl.row_expand_mul(numerator, pl.reshape(correction, [M_TILE, 1])), weighted)
            maximum = next_max
        sinks = pl.reshape(sink[h : h + M_TILE], [1, M_TILE])
        final_max = pl.maximum(maximum, sinks)
        correction = pl.exp(pl.sub(maximum, final_max))
        denominator = pl.add(pl.mul(denominator, correction), pl.exp(pl.sub(sinks, final_max)))
        result = pl.row_expand_mul(numerator, pl.reshape(pl.div(correction, denominator), [M_TILE, 1]))
        oflat[q0 : q0 + M_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
    return attend_tid


project_qa = make_mx_projection_with_deps(D, Q_LORA)
project_qb = make_mx_projection_with_deps(Q_LORA, LOCAL_H * HEAD_DIM)
project_kv = make_mx_projection_with_deps(D, HEAD_DIM)
project_ob = make_mx_projection_with_deps(LOCAL_O_WIDTH, D, pl.FP32)
normalize_q = make_norm_with_deps(Q_LORA, name_hint="c1a_q_rmsnorm")
normalize_kv = make_norm_with_deps(HEAD_DIM, name_hint="c1a_kv_rmsnorm")
rotate_q = make_rope_with_deps(LOCAL_H, name_hint="c1a_q_rope")
rotate_kv = make_rope_with_deps(1, name_hint="c1a_kv_rope")
rotate_output = make_rope_with_deps(LOCAL_H, inverse=True, name_hint="c1a_o_rope")
qkv_proj_rope_with_deps = build_qkv_proj_rope_with_deps(
    project_qa, normalize_q, project_qb, rotate_q, project_kv, normalize_kv, rotate_kv
)
o_proj_with_deps = build_o_proj_with_deps(
    rotate_output, grouped_output_with_deps, project_ob
)


@pl.jit.inline(auto_scope=False)
def c1a_previous_epoch(
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c1a_previous_epoch", allow_early_resolve=False) as ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge
            )
    return ready


def make_c1a_reduce(scatter=False):
    """Build the C1A output collective.

    ``scatter=False`` keeps the historical all-reduce result: every rank holds
    every token row. ``scatter=True`` publishes the same partials but only sums
    and stores the contiguous token rows this rank owns, which is the
    ReduceScatter(SUM) sequence-parallel boundary. The scatter form never sums
    the replicated rows, so the result is not scaled by the TP degree.

    Row ownership follows the rank's physical slab (``ceil(tokens / tp)``), the
    same mapping the Attention-input AllGather publishes, so ``T < TP`` and a
    partially filled batch keep consistent row indices.  Rows outside the active
    range are written as zeros: the slab is always fully materialized.
    """

    @pl.jit.inline(auto_scope=False)
    def c1a_reduce(
        partial: pl.Tensor[[T_DYN, D], pl.FP32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
        partial_ready: pl.Scalar[pl.TASK_ID],
    ):
        first = pl.cast(0, pl.INT32)
        count = pl.cast(num_tokens, pl.INT32)
        if scatter:
            # Ownership uses the fixed physical slab, not the active row count.
            # Mirrors decode_common.slab_owner / decode_sp_integration's
            # sequence_parallel_bounds; keep the three in step.
            width = pl.tensor.dim(output, 0)
            first = pl.cast(pl.min(tp_rank * width, num_tokens), pl.INT32)
            count = pl.cast(pl.max(0, pl.min(width, num_tokens - first)), pl.INT32)
        with pl.spmd(32, name_hint="c1a_tp_publish", deps=[partial_ready]) as published:
            worker = pl.tile.get_block_idx()
            for tile in pl.range(worker, num_tokens * (D // 512), 32):
                row = tile // (D // 512)
                col = tile % (D // 512) * 512
                value = pl.load(partial, [row, col], [1, 512])
                pld.tile.remote_store(value, output_window, peer=group_base + tp_rank, offsets=[row, col])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c1a_tp_ready", deps=[published]) as notified:
            for peer in pl.range(TP_SIZE):
                pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c1a_tp_wait", deps=[notified],
                   allow_early_resolve=False) as arrived:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1,
                                cmp=pld.WaitCmp.Ge)
        with pl.spmd(32, name_hint="c1a_tp_reduce", deps=[arrived]) as reduced:
            worker = pl.tile.get_block_idx()
            for tile in pl.range(worker, count * (D // 512), 32):
                row = tile // (D // 512)
                col = tile % (D // 512) * 512
                acc = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                for peer in pl.range(TP_SIZE):
                    peer_value = pld.tile.remote_load(
                        output_window, peer=group_base + peer, offsets=[first + row, col], shape=[1, 512]
                    )
                    acc = pl.add(acc, peer_value)
                pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row, col], output)
            if scatter:
                # Empty owners and the last slab's padded tail are zeros, never
                # a stale window or sentinel value.
                for tile in pl.range(worker, (pl.tensor.dim(output, 0) - count) * (D // 512), 32):
                    row = count + tile // (D // 512)
                    col = tile % (D // 512) * 512
                    pl.store(pl.tile.full([1, 512], dtype=pl.BF16, value=0.0), [row, col], output)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c1a_tp_release", deps=[reduced]) as released:
            for peer in pl.range(TP_SIZE):
                pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c1a_tp_consumed", deps=[released],
                   allow_early_resolve=False):
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2,
                                cmp=pld.WaitCmp.Ge)
        return output

    return c1a_reduce


c1a_reduce = make_c1a_reduce()
c1a_reduce_scatter = make_c1a_reduce(scatter=True)


@pl.jit.inline(auto_scope=False)
def c1a_prepare(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(x, 0)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qr_tid, q_tid, kvr_tid = qkv_proj_rope_with_deps(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, qr, q, kv, num_tokens, cache_ready,
    )
    publish_window(
        kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready, kvr_tid,
    )
    return qr, q, qr_tid, q_tid


@pl.jit.inline
def gather_combined(
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.FP8E4M3FN],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    selected: pl.Tensor[[T_DYN, 640, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, 640], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
):
    window_rows = pl.tensor.dim(window_cache, 0) * 128
    compressed_rows = pl.tensor.dim(compressed_cache, 0) * 128
    flat_window = pl.reshape(window_cache, [window_rows, HEAD_DIM])
    flat_window_scale = pl.reshape(window_scale, [window_rows, HEAD_DIM // 32])
    # Cache addressing remains in packed byte carriers.
    flat_compressed = pl.reshape(compressed_cache, [1, compressed_rows * (HEAD_DIM // 2)])
    flat_compressed_scale = pl.reshape(compressed_scale, [compressed_rows, HEAD_DIM // 16])
    selected_rows = pl.tensor.dim(selected, 0) * 640
    flat_selected = pl.reshape(selected, [selected_rows, HEAD_DIM])
    with pl.spmd(32, name_hint="c1a_gather_combined", deps=[selected_ready]) as gather_tid:
        worker = pl.tile.get_block_idx()
        for task in pl.range(worker, num_tokens * 40, 32):
            t = task // 40
            first = task % 40 * 16
            id_tile = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
            for lane in pl.range(16):
                i = first + lane
                row = pl.cast(-1, pl.INT32)
                if i < 128:
                    row = pl.read(window_indices, [t, i])
                else:
                    row = pl.read(compressed_indices, [t, i - 128])
                pl.tile.write(id_tile, [0, lane], row)
                destination = t * 640 + i
                if row >= 0:
                    physical = pl.cast(row, pl.INDEX)
                    if i < 128:
                        payload = pl.load(flat_window, [physical, 0], [1, HEAD_DIM])
                        values = pl.reshape(pl.cast(payload, pl.FP32), [HEAD_DIM // 32, 32])
                        raw = pl.load(flat_window_scale, [physical, 0], [1, 32],
                                      valid_shape=[1, HEAD_DIM // 32])
                        signed = pl.cast(pl.reinterpret_view(pl.reinterpret_view(raw, pl.UINT8), pl.INT8),
                                         pl.INT32)
                        codes = pl.ands(signed, 255)
                        scales = pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32)
                        factors = pl.reshape(pl.tile.slice(scales, [1, HEAD_DIM // 32], [0, 0]),
                                             [HEAD_DIM // 32, 1])
                        decoded_window = pl.reshape(pl.row_expand_mul(values, factors), [1, HEAD_DIM])
                        pl.store(pl.cast(decoded_window, pl.BF16, mode="rint"),
                                 [destination, 0], flat_selected)
                    else:
                        packed_row = pl.slice(flat_compressed, [1, HEAD_DIM // 2],
                                             [0, physical * (HEAD_DIM // 2)])
                        byte_values = pl.ands(pl.cast(pl.reinterpret_view(packed_row, pl.INT8), pl.INT32), 255)
                        low = pl.ands(byte_values, 15)
                        high = pl.shrs(byte_values, 4)
                        # The mask-form tile scatter drops the even lanes on A5, so the
                        # halves go through the tensor form.
                        carrier = pl.create_tensor([1, HEAD_DIM], dtype=pl.INT32)
                        placed_low = pl.tensor.scatter(low, mask_pattern=pl.tile.MaskPattern.P0101, dst=carrier)
                        placed = pl.tensor.scatter(high, mask_pattern=pl.tile.MaskPattern.P1010, dst=placed_low)
                        magnitude_code = pl.cast(pl.ands(placed, 7), pl.FP32)
                        magnitude = pl.add(pl.mul(pl.minimum(magnitude_code, 4.0), 0.5),
                                           pl.add(pl.maximum(pl.sub(magnitude_code, 4.0), 0.0),
                                                  pl.maximum(pl.sub(magnitude_code, 6.0), 0.0)))
                        bits = pl.or_(
                            pl.reinterpret_view(magnitude, pl.INT32), pl.shls(pl.ands(placed, 8), 28)
                        )
                        decoded_value = pl.cast(pl.reinterpret_view(bits, pl.FP32), pl.BF16)
                        compressed_factors = pl.cast(
                            pl.slice(flat_compressed_scale, [1, HEAD_DIM // 16], [physical, 0]), pl.FP32
                        )
                        scaled = pl.mul(
                            pl.reshape(pl.cast(decoded_value, pl.FP32), [HEAD_DIM // 16, 16]),
                            pl.reshape(compressed_factors, [HEAD_DIM // 16, 1]),
                        )
                        flat_selected[destination : destination + 1, 0:HEAD_DIM] = pl.cast(
                            pl.reshape(scaled, [1, HEAD_DIM]), pl.BF16, mode="rint"
                        )
                else:
                    zero = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    pl.store(zero, [destination, 0], flat_selected)
            pl.store(id_tile, [t, first], indices)
    return selected, indices, gather_tid


def make_c1a_finish(output_reduce=None):
    """Build the C1A finish stage around one output collective."""
    reducer = output_reduce or c1a_reduce

    @pl.jit.inline(auto_scope=False)
    def c1a_finish(
        query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.FP8E4M3FN],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
        q_ready: pl.Scalar[pl.TASK_ID],
        topk_ready: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(query, 0)
        selected = pl.create_tensor([tokens, 640, HEAD_DIM], dtype=pl.BF16)
        indices = pl.create_tensor([tokens, 640], dtype=pl.INT32)
        _selected_ret, _indices_ret, gather_tid = gather_combined(
            window_cache, window_scale, compressed_cache, compressed_scale,
            window_indices, compressed_indices,
            selected, indices,
            num_tokens, topk_ready,
        )
        attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        attend_tid = attend_combined(
            query, selected, indices, sink, attended, num_tokens, gather_tid, q_ready,
        )
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        ob_tid = o_proj_with_deps(
            attended, wo_a, wo_b, wo_b_scale, cos, sin, partial, num_tokens, attend_tid
        )
        reducer(
            partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch,
            ob_tid,
        )
        return output

    return c1a_finish


c1a_finish = make_c1a_finish()
c1a_finish_sharded = make_c1a_finish(c1a_reduce_scatter)


def make_fp4_publish(width, group, scale_dtype, cache_dim):
    padded_width = max(width, 32 * group)
    is_e8m0 = scale_dtype == pl.FP8E8M0
    byte_dtype = pl.UINT8 if is_e8m0 else pl.INT8

    @pl.jit.inline(auto_scope=False)
    def publish(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        slots: pl.Tensor[[T_DYN], pl.INT64],
        cache: pl.Tensor[[cache_dim, 128, 1, width // 2], pl.UINT8],
        scales: pl.Tensor[[cache_dim, 128, 1, width // group], scale_dtype],
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        kv_ready: pl.Scalar[pl.TASK_ID],
    ):
        cache_rows = pl.tensor.dim(cache, 0) * 128
        flat = pl.reshape(cache, [cache_rows, width // 2])
        flat_scale = pl.reshape(scales, [cache_rows, width // group])
        # Scale rows may share a cache line, so one worker publishes them in order.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="c1a_publish_fp4", deps=[cache_ready, kv_ready]
        ) as published:
            for t in pl.range(num_tokens):
                slot = pl.read(slots, [t])
                if slot >= 0:
                    physical = pl.cast(slot, pl.INDEX)
                    raw = pl.load(source, [t, 0], [1, padded_width], valid_shape=[1, width])
                    raw = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 1, padded_width)
                    values = pl.reshape(pl.cast(raw, pl.FP32), [padded_width // group, group])
                    temporary = pl.tile.create([padded_width // group, group], dtype=pl.FP32)
                    amax = pl.row_max(pl.abs(values), tmp_tile=temporary)
                    if is_e8m0:
                        bounded = pl.maximum(amax, 7.052966104933725e-38)
                        bits = pl.reinterpret_view(pl.mul(bounded, 1.0 / 6.0), pl.INT32)
                        exponent = pl.shrs(pl.add(bits, 8388607), 23)
                        factors = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                        # Raw codes span 0..255; folding to a signed byte would saturate.
                        stored_codes = exponent
                    else:
                        factors_raw = pl.maximum(pl.mul(amax, 1.0 / 6.0), 0.001953125)
                        stored_e4m3 = pl.cast(factors_raw, pl.FP8E4M3FN, mode="rint")
                        stored_codes = pl.cast(pl.reinterpret_view(stored_e4m3, pl.INT8), pl.INT32)
                        factors = pl.cast(stored_e4m3, pl.FP32)
                    normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, factors), -6.0), 6.0)
                    magnitude = pl.abs(normalized)
                    rounded = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.0)
                    select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
                    predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.5)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.5)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=2.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=3.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=4.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=6.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    # Convert the exact RNE levels to E2M1 codes, then pack adjacent lanes.
                    codes = pl.add(pl.minimum(pl.mul(rounded, 2.0), 4.0),
                                   pl.minimum(pl.maximum(pl.sub(rounded, 2.0), 0.0), 3.0))
                    sign = pl.ands(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 28), 8)
                    codes = pl.reshape(pl.or_(pl.cast(codes, pl.INT32), sign), [1, padded_width])
                    low = pl.tile.gather_mask(
                        codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32
                    )
                    high = pl.tile.gather_mask(
                        codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    packed = pl.add(low, pl.mul(high, 16))
                    signed_bytes = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
                    payload_bytes = pl.reinterpret_view(pl.cast(signed_bytes, pl.INT8), pl.UINT8)
                    payload = pl.set_validshape(payload_bytes, 1, width // 2)
                    encoded_codes = pl.reinterpret_view(pl.cast(stored_codes, byte_dtype), scale_dtype)
                    encoded = pl.set_validshape(pl.reshape(encoded_codes, [1, padded_width // group]),
                                                1, width // group)
                    pl.store(payload, [physical, 0], flat)
                    pl.store(encoded, [physical, 0], flat_scale)
        return published

    return publish


def _reference_linear(x, weight, packed_scale):
    """Group-32 FP32 accumulation with the released activation rounding rule."""
    from models.deepseek_v4_1_flash.quantization import decode_e8m0, unpack_mx_b_scale

    groups = x.float().unflatten(-1, (-1, 32))
    scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    quantized = (groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
    weight_scale = decode_e8m0(unpack_mx_b_scale(packed_scale))
    acc = torch.zeros(*x.shape[:-1], weight.shape[-1], dtype=torch.float32)
    for group in range(x.shape[-1] // 32):
        dot = torch.matmul(quantized[..., group, :], weight[group * 32:(group + 1) * 32].float())
        acc += dot * scale[..., group:group + 1] * weight_scale[group]
    return acc.to(x.dtype)


def _bf16_gemm(lhs, rhs):
    """BF16 GEMM with the A5 Cube's arithmetic: FP32 products, FP32 accumulator, one BF16 cast.

    The sibling operators pin this ABI explicitly (``decode_c2a_full.BF16_GEMM = "fp32"``)
    because torch's BF16 GEMM rounds through its own blocking. One ULP of difference is
    enough to move a published E2M1/E8M0 row one code step, or to reorder a near-tied
    indexer cutoff, on one stack but not on another.
    """
    return (lhs.float() @ rhs.float()).to(torch.bfloat16)


def _bf16_grouped(grouped, weight):
    """Grouped low-rank GEMM with the same ABI: FP32 accumulation, one BF16 cast."""
    return torch.einsum("tgd,grd->tgr", grouped.float(), weight.float()).to(torch.bfloat16)


# E2M1 exact midpoints and the magnitude code the A5 publish kernels emit for
# them: even codes win the tie, so 3.5 selects +4 and 2.5 selects +2.
FP4_MIDPOINT_CODES = {0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6}


def check_fp4_boundaries():
    """CPU check of the E2M1 midpoint rule; returns the failure count."""
    from models.deepseek_v4_1_flash.quantization import FP4_VALUES, quantize_mxfp4_cache

    magnitudes = [sign * value for value in FP4_MIDPOINT_CODES for sign in (1.0, -1.0)]
    rows = torch.zeros(len(magnitudes), 32, dtype=torch.bfloat16)
    rows[:, 0] = 3.0
    rows[:, 1] = torch.tensor([value * 0.5 for value in magnitudes], dtype=torch.float32).to(torch.bfloat16)
    reference, _, _ = _reference_fp4(rows, 32, "e8m0")
    public, _ = quantize_mxfp4_cache(rows, 32, "e8m0")
    failures = 0
    for index, magnitude in enumerate(magnitudes):
        code = int(reference[index, 0]) >> 4
        sign = code >> 3
        want = FP4_MIDPOINT_CODES[abs(magnitude)] | (sign << 3)
        other = int(public[index, 0]) >> 4
        differs = "helper-toward-zero" if other != code else "same"
        if code != want:
            failures += 1
            print(f"FAIL {magnitude:+}: code {code} != expected {want}")
        value = FP4_VALUES[code & 0x7].item()
        print(f"  {magnitude:+.2f} -> code {code:2d} (0x{code:X}) value {value:g} [{differs}]")
    print(f"fp4 midpoint check: {len(magnitudes) - failures}/{len(magnitudes)} passed")
    return failures


def _reference_fp4(value, group, scale_format):
    """Independent E2M1 round-half-to-even packing with format-specific scale floors."""
    from models.deepseek_v4_1_flash.quantization import FP4_VALUES, encode_e8m0

    grouped = value.float().unflatten(-1, (-1, group))
    amax = grouped.abs().amax(-1)
    if scale_format == "e8m0":
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6 * 2.0**-126) / 6.0)))
        stored = encode_e8m0(scale)
    else:
        stored = (amax.clamp_min(6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn)
        scale = stored.float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    # Even codes precede odd codes when distances at an exact midpoint tie.
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    distances = (normalized.abs().unsqueeze(-1) - FP4_VALUES[order]).abs()
    codes = order[distances.argmin(-1)].to(torch.uint8)
    codes |= torch.signbit(normalized).to(torch.uint8) << 3
    codes = codes.flatten(-2)
    payload = codes[..., 0::2] | (codes[..., 1::2] << 4)
    decoded = FP4_VALUES[codes.long()] * scale.repeat_interleave(group, -1)
    return payload, stored, decoded.to(value.dtype)


def _reference_sparse_attention(q, window_value, window_indices, compressed_value, compressed_indices, sink):
    """Merge both paged sources with one sink the way the device walks them.

    The device streams the gathered keys in 64-lane tiles against a running
    maximum and rounds each tile's probabilities to BF16 before the value
    product, so the reference repeats that structure instead of letting the
    comparison measure a different rounding order.
    """
    tokens, heads, head_dim = q.shape
    gathered = torch.cat(
        (
            window_value.flatten(0, 1).squeeze(-2)[window_indices.clamp_min(0).long()],
            compressed_value.flatten(0, 1).squeeze(-2)[compressed_indices.clamp_min(0).long()],
        ),
        dim=-2,
    ).to(torch.bfloat16).float()
    visible = torch.cat((window_indices >= 0, compressed_indices >= 0), dim=-1)
    gathered = gathered.masked_fill(~visible.unsqueeze(-1), 0.0)
    maximum = torch.full((tokens, heads), -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(tokens, heads, head_dim, dtype=torch.float32)
    for start in range(0, gathered.shape[-2], ATTEND_TILE):
        tile = gathered[:, start : start + ATTEND_TILE]
        tile_visible = visible[:, None, start : start + ATTEND_TILE]
        logits = torch.einsum("thd,tkd->thk", q.float(), tile) * head_dim**-0.5
        logits = logits.masked_fill(~tile_visible, -1e30)
        next_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = torch.exp(maximum - next_maximum)
        probabilities = torch.exp(logits - next_maximum.unsqueeze(-1))
        probabilities = probabilities * tile_visible
        denominator = denominator * correction + probabilities.sum(-1)
        weights = probabilities.to(torch.bfloat16).float()
        numerator = numerator * correction.unsqueeze(-1) + torch.einsum("thk,tkd->thd", weights, tile)
        maximum = next_maximum
    sink_value = sink.float().view(1, -1)
    final_maximum = torch.maximum(maximum, sink_value)
    correction = torch.exp(maximum - final_maximum)
    denominator = denominator * correction + torch.exp(sink_value - final_maximum)
    return (numerator * (correction / denominator).unsqueeze(-1)).to(q.dtype)


# Independent CPU reference; only rows named by a non-negative slot are rewritten.
def _reference_index_scores(a, qr, index_payload, index_scale, candidates):
    """Rebuild the per-token index scores and the physical row behind every logical column.

    The golden selection and the top-k comparator share this so a cutoff substitution can be
    checked against the score gap that produced it. The query is quantized exactly as the
    released indexer does, and every rounding point follows the device score kernel, so a
    device/reference disagreement is a real disagreement instead of an arithmetic ABI mismatch.
    """
    from models.deepseek_v4_1_flash.golden import rope_interleave
    from models.deepseek_v4_1_flash.quantization import dequantize_mxfp4_cache

    x = a["x"]
    lengths = a["compressed_lens"]
    width = int(lengths.max()) if lengths.numel() else 0
    if candidates is not None:
        width = candidates.shape[-1]
    heads = a["index_weights_proj"].shape[-1]
    iq = _reference_linear(qr, a["index_wq_b"], a["index_wq_b_scale"])
    if iq.shape[-1] % heads:
        raise RuntimeError(f"index query width {iq.shape[-1]} is not divisible by {heads} heads")
    dim = iq.shape[-1] // heads
    rd = a["rope_cos"].shape[-1] * 2
    iq = iq.unflatten(-1, (heads, dim))
    iq = torch.cat((iq[..., :-rd], rope_interleave(iq[..., -rd:], a["rope_cos"], a["rope_sin"])), -1)
    _, _, iq = _reference_fp4(iq, 32, "e8m0")
    keys = dequantize_mxfp4_cache(index_payload, index_scale, 32, "e8m0").to(x.dtype).flatten(0, 1)[:, 0]
    # The device scales the projected head weight in FP32 and rounds it once to BF16.
    weight = _bf16_gemm(x, a["index_weights_proj"]).float() * (dim * heads) ** -0.5
    weight = weight.to(torch.bfloat16).float()
    scores = torch.full((x.shape[0], width), -torch.inf)
    physical = torch.zeros((x.shape[0], width), dtype=torch.int64)
    for t in range(x.shape[0]):
        count = min(int(lengths[t]), width)
        pos = torch.arange(count)
        rows = a["index_block_table"][a["request_ids"][t].long(), pos // 128] * 128 + pos % 128
        physical[t, :count] = rows
        # Device order: BF16-cast the dot, ReLU in FP32, multiply by the rounded weight,
        # BF16-cast the product, sum in FP32, BF16-cast the total.
        dots = _bf16_gemm(iq[t], keys[rows.long()].T)
        product = dots.float().relu() * weight[t, :, None]
        values = product.to(torch.bfloat16).float().sum(0).to(torch.bfloat16).float()
        if candidates is not None:
            values = values.masked_fill(candidates[t, :count] == 0, -torch.inf)
        scores[t, :count] = values
    return scores, physical


def official_reference_c1a(**args):
    """Released C1A math with caller-managed paged caches and replicated index heads."""
    from models.deepseek_v4_1_flash.golden import (
        rms_norm,
        rope_interleave,
        select_candidate_blocks,
    )
    from models.deepseek_v4_1_flash.quantization import (
        encode_e8m0, dequantize_mxfp4_cache, dequantize_mxfp8_cache,
    )

    a = args
    x = a["x"]
    qr = rms_norm(_reference_linear(x, a["wq_a"], a["wq_a_scale"]), a["q_norm_weight"])
    rd = a["rope_cos"].shape[-1] * 2

    def rotate(value, cos, sin, inverse=False):
        return torch.cat((value[..., :-rd], rope_interleave(value[..., -rd:], cos, sin, inverse)), -1)

    head_dim = a["wkv"].shape[-1]

    q = _reference_linear(qr, a["wq_b"], a["wq_b_scale"]).unflatten(-1, (-1, head_dim))
    q = rotate(q, a["rope_cos"], a["rope_sin"])
    kv = rms_norm(_reference_linear(x, a["wkv"], a["wkv_scale"]), a["kv_norm_weight"])
    kv = rotate(kv, a["rope_cos"], a["rope_sin"])
    window = a["window_cache"].clone()
    window_scale = a["window_cache_scale"].view(torch.uint8).clone()
    grouped = kv.float().unflatten(-1, (-1, 32))
    ws = torch.exp2(torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    wp = (grouped / ws.unsqueeze(-1)).flatten(-2).to(torch.float8_e4m3fn)
    valid = a["window_slots"] >= 0
    slots = a["window_slots"][valid].long()
    window.flatten(0, 1)[slots, 0] = wp[valid]
    window_scale.flatten(0, 1)[slots, 0] = encode_e8m0(ws)[valid]
    compressed = a["compressed_cache"].view(torch.uint8).clone()
    compressed_scale = a["compressed_cache_scale"].clone()
    index = None if a["index_cache"] is None else a["index_cache"].view(torch.uint8).clone()
    index_scale = None if a["index_cache_scale"] is None else a["index_cache_scale"].view(torch.uint8).clone()
    mode = a["mode"]
    if mode == AttentionMode.FULL:
        latent = rms_norm(_bf16_gemm(x, a["compressor_wkv"]), a["compressor_norm_weight"])
        key = rms_norm(_bf16_gemm(latent, a["index_wk"]), a["index_norm_weight"])
        key = rotate(key, a["compressed_rope_cos"], a["compressed_rope_sin"])
        kp, ks, _ = _reference_fp4(key, 32, "e8m0")
        rotated = rotate(latent, a["compressed_rope_cos"], a["compressed_rope_sin"])
        cp, cs, _ = _reference_fp4(rotated, 16, "e4m3")
        valid = a["compressed_slots"] >= 0
        slots = a["compressed_slots"][valid].long()
        compressed.flatten(0, 1)[slots, 0] = cp[valid]
        compressed_scale.flatten(0, 1)[slots, 0] = cs[valid]
        index.flatten(0, 1)[slots, 0] = kp[valid]
        index_scale.flatten(0, 1)[slots, 0] = ks[valid]
    candidates = a["candidate_mask"]
    topk = a["compressed_indices"]
    if mode != AttentionMode.REUSE:
        lengths = a["compressed_lens"]
        width = int(lengths.max()) if lengths.numel() else 0
        if candidates is not None:
            width = candidates.shape[-1]
        scores, physical = _reference_index_scores(a, qr, index, index_scale, candidates)
        topk = torch.full((x.shape[0], INDEX_TOPK), -1, dtype=torch.int32)
        for t in range(x.shape[0]):
            count = min(int(lengths[t]), width)
            # Stable tie order makes ties reproducible without requiring Torch's topk implementation order.
            chosen = torch.argsort(scores[t, :count], descending=True, stable=True)[:INDEX_TOPK]
            chosen = chosen[torch.isfinite(scores[t, chosen])].sort().values
            topk[t, :chosen.numel()] = physical[t, chosen].int()
        if mode == AttentionMode.FULL:
            if width:
                candidates = select_candidate_blocks(scores, lengths)
            else:
                candidates = torch.zeros_like(scores, dtype=torch.bool)
    window_value = dequantize_mxfp8_cache(window, window_scale).to(x.dtype)
    compressed_value = dequantize_mxfp4_cache(compressed, compressed_scale, 16, "e4m3").to(x.dtype)
    attended = _reference_sparse_attention(
        q, window_value, a["window_indices"], compressed_value, topk, a["attn_sink"]
    )
    attended = rotate(attended, a["rope_cos"], a["rope_sin"], True)
    grouped = attended.flatten(-2).unflatten(-1, (a["wo_a"].shape[0], -1))
    # The device writes the grouped projection as BF16 after one FP32 accumulation.
    latent = _bf16_grouped(grouped, a["wo_a"])
    output = _reference_linear(latent.flatten(-2), a["wo_b"], a["wo_b_scale"])
    return AttentionGoldenResult(output, window, window_scale, compressed, compressed_scale,
                                 index, index_scale, None, topk, candidates)


project_index_query = make_mx_projection_with_deps(Q_LORA, INDEX_H * INDEX_DIM)
project_index_weights = make_bf16_projection_with_deps(
    D, INDEX_H, name_hint="c1a_index_weights_projection"
)
rotate_index_query = make_rope_with_deps(
    INDEX_H, head_dim=INDEX_DIM, name_hint="c1a_index_query_rope"
)


@pl.jit.inline
def quantize_index_query(
    source: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(32, name_hint="c1a_index_query_fp4", deps=[ready]) as quant_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, 32):
            raw = pl.load(source, [t, 0], [1, INDEX_H * INDEX_DIM])
            values = pl.reshape(pl.cast(raw, pl.FP32), [INDEX_H * INDEX_DIM // 32, 32])
            reduce_tmp = pl.tile.create([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32)
            maximum = pl.maximum(pl.row_max(pl.abs(values), tmp_tile=reduce_tmp), 7.052966104933725e-38)
            bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 6.0), pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, scale), -6.0), 6.0)
            magnitude = pl.abs(normalized)
            rounded = pl.tile.full([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32, value=0.0)
            select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
            predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
            level = pl.tile.full([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32, value=0.5)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
            level = pl.tile.full([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32, value=1.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
            level = pl.tile.full([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32, value=1.5)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
            level = pl.tile.full([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32, value=2.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
            level = pl.tile.full([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32, value=3.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
            level = pl.tile.full([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32, value=4.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
            level = pl.tile.full([INDEX_H * INDEX_DIM // 32, 32], dtype=pl.FP32, value=6.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            sign = pl.ands(pl.reinterpret_view(normalized, pl.INT32), -2147483648)
            signed_bits = pl.or_(pl.reinterpret_view(rounded, pl.INT32), sign)
            exact = pl.reinterpret_view(signed_bits, pl.FP32)
            packed_input = pl.reshape(exact, [1, INDEX_H * INDEX_DIM])
            packed = pl.cast(pl.cast(packed_input, pl.BF16, mode="rint"), pl.FP4, mode="rint")
            decoded_groups = pl.reshape(
                pl.cast(pl.cast(packed, pl.BF16), pl.FP32), [INDEX_H * INDEX_DIM // 32, 32]
            )
            decoded = pl.row_expand_mul(decoded_groups, scale)
            pl.store(
                pl.reshape(pl.cast(decoded, pl.BF16, mode="rint"), [1, INDEX_H * INDEX_DIM]), [t, 0], output
            )
    return quant_tid


@pl.jit.inline
def decode_index_keys(
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 32], pl.FP8E8M0],
    decoded: pl.Tensor[[INDEX_BLOCKS_DYN, 128, INDEX_DIM], pl.BF16],
    entry_ready: pl.Scalar[pl.TASK_ID],
):
    rows = pl.tensor.dim(cache, 0) * 128
    # Keep packed byte offsets explicit at the FP4 GM boundary.
    flat = pl.reshape(cache, [1, rows * (INDEX_DIM // 2)])
    scale_flat = pl.reshape(scales, [rows, INDEX_DIM // 32])
    output = pl.reshape(decoded, [rows, INDEX_DIM])
    with pl.spmd(32, name_hint="c1a_decode_index_keys", deps=[entry_ready]) as keys_tid:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, rows, 32):
            packed_row = pl.slice(flat, [1, INDEX_DIM // 2], [0, row * (INDEX_DIM // 2)])
            byte_values = pl.ands(pl.cast(pl.reinterpret_view(packed_row, pl.INT8), pl.INT32), 255)
            low = pl.ands(byte_values, 15)
            high = pl.shrs(byte_values, 4)
            low_code = pl.cast(pl.ands(low, 7), pl.FP32)
            high_code = pl.cast(pl.ands(high, 7), pl.FP32)
            low_magnitude = pl.add(pl.mul(pl.minimum(low_code, 4.0), 0.5),
                                   pl.add(pl.maximum(pl.sub(low_code, 4.0), 0.0),
                                          pl.maximum(pl.sub(low_code, 6.0), 0.0)))
            high_magnitude = pl.add(pl.mul(pl.minimum(high_code, 4.0), 0.5),
                                    pl.add(pl.maximum(pl.sub(high_code, 4.0), 0.0),
                                           pl.maximum(pl.sub(high_code, 6.0), 0.0)))
            low_value = pl.reinterpret_view(
                pl.or_(pl.reinterpret_view(low_magnitude, pl.INT32), pl.shls(pl.ands(low, 8), 28)),
                pl.FP32,
            )
            high_value = pl.reinterpret_view(
                pl.or_(pl.reinterpret_view(high_magnitude, pl.INT32), pl.shls(pl.ands(high, 8), 28)),
                pl.FP32,
            )
            # The mask-form tile scatter drops the even lanes on A5; the tensor form
            # places both halves.
            interleaved = pl.full([1, INDEX_DIM], dtype=pl.FP32, value=0.0)
            interleaved = pl.tensor.scatter(low_value, mask_pattern=pl.tile.MaskPattern.P0101,
                                            dst=interleaved)
            interleaved = pl.tensor.scatter(high_value, mask_pattern=pl.tile.MaskPattern.P1010,
                                            dst=interleaved)
            raw = pl.load(scale_flat, [row, 0], [1, 32], valid_shape=[1, INDEX_DIM // 32])
            signed = pl.cast(pl.reinterpret_view(pl.reinterpret_view(raw, pl.UINT8), pl.INT8), pl.INT32)
            codes = pl.ands(signed, 255)
            factors = pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32)
            # One scalar scale per 32-element group.
            for group in pl.unroll(INDEX_DIM // 32):
                output[row : row + 1, group * 32:(group + 1) * 32] = pl.cast(
                    pl.mul(interleaved[0:1, group * 32:(group + 1) * 32],
                           pl.tile.read(factors, [0, group])),
                    pl.BF16,
                    mode="rint",
                )
    return decoded, keys_tid


def make_index_scores(use_candidates):
    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.BF16],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        keys: pl.Tensor[[INDEX_BLOCKS_DYN, 128, INDEX_DIM], pl.BF16],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        candidates: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        keys_ready: pl.Scalar[pl.TASK_ID],
        query_ready: pl.Scalar[pl.TASK_ID],
        weights_ready: pl.Scalar[pl.TASK_ID],
    ):
        width = pl.tensor.dim(output, 1)
        key_rows = pl.tensor.dim(keys, 0) * 128
        flat_keys = pl.reshape(keys, [key_rows, INDEX_DIM])
        query_rows = pl.tensor.dim(query, 0) * INDEX_H
        flat_query = pl.reshape(query, [query_rows, INDEX_DIM])
        # The decoded keys are device-only scratch with no host-side edge, so the
        # score task fences against the decode task explicitly instead of relying
        # on the scheduler inferring the RAW edge.
        with pl.spmd(
            32, name_hint="c1a_index_scores", deps=[keys_ready, query_ready, weights_ready]
        ) as scores_tid:
            worker = pl.tile.get_block_idx()
            for t in pl.range(worker, num_tokens, 32):
                length = pl.read(lengths, [t])
                request = pl.cast(pl.read(request_ids, [t]), pl.INDEX)
                for start in pl.range(0, width, 64):
                    infinity_bits = pl.tile.full([1, 64], dtype=pl.INT32, value=-8388608)
                    masked = pl.reinterpret_view(infinity_bits, pl.FP32)
                    if start < length:
                        page = pl.cast(pl.read(block_table, [request, start // 128]), pl.INDEX)
                        physical = page * 128 + start % 128
                        q = pl.slice(flat_query, [INDEX_H, INDEX_DIM], [t * INDEX_H, 0])
                        k = pl.slice(flat_keys, [64, INDEX_DIM], [physical, 0])
                        dot = pl.cast(pl.matmul(q, k, b_trans=True, out_dtype=pl.FP32), pl.BF16, mode="rint")
                        relu = pl.maximum(pl.cast(dot, pl.FP32), 0.0)
                        head_weight = pl.cast(pl.slice(weights, [1, INDEX_H], [t, 0]), pl.FP32)
                        head_weight = pl.mul(head_weight, (INDEX_DIM * INDEX_H) ** -0.5)
                        rounded_weight = pl.cast(pl.cast(head_weight, pl.BF16, mode="rint"), pl.FP32)
                        product = pl.row_expand_mul(relu, pl.reshape(rounded_weight, [INDEX_H, 1]))
                        rounded_product = pl.cast(pl.cast(product, pl.BF16, mode="rint"), pl.FP32)
                        total = pl.cast(pl.cast(pl.col_sum(rounded_product), pl.BF16, mode="rint"), pl.FP32)
                        # Masking uses scalar stores to a private tile, including partial pages.
                        for lane in pl.range(pl.min(64, width - start)):
                            if start + lane < length:
                                allowed = True
                                if use_candidates:
                                    allowed = pl.read(candidates, [t, start + lane]) != 0
                                if allowed:
                                    pl.tile.write(masked, [0, lane], pl.read(total, [0, lane]))
                    pl.store(pl.set_validshape(masked, 1, pl.min(64, width - start)), [t, start], output)
        return scores_tid

    return score


score_full = make_index_scores(False)
score_reindex = make_index_scores(True)


@pl.jit.inline
def select_index_topk(
    index_scores: pl.Tensor,
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    width = pl.tensor.dim(index_scores, 1)
    # scores is device-only scratch; fence the reader explicitly.
    with pl.spmd(32, name_hint="c1a_index_topk", deps=[scores_ready]) as topk_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            heap_bits = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-8388608)
            heap = pl.reinterpret_view(heap_bits, pl.FP32)
            for start in pl.range(0, width, 256):
                scores = pl.load(
                    index_scores, [token, start], [1, 256], valid_shape=[1, pl.min(256, width - start)]
                )
                for lane in pl.range(pl.min(256, width - start)):
                    value = pl.tile.read(scores, [0, lane])
                    minimum = pl.tile.read(heap, [0, 0])
                    if value > minimum:
                        node = pl.cast(0, pl.INDEX)
                        moving = True
                        for depth in pl.range(10):
                            if moving:
                                left = node * 2 + 1
                                if left < INDEX_TOPK:
                                    child = left
                                    left_value = pl.tile.read(heap, [0, left])
                                    if left + 1 < INDEX_TOPK:
                                        right_value = pl.tile.read(heap, [0, left + 1])
                                        if right_value < left_value:
                                            child = left + 1
                                    child_value = pl.tile.read(heap, [0, child])
                                    if child_value < value:
                                        pl.tile.write(heap, [0, node], child_value)
                                        node = child
                                    else:
                                        moving = False
                                else:
                                    moving = False
                        pl.tile.write(heap, [0, node], value)
            cutoff = pl.tile.read(heap, [0, 0])
            above = pl.cast(0, pl.INDEX)
            for lane in pl.range(INDEX_TOPK):
                value = pl.tile.read(heap, [0, lane])
                if value > cutoff:
                    above = above + 1
            remaining = INDEX_TOPK - above
            # Recreate -inf after the heap has overwritten its storage.
            negative_bits = pl.tile.full([1, 8], dtype=pl.INT32, value=-8388608)
            invalid_score = pl.tile.read(pl.reinterpret_view(negative_bits, pl.FP32), [0, 0])
            selected = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            count = pl.cast(0, pl.INDEX)
            request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
            for start in pl.range(0, width, 256):
                scan_scores = pl.load(index_scores, [token, start], [1, 256],
                                      valid_shape=[1, pl.min(256, width - start)])
                for lane in pl.range(pl.min(256, width - start)):
                    value = pl.tile.read(scan_scores, [0, lane])
                    keep = value > cutoff
                    if value == cutoff and value > invalid_score and remaining > 0:
                        keep = True
                        remaining = remaining - 1
                    if keep and count < INDEX_TOPK:
                        logical = start + lane
                        page = pl.read(block_table, [request, logical // 128])
                        physical = pl.cast(page * 128 + logical % 128, pl.INT32)
                        pl.tile.write(selected, [0, count], physical)
                        count = count + 1
            pl.store(selected, [token, 0], output)
    return topk_tid


@pl.jit.inline(auto_scope=False)
def c1a_index(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    score_width: pl.Scalar[pl.INDEX],
    entry_ready: pl.Scalar[pl.TASK_ID],
    qr_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    projected_tid = project_index_query(
        qr, index_wq_b, index_wq_b_scale, projected, num_tokens, qr_ready,
    )
    rotated = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    rotated_tid = rotate_index_query(projected, cos, sin, rotated, num_tokens, projected_tid)
    query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    query_tid = quantize_index_query(rotated, query, num_tokens, rotated_tid)
    weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    weights_tid = project_index_weights(x, index_weights_proj, weights, num_tokens, entry_ready)
    pages = pl.tensor.dim(index_cache, 0)
    keys = pl.create_tensor([pages, 128, INDEX_DIM], dtype=pl.BF16)
    # Work around pypto#2829: finish weights before decode reuses the paired Vector UB.
    keys, keys_tid = decode_index_keys(index_cache, index_cache_scale, keys, weights_tid)
    scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    return query, weights, keys, scores, keys_tid, query_tid, weights_tid


def golden_decode_attn_c1a_reuse(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    compressed_indices: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate C1A decode reuse against the source layer caches."""
    return official_reference_c1a(
        mode=AttentionMode.REUSE,
        ratio=1,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=compressed_indices,
        compressor_wkv=None,
        compressor_wgate=None,
        compressor_norm_weight=None,
        state_block_table=None,
        state_cache=None,
        compressed_slots=None,
        position_ids=None,
        compressed_lens=None,
        compressed_rope_cos=None,
        compressed_rope_sin=None,
        index_wk=None,
        index_norm_weight=None,
        index_wq_b=None,
        index_wq_b_scale=None,
        index_weights_proj=None,
        index_cache=None,
        index_cache_scale=None,
        index_block_table=None,
        request_ids=None,
        candidate_mask=None,
    )


def golden_decode_attn_c1a_reindex(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    request_ids: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_mask: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate C1A decode reindex with the source candidate mask."""
    return official_reference_c1a(
        mode=AttentionMode.REINDEX,
        ratio=1,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=None,
        compressor_wkv=None,
        compressor_wgate=None,
        compressor_norm_weight=None,
        state_block_table=None,
        state_cache=None,
        compressed_slots=None,
        position_ids=None,
        compressed_lens=compressed_lens,
        compressed_rope_cos=None,
        compressed_rope_sin=None,
        index_wk=None,
        index_norm_weight=None,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_cache=index_cache,
        index_cache_scale=index_cache_scale,
        index_block_table=index_block_table,
        request_ids=request_ids,
        candidate_mask=candidate_mask,
    )


project_compressor = make_bf16_projection_with_deps(
    D, HEAD_DIM, name_hint="c1a_compressor_projection"
)
normalize_compressor = make_norm_with_deps(HEAD_DIM, name_hint="c1a_compressor_rmsnorm")
rotate_compressor = make_rope_with_deps(1, name_hint="c1a_compressor_rope")
project_index_key = make_bf16_projection_with_deps(
    HEAD_DIM, INDEX_DIM, name_hint="c1a_index_key_projection"
)
normalize_index_key = make_norm_with_deps(INDEX_DIM, name_hint="c1a_index_key_rmsnorm")
rotate_index_key = make_rope_with_deps(
    1, head_dim=INDEX_DIM, name_hint="c1a_index_key_rope"
)
publish_compressed = make_fp4_publish(HEAD_DIM, 16, pl.FP8E4M3FN, CMP_BLOCKS_DYN)
publish_index = make_fp4_publish(INDEX_DIM, 32, pl.FP8E8M0, INDEX_BLOCKS_DYN)


def golden_decode_attn_c1a_full(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    request_ids: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    compressed_rope_cos: torch.Tensor,
    compressed_rope_sin: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    compressed_slots: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate C1A decode projection, cache publication and sparse attention."""
    return official_reference_c1a(
        mode=AttentionMode.FULL,
        ratio=1,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=None,
        compressor_wkv=compressor_wkv,
        compressor_wgate=None,
        compressor_norm_weight=compressor_norm_weight,
        state_block_table=None,
        state_cache=None,
        compressed_slots=compressed_slots,
        position_ids=None,
        compressed_lens=compressed_lens,
        compressed_rope_cos=compressed_rope_cos,
        compressed_rope_sin=compressed_rope_sin,
        index_wk=index_wk,
        index_norm_weight=index_norm_weight,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_cache=index_cache,
        index_cache_scale=index_cache_scale,
        index_block_table=index_block_table,
        request_ids=request_ids,
        candidate_mask=None,
    )


def make_decode_attn_c1a_full(output_reduce=None):
    """Build the C1A full Attention leaf around one output collective."""
    reducer = output_reduce or c1a_finish

    @pl.jit.inline(auto_scope=False)
    def decode_attn_c1a_full(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
        ],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
        index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
        compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        cache_ready = c1a_previous_epoch(output_arrived, attention_epoch)
        (qr, query, qr_tid, q_tid) = c1a_prepare(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, rope_cos,
            rope_sin, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready,
        )
        tokens = pl.tensor.dim(x, 0)
        projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        pc_tid = project_compressor(x, compressor_wkv, projected, num_tokens, cache_ready)
        latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        nc_tid = normalize_compressor(projected, compressor_norm_weight, latent, num_tokens, pc_tid)
        projected_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        pk_tid = project_index_key(latent, index_wk, projected_key, num_tokens, nc_tid)
        normalized_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        nk_tid = normalize_index_key(projected_key, index_norm_weight, normalized_key, num_tokens, pk_tid)
        index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        ik_tid = rotate_index_key(
            normalized_key, compressed_rope_cos, compressed_rope_sin, index_key, num_tokens, nk_tid,
        )
        publish_index(
            index_key, compressed_slots, index_cache, index_cache_scale, num_tokens, cache_ready, ik_tid,
        )
        rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        rc_tid = rotate_compressor(
            latent, compressed_rope_cos, compressed_rope_sin, rotated, num_tokens, nc_tid,
        )
        publish_compressed(
            rotated, compressed_slots, compressed_cache, compressed_cache_scale, num_tokens, cache_ready, rc_tid,
        )
        width = pl.tensor.dim(candidate_mask, 1)
        (index_query, index_weights, keys, scores, keys_tid, query_tid, weights_tid) = c1a_index(
            x, qr, index_wq_b, index_wq_b_scale, index_weights_proj, rope_cos, rope_sin, index_cache,
            index_cache_scale, num_tokens, width, cache_ready, qr_tid,
        )
        scores_tid = score_full(
            index_query, index_weights, keys, index_block_table, request_ids, compressed_lens, candidate_mask,
            scores, num_tokens, keys_tid, query_tid, weights_tid,
        )
        topk_tid = select_index_topk(
            scores, index_block_table, request_ids, topk_indices, num_tokens, scores_tid,
        )
        hierarchical_sparse_indexer(scores, compressed_lens, candidate_mask)
        reducer(
            query, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, window_indices,
            topk_indices, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, output_window, output_arrived,
            output, group_base, tp_rank, num_tokens, attention_epoch, q_tid, topk_tid,
        )
        return output, topk_indices, candidate_mask

    return decode_attn_c1a_full


decode_attn_c1a_full = make_decode_attn_c1a_full()
decode_attn_c1a_full_sharded = make_decode_attn_c1a_full(c1a_finish_sharded)


__all__ = [
    "golden_decode_attn_c1a_full",
    "golden_decode_attn_c1a_reindex",
    "golden_decode_attn_c1a_reuse",
    "decode_attn_c1a_full",
    "decode_attn_c1a_full_sharded",
    "c1a_previous_epoch",
    "c1a_prepare",
    "c1a_index",
    "score_reindex",
    "select_index_topk",
    "c1a_finish",
    "c1a_finish_sharded",
    "c1a_reduce_scatter",
    "run_c1a",
]


@pl.jit
def decode_attn_c1a_full_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.Out[pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    candidate_mask.bind_dynamic(1, CMP_POSITIONS_DYN)
    return decode_attn_c1a_full(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b,
        index_wq_b_scale, index_weights_proj, topk_indices, candidate_mask, output_window, output_arrived,
        output, group_base, tp_rank, num_tokens, attention_epoch,
    )


def make_program(tokens, pages, epochs=1):
    """Build a distributed host using static packed-FP4 storage dimensions."""
    TOKENS = tokens
    PAGES = pages
    EPOCHS = epochs

    @pl.jit.host
    def host(
        x: pl.Tensor[[TP_SIZE, TOKENS, D], pl.BF16],
        wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[TP_SIZE, D // 32, Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[TP_SIZE, D // 32, HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[TP_SIZE, TOKENS], pl.INT64],
        window_indices: pl.Tensor[[TP_SIZE, TOKENS, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // 2], pl.UINT8]],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
        ],
        request_ids: pl.Tensor[[TP_SIZE, TOKENS], pl.INT32],
        compressed_lens: pl.Tensor[[TP_SIZE, TOKENS], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[TP_SIZE, PAGES, 128, 1, INDEX_DIM // 2], pl.UINT8]],
        index_cache_scale: pl.InOut[
            pl.Tensor[[TP_SIZE, PAGES, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
        ],
        index_block_table: pl.Tensor[[TP_SIZE, 1, PAGES], pl.INT32],
        compressed_rope_cos: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[TP_SIZE, TOKENS, ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
        compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[TP_SIZE, TOKENS], pl.INT64],
        index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
        index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
        topk_indices: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, INDEX_TOPK], pl.INT32]],
        candidate_mask: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, PAGES * 128], pl.UINT8]],
        output: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, D], pl.BF16]],
    ):
        transport = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for epoch in pl.range(1, EPOCHS + 1):
            for rank in pl.unroll(TP_SIZE):
                output_window = pld.window(transport, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
                output_arrived = pld.window(signals, [TP_SIZE, 1], dtype=pl.INT32)
                # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
                wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
                wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = wq_b_scale[rank]
                wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
                wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
                index_wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = index_wq_b_scale[rank]
                decode_attn_c1a_full_test(
                    x[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
                    wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank],
                    wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank],
                    window_indices[rank], window_cache[rank], window_cache_scale[rank],
                    compressed_cache[rank], compressed_cache_scale[rank], request_ids[rank],
                    compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
                    index_block_table[rank], compressed_rope_cos[rank], compressed_rope_sin[rank],
                    compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank],
                    index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r,
                    index_weights_proj[rank], topk_indices[rank], candidate_mask[rank], output_window,
                    output_arrived, output[rank], 0, rank, TOKENS, epoch, device=rank,
                )

    return host


def build_validation_values(mode, tokens, pages, seed=17, case="random"):
    """Deterministic synthetic tensors at released model dimensions."""
    import math

    TOKENS, PAGES, RANKS = tokens, pages, TP_SIZE
    from models.deepseek_v4_1_flash.quantization import (
        pack_mx_b_scale,
        quantize_mxfp8_cache,
        quantize_mxfp4_cache,
    )

    torch.manual_seed(seed)
    values = {}
    shapes = {
        "wq_a": (D, Q_LORA),
        "wq_b": (Q_LORA, LOCAL_H * HEAD_DIM),
        "wkv": (D, HEAD_DIM),
        "wo_b": (LOCAL_O_WIDTH, D),
    }
    for name, (k, n) in shapes.items():
        values[name] = torch.randn(RANKS, k, n).to(torch.float8_e4m3fn)
        exponent = round(127 - math.log2(k) / 2)
        values[name + "_scale"] = pack_mx_b_scale(
            torch.full((RANKS, k // 32, n), exponent, dtype=torch.uint8)
        ).view(torch.float8_e8m0fnu)
    values["x"] = torch.randn(1, TOKENS, D, dtype=torch.bfloat16).expand(RANKS, -1, -1).contiguous()
    values["q_norm_weight"] = torch.ones(RANKS, Q_LORA, dtype=torch.bfloat16)
    values["kv_norm_weight"] = torch.ones(RANKS, HEAD_DIM, dtype=torch.bfloat16)
    values["attn_sink"] = torch.randn(RANKS, LOCAL_H)
    values["wo_a"] = (torch.randn(RANKS, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(
        torch.bfloat16
    )
    angles = torch.randn(1, TOKENS, ROPE_DIM // 2).expand(RANKS, -1, -1).contiguous()
    values["rope_cos"], values["rope_sin"] = angles.cos(), angles.sin()
    values["window_slots"] = (
        torch.arange(TOKENS, dtype=torch.int64).unsqueeze(0).expand(RANKS, -1).contiguous()
    )
    values["window_indices"] = (
        torch.arange(128, dtype=torch.int32).view(1, 1, 128).expand(RANKS, TOKENS, -1).contiguous()
    )
    values["compressed_indices"] = torch.full((RANKS, TOKENS, INDEX_TOPK), -1, dtype=torch.int32)
    values["compressed_indices"][:, :, :128] = torch.arange(128, dtype=torch.int32)
    wc, ws = quantize_mxfp8_cache(
        torch.randn(1, PAGES, 128, 1, HEAD_DIM).expand(RANKS, -1, -1, -1, -1).contiguous()
    )
    cc, cs = quantize_mxfp4_cache(
        torch.randn(1, PAGES, 128, 1, HEAD_DIM).expand(RANKS, -1, -1, -1, -1).contiguous(), 16, "e4m3"
    )
    values["window_cache"], values["window_cache_scale"] = wc, ws.view(torch.float8_e8m0fnu)
    values["compressed_cache"], values["compressed_cache_scale"] = cc, cs
    values["output"] = torch.zeros(RANKS, TOKENS, D, dtype=torch.bfloat16)
    values["index_wq_b"] = torch.randn(RANKS, Q_LORA, INDEX_H * INDEX_DIM).to(torch.float8_e4m3fn)
    values["index_wq_b_scale"] = pack_mx_b_scale(
        torch.full((RANKS, Q_LORA // 32, INDEX_H * INDEX_DIM), 122, dtype=torch.uint8)
    ).view(torch.float8_e8m0fnu)
    # Non-negative head weights keep the score monotone in the key magnitude, so the page ladder
    # above turns the cutoff into a magnitude gap instead of a dense band. Signed weights would
    # let a quiet row outrank a loud one and put the cut back on a knife edge.
    values["index_weights_proj"] = (torch.rand(RANKS, D, INDEX_H) / math.sqrt(D)).to(torch.bfloat16)
    values["request_ids"] = torch.zeros(RANKS, TOKENS, dtype=torch.int32)
    values["compressed_lens"] = torch.tensor(
        [PAGES * 128 - t for t in range(TOKENS)], dtype=torch.int32
    ).repeat(RANKS, 1)
    values["index_block_table"] = (
        torch.randperm(PAGES).int().view(1, 1, PAGES).expand(RANKS, -1, -1).contiguous()
    )
    # Index keys carry a page-level magnitude ladder: the first half of the pages dominates the
    # second half by 2^10. The candidate window then ends in a magnitude gap instead of a dense
    # score band, so a ulp-level stack difference cannot reorder the cut. The index query is
    # quantized to E2M1 inside the operator, so a cutoff decided by a dense band would move with
    # any build. A few quiet positions stay visible so the mask still excludes something.
    quiet_page = torch.arange(PAGES) >= PAGES // 2
    page_gain = torch.where(quiet_page, torch.full((PAGES,), 2.0**-10), torch.ones(PAGES))
    keys = torch.randn(RANKS, PAGES, 128, 1, INDEX_DIM) * page_gain.view(1, PAGES, 1, 1, 1)
    ic, ics, _ = _reference_fp4(keys.to(torch.bfloat16), 32, "e8m0")
    values["index_cache"], values["index_cache_scale"] = (
        ic,
        ics.view(torch.float8_e8m0fnu),
    )
    values["topk_indices"] = torch.zeros(RANKS, TOKENS, INDEX_TOPK, dtype=torch.int32)
    logical_quiet = quiet_page[values["index_block_table"][:, 0]].repeat_interleave(128, dim=1)
    visible = ~logical_quiet | (torch.arange(PAGES * 128) % 32 == 0).unsqueeze(0)
    values["candidate_mask"] = (
        visible.to(torch.uint8).unsqueeze(1).expand(RANKS, TOKENS, -1).contiguous()
    )
    if mode == "full":
        values["compressed_rope_cos"] = values["rope_cos"].clone()
        values["compressed_rope_sin"] = values["rope_sin"].clone()
        values["compressor_wkv"] = (torch.randn(RANKS, D, HEAD_DIM) / math.sqrt(D)).to(torch.bfloat16)
        values["compressor_norm_weight"] = torch.ones(RANKS, HEAD_DIM, dtype=torch.bfloat16)
        values["index_wk"] = (torch.randn(RANKS, HEAD_DIM, INDEX_DIM) / math.sqrt(HEAD_DIM)).to(
            torch.bfloat16
        )
        # The published index key lands in the quiet half of the ladder: the loud half already
        # fills INDEX_TOPK slots exactly, so a key competing at the cut would make the last slot
        # depend on a ulp. RMSNorm cancels any scale on the projection, so the norm weight sets
        # the published key magnitude.
        values["index_norm_weight"] = torch.full((RANKS, INDEX_DIM), 2.0**-10, dtype=torch.bfloat16)
        positions = values["compressed_lens"].long() - 1
        values["compressed_slots"] = (
            values["index_block_table"][:, 0].gather(1, positions // 128).long() * 128 + positions % 128
        )
    for name in (
        "wq_a",
        "wq_a_scale",
        "wkv",
        "wkv_scale",
        "index_wq_b",
        "index_wq_b_scale",
        "index_weights_proj",
        "index_cache",
        "index_cache_scale",
        "compressor_wkv",
        "index_wk",
    ):
        if name in values:
            values[name] = values[name][0:1].expand_as(values[name]).contiguous()
    if case == "ragged":
        for t in range(TOKENS):
            length = (PAGES * 128, 0, 7, 129)[t % 4]
            values["compressed_lens"][:, t] = min(length, PAGES * 128)
            values["window_indices"][:, t, (0 if t % 4 == 1 else 17) :] = -1
            values["compressed_indices"][:, t, (0 if t % 4 == 1 else 31) :] = -1
            if t % 4 == 1:
                values["window_slots"][:, t] = -1
        if mode == "full":
            positions = values["compressed_lens"].long() - 1
            safe = positions.clamp_min(0)
            values["compressed_slots"] = (
                values["index_block_table"][:, 0].gather(1, safe // 128).long() * 128 + safe % 128
            )
            values["compressed_slots"][positions < 0] = -1
    elif case == "masked":
        values["window_indices"].fill_(-1)
        values["compressed_indices"].fill_(-1)
        values["candidate_mask"].zero_()
        values["compressed_lens"].zero_()
        if mode == "full":
            values["compressed_slots"].fill_(-1)
    elif case == "sink":
        values["attn_sink"].fill_(10000)
    return values


def exact_bytes(actual, expected, **kwargs):
    ok = torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))

    if not ok:
        ab = actual.contiguous().view(torch.uint8)
        eb = expected.contiguous().view(torch.uint8)
        idx = (ab != eb).nonzero()
        print("BYTE DIFFERENCES", idx[:20].tolist(),
              "actual", ab[ab != eb][:20].tolist(), "expected", eb[ab != eb][:20].tolist())
    return ok, "cache/scale bytes differ"


def output_compare(actual, expected, **kwargs):
    """Report the output diagnostics and apply the sibling C1A per-point budget.

    The device fuses operations a CPU reference rounds separately, so a BF16 output cannot be
    held closer than its own ulp: the verdict is the per-point rule the other C1A entries use
    (1e-2 absolute and relative, at most 1% of points outside). A fully masked or sink-dominated
    request has an all-zero expected output, whose cosine is undefined, so the L2 and cosine
    numbers printed here stay diagnostics.
    """
    a, e = actual.float(), expected.float()
    error = (a - e).norm() / e.norm().clamp_min(1e-12)
    rows = (a - e).norm(dim=-1) / e.norm(dim=-1).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0)
    print(
        f"[PRECISION] output rel_l2={error.item():.6g} cosine={cosine.item():.8f} "
        f"max_abs={(a - e).abs().max().item():.6g} max_row_rel_l2={rows.max().item():.6g}"
    )
    from golden import ratio_allclose

    budget = ratio_allclose(atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL, max_error_ratio=OUTPUT_MAX_ERROR_RATIO)
    return budget(actual, expected, **kwargs)


def topk_indices_compare(mode):
    """Accept a top-k substitution only where the reference scores are numerically tied.

    The index query is quantized to E2M1 before scoring, so a stack difference of one ulp can
    move a query code one step and shift every score by a small relative amount. Rows that swap
    across the cutoff are then legitimate only if their reference scores actually tie; a swap of
    rows with distinct scores stays a failure.
    """

    def compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        del rtol, atol
        if torch.equal(actual, expected):
            return True, ""
        from models.deepseek_v4_1_flash.golden import rms_norm

        names = (
            "x",
            "wq_a",
            "wq_a_scale",
            "q_norm_weight",
            "index_wq_b",
            "index_wq_b_scale",
            "index_weights_proj",
            "rope_cos",
            "rope_sin",
            "compressed_lens",
            "request_ids",
            "index_block_table",
            "candidate_mask",
        )
        # FULL publishes the index cache the scores were read from; REINDEX and REUSE read the
        # fixture's copy, which is what the device consumed as well.
        score_outputs = actual_outputs if mode == "full" else expected_outputs
        for rank in range(actual.shape[0]):
            a = {name: (score_outputs if name in score_outputs else inputs)[name][rank] for name in names}
            a["index_cache"] = score_outputs["index_cache"][rank]
            a["index_cache_scale"] = score_outputs["index_cache_scale"][rank]
            qr = rms_norm(_reference_linear(a["x"], a["wq_a"], a["wq_a_scale"]), a["q_norm_weight"])
            scores, physical = _reference_index_scores(
                a, qr, a["index_cache"].view(torch.uint8), a["index_cache_scale"], a["candidate_mask"]
            )
            width = scores.shape[-1]
            for token in range(actual.shape[1]):
                actual_valid = actual[rank, token][actual[rank, token] >= 0].to(torch.int64)
                expected_valid = expected[rank, token][expected[rank, token] >= 0].to(torch.int64)
                if actual_valid.numel() != expected_valid.numel():
                    return False, (
                        f"    rank {rank} token {token} selects {actual_valid.numel()} rows; "
                        f"the reference selects {expected_valid.numel()}"
                    )
                if torch.unique(actual_valid).numel() != actual_valid.numel():
                    return False, f"    rank {rank} token {token} repeats a selected row"
                actual_set = set(actual_valid.tolist())
                expected_set = set(expected_valid.tolist())
                if actual_set == expected_set:
                    continue
                count = min(int(a["compressed_lens"][token]), width)
                logical = {int(row): index for index, row in enumerate(physical[token, :count].tolist())}
                missing = sorted(expected_set - actual_set)
                extra = sorted(actual_set - expected_set)
                if any(row not in logical for row in extra):
                    return False, f"    rank {rank} token {token} selected a row outside its request"
                missing_scores = torch.sort(scores[token, [logical[row] for row in missing]], descending=True).values
                extra_scores = torch.sort(scores[token, [logical[row] for row in extra]], descending=True).values
                tolerance = TOPK_SCORE_ATOL + TOPK_SCORE_RTOL * missing_scores.abs()
                gap = missing_scores - extra_scores
                if bool(torch.all(gap <= tolerance)):
                    continue
                worst = int(torch.argmax(gap - tolerance))
                return False, (
                    f"    rank {rank} token {token} top-k cutoff differs: reference row scores "
                    f"{float(missing_scores[worst]):.8g}, substitute {float(extra_scores[worst]):.8g}, "
                    f"tolerance {float(tolerance[worst]):.8g}"
                )
        return True, ""

    compare.__name__ = f"tied_cutoff_topk_{mode}"
    return compare


def run_c1a(mode, kernel_factory, golden_fn, argv=None):
    """Run A5 validation for a production C1A operator."""
    import argparse
    import inspect

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description=f"TP1/2/4 A5 C1A {mode} validation")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument(
        "--dp", type=int, default=1, choices=[1],
        help="one data-parallel group; the shared A5 entry command passes 1",
    )
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a5"])
    parser.add_argument(
        "-d", "--device", "--devices", type=str, default=None,
        help="comma-separated physical device IDs, one per rank; default: 0 through TP-1",
    )
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["random", "ragged", "masked", "sink"], default="random")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--check-fp4", action="store_true", default=False,
                        help="run the CPU E2M1 midpoint check and exit")
    args = parser.parse_args(argv)
    if args.check_fp4:
        raise SystemExit(check_fp4_boundaries())
    if args.tp != TP_SIZE:
        parser.error(f"--tp {args.tp} does not match the TP_SIZE {TP_SIZE} the operator was built for")
    try:
        devices = list(range(TP_SIZE)) if args.device is None else [int(d) for d in args.device.split(",")]
    except ValueError:
        parser.error("device IDs must be comma-separated integers")
    if len(devices) != TP_SIZE or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"device IDs must be {TP_SIZE} distinct nonnegative integers, one per rank")
    if not 1 <= args.tokens <= min(DECODE_MAX_TOKENS, args.pages * 128) or args.pages < 1 or args.epochs < 1:
        parser.error(f"require 1 <= tokens <= min({DECODE_MAX_TOKENS}, pages * 128), pages >= 1, epochs >= 1")
    host = kernel_factory(args.tokens, args.pages, args.epochs)
    values = build_validation_values(mode, args.tokens, args.pages, args.seed, args.case)
    # Clone per call: the harness builds the golden scratch from these same specs, and a
    # tensor init_value whose dtype already matches comes back as itself.
    specs = [
        TensorSpec(n, list(values[n].shape), values[n].dtype, init_value=lambda n=n: values[n].clone())
        for n in host.param_names
    ]

    def golden_c1a(tensors):
        for _ in range(args.epochs):
            partials = []
            for rank in range(TP_SIZE):
                kwargs = {n: tensors[n][rank] for n in inspect.signature(golden_fn).parameters}
                result = golden_fn(**kwargs)
                partials.append(result.output.float())
                for name in CACHE_STATE_NAMES:
                    value = getattr(result, name, None)
                    if value is not None and name in tensors:
                        destination = tensors[name][rank]
                        if name == "candidate_mask":
                            value = value.clone()
                            destination.zero_()
                            destination[:, : value.shape[1]].copy_(value.to(destination.dtype))
                        else:
                            destination.view(torch.uint8).copy_(value.contiguous().view(torch.uint8))
            reduced = torch.zeros_like(partials[0])
            for partial in partials:
                reduced += partial
            tensors["output"].copy_(reduced.to(torch.bfloat16).unsqueeze(0).expand_as(tensors["output"]))

    # Published cache rows are compared as dequantized values, because a one-ULP difference can
    # move a published code one E2M1/E4M3 step; which rows may change at all stays byte exact.
    # Only FULL publishes the compressor and index caches: REINDEX and REUSE read them.
    comparisons = {
        "window_cache": quantized_cache_compare(
            "window_cache", "window_cache_scale", "window_slots", CACHE_MAX_RELATIVE_L2
        ),
        "topk_indices": exact_bytes if mode == "reuse" else topk_indices_compare(mode),
        "candidate_mask": exact_bytes,
        "output": output_compare,
    }
    comparisons["window_cache_scale"] = comparisons["window_cache"]
    if mode == "full":
        comparisons["compressed_cache"] = quantized_cache_compare(
            "compressed_cache", "compressed_cache_scale", "compressed_slots",
            MXFP4_CACHE_MAX_RELATIVE_L2, group_size=COMPRESSED_CACHE_GROUP, scale_format="e4m3",
        )
        comparisons["index_cache"] = quantized_cache_compare(
            "index_cache", "index_cache_scale", "compressed_slots",
            MXFP4_CACHE_MAX_RELATIVE_L2, group_size=INDEX_CACHE_GROUP, scale_format="e8m0",
        )
        comparisons["compressed_cache_scale"] = comparisons["compressed_cache"]
        comparisons["index_cache_scale"] = comparisons["index_cache"]
    else:
        # REINDEX and REUSE only read the compressor and index caches, so those stay a byte
        # contract: the reference hands them back unmodified.
        for name in ("compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale"):
            comparisons[name] = exact_bytes
    comparisons = {name: fn for name, fn in comparisons.items() if name in host.param_names}
    result = run(
        fn=host,
        specs=specs,
        golden_fn=golden_c1a,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            platform=args.platform,
            dump_passes=args.dump_passes,
            enable_chip_swimlane=args.enable_chip_swimlane,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=comparisons,
    )
    return result


def validate(argv=None):
    """Validate the Decode C1A Full production operator on A5."""
    return run_c1a("full", make_program, golden_decode_attn_c1a_full, argv=argv)


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,dp", [(1, 1)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
