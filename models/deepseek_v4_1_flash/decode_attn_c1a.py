# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A full attention."""

import pypto.language as pl

from sparse_attention import attend_selected, gather_packed
import pypto.language.distributed as pld
import torch

from golden import ScalarSpec, TensorSpec, mapped_pool_ratio_allclose, ratio_allclose, run
from utils import AttentionGoldenResult
from quant import quantize_grouped_mxfp8_weight
from quant import golden_bf16_gemm, golden_mxfp8_linear
from qkv_proj_rope import (
    golden_rope_tail,
    make_bf16_projection_with_deps,
    make_mx_projection_with_deps,
    make_rope_with_deps,
)
from rmsnorm import rms_norm
from config import (
    AttentionMode,
    BLOCK_SIZE,
    COMPRESSED_CACHE_GROUP,
    FLASH as M,
    DECODE_MAX_TOKENS,
    INDEX_CACHE_GROUP,
    MAX_BATCH_PER_DP,
    WINDOW_CACHE_GROUP,
    B_DYN,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    INDEX_POOL_BLOCKS_DYN,
    TABLE_DYN,
    T_DYN,
    TP_SIZE,
)


from config import LOCAL_T_DYN
from rope_tables import precompute_rope_tables
from hierarchical_sparse_indexer import (
    golden_select_candidate_block_pool, index_pool_block_capacity, select_candidate_block_pool,
)
from decode_indexer import (
    make_indexer_prepare, make_fp4_publish,
    score_full,
    select_index_topk,
)
from o_proj import (
    golden_grouped_output,
    grouped_output_with_deps,
    make_o_proj_with_deps as build_o_proj_with_deps,
)
from qkv_proj_rope import (
    make_qkv_proj_rope_with_deps as build_qkv_proj_rope_with_deps,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk

# Model configuration.
ATTEND_TILE = 64

# Tiling.
N_TILE = 128

# Paged state one rank owns.
CACHE_STATE_NAMES = (
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "index_cache",
    "index_cache_scale",
    "topk_indices",
    "pool_block_ids",
    "pool_block_counts",
)

# Validation budgets, matching the sibling C1A entries.
OUTPUT_ATOL = 1e-2
OUTPUT_RTOL = 1e-2
OUTPUT_MAX_ERROR_RATIO = 0.01


@pl.jit.inline
def publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
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
            scale = pl.cast(pl.mul(amax, 1.0 / 448.0), pl.BF16, mode="rint")
            payload = pl.cast(pl.row_expand_div(value, pl.cast(scale, pl.FP32)), pl.FP8E4M3FN, mode="rint")
            flat[slot : slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
            scale_flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reshape(scale, [1, HEAD_DIM // 16]), 1, HEAD_DIM // 32
            )
    return cache, scales


@pl.jit.inline(auto_scope=False)
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
    return attend_selected(
        query, selected, indices, sink, output, num_tokens, selected_ready, query_ready,
    )


project_qa = make_mx_projection_with_deps(D, Q_LORA)
project_qb = make_mx_projection_with_deps(Q_LORA, LOCAL_H * HEAD_DIM)
project_kv = make_mx_projection_with_deps(D, HEAD_DIM)
project_ob = make_mx_projection_with_deps(LOCAL_O_WIDTH, D, pl.FP32)
rotate_q = make_rope_with_deps(LOCAL_H, max_workers=64, name_hint="c1a_q_rope")
rotate_kv = make_rope_with_deps(1, name_hint="c1a_kv_rope")
rotate_output = make_rope_with_deps(LOCAL_H, inverse=True, max_workers=64, name_hint="c1a_o_rope")
qkv_proj_rope_with_deps = build_qkv_proj_rope_with_deps(
    project_qa, project_qb, rotate_q, project_kv, rotate_kv
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
        output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
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
            # Mirrors decode_cp_allgather.slab_owner / utils'
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
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
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


@pl.jit.inline(auto_scope=False)
def gather_combined(
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    selected: pl.Tensor[[T_DYN, 640, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, 640], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
):
    return gather_packed(
        window_cache, window_scale, compressed_cache, compressed_scale,
        window_indices, compressed_indices, selected, indices, num_tokens, selected_ready,
    )


def make_c1a_finish(output_reduce=None):
    """Build the C1A finish stage around one output collective."""
    reducer = output_reduce or c1a_reduce

    @pl.jit.inline(auto_scope=False)
    def c1a_finish(
        query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
        wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
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
            attended, wo_a, wo_a_scale, wo_b, wo_b_scale, cos, sin, partial, num_tokens, attend_tid
        )
        reducer(
            partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch,
            ob_tid,
        )
        return output

    return c1a_finish


c1a_finish = make_c1a_finish()
c1a_finish_sharded = make_c1a_finish(c1a_reduce_scatter)


# E2M1 exact midpoints and the magnitude code the A5 publish kernels emit for
# them: even codes win the tie, so 3.5 selects +4 and 2.5 selects +2.
FP4_MIDPOINT_CODES = {0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6}


def check_fp4_boundaries():
    """CPU check of the E2M1 midpoint rule; returns the failure count."""
    from quant import FP4_VALUES, quantize_mxfp4_cache

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
    from quant import FP4_VALUES, encode_e8m0

    grouped = value.float().unflatten(-1, (-1, group))
    amax = grouped.abs().amax(-1)
    if scale_format == "e8m0":
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6 * 2.0**-126) / 6.0)))
        stored = encode_e8m0(scale)
    else:
        stored = (amax.clamp_min(6 * 2.0**-9) / 6.0).to(torch.bfloat16)
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
def _reference_index_scores(a, qr, index_payload, index_scale):
    """Rebuild the per-token index scores and the physical row behind every logical column."""
    from qkv_proj_rope import golden_rope_interleave as rope_interleave
    from quant import dequantize_mxfp4_cache

    x = a["x"]
    lengths = a["compressed_lens"]
    # Full scores every table position.  Decode Reindex materializes a compact
    # score row from this exact dense reference after selecting its block pool.
    width = a["index_block_table"].shape[-1] * 128
    heads = a["index_weights_proj"].shape[-1]
    iq = golden_mxfp8_linear(qr, a["index_wq_b"], a["index_wq_b_scale"])
    if iq.shape[-1] % heads:
        raise RuntimeError(f"index query width {iq.shape[-1]} is not divisible by {heads} heads")
    dim = iq.shape[-1] // heads
    rd = a["rope_cos"].shape[-1] * 2
    iq = iq.unflatten(-1, (heads, dim))
    iq = torch.cat((iq[..., :-rd], rope_interleave(iq[..., -rd:], a["rope_cos"], a["rope_sin"])), -1)
    _, _, iq = _reference_fp4(iq, 32, "e8m0")
    keys = dequantize_mxfp4_cache(index_payload, index_scale, 32, "e8m0").to(x.dtype).flatten(0, 1)[:, 0]
    # The device scales the projected head weight in FP32 and rounds it once to BF16.
    weight = golden_bf16_gemm(x, a["index_weights_proj"]).float() * (dim * heads) ** -0.5
    weight = weight.to(torch.bfloat16).float()
    scores = torch.full((x.shape[0], width), -torch.inf)
    physical = torch.full((x.shape[0], width), -1, dtype=torch.int64)
    for t in range(x.shape[0]):
        count = max(0, min(int(lengths[t]), width))
        pos = torch.arange(count)
        pages = a["index_block_table"][a["request_ids"][t].long(), pos // 128]
        valid = pages >= 0
        valid_pos = pos[valid]
        rows = pages[valid] * 128 + valid_pos % 128
        physical[t, valid_pos] = rows
        if not valid_pos.numel():
            continue
        # Device order: BF16-cast the dot, ReLU in FP32, multiply by the rounded weight,
        # BF16-cast the product, sum in FP32, BF16-cast the total.
        dots = golden_bf16_gemm(iq[t], keys[rows.long()].T)
        product = dots.float().relu() * weight[t, :, None]
        values = product.to(torch.bfloat16).float().sum(0).to(torch.bfloat16).float()
        scores[t, valid_pos] = values
    return scores, physical


def _reference_pool_index_scores(scores, physical, pool_block_ids, pool_block_counts):
    """Pack dense logical scores into the Reindex block-pool score layout."""
    tokens, dense_width = scores.shape
    capacity = pool_block_ids.shape[-1]
    compact_width = capacity * M.candidate_block_size
    compact_scores = torch.full((tokens, compact_width), -torch.inf, dtype=scores.dtype)
    compact_physical = torch.full((tokens, compact_width), -1, dtype=physical.dtype)
    for token in range(tokens):
        count = max(0, min(int(pool_block_counts[token]), capacity))
        for slot in range(count):
            block = int(pool_block_ids[token, slot])
            logical_begin = block * M.candidate_block_size
            if block < 0 or logical_begin >= dense_width:
                continue
            logical_end = min(logical_begin + M.candidate_block_size, dense_width)
            compact_begin = slot * M.candidate_block_size
            compact_end = compact_begin + (logical_end - logical_begin)
            compact_scores[token, compact_begin:compact_end] = scores[token, logical_begin:logical_end]
            compact_physical[token, compact_begin:compact_end] = physical[token, logical_begin:logical_end]
    return compact_scores, compact_physical


def golden_reference_c1a(**args):
    """Released C1A math with caller-managed paged caches and replicated index heads."""
    from rmsnorm import golden_rms_norm
    from quant import (
        dequantize_mxfp4_cache, dequantize_mxfp8_cache,
    )

    a = args
    x = a["x"]
    qr = golden_rms_norm(golden_mxfp8_linear(x, a["wq_a"], a["wq_a_scale"]), a["q_norm_weight"])
    head_dim = a["wkv"].shape[-1]

    q = golden_mxfp8_linear(qr, a["wq_b"], a["wq_b_scale"]).unflatten(-1, (-1, head_dim))
    q = golden_rope_tail(q, a["rope_cos"], a["rope_sin"])
    kv = golden_rms_norm(golden_mxfp8_linear(x, a["wkv"], a["wkv_scale"]), a["kv_norm_weight"])
    kv = golden_rope_tail(kv, a["rope_cos"], a["rope_sin"])
    window = a["window_cache"].clone()
    window_scale = a["window_cache_scale"].clone()
    grouped = kv.float().unflatten(-1, (-1, 32))
    ws = (grouped.abs().amax(-1).clamp_min(1e-4) / 448.0).to(torch.bfloat16)
    wp = (grouped / ws.float().unsqueeze(-1)).clamp(-448, 448).flatten(-2).to(torch.float8_e4m3fn)
    valid = a["window_slots"] >= 0
    slots = a["window_slots"][valid].long()
    window.view(torch.uint8).flatten(0, 1)[slots, 0] = wp.view(torch.uint8)[valid]
    window_scale.flatten(0, 1)[slots, 0] = ws[valid]
    compressed = a["compressed_cache"].view(torch.uint8).clone()
    compressed_scale = a["compressed_cache_scale"].clone()
    index = None if a["index_cache"] is None else a["index_cache"].view(torch.uint8).clone()
    index_scale = None if a["index_cache_scale"] is None else a["index_cache_scale"].view(torch.uint8).clone()
    mode = a["mode"]
    if mode == AttentionMode.FULL:
        latent = golden_rms_norm(golden_bf16_gemm(x, a["compressor_wkv"]), a["compressor_norm_weight"])
        key = golden_rms_norm(golden_bf16_gemm(latent, a["index_wk"]), a["index_norm_weight"])
        key = golden_rope_tail(key, a["compressed_rope_cos"], a["compressed_rope_sin"])
        kp, ks, _ = _reference_fp4(key, 32, "e8m0")
        rotated = golden_rope_tail(latent, a["compressed_rope_cos"], a["compressed_rope_sin"])
        cp, cs, _ = _reference_fp4(rotated, 16, "bf16")
        valid = a["compressed_slots"] >= 0
        slots = a["compressed_slots"][valid].long()
        compressed.flatten(0, 1)[slots, 0] = cp[valid]
        compressed_scale.view(torch.uint8).flatten(0, 1)[slots, 0] = cs.view(torch.uint8)[valid]
        index.flatten(0, 1)[slots, 0] = kp[valid]
        index_scale.flatten(0, 1)[slots, 0] = ks[valid]
    pool_block_ids = a.get("pool_block_ids")
    pool_block_counts = a.get("pool_block_counts")
    topk = a["compressed_indices"]
    if mode != AttentionMode.REUSE:
        lengths = a["compressed_lens"]
        scores, physical = _reference_index_scores(a, qr, index, index_scale)
        if mode == AttentionMode.REINDEX:
            if pool_block_ids is None or pool_block_counts is None:
                raise ValueError("C1A Reindex requires compact block-pool metadata")
            scores, physical = _reference_pool_index_scores(scores, physical, pool_block_ids, pool_block_counts)
        topk = torch.full((x.shape[0], INDEX_TOPK), -1, dtype=torch.int32)
        for t in range(x.shape[0]):
            if mode == AttentionMode.REINDEX:
                count = min(int(pool_block_counts[t]) * M.candidate_block_size, scores.shape[-1])
            else:
                count = min(int(lengths[t]), scores.shape[-1])
            # Stable tie order makes ties reproducible without requiring Torch's topk implementation order.
            chosen = torch.argsort(scores[t, :count], descending=True, stable=True)[:INDEX_TOPK]
            chosen = chosen[torch.isfinite(scores[t, chosen])].sort().values
            topk[t, :chosen.numel()] = physical[t, chosen].int()
        if mode == AttentionMode.FULL:
            pool_block_ids, pool_block_counts = golden_select_candidate_block_pool(scores, lengths)
    window_value = dequantize_mxfp8_cache(window, window_scale).to(x.dtype)
    compressed_value = dequantize_mxfp4_cache(compressed, compressed_scale, 16, "bf16").to(x.dtype)
    attended = _reference_sparse_attention(
        q, window_value, a["window_indices"], compressed_value, topk, a["attn_sink"]
    )
    attended = golden_rope_tail(attended, a["rope_cos"], a["rope_sin"], True)
    grouped = attended.flatten(-2).unflatten(-1, (a["wo_a"].shape[0], -1))
    # The device writes the grouped projection as BF16 after one FP32 accumulation.
    latent = golden_grouped_output(grouped, a["wo_a"], a["wo_a_scale"])
    output = golden_mxfp8_linear(latent.flatten(-2), a["wo_b"], a["wo_b_scale"])
    return AttentionGoldenResult(
        output, window, window_scale, compressed, compressed_scale, index, index_scale, None, topk,
        pool_block_ids=pool_block_ids, pool_block_counts=pool_block_counts,
    )


# Bind the shared C1A MX projection into the independent indexer preparation.
c1a_index = make_indexer_prepare(
    make_mx_projection_with_deps(Q_LORA, INDEX_H * INDEX_DIM)
)


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
    wo_a_scale: torch.Tensor,
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
    return golden_reference_c1a(
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
        wo_a=wo_a, wo_a_scale=wo_a_scale,
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
        pool_block_ids=None,
        pool_block_counts=None,
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
    wo_a_scale: torch.Tensor,
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
    pool_block_ids: torch.Tensor,
    pool_block_counts: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate C1A Decode Reindex using the immutable Full block pool."""
    return golden_reference_c1a(
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
        wo_a=wo_a, wo_a_scale=wo_a_scale,
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
        pool_block_ids=pool_block_ids,
        pool_block_counts=pool_block_counts,
    )


project_compressor = make_bf16_projection_with_deps(
    D, HEAD_DIM, name_hint="c1a_compressor_projection"
)
rotate_compressor = make_rope_with_deps(1, name_hint="c1a_compressor_rope")
project_index_key = make_bf16_projection_with_deps(
    HEAD_DIM, INDEX_DIM, name_hint="c1a_index_key_projection"
)
rotate_index_key = make_rope_with_deps(
    1, head_dim=INDEX_DIM, name_hint="c1a_index_key_rope"
)
publish_compressed = make_fp4_publish(HEAD_DIM, 16, pl.BF16, CMP_BLOCKS_DYN)
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
    wo_a_scale: torch.Tensor,
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
    return golden_reference_c1a(
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
        wo_a=wo_a, wo_a_scale=wo_a_scale,
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
        pool_block_ids=None,
        pool_block_counts=None,
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
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
        wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
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
        pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
        pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
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
        norm_blocks = (num_tokens + 7) // 8
        with pl.spmd(norm_blocks, name_hint="c1a_compressor_rmsnorm", deps=[pc_tid]) as nc_tid:
            norm_worker = pl.tile.get_block_idx()
            rms_norm(projected, compressor_norm_weight, latent, norm_worker, num_tokens, width=HEAD_DIM)
        projected_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        pk_tid = project_index_key(latent, index_wk, projected_key, num_tokens, nc_tid)
        normalized_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        norm_blocks = (num_tokens + 7) // 8
        with pl.spmd(norm_blocks, name_hint="c1a_index_key_rmsnorm", deps=[pk_tid]) as nk_tid:
            norm_worker = pl.tile.get_block_idx()
            rms_norm(projected_key, index_norm_weight, normalized_key, norm_worker, num_tokens, width=INDEX_DIM)
        index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        ik_tid = rotate_index_key(
            normalized_key, compressed_rope_cos, compressed_rope_sin, index_key, num_tokens, nk_tid,
        )
        index_ready = publish_index(
            index_key, compressed_slots, index_cache, index_cache_scale, num_tokens, cache_ready, ik_tid,
        )
        rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        rc_tid = rotate_compressor(latent, compressed_rope_cos, compressed_rope_sin, rotated, num_tokens, nc_tid)
        publish_compressed(
            rotated, compressed_slots, compressed_cache, compressed_cache_scale, num_tokens, cache_ready, rc_tid,
        )
        # Full score width follows the logical page-table capacity.
        score_width = pl.tensor.dim(index_block_table, 1) * 128
        (index_query, index_query_scale, index_weights, scores, query_tid, weights_tid) = c1a_index(
            x, qr, index_wq_b, index_wq_b_scale, index_weights_proj, rope_cos, rope_sin,
            num_tokens, score_width, cache_ready, qr_tid,
        )
        scores_tid = score_full(
            index_query, index_query_scale, index_weights, index_cache, index_cache_scale,
            index_block_table, request_ids, compressed_lens,
            scores, num_tokens, index_ready, query_tid, weights_tid,
        )
        topk_tid = select_index_topk(scores, index_block_table, request_ids, topk_indices, num_tokens, scores_tid)
        pool_tid = select_candidate_block_pool(
            scores, compressed_lens, pool_block_ids, pool_block_counts, num_tokens, scores_tid,
        )
        selection_ready = pl.system.task_dummy(deps=[topk_tid, pool_tid])
        reducer(
            query, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, window_indices,
            topk_indices, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, output_window, output_arrived,
            output, group_base, tp_rank, num_tokens, attention_epoch, q_tid, selection_ready,
        )
        return output, topk_indices, pool_block_ids, pool_block_counts

    return decode_attn_c1a_full


decode_attn_c1a_full = make_decode_attn_c1a_full()
decode_attn_c1a_full_sharded = make_decode_attn_c1a_full(c1a_finish_sharded)


@pl.jit
def l2_decode_attn_c1a(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
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
    pool_block_ids: pl.Out[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
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
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    return decode_attn_c1a_full(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b,
        index_wq_b_scale, index_weights_proj, topk_indices, pool_block_ids, pool_block_counts,
        output_window, output_arrived,
        output, group_base, tp_rank, num_tokens, attention_epoch,
    )


@pl.jit.host
def l3_decode_attn_c1a(
    x: pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TP_SIZE, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TP_SIZE, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.Out[pl.Tensor[[TP_SIZE, T_DYN], pl.INT32]],
    output: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(1, T_DYN)
    rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
    transport = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
    signals = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
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
        l2_decode_attn_c1a(
            x[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
            wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank],
            wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank],
            window_indices[rank], window_cache[rank], window_cache_scale[rank],
            compressed_cache[rank], compressed_cache_scale[rank], request_ids[rank],
            compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
            index_block_table[rank], compressed_rope_cos[rank], compressed_rope_sin[rank],
            compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r,
            index_weights_proj[rank], topk_indices[rank], pool_block_ids[rank], pool_block_counts[rank], output_window,
            output_arrived, output[rank], 0, rank, num_tokens, attention_epoch, device=rank,
        )



def build_validation_values(mode, tokens, pages, requests=1, seed=17, case="random"):
    """Deterministic synthetic tensors at released model dimensions."""
    import math

    TOKENS, PAGES, RANKS, REQUESTS = tokens, pages, TP_SIZE, requests
    from quant import (
        pack_mx_b_scale,
        quantize_mxfp8_cache,
        quantize_mxfp4_cache,
    )

    torch.manual_seed(seed)
    # Balanced token-to-request split. Every request carries the same page
    # permutation, so the page-addressable caches stay conflict-free while the
    # operator still routes TOKENS tokens through REQUESTS table rows.
    lengths = [TOKENS // REQUESTS + (1 if index < TOKENS % REQUESTS else 0) for index in range(REQUESTS)]
    token_to_req = torch.repeat_interleave(
        torch.arange(REQUESTS), torch.tensor(lengths)
    ).to(torch.int32)
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
    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(
        torch.randn(RANKS, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)
    )
    positions = torch.arange(PAGES * 128 - 1, PAGES * 128 - TOKENS - 1, -1)
    freqs_cos, freqs_sin = precompute_rope_tables(PAGES * 128, compressed_attention=True)
    values["rope_cos"] = freqs_cos[positions].unsqueeze(0).expand(RANKS, -1, -1).contiguous()
    values["rope_sin"] = freqs_sin[positions].unsqueeze(0).expand(RANKS, -1, -1).contiguous()
    values["window_slots"] = (
        torch.arange(TOKENS, dtype=torch.int64).unsqueeze(0).expand(RANKS, -1).contiguous()
    )
    values["window_indices"] = (
        torch.arange(128, dtype=torch.int32).view(1, 1, 128).expand(RANKS, TOKENS, -1).contiguous()
    )
    values["compressed_indices"] = torch.full((RANKS, TOKENS, INDEX_TOPK), -1, dtype=torch.int32)
    values["compressed_indices"][:, :, :128] = torch.arange(128, dtype=torch.int32)
    wc, ws = quantize_mxfp8_cache(torch.randn(1, PAGES, 128, 1, HEAD_DIM).expand(RANKS, -1, -1, -1, -1).contiguous())
    cc, cs = quantize_mxfp4_cache(
        torch.randn(1, PAGES, 128, 1, HEAD_DIM).expand(RANKS, -1, -1, -1, -1).contiguous(), 16, "bf16"
    )
    values["window_cache"], values["window_cache_scale"] = wc, ws
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
    values["request_ids"] = token_to_req.unsqueeze(0).expand(RANKS, -1).contiguous()
    values["compressed_lens"] = torch.tensor(
        [PAGES * 128 - t for t in range(TOKENS)], dtype=torch.int32
    ).repeat(RANKS, 1)
    values["index_block_table"] = (
        torch.randperm(PAGES).int().view(1, 1, PAGES).expand(RANKS, REQUESTS, -1).contiguous()
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
    if mode == "full":
        values["compressed_rope_cos"] = values["rope_cos"].clone()
        values["compressed_rope_sin"] = values["rope_sin"].clone()
        values["compressor_wkv"] = (torch.randn(RANKS, D, HEAD_DIM) / math.sqrt(D)).to(torch.bfloat16)
        values["compressor_norm_weight"] = torch.ones(RANKS, HEAD_DIM, dtype=torch.bfloat16)
        values["index_wk"] = (torch.randn(RANKS, HEAD_DIM, INDEX_DIM) / math.sqrt(HEAD_DIM)).to(torch.bfloat16)
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
        values["compressed_lens"].zero_()
        if mode == "full":
            values["compressed_slots"].fill_(-1)
    elif case == "sink":
        values["attn_sink"].fill_(10000)

    # Build sorted complete-block fixtures with the newest causal block.
    pool_capacity = index_pool_block_capacity(PAGES * 128)
    pool_ids = torch.full((RANKS, TOKENS, pool_capacity), -1, dtype=torch.int32)
    pool_counts = torch.zeros((RANKS, TOKENS), dtype=torch.int32)
    for rank in range(RANKS):
        for token in range(TOKENS):
            visible = max(0, min(int(values["compressed_lens"][rank, token]), PAGES * 128))
            block_count = (visible + M.candidate_block_size - 1) // M.candidate_block_size
            selected = min(block_count, pool_capacity)
            if selected == 0:
                continue
            if selected == block_count:
                chosen = torch.arange(selected, dtype=torch.int32)
            else:
                chosen = torch.cat(
                    (
                        torch.arange(selected - 1, dtype=torch.int32),
                        torch.tensor([block_count - 1], dtype=torch.int32),
                    )
                )
            pool_ids[rank, token, :selected] = chosen
            pool_counts[rank, token] = selected
    values["pool_block_ids"] = pool_ids
    values["pool_block_counts"] = pool_counts
    return values


CACHE_SLOTS = {
    "window_cache": "window_slots",
    "window_cache_scale": "window_slots",
    "compressed_cache": "compressed_slots",
    "compressed_cache_scale": "compressed_slots",
    "index_cache": "compressed_slots",
    "index_cache_scale": "compressed_slots",
}


def cache_compare(world_size, tokens, mode):
    """Mapped rows must match byte for byte and every other cache row must stay unchanged.

    Only FULL publishes the compressor and index caches: REINDEX and REUSE read them, so the
    reference hands those back unmodified and they stay an exact byte contract.
    """
    options = dict(
        mapping_shape=(world_size, tokens),
        block_size=BLOCK_SIZE,
        leading_rank_axis=True,
        atol=0.0,
        rtol=0.0,
        max_error_ratio=0.0,
    )
    compare = {
        name: mapped_pool_ratio_allclose(slots, pool_name=name, **options)
        for name, slots in CACHE_SLOTS.items()
        if mode == "full" or name.startswith("window")
    }
    if mode != "full":
        for name in ("compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale"):
            compare[name] = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    return compare


if __name__ == "__main__":
    import argparse
    import inspect

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1/2/4 A5 C1A full validation")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument(
        "--dp", type=int, default=1, choices=[1],
        help="one data-parallel group; the shared A5 entry command passes 1",
    )
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a5", "a5sim"])
    parser.add_argument(
        "-d", "--device", "--devices", type=str, default=None,
        help="comma-separated physical device IDs, one per rank; default: 0 through TP-1",
    )
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--requests", type=int, default=1)
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
    args = parser.parse_args()
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
    if not 1 <= args.tokens <= min(DECODE_MAX_TOKENS, args.pages * 128) or args.pages < 1:
        parser.error(f"require 1 <= tokens <= min({DECODE_MAX_TOKENS}, pages * 128), pages >= 1")
    if not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"require 1 <= requests <= min({MAX_BATCH_PER_DP}, tokens {args.tokens})")
    host = l3_decode_attn_c1a
    values = build_validation_values("full", args.tokens, args.pages, args.requests, args.seed, args.case)
    # Independent initial storage for each harness allocation.
    specs = [
        TensorSpec(n, list(values[n].shape), values[n].dtype, init_value=lambda n=n: values[n].clone())
        for n in host.param_names if n not in ("num_tokens", "attention_epoch")
    ]

    specs.append(ScalarSpec("num_tokens", torch.int32, args.tokens))
    specs.append(ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True))

    def golden_c1a(tensors):
        partials = []
        for rank in range(TP_SIZE):
            kwargs = {n: tensors[n][rank] for n in inspect.signature(golden_decode_attn_c1a_full).parameters}
            result = golden_decode_attn_c1a_full(**kwargs)
            partials.append(result.output.float())
            for name in CACHE_STATE_NAMES:
                value = getattr(result, name, None)
                if value is not None and name in tensors:
                    destination = tensors[name][rank]
                    destination.view(torch.uint8).copy_(value.contiguous().view(torch.uint8))
        reduced = torch.zeros_like(partials[0])
        for partial in partials:
            reduced += partial
        tensors["output"].copy_(reduced.to(torch.bfloat16).unsqueeze(0).expand_as(tensors["output"]))

    comparisons = {
        "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
        "pool_block_ids": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
        "pool_block_counts": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
        "output": ratio_allclose(atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL, max_error_ratio=OUTPUT_MAX_ERROR_RATIO),
        **cache_compare(len(devices), args.tokens, "full"),
    }
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
    if not result.passed:
        raise SystemExit(result.error or 1)
