# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill sliding-window attention for encoder layers 0 and 1."""


import pypto.language as pl
import pypto.language.distributed as pld
import torch

from utils import AttentionGoldenResult
from o_proj import make_prefill_tp_output_all_reduce
from config import (
    WINDOW_CACHE_GROUP,
    FLASH as M,
    ORI_BLOCKS_DYN,
    T_DYN,
    TP_SIZE,
)

from sparse_attention import (
    SWA_RUN_DESCRIPTOR_WIDTH as RUN_DESCRIPTOR_WIDTH,
    SWA_RUN_PARTS as RUN_PARTS,
    analyze_index_runs,
    attend_shared_cache,
    decode_shared_cache_after_publish,
    build_swa_leaf_specs,
    golden_swa_attention,
    golden_swa_leaf,
)
from o_proj import prefill_o_proj
from qkv_proj_rope import (
    prefill_kv_proj_rope,
    prefill_q_proj_qr,
    prefill_q_proj_rope,
)


import argparse
import os
import functools
from golden import run, ratio_allclose
from config import EP_SIZE, MAX_BATCH_PER_DP
from utils import swa_window_cache_compare

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


# tiling
QUERY_TILE = 128
WORKER_TILE = 64

# Communication capacity.
PREFILL_MAX_TOKENS = 8192

prefill_tp_output_all_reduce = make_prefill_tp_output_all_reduce(PREFILL_MAX_TOKENS)


@pl.jit.inline
def prefill_publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(WORKER_TILE, name_hint="prefill_swa_cache_publish", deps=[cache_ready]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, WORKER_TILE):
            slot_i64 = pl.read(slots, [t])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                source = pl.slice(kv, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
                source_padded = pl.fillpad(source, pad_value=pl.PadValue.zero)
                source = pl.set_validshape(source_padded, 1, HEAD_DIM * 2)
                source_fp32 = pl.cast(source, pl.FP32)
                value = pl.reshape(source_fp32, [HEAD_DIM // 16, 32])
                absolute = pl.abs(value)
                row_maximum = pl.row_max(absolute)
                amax = pl.maximum(row_maximum, 1e-4)
                raw_scale = pl.mul(amax, 1.0 / 448.0)
                scale = pl.cast(raw_scale, pl.BF16, mode="rint")
                scale_fp32 = pl.cast(scale, pl.FP32)
                normalized = pl.row_expand_div(value, scale_fp32)
                payload = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
                payload_row = pl.reshape(payload, [1, HEAD_DIM * 2])
                flat[slot:slot + 1, :] = pl.set_validshape(payload_row, 1, HEAD_DIM)
                scale_row = pl.reshape(scale, [1, HEAD_DIM // 16])
                scale_flat[slot:slot + 1, :] = pl.set_validshape(scale_row, 1, HEAD_DIM // 32)
    return publish_tid


@pl.jit.inline(auto_scope=False)
def prefill_attn_swa_partial(
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
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write FP32 head-TP partials for packed causal SWA; write slots must be unique."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_prefill_previous_epoch", allow_early_resolve=False) as cache_ready:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    tokens = pl.tensor.dim(x, 0)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    # Gate every projection tile before scratch storage is reused by a new epoch.
    with pl.spmd(WORKER_TILE, name_hint="prefill_swa_begin", deps=[cache_ready]):
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, num_tokens, WORKER_TILE):
            kv_projection[row : row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    prefill_kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, kv, num_tokens)
    chunk_done = prefill_publish_window(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
    cache_blocks = pl.tensor.dim(window_cache, 0)
    cache_rows = cache_blocks * 128
    decoded_cache = pl.create_tensor([cache_blocks, 128, 1, HEAD_DIM], dtype=pl.BF16)
    zero_row = pl.create_tensor([32, HEAD_DIM], dtype=pl.BF16)
    # Decode once for the entire packed prompt. The explicit publication
    # dependency protects the hoisted reader independently of query staging.
    decoded_ready = decode_shared_cache_after_publish(window_cache, window_cache_scale, decoded_cache, zero_row, chunk_done)

    chunk_x = pl.create_tensor([QUERY_TILE, D], dtype=pl.BF16)
    chunk_cos = pl.create_tensor([QUERY_TILE, ROPE_DIM // 2], dtype=pl.FP32)
    chunk_sin = pl.create_tensor([QUERY_TILE, ROPE_DIM // 2], dtype=pl.FP32)
    chunk_indices = pl.create_tensor([QUERY_TILE, 128], dtype=pl.INT32)
    qr = pl.create_tensor([QUERY_TILE, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    descriptors = pl.create_tensor([QUERY_TILE, RUN_PARTS, RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
    attended = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    chunk_partial = pl.create_tensor([QUERY_TILE, D], dtype=pl.FP32)

    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        # The prior chunk must finish reading every reusable scratch buffer.
        with pl.spmd(WORKER_TILE, name_hint="prefill_swa_stage", deps=[chunk_done]) as stage_tid:
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, active, WORKER_TILE):
                source_row = start + row
                for col in pl.range(0, D, 512):
                    chunk_x[row:row + 1, col:col + 512] = x[source_row:source_row + 1, col:col + 512]
                chunk_cos[row:row + 1, :] = rope_cos[source_row:source_row + 1, :]
                chunk_sin[row:row + 1, :] = rope_sin[source_row:source_row + 1, :]
                chunk_indices[row:row + 1, :] = window_indices[source_row:source_row + 1, :]
        prefill_q_proj_qr(chunk_x, wq_a, wq_a_scale, q_norm_weight, qr, active)
        prefill_q_proj_rope(qr, wq_b, wq_b_scale, chunk_cos, chunk_sin, q, active)
        index_ready = analyze_index_runs(chunk_indices, descriptors, active, cache_rows)
        attend_shared_cache(
            q, decoded_cache, zero_row,
            descriptors, chunk_indices, attn_sink,
            attended, active, decoded_ready, index_ready,
        )
        prefill_o_proj(attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_cos, chunk_sin, chunk_partial, active)
        with pl.spmd(WORKER_TILE, name_hint="prefill_swa_collect") as collect_tid:
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, active, WORKER_TILE):
                for col in pl.range(0, D, 512):
                    partial[start + row:start + row + 1, col:col + 512] = chunk_partial[row:row + 1, col:col + 512]
        chunk_done = collect_tid
    return partial


def make_prefill_attn_swa(capacity=PREFILL_MAX_TOKENS, reducer=prefill_tp_output_all_reduce):
    """Build head-TP attention around an output transport capacity."""
    @pl.jit.inline(auto_scope=False)
    def prefill_attn_swa(
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
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Return replicated SWA output for the standalone attention interface."""
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        prefill_attn_swa_partial(
            x,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            output_arrived, partial, num_tokens, attention_epoch,
        )
        reducer(partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch)
        return output

    return prefill_attn_swa


prefill_attn_swa = make_prefill_attn_swa()


def golden_prefill_attn_swa(
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
) -> AttentionGoldenResult:
    return golden_swa_attention(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
    )


__all__ = [
    "golden_prefill_attn_swa", "prefill_attn_swa", "prefill_attn_swa_partial", "make_prefill_attn_swa",
]


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_attn_swa(
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
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    prefill_attn_swa(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
        output_window, output_arrived, output,
        rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, num_tokens, attention_epoch,
    )
    return output, window_cache, window_cache_scale


@pl.jit.host
def l3_prefill_attn_swa(
    x: pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    output: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    rope_cos.bind_dynamic(1, T_DYN)
    x.bind_dynamic(0, RANKS_DYN)
    x.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    data_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        data = pld.window(data_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        signal = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        l2_prefill_attn_swa(
            x[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
            wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank],
            window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank],
            output[rank], data, signal, rank, num_tokens, attention_epoch, device=rank,
        )


if __name__ == "__main__":
    mode = "prefill"
    capacity = PREFILL_MAX_TOKENS
    parser = argparse.ArgumentParser(description=f"DeepSeek V4.1 {mode} SWA: A5 precision and timing")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=[2, 4, 8])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", "--batch", type=int, default=257)
    parser.add_argument("--requests", type=int, help="packed requests; default min(tokens, 4)")
    parser.add_argument("--case", default="mixed", choices=("mixed", "prefix", "shuffle", "masked", "zero", "sink"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true", help="save validated inputs and golden outputs for replay")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        parser.error(f"--tp {args.tp} does not match import-time TP_SIZE {TP_SIZE}")
    from pypto.ir import DistributedConfig

    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    args.requests = args.requests if args.requests is not None else min(args.tokens, 4)
    try:
        devices = list(range(TP_SIZE * args.dp)) if args.device is None else [int(d) for d in args.device.split(",")]
    except ValueError:
        parser.error("device IDs must be comma-separated integers")
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("device IDs must be distinct nonnegative integers, with count TP * DP")
    if not 1 <= args.tokens <= capacity or not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"tokens must be in [1, {capacity}]; requests in [1, min({MAX_BATCH_PER_DP}, tokens)]")
    if args.bench and args.enable_chip_swimlane:
        parser.error("benchmark epoch stepping and multi-pass chip swimlane must be run separately")
    torch.set_num_threads(8)
    print(
        f"[SWA] mode={mode} tokens={args.tokens} requests={args.requests} TP={TP_SIZE} DP={args.dp} "
        f"case={args.case} seed={args.seed} devices={devices}",
    )
    if args.bench:
        print(
            "[SWA] Resident device timing excludes compilation, input generation and CPU golden; "
            "each dispatch advances the communication epoch.",
        )
    validation_program = l3_prefill_attn_swa
    validation_specs = build_swa_leaf_specs(args, mode)
    validation_distribution = DistributedConfig(device_ids=devices, num_sub_workers=0)
    validation_config = dict(
        platform=args.platform, distributed_config=validation_distribution,
        enable_chip_swimlane=args.enable_chip_swimlane, enable_dep_gen=args.enable_dep_gen,
    )
    output_compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128)
    if args.case in ("masked", "zero", "sink"):
        output_compare = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    validation_comparisons = {
        "output": output_compare,
        # Projection/RMSNorm BF16 rounding can cross an FP8 boundary. Bound
        # every mapped payload by 2**-9 + abs(expected)/8 and scales by 2%;
        # untouched values remain exact, with no out-of-bound allowance.
        **swa_window_cache_compare(
            len(devices), args.tokens, payload_atol=2 ** -9,
            payload_rtol=2 ** -3, scale_rtol=0.02,
        ),
    }
    result = run(
        fn=validation_program,
        specs=validation_specs,
        golden_fn=functools.partial(golden_swa_leaf, mode=mode),
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        config=validation_config,
        compare_fn=validation_comparisons,
    )
    print(f"[SWA] work_dir={result.work_dir}")
    if args.compile_only and result.passed:
        print("[SWA] Compilation passed; device accuracy was NOT validated.")
    elif args.save_data and result.passed:
        print(f"[SWA] Validated snapshot: {result.work_dir}/data")
    if not result.passed:
        raise SystemExit(result.error or 1)
