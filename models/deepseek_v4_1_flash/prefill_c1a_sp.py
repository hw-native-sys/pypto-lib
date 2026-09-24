# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sequence-parallel mHC composition for packed C1A Full, Reindex and Reuse."""

import argparse
import inspect
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir import DistributedConfig

from golden import ScalarSpec, TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import (
    D,
    HC_MULT,
    TP_SIZE,
    INDEX_TOPK,
    PREFILL_MAX_TOKENS,
    Q_LORA,
    LOCAL_H,
    HEAD_DIM,
    LOCAL_O_WIDTH,
    INDEX_H,
    INDEX_DIM,
)
from models.deepseek_v4_1_flash.attention_sp import (
    SP_T_DYN,
    prefill_sp_input_allgather,
    prefill_sp_output_reduce_scatter,
    prefill_sp_post,
)
from models.deepseek_v4_1_flash import prefill_attn_c1a_full as full
from models.deepseek_v4_1_flash import prefill_attn_c1a_reindex as reindex
from models.deepseek_v4_1_flash import prefill_attn_c1a_reuse as reuse
from models.deepseek_v4_1_flash import prefill_c2a_full as common
from models.deepseek_v4_1_flash.prefill_c2a_full import attention_hc_pre
from models.deepseek_v4_1_flash.prefill_c1a_test_utils import (
    make_fixture_values,
    CASE_NAMES,
    attention_output_compare,
    topk_indices_compare,
)


LEAF_NAMES = (
    "wq_a",
    "wq_a_scale",
    "q_norm_weight",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "wkv_scale",
    "kv_norm_weight",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "rope_cos",
    "rope_sin",
    "window_slots",
    "window_indices",
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "request_ids",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "compressed_rope_cos",
    "compressed_rope_sin",
    "compressor_wkv",
    "compressor_norm_weight",
    "compressed_slots",
    "index_wk",
    "index_norm_weight",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
    "topk_indices",
    "candidate_mask",
    "compressed_indices",
)


def make_rank(mode, epochs=1):
    """Build a device entry taking caller-owned local residuals and global cache metadata."""
    if mode not in ("full", "reindex", "reuse"):
        raise ValueError("mode must be full, reindex or reuse")
    is_full = mode == "full"
    is_reindex = mode == "reindex"
    full_operator = full.make_prefill_attn_c1a_full(full.paged_indexer, prefill_sp_output_reduce_scatter, SP_T_DYN)
    reindex_operator = reindex.make_prefill_attn_c1a_reindex(
        reindex.paged_indexer, prefill_sp_output_reduce_scatter, SP_T_DYN
    )
    reuse_operator = reuse.make_prefill_attn_c1a_reuse(prefill_sp_output_reduce_scatter, SP_T_DYN)

    @pl.jit
    def c1a_sp_rank(
        x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.InOut[pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8]],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN]
        ],
        request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8]],
        index_cache_scale: pl.InOut[
            pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0]
        ],
        index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.BF16],
        compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.InOut[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        candidate_mask: pl.InOut[pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8]],
        compressed_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
        attn_input: pl.Out[pl.Tensor[[C.T_DYN, D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
        input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
        input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
        output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        num_tokens: pl.Tensor[[1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        x_hc.bind_dynamic(0, SP_T_DYN)
        pre_mix.bind_dynamic(0, SP_T_DYN)
        rope_cos.bind_dynamic(0, C.T_DYN)
        rope_sin.bind_dynamic(0, C.T_DYN)
        window_slots.bind_dynamic(0, C.T_DYN)
        window_indices.bind_dynamic(0, C.T_DYN)
        window_cache.bind_dynamic(0, C.ORI_BLOCKS_DYN)
        window_cache_scale.bind_dynamic(0, C.ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(0, C.CMP_BLOCKS_DYN)
        compressed_cache_scale.bind_dynamic(0, C.CMP_BLOCKS_DYN)
        request_ids.bind_dynamic(0, C.T_DYN)
        compressed_lens.bind_dynamic(0, C.T_DYN)
        index_cache.bind_dynamic(0, C.INDEX_BLOCKS_DYN)
        index_cache_scale.bind_dynamic(0, C.INDEX_BLOCKS_DYN)
        index_block_table.bind_dynamic(0, C.B_DYN)
        index_block_table.bind_dynamic(1, C.TABLE_DYN)
        compressed_rope_cos.bind_dynamic(0, C.T_DYN)
        compressed_rope_sin.bind_dynamic(0, C.T_DYN)
        compressed_slots.bind_dynamic(0, C.T_DYN)
        topk_indices.bind_dynamic(0, C.T_DYN)
        candidate_mask.bind_dynamic(0, C.T_DYN)
        candidate_mask.bind_dynamic(1, C.CMP_POSITIONS_DYN)
        compressed_indices.bind_dynamic(0, C.T_DYN)
        attn_input.bind_dynamic(0, C.T_DYN)
        attn_output.bind_dynamic(0, SP_T_DYN)
        next_pre_mix.bind_dynamic(0, SP_T_DYN)
        x_hc_out.bind_dynamic(0, SP_T_DYN)
        active = pl.read(num_tokens, [0])
        tokens = pl.tensor.dim(x_hc, 0)
        group_tokens = pl.tensor.dim(attn_input, 0)
        for step in pl.range(epochs):
            post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
            residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
            local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
            attention_hc_pre(
                x_hc,
                pre_mix,
                hc_attn_fn,
                hc_attn_scale,
                hc_attn_base,
                attn_norm_weight,
                next_pre_mix,
                post_mix,
                residual_mix,
                local_hidden,
            )
            prefill_sp_input_allgather(
                local_hidden,
                input_window,
                input_arrived,
                attn_input,
                rank // TP_SIZE * TP_SIZE,
                rank % TP_SIZE,
                active,
                attention_epoch + step,
            )
            if is_full or is_reindex:
                for row in pl.spmd(group_tokens, name_hint="c1a_sp_inactive_topk"):
                    if row >= active:
                        topk_indices[row : row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            if is_full:
                positions = pl.tensor.dim(candidate_mask, 1)
                for row in pl.spmd(group_tokens, name_hint="c1a_sp_inactive_candidates"):
                    if row >= active:
                        for col in pl.range(0, positions, 128):
                            candidate_mask[row : row + 1, col : col + 128] = pl.cast(
                                pl.full([1, 128], dtype=pl.INT32, value=0), pl.UINT8
                            )
            if is_full:
                full_operator(
                    attn_input,
                    wq_a,
                    wq_a_scale,
                    q_norm_weight,
                    wq_b,
                    wq_b_scale,
                    wkv,
                    wkv_scale,
                    kv_norm_weight,
                    attn_sink,
                    wo_a,
                    wo_b,
                    wo_b_scale,
                    rope_cos,
                    rope_sin,
                    window_slots,
                    window_indices,
                    window_cache,
                    window_cache_scale,
                    compressed_cache,
                    compressed_cache_scale,
                    request_ids,
                    compressed_lens,
                    index_cache,
                    index_cache_scale,
                    index_block_table,
                    compressed_rope_cos,
                    compressed_rope_sin,
                    compressor_wkv,
                    compressor_norm_weight,
                    compressed_slots,
                    index_wk,
                    index_norm_weight,
                    index_wq_b,
                    index_wq_b_scale,
                    index_weights_proj,
                    topk_indices,
                    candidate_mask,
                    output_window,
                    output_arrived,
                    attn_output,
                    rank // TP_SIZE * TP_SIZE,
                    rank % TP_SIZE,
                    active,
                    attention_epoch + step,
                )
            elif is_reindex:
                reindex_operator(
                    attn_input,
                    wq_a,
                    wq_a_scale,
                    q_norm_weight,
                    wq_b,
                    wq_b_scale,
                    wkv,
                    wkv_scale,
                    kv_norm_weight,
                    attn_sink,
                    wo_a,
                    wo_b,
                    wo_b_scale,
                    rope_cos,
                    rope_sin,
                    window_slots,
                    window_indices,
                    window_cache,
                    window_cache_scale,
                    compressed_cache,
                    compressed_cache_scale,
                    request_ids,
                    compressed_lens,
                    index_cache,
                    index_cache_scale,
                    index_block_table,
                    candidate_mask,
                    index_wq_b,
                    index_wq_b_scale,
                    index_weights_proj,
                    topk_indices,
                    output_window,
                    output_arrived,
                    attn_output,
                    rank // TP_SIZE * TP_SIZE,
                    rank % TP_SIZE,
                    active,
                    attention_epoch + step,
                )
            else:
                reuse_operator(
                    attn_input,
                    wq_a,
                    wq_a_scale,
                    q_norm_weight,
                    wq_b,
                    wq_b_scale,
                    wkv,
                    wkv_scale,
                    kv_norm_weight,
                    attn_sink,
                    wo_a,
                    wo_b,
                    wo_b_scale,
                    rope_cos,
                    rope_sin,
                    window_slots,
                    window_indices,
                    window_cache,
                    window_cache_scale,
                    compressed_cache,
                    compressed_cache_scale,
                    compressed_indices,
                    output_window,
                    output_arrived,
                    attn_output,
                    rank // TP_SIZE * TP_SIZE,
                    rank % TP_SIZE,
                    active,
                    attention_epoch + step,
                )
            prefill_sp_post(
                attn_output, x_hc, post_mix, residual_mix, x_hc_out, next_pre_mix, rank % TP_SIZE, active,
            )
        return x_hc_out

    return c1a_sp_rank


def make_program(mode, world_size, epochs=1):
    """Allocate retained windows and dispatch C1A within contiguous TP groups."""
    if world_size % TP_SIZE or world_size < TP_SIZE:
        raise ValueError("world_size must contain whole TP groups")
    rank_entry = make_rank(mode, epochs)

    @pl.jit.host
    def c1a_sp_group(
        x_hc: pl.Tensor[[world_size, SP_T_DYN, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[world_size, SP_T_DYN, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[world_size, C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[world_size, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[world_size, C.MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[world_size, D], pl.BF16],
        wq_a: pl.Tensor[[world_size, C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[world_size, C.D // 32, C.Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[world_size, C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[world_size, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[world_size, C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[world_size, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[world_size, C.D // 32, C.HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[world_size, C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[world_size, C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[world_size, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[world_size, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[world_size, C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[world_size, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[world_size, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[world_size, C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[world_size, C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[world_size, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[world_size, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.InOut[
            pl.Tensor[[world_size, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8]
        ],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[
                [world_size, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
            ]
        ],
        request_ids: pl.Tensor[[world_size, C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[world_size, C.T_DYN], pl.INT32],
        index_cache: pl.InOut[
            pl.Tensor[[world_size, C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8]
        ],
        index_cache_scale: pl.InOut[
            pl.Tensor[
                [world_size, C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
            ]
        ],
        index_block_table: pl.Tensor[[world_size, C.B_DYN, C.TABLE_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[world_size, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[world_size, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[world_size, C.D, C.HEAD_DIM], pl.BF16],
        compressor_norm_weight: pl.Tensor[[world_size, C.HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[world_size, C.T_DYN], pl.INT64],
        index_wk: pl.Tensor[[world_size, C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[world_size, C.INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[world_size, C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[world_size, C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0],
        index_weights_proj: pl.Tensor[[world_size, C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.InOut[pl.Tensor[[world_size, C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        candidate_mask: pl.InOut[pl.Tensor[[world_size, C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8]],
        compressed_indices: pl.Tensor[[world_size, C.T_DYN, C.INDEX_TOPK], pl.INT32],
        attn_input: pl.Out[pl.Tensor[[world_size, C.T_DYN, D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[world_size, SP_T_DYN, D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[world_size, SP_T_DYN, HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[world_size, SP_T_DYN, HC_MULT, D], pl.FP32]],
        num_tokens: pl.Tensor[[world_size, 1], pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        x_hc.bind_dynamic(1, SP_T_DYN)
        pre_mix.bind_dynamic(1, SP_T_DYN)
        rope_cos.bind_dynamic(1, C.T_DYN)
        rope_sin.bind_dynamic(1, C.T_DYN)
        window_slots.bind_dynamic(1, C.T_DYN)
        window_indices.bind_dynamic(1, C.T_DYN)
        window_cache.bind_dynamic(1, C.ORI_BLOCKS_DYN)
        window_cache_scale.bind_dynamic(1, C.ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(1, C.CMP_BLOCKS_DYN)
        compressed_cache_scale.bind_dynamic(1, C.CMP_BLOCKS_DYN)
        request_ids.bind_dynamic(1, C.T_DYN)
        compressed_lens.bind_dynamic(1, C.T_DYN)
        index_cache.bind_dynamic(1, C.INDEX_BLOCKS_DYN)
        index_cache_scale.bind_dynamic(1, C.INDEX_BLOCKS_DYN)
        index_block_table.bind_dynamic(1, C.B_DYN)
        index_block_table.bind_dynamic(2, C.TABLE_DYN)
        compressed_rope_cos.bind_dynamic(1, C.T_DYN)
        compressed_rope_sin.bind_dynamic(1, C.T_DYN)
        compressed_slots.bind_dynamic(1, C.T_DYN)
        topk_indices.bind_dynamic(1, C.T_DYN)
        candidate_mask.bind_dynamic(1, C.T_DYN)
        candidate_mask.bind_dynamic(2, C.CMP_POSITIONS_DYN)
        compressed_indices.bind_dynamic(1, C.T_DYN)
        attn_input.bind_dynamic(1, C.T_DYN)
        attn_output.bind_dynamic(1, SP_T_DYN)
        next_pre_mix.bind_dynamic(1, SP_T_DYN)
        x_hc_out.bind_dynamic(1, SP_T_DYN)
        input_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        input_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        output_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        output_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(world_size):
            input_window = pld.window(input_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
            input_arrived = pld.window(input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            output_window = pld.window(output_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
            output_arrived = pld.window(output_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
            wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[
                rank
            ]
            wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
            wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
            index_wq_b_scale_r: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = (
                index_wq_b_scale[rank]
            )
            rank_entry(
                x_hc[rank],
                pre_mix[rank],
                hc_attn_fn[rank],
                hc_attn_scale[rank],
                hc_attn_base[rank],
                attn_norm_weight[rank],
                wq_a[rank],
                wq_a_scale_r,
                q_norm_weight[rank],
                wq_b[rank],
                wq_b_scale_r,
                wkv[rank],
                wkv_scale_r,
                kv_norm_weight[rank],
                attn_sink[rank],
                wo_a[rank],
                wo_b[rank],
                wo_b_scale_r,
                rope_cos[rank],
                rope_sin[rank],
                window_slots[rank],
                window_indices[rank],
                window_cache[rank],
                window_cache_scale[rank],
                compressed_cache[rank],
                compressed_cache_scale[rank],
                request_ids[rank],
                compressed_lens[rank],
                index_cache[rank],
                index_cache_scale[rank],
                index_block_table[rank],
                compressed_rope_cos[rank],
                compressed_rope_sin[rank],
                compressor_wkv[rank],
                compressor_norm_weight[rank],
                compressed_slots[rank],
                index_wk[rank],
                index_norm_weight[rank],
                index_wq_b[rank],
                index_wq_b_scale_r,
                index_weights_proj[rank],
                topk_indices[rank],
                candidate_mask[rank],
                compressed_indices[rank],
                attn_input[rank],
                attn_output[rank],
                next_pre_mix[rank],
                x_hc_out[rank],
                input_window,
                input_arrived,
                output_window,
                output_arrived,
                num_tokens[rank],
                rank,
                attention_epoch,
                device=rank,
            )

    return c1a_sp_group


STATE_NAMES = {
    "full": (
        "window_cache",
        "window_cache_scale",
        "compressed_cache",
        "compressed_cache_scale",
        "index_cache",
        "index_cache_scale",
        "topk_indices",
        "candidate_mask",
    ),
    "reindex": ("window_cache", "window_cache_scale", "topk_indices"),
    "reuse": ("window_cache", "window_cache_scale"),
}
TOKEN_NAMES = {
    "rope_cos",
    "rope_sin",
    "window_slots",
    "window_indices",
    "compressed_indices",
    "request_ids",
    "compressed_lens",
    "candidate_mask",
    "compressed_rope_cos",
    "compressed_rope_sin",
    "compressed_slots",
}


def reference_attention(mode, epochs, tensors, x, ranks):
    """Run the independent C1A reference in full token order before sharding."""
    module = {"full": full, "reindex": reindex, "reuse": reuse}[mode]
    reference = getattr(module, f"golden_prefill_attn_c1a_{mode}")
    names = tuple(inspect.signature(reference).parameters)
    active = int(tensors["num_tokens"][ranks[0], 0])
    partials, results = [], []
    for rank in ranks:
        values = {name: x[:active] if name == "x" else tensors[name][rank] for name in names}
        for name in TOKEN_NAMES.intersection(values):
            values[name] = values[name][:active]
        result = {name: tensors[name][rank].clone() for name in STATE_NAMES[mode]}
        partial = torch.zeros_like(x, dtype=torch.float32)
        if mode != "reuse":
            result["topk_indices"][active:] = -1
        if mode == "full":
            result["candidate_mask"].zero_()
        if active:
            for _ in range(epochs):
                computed = reference(**values)
                for name in STATE_NAMES[mode]:
                    if name in values and "cache" in name:
                        values[name] = getattr(computed, name)
            partial[:active] = computed.output
            for name in STATE_NAMES[mode]:
                if name == "topk_indices":
                    result[name][:active] = computed.topk_indices
                elif name == "candidate_mask":
                    block_size = C.FLASH.candidate_block_size
                    for row in range(active):
                        selected = torch.nonzero(computed.candidate_mask[row], as_tuple=False).flatten()
                        for block in torch.unique(selected // block_size):
                            start = int(block) * block_size
                            result[name][row, start : start + block_size] = 1
                else:
                    result[name] = getattr(computed, name)
        partials.append(partial)
        results.append(result)
    return sum(partials).bfloat16(), results


def build_specs(args, initial_state):
    """Make distinct DP requests, contiguous local residuals, and global-order metadata."""
    fixture = make_fixture_values(args.tokens, args.case)
    tokens = fixture["x"].shape[1]
    world = TP_SIZE * args.dp
    local = (tokens + TP_SIZE - 1) // TP_SIZE
    if any(count < 0 or count > tokens for count in args.dp_tokens):
        raise ValueError(f"active counts must be in [0, {tokens}]")
    values = {}
    for name in LEAF_NAMES:
        if name == "topk_indices":
            value = torch.full((TP_SIZE, tokens, C.INDEX_TOPK), -1, dtype=torch.int32)
        else:
            value = fixture[name]
        values[name] = torch.cat([value.view(torch.uint8)] * args.dp).view(value.dtype)
    hc_keys = common.HC_INPUT_NAMES
    groups = [common.make_hc_inputs(tokens, args.seed + group * 7919, "mixed") for group in range(args.dp)]
    for name in hc_keys:
        if name in ("x_hc", "pre_mix"):
            values[name] = torch.cat([common.token_shards(group[name], local) for group in groups])
        else:
            values[name] = torch.stack([group[name] for group in groups]).repeat_interleave(TP_SIZE, 0)
    for rank in range(world):
        active = args.dp_tokens[rank // TP_SIZE]
        valid = max(0, min(local, active - rank % TP_SIZE * local))
        values["x_hc"][rank, valid:] = 17
        for name in ("window_slots", "compressed_slots", "window_indices", "compressed_indices"):
            values[name][rank, active:] = -1
        values["candidate_mask"][rank, active:] = 0
        values["compressed_lens"][rank, active:] = 0
    initial_state.update({name: values[name].clone() for name in set().union(*STATE_NAMES.values())})
    specs = [
        TensorSpec(
            name, list(values[name].shape), values[name].dtype, init_value=values[name], resident="stacked"
        )
        for name in hc_keys + LEAF_NAMES
    ]
    for name, shape, dtype in (
        ("attn_input", [tokens, D], torch.bfloat16),
        ("attn_output", [local, D], torch.bfloat16),
        ("next_pre_mix", [local, HC_MULT], torch.float32),
        ("x_hc_out", [local, HC_MULT, D], torch.float32),
    ):
        specs.append(TensorSpec(name, [world, *shape], dtype, resident="stacked"))
    counts = torch.tensor(args.dp_tokens, dtype=torch.int32).repeat_interleave(TP_SIZE).reshape(-1, 1)
    specs.append(TensorSpec("num_tokens", [world, 1], torch.int32, init_value=counts, resident="stacked"))
    specs.append(ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True))
    return specs


def make_compare(mode, epochs, initial_state):
    """Apply existing row bounds to every shard and validate all cache side effects."""
    from models.deepseek_v4_1_flash.attention_common import quantized_cache_compare
    from models.deepseek_v4_1_flash.prefill_c1a_test_utils import (
        CACHE_MAX_RELATIVE_L2,
        MXFP4_CACHE_MAX_RELATIVE_L2,
    )

    staged = common.StagedAttentionReference(
        mode,
        epochs,
        initial_state,
        attention_reference=reference_attention,
        state_names=STATE_NAMES[mode],
    )
    def group_values(values, base, active):
        grouped = {}
        for name, value in values.items():
            if name == "num_tokens":
                grouped[name] = active
                continue
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                grouped[name] = value
                continue
            value = value[base:base + TP_SIZE]
            if name in TOKEN_NAMES or name in ("x", "topk_indices"):
                value = value[:, :active]
            grouped[name] = value
        return grouped

    def selected_compare(name):
        """Reuse the standalone selection-quality and selected-output contracts per DP group."""
        check = attention_output_compare(mode) if name == "attn_output" else topk_indices_compare(mode)

        def compare(actual, expected, *, inputs, actual_outputs, expected_outputs, **kwargs):
            values = {**initial_state, **inputs, "x": actual_outputs["attn_input"]}
            for base in range(0, actual.shape[0], TP_SIZE):
                active = int(inputs["num_tokens"][base, 0])
                group = slice(base, base + TP_SIZE)
                if name == "attn_output":
                    value = actual[group].flatten(0, 1)
                    reference = expected[group].flatten(0, 1)
                    if not bool((value[active:] == 0).all()):
                        return False, f"DP group {base // TP_SIZE} has nonzero attention padding"
                    value = value[:active].unsqueeze(0).expand(TP_SIZE, -1, -1)
                    reference = reference[:active].unsqueeze(0).expand_as(value)
                else:
                    if not bool((actual[group, active:] == -1).all()):
                        return False, f"DP group {base // TP_SIZE} has invalid Top-K padding"
                    value, reference = actual[group, :active], expected[group, :active]
                if not active:
                    continue
                expected_group = group_values(expected_outputs, base, active)
                if name == "attn_output":
                    expected_group["output"] = reference
                passed, detail = check(
                    value, reference, inputs=group_values(values, base, active),
                    actual_outputs=group_values(actual_outputs, base, active),
                    expected_outputs=expected_group, rtol=kwargs.get("rtol", 0), atol=kwargs.get("atol", 0),
                )
                if not passed:
                    return False, f"DP group {base // TP_SIZE}: {detail}"
            return True, "standalone C1A selection and row bounds; exact inactive padding"

        return compare

    compares = {
        "attn_input": common.compare_group_leaders(
            "attn_input", ratio_allclose(atol=1e-4, rtol=1.0 / 128), common.ROW_BUDGET
        ),
        "next_pre_mix": common.compare_shards("next_pre_mix", ratio_allclose(atol=2.5e-5, rtol=5e-3), 5e-3),
        "attn_output": staged.compare(
            "attn_output",
            selected_compare("attn_output"),
        ),
        "x_hc_out": common.compare_x_hc_out,
    }
    cache_checks = {
        "window": quantized_cache_compare(
            "window_cache", "window_cache_scale", "window_slots", CACHE_MAX_RELATIVE_L2
        ),
        "compressed": quantized_cache_compare(
            "compressed_cache",
            "compressed_cache_scale",
            "compressed_slots",
            MXFP4_CACHE_MAX_RELATIVE_L2,
            group_size=C.COMPRESSED_CACHE_GROUP,
            scale_format="e4m3",
        ),
        "index": quantized_cache_compare(
            "index_cache",
            "index_cache_scale",
            "compressed_slots",
            MXFP4_CACHE_MAX_RELATIVE_L2,
            group_size=C.INDEX_CACHE_GROUP,
            scale_format="e8m0",
        ),
    }
    for name in STATE_NAMES[mode]:
        if "cache" in name:
            compares[name] = staged.compare(name, cache_checks[name.split("_")[0]])
        elif name == "topk_indices":
            compares[name] = staged.compare(name, selected_compare(name))
        else:
            compares[name] = staged.compare(
                name,
                lambda actual, expected, **kwargs: (
                    torch.equal(actual, expected),
                    "candidate blocks must match the reference",
                ),
            )
    for name in (
        "compressed_cache",
        "compressed_cache_scale",
        "index_cache",
        "index_cache_scale",
        "topk_indices",
        "candidate_mask",
    ):
        if name not in STATE_NAMES[mode]:
            compares[name] = lambda actual, expected, **kwargs: (
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)),
                "read-only state is unchanged",
            )
    return compares


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("-d", "--device", required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4], default=TP_SIZE)
    parser.add_argument("--dp", type=int, choices=[1, 2], default=1)
    parser.add_argument("--mode", choices=["full", "reindex", "reuse"], default="full")
    parser.add_argument("--case", choices=CASE_NAMES, default="causal")
    parser.add_argument("--tokens", type=int, default=7)
    parser.add_argument("--dp-tokens", help="active count of each DP group")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 128 or not 1 <= args.epochs <= 1000:
        parser.error("tokens must be in [1, 128] and epochs in [1, 1000]")
    devices = [int(value) for value in args.device.split(",")]
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices):
        parser.error("device IDs must name TP*DP distinct devices")
    fixture_tokens = {"mixed": 8, "multi_1k": 3, "long": 3}.get(args.case, args.tokens)
    args.dp_tokens = (
        [int(value) for value in args.dp_tokens.split(",")] if args.dp_tokens else [fixture_tokens] * args.dp
    )
    if len(args.dp_tokens) != args.dp:
        parser.error("provide one active count per DP group")
    torch.set_num_threads(8)
    initial_state = {}
    specs = build_specs(args, initial_state)
    result = run(
        fn=make_program(args.mode, len(devices), args.epochs),
        specs=specs,
        golden_fn=common.make_golden(
            args.mode,
            args.epochs,
            attention_reference=reference_attention,
            state_names=STATE_NAMES[args.mode],
        ),
        compare_fn=make_compare(args.mode, args.epochs, initial_state),
        compile_only=args.compile_only,
        config=dict(
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        ),
    )
    print(f"[C1A-SP] mode={args.mode} work_dir={result.work_dir}")
    if args.compile_only:
        print("[C1A-SP] Compilation only; device accuracy was NOT validated.")
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
