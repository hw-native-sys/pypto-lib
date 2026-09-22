# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A full attention wired through mHC.

The attention operator stays in ``decode_attn_c1a_full.py``; this entry consumes the staggered
``pre_mix`` the previous sub-layer produced (identity one-hot at the very first site),
computes this site's coefficients with ``mhc_mixes``, collapses with ``mhc_pre``, runs the
operator, and expands the residual with ``mhc_post``. ``post_mix`` and ``residual_mix``
apply immediately, while the computed ``pre_mix`` is handed to the next sub-layer. The
module also hosts the shared HC fixture, goldens, and validation harness the reindex and
reuse entries reuse.
"""

import inspect
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
from models.deepseek_v4_1_flash.attention_common import quantized_cache_compare
from models.deepseek_v4_1_flash.config import (
    B_DYN,
    CMP_BLOCKS_DYN,
    CMP_POSITIONS_DYN,
    COMPRESSED_CACHE_GROUP,
    D,
    DECODE_MAX_TOKENS,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP,
    INDEX_DIM,
    INDEX_H,
    INDEX_TOPK,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MIX_HC,
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
from models.deepseek_v4_1_flash.decode_attn_c1a_full import (
    CACHE_MAX_RELATIVE_L2,
    CACHE_STATE_NAMES,
    MXFP4_CACHE_MAX_RELATIVE_L2,
    OUTPUT_ATOL,
    OUTPUT_MAX_ERROR_RATIO,
    OUTPUT_RTOL,
    build_validation_values,
    check_fp4_boundaries,
    decode_attn_c1a_full,
    exact_bytes,
    golden_decode_attn_c1a_full,
    topk_indices_compare,
)
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes, mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post, mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre, mhc_pre
from models.deepseek_v4_1_flash.golden import rms_norm as golden_rms_norm
from models.deepseek_v4_1_flash.rmsnorm import rms_norm


@pl.jit.inline(auto_scope=False)
def decode_c1a_full(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
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
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normalized_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    # The coefficients are staggered: collapse with the pre-mix the previous sub-layer
    # produced, apply post/residual immediately, and hand this site's pre-mix forward.
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    mhc_pre(x_hc, pre_mix, hidden)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    active_hidden = pl.slice(hidden, [active_tokens, D], [0, 0])
    active_normalized = pl.slice(normalized_hidden, [active_tokens, D], [0, 0])
    rms_norm(active_hidden, attn_norm_weight, active_normalized)
    decode_attn_c1a_full(
        normalized_hidden, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b,
        index_wq_b_scale, index_weights_proj, topk_indices, candidate_mask, output_window, output_arrived,
        attn_out, group_base, tp_rank, num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


@pl.jit
def decode_c1a_full_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
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
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    candidate_mask.bind_dynamic(1, CMP_POSITIONS_DYN)
    return decode_c1a_full(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots,
        window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        request_ids, compressed_lens, index_cache, index_cache_scale, index_block_table, compressed_rope_cos,
        compressed_rope_sin, compressor_wkv, compressor_norm_weight, compressed_slots, index_wk,
        index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, candidate_mask,
        output_window, output_arrived, output, next_pre_mix, group_base, tp_rank, num_tokens,
        attention_epoch,
    )


__all__ = [
    "build_hc_validation_values",
    "decode_c1a_full",
    "golden_c1a_hc_case",
    "golden_decode_c1a_full_case",
    "run_c1a_hc",
]


def make_program(tokens, pages, epochs=1):
    """Build a distributed host using static packed-FP4 storage dimensions."""
    TOKENS = tokens
    PAGES = pages
    EPOCHS = epochs

    @pl.jit.host
    def host(
        x_hc: pl.Tensor[[TP_SIZE, TOKENS, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[TP_SIZE, TOKENS, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
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
        output: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, HC_MULT, D], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, TOKENS, HC_MULT], pl.FP32]],
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
                decode_c1a_full_test(
                    x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank],
                    hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank],
                    wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank],
                    wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank],
                    wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
                    window_cache[rank], window_cache_scale[rank], compressed_cache[rank],
                    compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank],
                    index_cache_scale[rank], index_block_table[rank], compressed_rope_cos[rank],
                    compressed_rope_sin[rank], compressor_wkv[rank], compressor_norm_weight[rank],
                    compressed_slots[rank], index_wk[rank], index_norm_weight[rank], index_wq_b[rank],
                    index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], candidate_mask[rank],
                    output_window, output_arrived, output[rank], next_pre_mix[rank], 0, rank, TOKENS,
                    epoch, device=rank,
                )

    return host


def build_hc_validation_values(mode, tokens, pages, seed=17, case="random"):
    """Attention fixture plus the HC stream and mixing parameters."""
    import math

    values = build_validation_values(mode, tokens, pages, seed, case)
    del values["x"]
    generator = torch.Generator().manual_seed(seed)
    values["x_hc"] = torch.randn(TP_SIZE, tokens, HC_MULT, D, generator=generator)
    # The coefficients are staggered: this site collapses with the pre-mix the previous
    # sub-layer produced (identity one-hot at the very first site). The fixture carries a
    # general sigmoid-distributed mix instead of the seed so every lane participates.
    values["pre_mix"] = torch.sigmoid(torch.randn(TP_SIZE, tokens, HC_MULT, generator=generator))
    values["hc_attn_fn"] = torch.randn(TP_SIZE, MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)
    values["hc_attn_scale"] = torch.randn(TP_SIZE, 3, generator=generator)
    values["hc_attn_base"] = torch.randn(TP_SIZE, MIX_HC, generator=generator)
    values["attn_norm_weight"] = (torch.randn(TP_SIZE, D, generator=generator) * 0.1 + 1).to(torch.bfloat16)
    values["output"] = torch.zeros(TP_SIZE, tokens, HC_MULT, D, dtype=torch.float32)
    values["next_pre_mix"] = torch.zeros(TP_SIZE, tokens, HC_MULT, dtype=torch.float32)
    return values


def output_hc_compare(actual, expected, **kwargs):
    """Report the expanded-output diagnostics and apply the per-point budget.

    The expansion rounds through BF16 and back to FP32, which both stacks do at the same
    point, so the same per-point rule the sublayer entries use applies row by row.
    """
    a, e = actual.float(), expected.float()
    a_rows, e_rows = a.reshape(-1, e.shape[-1]), e.reshape(-1, e.shape[-1])
    error = (a_rows - e_rows).norm() / e_rows.norm().clamp_min(1e-12)
    rows = (a_rows - e_rows).norm(dim=-1) / e_rows.norm(dim=-1).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0)
    print(
        f"[PRECISION] output rel_l2={error.item():.6g} cosine={cosine.item():.8f} "
        f"max_abs={(a - e).abs().max().item():.6g} max_row_rel_l2={rows.max().item():.6g}"
    )
    from golden import ratio_allclose

    budget = ratio_allclose(atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL, max_error_ratio=OUTPUT_MAX_ERROR_RATIO)
    return budget(actual, expected, **kwargs)


def next_pre_mix_compare(actual, expected, **kwargs):
    """Report the staggered coefficient diagnostics under the mHC budget.

    The next sub-layer collapses with this coefficient, so it is held to the tolerance the
    mHC coefficient module applies to its own pre_mix output.
    """
    a, e = actual.float(), expected.float()
    error = (a - e).norm() / e.norm().clamp_min(1e-12)
    print(
        f"[PRECISION] next_pre_mix rel_l2={error.item():.6g} "
        f"max_abs={(a - e).abs().max().item():.6g}"
    )
    from golden import ratio_allclose

    budget = ratio_allclose(atol=2.5e-5, rtol=5e-3)
    return budget(actual, expected, **kwargs)


def hc_topk_indices_compare(mode):
    """Tie-tolerant top-k comparison for the mHC entries.

    The attention comparator rebuilds the index scores from the collapsed stream, which the
    mHC entries carry as HC tensors rather than ``x``; derive it here from the staggered
    pre-mix and hand it over under that name before delegating.
    """
    inner = topk_indices_compare(mode)

    def compare(actual, expected, **kwargs):
        inputs = kwargs.get("inputs")
        if inputs is not None and "x" not in inputs:
            hidden = [
                golden_rms_norm(
                    golden_mhc_pre(inputs["x_hc"][rank], inputs["pre_mix"][rank]),
                    inputs["attn_norm_weight"][rank],
                )
                for rank in range(TP_SIZE)
            ]
            inputs["x"] = torch.stack(hidden)
        return inner(actual, expected, **kwargs)

    compare.__name__ = f"hc_tied_cutoff_topk_{mode}"
    return compare


def golden_c1a_hc_case(tensors, attn_golden, epochs=1):
    """Reference for an mHC-wired entry: mixes -> attention -> expansion.

    One DP group collapses its stream with the staggered pre-mix the fixture carries, normalizes
    it, runs the attention golden per rank, hands this site's computed pre-mix
    to the next sub-layer, publishes the cache state the operator returned, reduces the
    attention partials, and expands the reduced result back onto the four streams.
    """
    parameters = [name for name in inspect.signature(attn_golden).parameters if name != "x"]
    for _ in range(epochs):
        partials = []
        mixes = []
        for rank in range(TP_SIZE):
            x_hc = tensors["x_hc"][rank]
            next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
                x_hc, tensors["hc_attn_fn"][rank], tensors["hc_attn_scale"][rank], tensors["hc_attn_base"][rank]
            )
            tensors["next_pre_mix"][rank].copy_(next_pre_mix)
            hidden = golden_mhc_pre(x_hc, tensors["pre_mix"][rank])
            normalized_hidden = golden_rms_norm(hidden, tensors["attn_norm_weight"][rank])
            result = attn_golden(x=normalized_hidden, **{name: tensors[name][rank] for name in parameters})
            partials.append(result.output.float())
            mixes.append((x_hc, post_mix, residual_mix))
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
        sublayer = reduced.to(torch.bfloat16)
        for rank, (x_hc, post_mix, residual_mix) in enumerate(mixes):
            tensors["output"][rank].copy_(golden_mhc_post(sublayer, x_hc, post_mix, residual_mix))


def golden_decode_c1a_full_case(tensors, epochs=1):
    """Fill the mHC-wired full attention entry expectations from the operator reference."""
    golden_c1a_hc_case(tensors, golden_decode_attn_c1a_full, epochs)


def run_c1a_hc(mode, kernel_factory, golden_case, argv=None):
    """Run A5 validation for an mHC-wired C1A operator."""
    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description=f"TP1/2/4 A5 mHC C1A {mode} validation")
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
    values = build_hc_validation_values(mode, args.tokens, args.pages, args.seed, args.case)
    # Clone per call: the harness builds the golden scratch from these same specs, and a
    # tensor init_value whose dtype already matches comes back as itself.
    specs = [
        TensorSpec(n, list(values[n].shape), values[n].dtype, init_value=lambda n=n: values[n].clone())
        for n in host.param_names
    ]

    def golden_fn(tensors):
        golden_case(tensors, args.epochs)

    # Published cache rows are compared as dequantized values, because a one-ULP difference can
    # move a published code one E2M1/E4M3 step; which rows may change at all stays byte exact.
    # Only FULL publishes the compressor and index caches: REINDEX and REUSE read them.
    comparisons = {
        "window_cache": quantized_cache_compare(
            "window_cache", "window_cache_scale", "window_slots", CACHE_MAX_RELATIVE_L2
        ),
        "topk_indices": exact_bytes if mode == "reuse" else hc_topk_indices_compare(mode),
        "candidate_mask": exact_bytes,
        "next_pre_mix": next_pre_mix_compare,
        "output": output_hc_compare,
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
        golden_fn=golden_fn,
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
    """Validate the mHC-wired decode C1A full attention entry on A5."""
    return run_c1a_hc("full", make_program, golden_decode_c1a_full_case, argv=argv)


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

    @pytest.mark.parametrize("tp,dp", [(1, 1), (4, 1)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
