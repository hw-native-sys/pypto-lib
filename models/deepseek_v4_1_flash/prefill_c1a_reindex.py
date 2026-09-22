# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C1A reindex attention wired through delayed mHC mixing."""

# ci: no-sim
# ci: a5

import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from golden import ScalarSpec, TensorSpec, ratio_allclose, run

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import quantized_cache_compare
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes, mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post, mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre, mhc_pre
from models.deepseek_v4_1_flash.golden import rms_norm as golden_rms_norm
from models.deepseek_v4_1_flash.prefill_attn_c1a_reindex import (
    REINDEX_INPUT_NAMES,
    golden_prefill_c1a_reindex as golden_prefill_attn_c1a_reindex,
    prefill_c1a_reindex as prefill_attn_c1a_reindex,
)
from models.deepseek_v4_1_flash.prefill_c1a_test_utils import (
    CACHE_MAX_RELATIVE_L2,
    CASE_DEFAULT,
    CASE_MAX_TOKENS,
    CASE_NAMES,
    apply_distributed_golden,
    attn_input_compare,
    attention_output_compare,
    golden_c1a_attention_input,
    golden_prefill_tp_attention,
    hc_hidden_compare,
    hc_output_compare,
    make_fixture_values,
    topk_indices_compare,
)
from models.deepseek_v4_1_flash.rmsnorm import rms_norm


D = C.D
HC_MULT = C.HC_MULT
HEAD_DIM = C.HEAD_DIM
INDEX_DIM = C.INDEX_DIM
INDEX_H = C.INDEX_H
LOCAL_H = C.LOCAL_H
LOCAL_O_WIDTH = C.LOCAL_O_WIDTH
PREFILL_MAX_TOKENS = C.PREFILL_MAX_TOKENS
Q_LORA = C.Q_LORA
TP_SIZE = C.TP_SIZE
PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4

golden_prefill_c1a_reindex = golden_prefill_attn_c1a_reindex


@pl.jit.inline(auto_scope=False)
def prefill_c1a_reindex(
    x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
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
    window_cache: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[
        [C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0
    ],
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8],
    index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    hidden: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    attn_input: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    attn_out: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    output: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    next_pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run official mHC mixes, collapse, attention RMSNorm, C1A, then mHC post."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    mhc_pre(x_hc, pre_mix, hidden)
    rms_norm(hidden, attn_norm_weight, attn_input)
    prefill_attn_c1a_reindex(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, candidate_mask,
        index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices,
        output_window, output_arrived, attn_out,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    mhc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


def golden_prefill_c1a_reindex_hc(
    x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, *attention_args
):
    """Reference the official delayed pre-mix ordering around C1A reindex."""
    if x_hc.ndim == 4 and x_hc.shape[0] == TP_SIZE:
        x_hc = x_hc[0]
        pre_mix = pre_mix[0]
        hc_attn_fn = hc_attn_fn[0]
        hc_attn_scale = hc_attn_scale[0]
        hc_attn_base = hc_attn_base[0]
        attn_norm_weight = attn_norm_weight[0]
    next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base
    )
    hidden = golden_mhc_pre(x_hc, pre_mix)
    attn_input = golden_rms_norm(hidden.to(torch.bfloat16), attn_norm_weight)
    sublayer, result = golden_prefill_tp_attention(
        golden_prefill_attn_c1a_reindex, attn_input, attention_args, TP_SIZE
    )
    output = golden_mhc_post(sublayer, x_hc, post_mix, residual_mix)
    return output, next_pre_mix, result


@pl.jit
def prefill_c1a_reindex_test(
    x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
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
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8],
    index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
    output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    next_pre_mix: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
    output: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one TP rank of the delayed-mix C1A reindex path."""
    x_hc.bind_dynamic(0, C.T_DYN)
    output.bind_dynamic(0, C.T_DYN)
    return prefill_c1a_reindex(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, candidate_mask,
        index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices,
        output_window, output_arrived, hidden, attn_input, attn_out, output, next_pre_mix,
        0, tp_rank, num_tokens, 1,
    )


@pl.jit.host
def l3_prefill_c1a_reindex_test(
    x_hc: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    pre_mix: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.TP_SIZE, C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[C.TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.TP_SIZE, C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.TP_SIZE, C.D], pl.BF16],
    wq_a: pl.Tensor[[C.TP_SIZE, C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.TP_SIZE, C.D // 32, C.Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[C.TP_SIZE, C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.TP_SIZE, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.TP_SIZE, C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[C.TP_SIZE, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.TP_SIZE, C.D // 32, C.HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[C.TP_SIZE, C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.TP_SIZE, C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.TP_SIZE, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.TP_SIZE, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.TP_SIZE, C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.TP_SIZE, C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.TP_SIZE, C.T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[
        pl.Tensor[[C.TP_SIZE, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[[C.TP_SIZE, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    compressed_cache: pl.Tensor[
        [C.TP_SIZE, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8
    ],
    compressed_cache_scale: pl.Tensor[
        [C.TP_SIZE, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    request_ids: pl.Tensor[[C.TP_SIZE, C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.TP_SIZE, C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[
        [C.TP_SIZE, C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8
    ],
    index_cache_scale: pl.Tensor[
        [C.TP_SIZE, C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[C.TP_SIZE, C.B_DYN, C.TABLE_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8],
    index_wq_b: pl.Tensor[[C.TP_SIZE, C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [C.TP_SIZE, C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0
    ],
    index_weights_proj: pl.Tensor[[C.TP_SIZE, C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.INDEX_TOPK], pl.INT32]],
    next_pre_mix: pl.Out[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.HC_MULT], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.D], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.D], pl.BF16]],
    output: pl.Out[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output_window_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
    output_arrived_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        output_window = pld.window(output_window_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        output_arrived = pld.window(output_arrived_buf, [TP_SIZE, 1], dtype=pl.INT32)
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = index_wq_b_scale[rank]
        prefill_c1a_reindex_test(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
            wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank],
            wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank],
            window_indices[rank], window_cache[rank], window_cache_scale[rank],
            compressed_cache[rank], compressed_cache_scale[rank], request_ids[rank],
            compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
            index_block_table[rank], candidate_mask[rank], index_wq_b[rank],
            index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank],
            output_window, output_arrived, next_pre_mix[rank], hidden[rank], attn_input[rank], attn_out[rank],
            output[rank], rank, num_tokens, device=rank,
        )


def build_hc_tensor_specs(token_count=32, case_name=CASE_DEFAULT, active_tokens=None):
    """Build C1A inputs plus a pre-mix produced by the preceding mHC block."""
    values = make_fixture_values(token_count, case_name)
    tokens = values["x"].shape[1]
    active_tokens = tokens if active_tokens is None else active_tokens
    if not 0 <= active_tokens <= tokens:
        raise ValueError(f"active_tokens must be in [0, {tokens}], got {active_tokens}")
    generator = torch.Generator().manual_seed(2026)
    x_hc = torch.randn(tokens, C.HC_MULT, C.D, generator=generator).bfloat16().float()
    previous_x_hc = torch.randn(tokens, C.HC_MULT, C.D, generator=generator).bfloat16().float()
    hc_attn_fn = torch.randn(C.MIX_HC, C.HC_DIM, generator=generator) / math.sqrt(C.HC_DIM)
    hc_attn_scale = torch.randn(3, generator=generator)
    hc_attn_base = torch.randn(C.MIX_HC, generator=generator)
    pre_mix, _, _ = golden_mhc_mixes(previous_x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    hc_values = {
        "x_hc": x_hc.unsqueeze(0).repeat(C.TP_SIZE, 1, 1, 1),
        "pre_mix": pre_mix.unsqueeze(0).repeat(C.TP_SIZE, 1, 1),
        "hc_attn_fn": hc_attn_fn.unsqueeze(0).repeat(C.TP_SIZE, 1, 1),
        "hc_attn_scale": hc_attn_scale.unsqueeze(0).repeat(C.TP_SIZE, 1),
        "hc_attn_base": hc_attn_base.unsqueeze(0).repeat(C.TP_SIZE, 1),
        "attn_norm_weight": values["attn_norm_weight"],
    }
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in hc_values.items()
    ]
    specs += [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name])
        for name in REINDEX_INPUT_NAMES if name != "x"
    ]
    specs += [
        TensorSpec("topk_indices", [C.TP_SIZE, tokens, C.INDEX_TOPK], torch.int32),
        TensorSpec("next_pre_mix", [C.TP_SIZE, tokens, C.HC_MULT], torch.float32),
        TensorSpec("hidden", [C.TP_SIZE, tokens, C.D], torch.bfloat16),
        TensorSpec("attn_input", [C.TP_SIZE, tokens, C.D], torch.bfloat16),
        TensorSpec(
            "attn_out", [C.TP_SIZE, tokens, C.D], torch.bfloat16,
            init_value=torch.zeros(C.TP_SIZE, tokens, C.D, dtype=torch.bfloat16),
        ),
        TensorSpec("output", [C.TP_SIZE, tokens, C.HC_MULT, C.D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]
    return specs


def golden_prefill_c1a_reindex_case(tensors):
    """Reference delayed pre-mix, current coefficient generation, attention, and post-mix."""
    next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"][0], tensors["hc_attn_fn"][0],
        tensors["hc_attn_scale"][0], tensors["hc_attn_base"][0],
    )
    hidden = golden_mhc_pre(tensors["x_hc"][0], tensors["pre_mix"][0])
    attn_input = golden_c1a_attention_input(hidden, tensors["attn_norm_weight"][0])
    active = int(tensors["num_tokens"])
    output = tensors["output"]
    tensors["x"] = attn_input.unsqueeze(0).expand(C.TP_SIZE, -1, -1)
    tensors["output"] = tensors["attn_out"]
    apply_distributed_golden("reindex", golden_prefill_attn_c1a_reindex, tensors, active)
    tensors["attn_out"][:, active:].zero_()
    tensors["attn_input"][:] = attn_input.unsqueeze(0)
    hc_output = golden_mhc_post(tensors["attn_out"][0], tensors["x_hc"][0], post_mix, residual_mix)
    tensors["output"] = output
    tensors["next_pre_mix"][:] = next_pre_mix.unsqueeze(0)
    tensors["hidden"][:] = hidden.unsqueeze(0)
    tensors["output"][:] = hc_output.unsqueeze(0)


def wrap_attention_compare(compare):
    """Present HC workspace names through the attention-only comparator ABI."""

    def compare_hc(actual, expected, *, inputs, actual_outputs, expected_outputs, **kwargs):
        attention_inputs = {**inputs, "x": expected_outputs["attn_input"]}
        attention_actual = {**actual_outputs, "output": actual_outputs["attn_out"]}
        attention_expected = {**expected_outputs, "output": expected_outputs["attn_out"]}
        return compare(
            actual,
            expected,
            inputs=attention_inputs,
            actual_outputs=attention_actual,
            expected_outputs=attention_expected,
            **kwargs,
        )

    return compare_hc


def validate(argv=None):
    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C1A reindex with mHC validation")
    parser.add_argument("-p", "--platform", default="a5", choices=("a5",))
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(C.TP_SIZE)))
    parser.add_argument("--tp", type=int, default=C.TP_SIZE, choices=(1, 2, 4))
    parser.add_argument("--dp", type=int, default=1, choices=(1,))
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--case", default=CASE_DEFAULT, choices=CASE_NAMES)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args(argv)
    if args.tp != C.TP_SIZE:
        parser.error(f"--tp was parsed as TP{C.TP_SIZE}, got --tp {args.tp}")
    if args.case == "causal" and not 1 <= args.tokens <= CASE_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {CASE_MAX_TOKENS}]")
    if args.active_tokens is not None and not 0 <= args.active_tokens <= args.tokens:
        parser.error(f"--active-tokens must be in [0, {args.tokens}]")
    devices = [int(device) for device in args.device.split(",")]
    if len(devices) != C.TP_SIZE:
        parser.error(f"need exactly {C.TP_SIZE} devices, got {devices}")
    window_cache_compare = quantized_cache_compare(
        "window_cache", "window_cache_scale", "window_slots", CACHE_MAX_RELATIVE_L2,
    )
    result = run(
        fn=l3_prefill_c1a_reindex_test,
        specs=build_hc_tensor_specs(args.tokens, args.case, args.active_tokens),
        golden_fn=golden_prefill_c1a_reindex_case,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config={
            "platform": args.platform,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "dump_passes": args.dump_passes,
            "enable_chip_swimlane": args.enable_chip_swimlane,
            "ring_heap": PREFILL_ATTN_RING_HEAP,
        },
        compare_fn={
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "hidden": hc_hidden_compare(),
            "attn_input": attn_input_compare(),
            "attn_out": wrap_attention_compare(attention_output_compare("reindex")),
            "output": hc_output_compare(),
            "window_cache": window_cache_compare,
            "window_cache_scale": window_cache_compare,
            "topk_indices": wrap_attention_compare(topk_indices_compare("reindex")),
        },
    )
    return result


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

__all__ = [
    "build_hc_tensor_specs",
    "golden_prefill_c1a_reindex",
    "golden_prefill_c1a_reindex_hc",
    "l3_prefill_c1a_reindex_test",
    "prefill_attn_c1a_reindex",
    "prefill_c1a_reindex",
    "prefill_c1a_reindex_test",
]

_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    main()
