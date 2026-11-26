# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C1A reuse decode attention with mHC."""

import pypto.language as pl
import pypto.language.distributed as pld
from .config import TP_SIZE
from .config import COMPRESSED_CACHE_GROUP, WINDOW_CACHE_GROUP, FLASH as M
from .dsl_types import CMP_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN, LOCAL_T_DYN
from .hc import hc_decode_mixes, hc_head_norm_quantize_decode
from .qkv_proj_rope import decode_qkv_proj_rope
from .sparse_attention import (
    gather_packed, attend_decode_selected, publish_window_row, GATHER_WORKERS as GATHER_WORKER_TILE,
    WORKERS as ATTEND_WORKER_TILE,
)
from .o_proj import decode_o_proj_nz_from_attention as o_proj, decode_tp_output_post


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
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
INDEX_TOPK = M.index_topk


@pl.jit.inline(auto_scope=False)
def decode_c1a_reuse(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    ffn_input: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    output_window: pld.DistributedTensor,
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    selected: pl.Tensor[[LOCAL_T_DYN, 640, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[LOCAL_T_DYN, 640], pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    proj_payload, proj_scales, x_ready = hc_head_norm_quantize_decode(
        x_hc, pre_mix, attn_norm_weight, x_norm, num_tokens, zero_padding=False,
    )

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    decode_qkv_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight,
        rope_cos, rope_sin, qr, query, kv, num_tokens,
    )
    publish_window_row(kv, window_slots, window_cache, window_cache_scale, num_tokens)
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected_ret, indices_ret, gather_tid = gather_packed(
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        window_indices, compressed_indices, selected, indices, num_tokens, x_ready, max_workers=GATHER_WORKER_TILE,
        name_hint="c1a_gather", gather_keys=128, decode=True,
    )
    attend_decode_selected(
        query, selected, indices, attn_sink, attended, num_tokens, gather_tid, 64, ATTEND_WORKER_TILE, rope_cos, rope_sin,
    )
    hc_decode_mixes(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix,
        num_tokens,
    )
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    o_proj(
        attended, wo_a, wo_b, rope_cos, rope_sin, attn_out, num_tokens,
    )
    decode_tp_output_post(
        attn_out, output_window, output_arrived, group_base, tp_rank, num_tokens, attention_epoch,
        x_hc, post_mix, residual_mix, next_pre_mix, output, ffn_input,
    )
    return output
