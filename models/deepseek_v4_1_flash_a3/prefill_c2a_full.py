# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C2A Full wired through mHC, and the validation shared with Reuse.

One attention sublayer of the text backbone, in the order the official ``Block.forward`` runs it::

    x, attn_pre, attn_post, attn_comb = hc_pre(x_hc, pre_mix)
    x = rms_norm(x)                                 # pre_mix: the previous sublayer's delayed mix
    x = allgather(x)                                # normalized input in full token order
    x = reduce_scatter(attention_partial(x))         # FP32 head-TP sum, local token rows
    x_hc = hc_post(x, x_hc, attn_post, attn_comb)

``attn_pre`` comes back as ``next_pre_mix``; the FFN sublayer of the same layer collapses its input
with it.
"""

import pypto.language as pl
from .activation_quant import quantize_projection_activation
import pypto.language.distributed as pld
import torch

from .hc import hc_pre
from .config import (
    FLASH as M, TP_SIZE, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, PREFILL_MAX_TOKENS, STATE_CAPACITY,
    WINDOW_CACHE_GROUP,
)
from .dsl_types import (
    B_DYN, CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN, TABLE_DYN, Q_START_DYN,
    STATE_BLOCKS_DYN, SP_T_DYN,
)
from .compressor import STATE_WIDTH, c2a_compressor_project, c2a_compressor_pool_state, c2a_compressor_publish
from .prefill_cp_allgather import prefill_sp_input_allgather
from .o_proj import (
    retire_prefill_signals,
    prefill_sp_output_reduce_scatter, prefill_o_proj_nz as o_proj,
)
from .hc import prefill_sp_post
from .rmsnorm import rms_norm_tiled
from .rope import ROPE_ROWS_DYN, materialize_rope_rows
from .qkv_proj_rope import q_proj_rope, kv_proj_rope
from .sparse_attention import PACKED_KEY_TILE, attend_packed, prefill_publish_window
from .prefill_indexer import prefill_indexer, publish_c2a_index_key

# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4

D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk


@pl.jit.inline(auto_scope=False)
def prefill_c2a_full_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
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
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    shared_prefix: pl.Scalar[pl.INT32] = -1,
):
    """Return replicated output for standalone C2A attention callers."""
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="c2a_previous_epoch", allow_early_resolve=False
    ) as cache_ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                cmp=pld.WaitCmp.Ge,
            )
    tokens = pl.tensor.dim(x, 0)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    proj_payload, proj_scales = quantize_projection_activation(x, num_tokens, D, True, True)
    _, query, qb_payload, qb_scales = q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, query,
        num_tokens,
    )
    kv_proj_rope(
        proj_payload, proj_scales,
        wkv, wkv_scale, kv_norm_weight,
        rope_cos, rope_sin,
        kv_projection, window_kv,
        num_tokens,
    )
    window_publish_tid = prefill_publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)

    compressor_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    compressor_score = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    c2a_compressor_project(x, compressor_wkv, compressor_wgate, compressor_kv, compressor_score, num_tokens)
    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    pool_tid, state_tid = c2a_compressor_pool_state(
        compressor_kv, compressor_score, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, compressor_norm_weight, latent, num_tokens, cache_ready,
    )

    # Index keys come from the unrotated latent; the compressed cache stores the rotated one.
    index_publish_tid = publish_c2a_index_key(
        latent, index_wk, index_norm_weight, compressed_rope_cos, compressed_rope_sin,
        compressed_slots, index_cache, index_cache_scale, num_tokens, pool_tid, cache_ready,
    )
    compressed_publish_tid = c2a_compressor_publish(
        latent, compressed_rope_cos, compressed_rope_sin, compressed_slots,
        compressed_cache, compressed_cache_scale, num_tokens, cache_ready, pool_tid, max_workers=64,
    )

    select_tid = prefill_indexer(
        x, qb_payload, qb_scales, token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table,
        rope_cos, rope_sin, index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, compressed_lens, topk_indices, num_tokens, index_publish_tid, pool=False, c2a=True,
    )

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    cache_published = pl.system.task_dummy(deps=[select_tid, state_tid, window_publish_tid, compressed_publish_tid])
    attended = attend_packed(
        query, window_indices, window_cache, window_cache_scale, topk_indices,
        compressed_cache, compressed_cache_scale, attn_sink, attended, num_tokens, cache_published,
        key_tile=PACKED_KEY_TILE, token_tile=128, shared_prefix=shared_prefix,
    )

    o_proj(
        attended, wo_a, wo_b, rope_cos, rope_sin, partial, num_tokens,
    )
    return partial


@pl.jit.inline
def prefill_c2a_full_sp(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
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
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    x_hc_out: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Full attention sublayer between mHC pre and post.

    ``x_hc`` and ``pre_mix`` contain contiguous local token rows. ``num_tokens`` is
    the valid global count of this DP group; metadata follows the gathered group order.
    ``attn_input`` is gathered, while ``attn_output``, ``next_pre_mix`` and ``x_hc_out``
    stay local. Inactive local outputs are zero. Outputs must not alias inputs.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hc_collapsed, next_pre_mix, post_mix, residual_mix,
        prefill=True,
    )
    hc_norm_tokens = pl.tensor.dim(hc_collapsed, 0)
    norm_tiles = (hc_norm_tokens + 7) // 8
    norm_workers = pl.max(1, pl.min(norm_tiles, pl.system.available_aiv_count()))
    for norm_worker in pl.spmd(norm_workers, name_hint="rms_norm"):
        for hc_norm_block in pl.range(norm_worker, norm_tiles, norm_workers):
            rms_norm_tiled(hc_collapsed, attn_norm_weight, local_hidden, hc_norm_block, hc_norm_tokens, output_fp32=False, compensated=False)
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, attn_input,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    group_tokens = pl.tensor.dim(attn_input, 0)
    partial = pl.create_tensor([group_tokens, D], dtype=pl.BF16)
    topk_tiles = group_tokens
    topk_workers = pl.max(1, pl.min(topk_tiles, pl.system.available_aiv_count()))
    for topk_worker in pl.spmd(topk_workers, name_hint="sp_topk_padding"):
        for row in pl.range(topk_worker, topk_tiles, topk_workers):
            if row >= num_tokens:
                topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
    if num_tokens > 0:
        prefill_c2a_full_attention(
            attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens, index_cache, index_cache_scale, index_block_table, position_ids, compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate, query_start_loc, state_block_table, state_cache, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, output_arrived, partial, num_tokens, attention_epoch,
        )
    prefill_sp_output_reduce_scatter(
        partial, output_window, output_arrived, attn_output,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    prefill_sp_post(
        attn_output, x_hc, post_mix, residual_mix, x_hc_out, next_pre_mix, tp_rank, num_tokens,
    )
    return x_hc_out


MX_GROUP = M.weight_block_size[0]

CMP_SCALES = HEAD_DIM // 16

IDX_SCALES = INDEX_DIM // 32

CMP_PACKED = HEAD_DIM // 2

IDX_PACKED = INDEX_DIM // 2

WINDOW = M.sliding_window

ATTEND_TILE = 64

E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

RATIO = 2

TOPK = M.index_topk

CMP_GROUP = COMPRESSED_CACHE_GROUP

IDX_GROUP = INDEX_CACHE_GROUP


INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a",  "wo_b",  "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
    "compressed_cache", "compressed_cache_scale", "token_to_req_indices", "compressed_lens",
    "index_cache", "index_cache_scale", "index_block_table", "position_ids",
    "compressed_rope_cos", "compressed_rope_sin", "compressor_wkv", "compressor_wgate",
    "query_start_loc", "state_block_table", "state_cache", "compressor_norm_weight", "compressed_slots",
    "index_wk", "index_norm_weight", "index_wq_b", "index_wq_b_scale", "index_weights_proj",
)

FULL_ROPE_NAMES = {
    "rope_cos": "freqs_cos", "rope_sin": "freqs_sin",
    "compressed_rope_cos": "compressed_freqs_cos",
    "compressed_rope_sin": "compressed_freqs_sin",
}

PROGRAM_INPUT_NAMES = tuple(
    entry
    for name in INPUT_NAMES
    for entry in (("compressed_rope_positions", name) if name == "compressor_wkv"
                  else (FULL_ROPE_NAMES.get(name, name),))
)

_PREFIXES = {
    ("mixed", "decode"): (1, 63, 128, 255, 0, 511, 129, 256),
    ("mixed", "prefill"): (0, 1, 64, 128, 0, 255, 2, 127),
    ("long", "decode"): (1024, 1027, 1290, 63, 1535, 128, 2048, 255),
    ("long", "prefill"): (1024, 1027, 1291, 62, 1534, 129, 2049, 254),
}


MUTABLE_NAMES = (
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "index_cache",
    "index_cache_scale",
    "state_cache",
)

SHARDED_NAMES = ("wq_b", "wq_b_scale", "attn_sink", "wo_a",  "wo_b", "wo_b_scale")

REUSE_INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a",  "wo_b",  "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale", "compressed_cache",
    "compressed_cache_scale", "compressed_indices",
)

REUSE_MUTABLE_NAMES = ("window_cache", "window_cache_scale")

REUSE_SHARDED_NAMES = ("wq_b", "wq_b_scale", "attn_sink", "wo_a",  "wo_b", "wo_b_scale")


MODES = {
    "full": (INPUT_NAMES, MUTABLE_NAMES, SHARDED_NAMES),
    "reuse": (REUSE_INPUT_NAMES, REUSE_MUTABLE_NAMES, REUSE_SHARDED_NAMES),
}
# Per-rank attention results besides the TP-reduced output.
ATTENTION_STATE = {"full": MUTABLE_NAMES + ("topk_indices",), "reuse": REUSE_MUTABLE_NAMES}


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_c2a_full(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
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
    freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.UINT8]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_positions: pl.Tensor[[T_DYN], pl.INT32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Bind the runtime shapes and run the sublayer once on one rank."""
    freqs_cos.bind_dynamic(0, ROPE_ROWS_DYN)
    x_hc.bind_dynamic(0, SP_T_DYN)
    attn_input.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    query_start_loc.bind_dynamic(0, Q_START_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    active = pl.read(num_tokens, [0])
    tokens = pl.tensor.dim(attn_input, 0)
    rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    compressed_rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    compressed_rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    materialize_rope_rows(freqs_cos, freqs_sin, position_ids, active, rope_cos, rope_sin, False)
    materialize_rope_rows(
        compressed_freqs_cos, compressed_freqs_sin, compressed_rope_positions, active,
        compressed_rope_cos, compressed_rope_sin, False,
    )
    prefill_c2a_full_sp(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens, index_cache, index_cache_scale, index_block_table, position_ids, compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate, query_start_loc, state_block_table, state_cache, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, input_window, input_arrived, output_window, output_arrived, attn_input, attn_output, next_pre_mix, x_hc_out, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active, attention_epoch,
    )
    retire_prefill_signals(x_hc_out, input_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    retire_prefill_signals(x_hc_out, output_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    return (
        x_hc_out, next_pre_mix, attn_output, attn_input, topk_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, index_cache, index_cache_scale, state_cache,
    )


@pl.jit.host
def l3_prefill_c2a_full(
    x_hc: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[RANKS_DYN, 1, Q_LORA], pl.FP32],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[RANKS_DYN, 1, LOCAL_H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[RANKS_DYN, 1, HEAD_DIM], pl.FP32],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    freqs_cos: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16]
    ],
    token_to_req_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.UINT8]],
    index_block_table: pl.Tensor[[RANKS_DYN, B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_freqs_cos: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_freqs_sin: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_positions: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressor_wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
    compressor_wgate: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[RANKS_DYN, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS_DYN, B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[
        pl.Tensor[[RANKS_DYN, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]
    ],
    compressor_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[RANKS_DYN, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS_DYN, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, INDEX_H * INDEX_DIM], pl.INT8],
    index_wq_b_scale: pl.Tensor[[RANKS_DYN, 1, INDEX_H * INDEX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[RANKS_DYN, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32]],
    attn_input: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Allocate the TP communication windows and launch one rank entry per device."""
    freqs_cos.bind_dynamic(1, ROPE_ROWS_DYN)
    x_hc.bind_dynamic(0, RANKS_DYN)
    x_hc.bind_dynamic(1, SP_T_DYN)
    attn_input.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    input_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    input_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    data_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    # These signal windows are fresh on every standalone launch.
    for rank in pl.range(pld.world_size()):
        input_data = pld.window(input_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        input_signal = pld.window(input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        data = pld.window(data_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        # Each rank consumes packed MX_B_NN scale rows.
        wq_a_scale_r: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND] = wkv_scale[rank]
        index_wq_b_scale_r: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND] = index_wq_b_scale[rank]
        l2_prefill_c2a_full(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank], freqs_cos[rank], freqs_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank], token_to_req_indices[rank], compressed_lens[rank], index_cache[rank], index_cache_scale[rank], index_block_table[rank], position_ids[rank], compressed_freqs_cos[rank], compressed_freqs_sin[rank], compressed_rope_positions[rank], compressor_wkv[rank], compressor_wgate[rank], query_start_loc[rank], state_block_table[rank], state_cache[rank], compressor_norm_weight[rank], compressed_slots[rank], index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], attn_input[rank], attn_output[rank], next_pre_mix[rank], x_hc_out[rank], input_data, input_signal, data, signal, rank, num_tokens[rank], 1, device=rank,
        )


PREFILL_C2A_FULL_RING_HEAP = (2 * 1024 * 1024 * 1024,) * 4
