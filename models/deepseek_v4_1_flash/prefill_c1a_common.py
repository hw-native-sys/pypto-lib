# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared ratio-1 compressed-attention kernels for packed prefill."""

import pypto.language as pl
import pypto.language.distributed as pld

from sparse_attention import attend_packed

from config import (
    COMPRESSED_CACHE_GROUP,
    INDEX_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
    FLASH as M,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    T_DYN,
    TP_SIZE,
    B_DYN,
    CMP_POSITIONS_DYN,
    PREFILL_MAX_TOKENS,
    Q_START_DYN,
    STATE_BLOCKS_DYN,
    STATE_CAPACITY,
    TABLE_DYN,
)


from o_proj import prefill_o_proj
from qkv_proj_rope import kv_proj_rope, q_proj_rope


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk
STATE_WIDTH = 2 * HEAD_DIM

M_TILE = 16
N_TILE = 128
K_TILE = 256
ATTENTION_TILE = 32


@pl.jit.inline
def publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(num_tokens, name_hint="c1a_cache_publish", deps=[cache_ready]) as publish_tid:
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
            flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
            signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
            codes = pl.cast(signed_exponent, pl.INT8)
            encoded = pl.reinterpret_view(pl.reinterpret_view(codes, pl.UINT8), pl.FP8E8M0)
            encoded_row = pl.reshape(encoded, [1, HEAD_DIM // 16])
            encoded_valid = pl.set_validshape(encoded_row, 1, HEAD_DIM // 32)
            scale_flat[slot:slot + 1, :] = encoded_valid
    return cache, scales

SOFTMAX_SCALE = HEAD_DIM**-0.5


@pl.jit.inline
def publish_compressed_cache(
    value: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    scales: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        pl.FP8E4M3FN,
    ],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Publish ratio-1 KV rows as group-16 MXFP4 with E4M3 scales."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    cache_flat = pl.reshape(cache, [cache_rows, HEAD_DIM // 2])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // COMPRESSED_CACHE_GROUP])
    with pl.spmd(num_tokens, name_hint="c1a_compressed_publish") as publish_tid:
        token = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [token])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.cast(pl.load(value, [token, 0], [1, HEAD_DIM]), pl.FP32)
            grouped = pl.reshape(source, [HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP])
            maximum_tmp = pl.create_tile([HEAD_DIM // COMPRESSED_CACHE_GROUP, 128], dtype=pl.FP32)
            maximum = pl.row_max(pl.abs(grouped), tmp_tile=maximum_tmp)
            raw_scale = pl.minimum(
                pl.maximum(pl.mul(maximum, 1.0 / 6.0), 2.0**-9),
                448.0,
            )
            stored_scale = pl.cast(raw_scale, pl.FP8E4M3FN, mode="rint")
            scale = pl.cast(stored_scale, pl.FP32)
            normalized = pl.row_expand_div(grouped, scale)
            normalized = pl.reshape(normalized, [1, HEAD_DIM])
            magnitude = pl.minimum(pl.abs(normalized), 6.0)
            lower = pl.cast(
                pl.add(pl.mul(pl.minimum(magnitude, 2.0), 2.0), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            middle = pl.cast(
                pl.add(pl.minimum(pl.maximum(pl.sub(magnitude, 2.0), 0.0), 2.0), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            upper = pl.cast(
                pl.add(pl.mul(pl.maximum(pl.sub(magnitude, 4.0), 0.0), 0.5), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            payload_codes = pl.add(pl.add(lower, middle), upper)
            bits = pl.reinterpret_view(normalized, pl.INT32)
            sign = pl.ands(pl.shrs(bits, 31), 1)
            payload_codes = pl.add(payload_codes, pl.mul(sign, 8))
            pair_ids = pl.tile.arange(0, [1, HEAD_DIM // 2], dtype=pl.INT32)
            low_indices = pl.mul(pair_ids, 2)
            high_indices = pl.add(low_indices, 1)
            low_tmp = pl.create_tile([1, HEAD_DIM // 2], dtype=pl.INT32)
            high_tmp = pl.create_tile([1, HEAD_DIM // 2], dtype=pl.INT32)
            low = pl.tile.gather(payload_codes, low_indices, low_tmp)
            high = pl.tile.gather(payload_codes, high_indices, high_tmp)
            payload_bytes = pl.reshape(
                pl.cast(pl.add(low, pl.shls(high, 4)), pl.UINT8),
                [1, HEAD_DIM // 2],
            )
            pl.store(payload_bytes, [slot, 0], cache_flat)
            pl.store(
                pl.reshape(stored_scale, [1, HEAD_DIM // COMPRESSED_CACHE_GROUP]),
                [slot, 0],
                scale_flat,
            )
    return publish_tid


@pl.jit.inline
def publish_index_cache(
    value: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    scales: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP],
        pl.FP8E8M0,
    ],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Publish index-key rows as group-32 MXFP4 with E8M0 scales."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    cache_flat = pl.reshape(cache, [cache_rows, INDEX_DIM // 2])
    scale_flat = pl.reshape(scales, [cache_rows, INDEX_DIM // INDEX_CACHE_GROUP])
    with pl.spmd(num_tokens, name_hint="c1a_index_publish") as publish_tid:
        token = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [token])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.cast(pl.load(value, [token, 0], [1, INDEX_DIM]), pl.FP32)
            source_groups = pl.reshape(source, [INDEX_DIM // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP])
            grouped = pl.tile.full([8, INDEX_CACHE_GROUP], dtype=pl.FP32, value=0.0)
            grouped[:INDEX_DIM // INDEX_CACHE_GROUP, :] = source_groups
            maximum_tmp = pl.create_tile([8, 128], dtype=pl.FP32)
            maximum = pl.row_max(pl.abs(grouped), tmp_tile=maximum_tmp)
            raw_scale = pl.maximum(pl.mul(maximum, 1.0 / 6.0), 2.0**-127)
            bits = pl.reinterpret_view(raw_scale, pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            normalized = pl.reshape(pl.row_expand_div(grouped, scale), [1, 8 * INDEX_CACHE_GROUP])
            padded = pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
            padded[:, :8 * INDEX_CACHE_GROUP] = normalized
            magnitude = pl.minimum(pl.abs(padded), 6.0)
            lower = pl.cast(
                pl.add(pl.mul(pl.minimum(magnitude, 2.0), 2.0), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            middle = pl.cast(
                pl.add(pl.minimum(pl.maximum(pl.sub(magnitude, 2.0), 0.0), 2.0), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            upper = pl.cast(
                pl.add(pl.mul(pl.maximum(pl.sub(magnitude, 4.0), 0.0), 0.5), 0.4999),
                pl.INT32,
                mode="trunc",
            )
            payload_codes = pl.add(pl.add(lower, middle), upper)
            bits = pl.reinterpret_view(padded, pl.INT32)
            sign = pl.ands(pl.shrs(bits, 31), 1)
            payload_codes = pl.add(payload_codes, pl.mul(sign, 8))
            pair_ids = pl.tile.arange(0, [1, HEAD_DIM // 2], dtype=pl.INT32)
            low_indices = pl.mul(pair_ids, 2)
            high_indices = pl.add(low_indices, 1)
            low_tmp = pl.create_tile([1, HEAD_DIM // 2], dtype=pl.INT32)
            high_tmp = pl.create_tile([1, HEAD_DIM // 2], dtype=pl.INT32)
            low = pl.tile.gather(payload_codes, low_indices, low_tmp)
            high = pl.tile.gather(payload_codes, high_indices, high_tmp)
            packed = pl.reshape(
                pl.cast(pl.add(low, pl.shls(high, 4)), pl.UINT8),
                [1, HEAD_DIM // 2],
            )
            payload_bytes = pl.tile.slice(packed, [1, INDEX_DIM // 2], [0, 0])
            pl.store(payload_bytes, [slot, 0], cache_flat)
            exponent_row = pl.reshape(exponent, [1, 8])
            exponent_padded = pl.tile.full([1, 32], dtype=pl.INT32, value=0)
            exponent_padded[:, :8] = exponent_row
            signed_exponent = pl.sub(
                exponent_padded,
                pl.mul(pl.shrs(exponent_padded, 7), 256),
            )
            codes = pl.reinterpret_view(pl.cast(signed_exponent, pl.INT8), pl.UINT8)
            encoded = pl.reinterpret_view(codes, pl.FP8E8M0)
            encoded = pl.tile.set_validshape(
                encoded,
                1,
                INDEX_DIM // INDEX_CACHE_GROUP,
            )
            pl.store(
                encoded,
                [slot, 0],
                scale_flat,
            )
    return publish_tid


@pl.jit.inline(auto_scope=False)
def attend_sparse_cache(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        pl.FP8E4M3FN,
    ],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Gather each cache row once, then compute sparse attention on Cube."""
    return attend_packed(
        query, window_indices, window_cache, window_scale, compressed_indices,
        compressed_cache, compressed_scale, sink, output, num_tokens,
    )


@pl.jit.inline(auto_scope=False)
def prefill_c1a_partial(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
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
    window_cache_scale: pl.Tensor[
        [ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
        pl.FP8E8M0,
    ],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        pl.FP8E4M3FN,
    ],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Compute one TP rank's ratio-1 compressed-attention output."""
    tokens = pl.tensor.dim(x, 0)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    q_proj_rope(query_latent, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens)
    cache_ready = pl.system.task_dummy(deps=[])
    publish_window(
        window_kv,
        window_slots,
        window_cache,
        window_cache_scale,
        num_tokens,
        cache_ready,
    )

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attend_sparse_cache(
        query,
        window_indices,
        window_cache,
        window_cache_scale,
        compressed_indices,
        compressed_cache,
        compressed_cache_scale,
        attn_sink,
        attended,
        num_tokens,
    )
    prefill_o_proj(attended, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, output, num_tokens)
    return output


__all__ = [
    "attend_sparse_cache",
    "make_prefill_attention",
    "make_prefill_publish_decoder",
    "prefill_c1a_partial",
    "publish_compressed_cache",
    "publish_index_cache",
]


def make_prefill_attention():
    """Compose existing attention leaves for the resident CED graph.

    Modes are SWA, C2A full/reuse, and C1A initial-index/reindex/reuse (0–5).
    The C1A source cache is published separately over all encoder rows before
    the decoder consumes its bounded replay rows. Imports stay inside this
    factory because the C1A leaves themselves use this module's cache kernels.
    """
    from prefill_attn_swa import prefill_attn_swa
    from prefill_attn_c2a_full import prefill_attn_c2a_full
    from prefill_attn_c2a_reuse import prefill_attn_c2a_reuse
    from prefill_attn_c1a_reindex import (
        make_prefill_attn_c1a_reindex,
        prefill_attn_c1a_reindex,
    )
    from prefill_attn_c1a_reuse import prefill_attn_c1a_reuse
    from prefill_c1a_indexer import make_paged_indexer

    initial_index = make_prefill_attn_c1a_reindex(make_paged_indexer(use_candidates=False))

    @pl.jit.inline(auto_scope=False)
    def prefill_attention(
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
        token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
        ],
        index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        position_ids: pl.Tensor[[T_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
        compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
        query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
        compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
        topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
        mode: pl.Scalar[pl.INT32],
    ):
        if mode == 0:
            prefill_attn_swa(
                x, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale, output_window,
                output_arrived, output, group_base, tp_rank, num_tokens,
                attention_epoch,
            )
        elif mode == 1:
            prefill_attn_c2a_full(
                x, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
                compressed_cache_scale, token_to_req_indices, compressed_lens, index_cache, index_cache_scale,
                index_block_table, position_ids, compressed_rope_cos, compressed_rope_sin, compressor_wkv,
                compressor_wgate, query_start_loc, state_block_table, state_cache, compressor_norm_weight,
                compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
                index_weights_proj, topk_indices, output_window, output_arrived, output,
                group_base, tp_rank, num_tokens, attention_epoch,
            )
        elif mode == 2:
            prefill_attn_c2a_reuse(
                x, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
                compressed_cache_scale, topk_indices, output_window, output_arrived, output,
                group_base, tp_rank, num_tokens, attention_epoch,
            )
        elif mode == 3:
            initial_index(
                x, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
                compressed_cache_scale, token_to_req_indices, compressed_lens, index_cache, index_cache_scale,
                index_block_table, candidate_mask, index_wq_b, index_wq_b_scale, index_weights_proj,
                topk_indices, output_window, output_arrived, output, group_base,
                tp_rank, num_tokens, attention_epoch,
            )
        elif mode == 4:
            prefill_attn_c1a_reindex(
                x, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
                compressed_cache_scale, token_to_req_indices, compressed_lens, index_cache, index_cache_scale,
                index_block_table, candidate_mask, index_wq_b, index_wq_b_scale, index_weights_proj,
                topk_indices, output_window, output_arrived, output, group_base,
                tp_rank, num_tokens, attention_epoch,
            )
        elif mode == 5:
            prefill_attn_c1a_reuse(
                x, wq_a, wq_a_scale, q_norm_weight, wq_b,
                wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
                compressed_cache_scale, topk_indices, output_window, output_arrived, output,
                group_base, tp_rank, num_tokens, attention_epoch,
            )
        return output

    return prefill_attention


def make_prefill_publish_decoder():
    """Reuse the C1A source stages over the complete encoder output."""
    from prefill_attn_c1a_full import (
        normalize_compressed,
        normalize_index_key,
        project_compressed,
        project_index_key,
        rotate_compressed,
        rotate_index_key,
    )

    @pl.jit.inline(auto_scope=False)
    def prefill_publish_decoder(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
        compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
        compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
        ],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
        ],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x, 0)
        compressed_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        project_compressed(x, compressor_wkv, compressed_projection, num_tokens)
        compressed_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        normalize_compressed(
            compressed_projection, compressor_norm_weight, compressed_latent, num_tokens,
        )
        compressed_rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        rotate_compressed(
            compressed_latent, compressed_rope_cos, compressed_rope_sin, compressed_rotated, num_tokens,
        )
        compressed_ready = publish_compressed_cache(
            compressed_rotated, compressed_slots, compressed_cache, compressed_cache_scale, num_tokens,
        )
        index_projection = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        project_index_key(compressed_latent, index_wk, index_projection, num_tokens)
        index_normalized = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        normalize_index_key(index_projection, index_norm_weight, index_normalized, num_tokens)
        index_rotated = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
        rotate_index_key(
            index_normalized, compressed_rope_cos, compressed_rope_sin, index_rotated, num_tokens,
        )
        index_ready = publish_index_cache(
            index_rotated, compressed_slots, index_cache, index_cache_scale, num_tokens,
        )
        ready = pl.system.task_dummy(deps=[compressed_ready, index_ready])
        return ready

    return prefill_publish_decoder
