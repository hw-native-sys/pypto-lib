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

from models.deepseek_v4_1_flash.config import (
    CMP_BLOCKS_DYN,
    COMPRESSED_CACHE_GROUP,
    D,
    HEAD_DIM,
    INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP,
    INDEX_DIM,
    INDEX_TOPK,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCKS_DYN,
    Q_LORA,
    ROPE_DIM,
    T_DYN,
    WINDOW_CACHE_GROUP,
)
from models.deepseek_v4_1_flash.o_proj import prefill_o_proj
from models.deepseek_v4_1_flash.qkv_proj_rope import kv_proj_rope, q_proj_rope

from models.deepseek_v4_1_flash.attention_ops import prefill_linear_policy_inline
from models.deepseek_v4_1_flash.attention_ops import prefill_norm_inline
from models.deepseek_v4_1_flash.attention_ops import prefill_packed_linear_inline
from models.deepseek_v4_1_flash.attention_ops import prefill_rope_inline
from models.deepseek_v4_1_flash.attention_ops import prefill_sparse_attention_inline
from models.deepseek_v4_1_flash.attention_ops import prefill_sparse_attention_official_inline
from models.deepseek_v4_1_flash.compressor import prefill_pool_pairs_inline
from models.deepseek_v4_1_flash.config import H
from models.deepseek_v4_1_flash.config import INDEX_H
from models.deepseek_v4_1_flash.config import O_GROUPS
from models.deepseek_v4_1_flash.hierarchical_sparse_indexer import prefill_index_scores_inline
from models.deepseek_v4_1_flash.hierarchical_sparse_indexer import prefill_index_scores_official_inline
from models.deepseek_v4_1_flash.hierarchical_sparse_indexer import prefill_index_topk_inline
from models.deepseek_v4_1_flash.quantization import prefill_quantize_fp8_inline
from models.deepseek_v4_1_flash.quantization import prefill_quantize_index_fp4_inline
from models.deepseek_v4_1_flash.quantization import prefill_quantize_kv_fp4_inline
from models.deepseek_v4_1_flash.quantization import prefill_round_inline

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


@pl.jit.inline
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
    """Merge window MXFP8 and compressed MXFP4 attention in one online softmax."""
    window_rows = pl.tensor.dim(window_cache, 0) * 128
    window_flat = pl.reshape(window_cache, [window_rows, HEAD_DIM])
    window_scale_flat = pl.reshape(window_scale, [window_rows, HEAD_DIM // 32])
    compressed_rows = pl.tensor.dim(compressed_cache, 0) * 128
    compressed_flat = pl.reshape(compressed_cache, [compressed_rows, HEAD_DIM // 2])
    compressed_scale_flat = pl.reshape(
        compressed_scale,
        [compressed_rows, HEAD_DIM // COMPRESSED_CACHE_GROUP],
    )
    query_flat = pl.reshape(query, [pl.tensor.dim(query, 0) * LOCAL_H, HEAD_DIM])
    output_flat = pl.reshape(output, [pl.tensor.dim(output, 0) * LOCAL_H, HEAD_DIM])
    head_blocks = LOCAL_H // M_TILE
    for block in pl.spmd(num_tokens * head_blocks, name_hint="c1a_sparse_attention"):
        token = block // head_blocks
        head = block % head_blocks * M_TILE
        query_row = token * LOCAL_H + head
        query_tile = query_flat[query_row:query_row + M_TILE, :]
        maximum = pl.full([1, M_TILE], dtype=pl.FP32, value=-1e30)
        denominator = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
        numerator = pl.full([M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
        numerator_patch = pl.full([1, 16], dtype=pl.FP32, value=0.0)

        for part in pl.range(128 // ATTENTION_TILE):
            kv = pl.full([ATTENTION_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            valid = pl.full([1, ATTENTION_TILE], dtype=pl.FP32, value=0.0)
            for lane in pl.range(ATTENTION_TILE):
                index_column = part * ATTENTION_TILE + lane
                row_i32 = pl.read(window_indices, [token, index_column])
                if row_i32 >= 0:
                    row = pl.cast(row_i32, pl.INDEX)
                    payload = pl.reshape(
                        pl.cast(window_flat[row:row + 1, :], pl.FP32),
                        [HEAD_DIM // 32, 32],
                    )
                    scale_row = pl.slice(
                        window_scale_flat,
                        [1, 32],
                        [row, 0],
                        valid_shape=[1, HEAD_DIM // 32],
                    )
                    raw_codes = pl.reinterpret_view(scale_row, pl.UINT8)
                    signed_codes = pl.cast(pl.reinterpret_view(raw_codes, pl.INT8), pl.INT32)
                    codes = pl.ands(signed_codes, 255)
                    scale_bits = pl.maximum(pl.shls(codes, 23), 4194304)
                    scale_value = pl.reinterpret_view(scale_bits, pl.FP32)
                    scale = pl.reshape(scale_value[:, :HEAD_DIM // 32], [HEAD_DIM // 32, 1])
                    decoded = pl.cast(pl.row_expand_mul(payload, scale), pl.BF16, mode="rint")
                    kv[lane:lane + 1, :] = pl.reshape(decoded, [1, HEAD_DIM])
                    pl.write(valid, [0, lane], 1.0)
            scores = pl.mul(pl.matmul(query_tile, kv, b_trans=True), SOFTMAX_SCALE)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            scores = pl.col_expand_add(scores, bias)
            next_maximum = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, M_TILE]))
            correction = pl.exp(pl.sub(maximum, next_maximum))
            score_exp = pl.exp(pl.row_expand_sub(scores, pl.reshape(next_maximum, [M_TILE, 1])))
            probability = pl.col_expand_mul(score_exp, valid)
            denominator = pl.add(
                pl.mul(denominator, correction),
                pl.reshape(pl.row_sum(probability), [1, M_TILE]),
            )
            weighted = pl.matmul(pl.cast(probability, pl.BF16, mode="rint"), kv)
            # Recompute the first A5 PV output vector on Vec before overwriting it below.
            patch_products = pl.row_expand_mul(
                pl.cast(kv[:, :16], pl.FP32),
                pl.reshape(probability[0:1, :], [ATTENTION_TILE, 1]),
            )
            weighted_patch = pl.reshape(
                pl.row_sum(pl.transpose(patch_products, axis1=0, axis2=1)),
                [1, 16],
            )
            numerator_patch = pl.add(
                pl.mul(numerator_patch, pl.read(correction, [0, 0])),
                weighted_patch,
            )
            numerator = pl.add(
                pl.row_expand_mul(numerator, pl.reshape(correction, [M_TILE, 1])),
                weighted,
            )
            maximum = next_maximum

        for part in pl.range(INDEX_TOPK // ATTENTION_TILE):
            kv = pl.full([ATTENTION_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            valid = pl.full([1, ATTENTION_TILE], dtype=pl.FP32, value=0.0)
            for lane in pl.range(ATTENTION_TILE):
                index_column = part * ATTENTION_TILE + lane
                row_i32 = pl.read(compressed_indices, [token, index_column])
                if row_i32 >= 0:
                    row = pl.cast(row_i32, pl.INDEX)
                    payload_bytes = compressed_flat[row:row + 1, :]
                    payload_signed = pl.reinterpret_view(payload_bytes, pl.INT8)
                    payload_i32 = pl.ands(pl.cast(payload_signed, pl.INT32), 255)
                    low = pl.ands(payload_i32, 15)
                    high = pl.ands(pl.shrs(payload_i32, 4), 15)
                    low = pl.reshape(low, [1, HEAD_DIM // 2])
                    high = pl.reshape(high, [1, HEAD_DIM // 2])
                    combined_codes = pl.concat(low, high)
                    output_ids = pl.tile.arange(0, [1, HEAD_DIM], dtype=pl.INT32)
                    pair_ids = pl.shrs(output_ids, 1)
                    parity = pl.ands(output_ids, 1)
                    code_indices = pl.add(pair_ids, pl.mul(parity, HEAD_DIM // 2))
                    payload_codes = pl.gather(combined_codes, index=code_indices)
                    magnitude_codes = pl.ands(payload_codes, 7)
                    magnitude = pl.mul(pl.cast(magnitude_codes, pl.FP32), 0.5)
                    extra = pl.minimum(pl.maximum(pl.sub(magnitude_codes, 4), 0), 1)
                    magnitude = pl.add(
                        magnitude,
                        pl.mul(pl.cast(extra, pl.FP32), 0.5),
                    )
                    extra = pl.minimum(pl.maximum(pl.sub(magnitude_codes, 5), 0), 1)
                    magnitude = pl.add(
                        magnitude,
                        pl.mul(pl.cast(extra, pl.FP32), 0.5),
                    )
                    extra = pl.minimum(pl.maximum(pl.sub(magnitude_codes, 6), 0), 1)
                    magnitude = pl.add(
                        magnitude,
                        pl.mul(pl.cast(extra, pl.FP32), 1.5),
                    )
                    sign = pl.cast(pl.ands(pl.shrs(payload_codes, 3), 1), pl.FP32)
                    sign_value = pl.add(pl.mul(sign, -2.0), 1.0)
                    decoded_payload = pl.mul(magnitude, sign_value)
                    payload = pl.reshape(
                        decoded_payload,
                        [HEAD_DIM // COMPRESSED_CACHE_GROUP, COMPRESSED_CACHE_GROUP],
                    )
                    scale = pl.reshape(
                        pl.cast(compressed_scale_flat[row:row + 1, :], pl.FP32),
                        [HEAD_DIM // COMPRESSED_CACHE_GROUP, 1],
                    )
                    decoded = pl.cast(pl.row_expand_mul(payload, scale), pl.BF16, mode="rint")
                    kv[lane:lane + 1, :] = pl.reshape(decoded, [1, HEAD_DIM])
                    pl.write(valid, [0, lane], 1.0)
            scores = pl.mul(pl.matmul(query_tile, kv, b_trans=True), SOFTMAX_SCALE)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            scores = pl.col_expand_add(scores, bias)
            next_maximum = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, M_TILE]))
            correction = pl.exp(pl.sub(maximum, next_maximum))
            score_exp = pl.exp(pl.row_expand_sub(scores, pl.reshape(next_maximum, [M_TILE, 1])))
            probability = pl.col_expand_mul(score_exp, valid)
            denominator = pl.add(
                pl.mul(denominator, correction),
                pl.reshape(pl.row_sum(probability), [1, M_TILE]),
            )
            weighted = pl.matmul(pl.cast(probability, pl.BF16, mode="rint"), kv)
            # Recompute the first A5 PV output vector on Vec before overwriting it below.
            patch_products = pl.row_expand_mul(
                pl.cast(kv[:, :16], pl.FP32),
                pl.reshape(probability[0:1, :], [ATTENTION_TILE, 1]),
            )
            weighted_patch = pl.reshape(
                pl.row_sum(pl.transpose(patch_products, axis1=0, axis2=1)),
                [1, 16],
            )
            numerator_patch = pl.add(
                pl.mul(numerator_patch, pl.read(correction, [0, 0])),
                weighted_patch,
            )
            numerator = pl.add(
                pl.row_expand_mul(numerator, pl.reshape(correction, [M_TILE, 1])),
                weighted,
            )
            maximum = next_maximum

        sinks = pl.reshape(sink[head:head + M_TILE], [1, M_TILE])
        final_maximum = pl.maximum(maximum, sinks)
        correction = pl.exp(pl.sub(maximum, final_maximum))
        denominator = pl.add(
            pl.mul(denominator, correction),
            pl.exp(pl.sub(sinks, final_maximum)),
        )
        normalization = pl.reshape(pl.div(correction, denominator), [M_TILE, 1])
        result = pl.row_expand_mul(numerator, normalization)
        output_flat[query_row:query_row + M_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
        patch_result = pl.mul(numerator_patch, pl.read(normalization, [0, 0]))
        output_flat[query_row:query_row + 1, :16] = pl.cast(patch_result, pl.BF16, mode="rint")
    return output


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
    "prefill_c1a_partial",
    "publish_compressed_cache",
    "publish_index_cache",
]


# Resident CED prefill precision variants.
_PREFILL_ATTENTION_C = pl.dynamic("V41_ATTN_C")


_PREFILL_ATTENTION_T = pl.dynamic("V41_ATTN_T")


_PREFILL_ATTENTION_P = pl.dynamic("V41_ATTN_P")


_PREFILL_ATTENTION_R = pl.dynamic("V41_ATTN_R")


@pl.jit.inline(auto_scope=False)
def prefill_publish_inline(
    values: pl.Tensor[[_PREFILL_ATTENTION_T, _PREFILL_ATTENTION_C], pl.FP32],
    slots: pl.Tensor[[_PREFILL_ATTENTION_T], pl.INT32],
    cache: pl.Tensor[[_PREFILL_ATTENTION_R, _PREFILL_ATTENTION_P, _PREFILL_ATTENTION_C], pl.FP32],
):
    rows = pl.tensor.dim(values, 0)
    width = pl.tensor.dim(values, 1)
    capacity = pl.tensor.dim(cache, 1)
    for worker in pl.spmd(32, name_hint="prefill_publish_cache"):
        for row in pl.range(worker, rows, 32):
            slot = pl.cast(pl.read(slots, [row]), pl.INDEX)
            if slot >= 0:
                for col in pl.range(0, width, 128):
                    active = pl.min(128, width - col)
                    value = pl.slice(values, [1, 128], [row, col], valid_shape=[1, active])
                    cache[
                        slot // capacity : slot // capacity + 1,
                        slot % capacity : slot % capacity + 1,
                        col : col + 128,
                    ] = pl.reshape(value, [1, 1, 128])
    return cache


@pl.jit.inline(auto_scope=False)
def prefill_transpose_inline(
    x: pl.Tensor[[_PREFILL_ATTENTION_R, 128, INDEX_DIM], pl.FP32],
    out: pl.Tensor[[INDEX_DIM, _PREFILL_ATTENTION_C], pl.FP32],
):
    rows = pl.tensor.dim(x, 0) * 128
    cols = INDEX_DIM
    source = pl.reshape(x, [rows, INDEX_DIM])
    for worker in pl.spmd(32, name_hint="prefill_transpose_cache"):
        for block in pl.range(worker, (rows + 31) // 32 * ((cols + 31) // 32), 32):
            row = block // ((cols + 31) // 32) * 32
            col = block % ((cols + 31) // 32) * 32
            nr = pl.min(32, rows - row)
            nc = pl.min(32, cols - col)
            value = pl.load(source, [row, col], [32, 32], valid_shape=[nr, nc])
            out = pl.store(pl.transpose(value, 0, 1), [col, row], out)
    return out


_PREFILL_ATTENTION_V = pl.dynamic("V41_ATTN_V")


_PREFILL_ATTENTION_BG = pl.dynamic("V41_ATTN_BG")


@pl.jit.inline(auto_scope=False)
def prefill_publish_latent_inline(
    latent: pl.Tensor[[_PREFILL_ATTENTION_T, HEAD_DIM], pl.FP32],
    dense_bank: pl.Tensor[[_PREFILL_ATTENTION_V], pl.FP32],
    dense_offsets: pl.Tensor[[40, 20], pl.INT32],
    layer: pl.Scalar[pl.INDEX],
    cos: pl.Tensor[[_PREFILL_ATTENTION_T, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[_PREFILL_ATTENTION_T, ROPE_DIM // 2], pl.FP32],
    slots: pl.Tensor[[_PREFILL_ATTENTION_T], pl.INT32],
    global_cache: pl.Tensor[[_PREFILL_ATTENTION_BG, 128, HEAD_DIM], pl.FP32],
    index_cache: pl.Tensor[[_PREFILL_ATTENTION_BG, 128, INDEX_DIM], pl.FP32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(latent, 0)
    latent_rows = pl.reshape(latent, [rows, HEAD_DIM])
    cos_rows = pl.reshape(cos, [rows, ROPE_DIM // 2])
    sin_rows = pl.reshape(sin, [rows, ROPE_DIM // 2])
    slots_rows = pl.reshape(slots, [rows])
    wk_offset = pl.cast(pl.read(dense_offsets, [layer, 16]), pl.INDEX)
    call_view_1 = dense_bank[wk_offset : wk_offset + HEAD_DIM * INDEX_DIM]
    wk = pl.reshape(call_view_1, [HEAD_DIM, INDEX_DIM])
    key_norm_offset = pl.cast(pl.read(dense_offsets, [layer, 17]), pl.INDEX)
    call_view_2 = dense_bank[key_norm_offset : key_norm_offset + INDEX_DIM]
    key_norm = pl.reshape(call_view_2, [INDEX_DIM])
    key_projection = pl.create_tensor([rows, INDEX_DIM], dtype=pl.FP32)
    prefill_linear_policy_inline(latent_rows, wk, key_projection, official, 1)
    keys = pl.create_tensor([rows, INDEX_DIM], dtype=pl.FP32)
    prefill_norm_inline(key_projection, key_norm, keys, official)
    rotated_kv = pl.create_tensor([rows, 1, HEAD_DIM], dtype=pl.FP32)
    call_view_3 = pl.reshape(latent_rows, [rows, 1, HEAD_DIM])
    prefill_rope_inline(call_view_3, cos_rows, sin_rows, rotated_kv, 1.0, official)
    rotated_index = pl.create_tensor([rows, 1, INDEX_DIM], dtype=pl.FP32)
    call_view_4 = pl.reshape(keys, [rows, 1, INDEX_DIM])
    prefill_rope_inline(call_view_4, cos_rows, sin_rows, rotated_index, 1.0, official)
    kv_values = pl.reshape(rotated_kv, [rows, HEAD_DIM])
    index_values = pl.reshape(rotated_index, [rows, INDEX_DIM])
    if official == 1:
        quantized_kv = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
        quantized_index = pl.create_tensor([rows, INDEX_DIM], dtype=pl.FP32)
        prefill_quantize_kv_fp4_inline(kv_values, quantized_kv)
        kv_values = quantized_kv
        prefill_quantize_index_fp4_inline(index_values, quantized_index)
        index_values = quantized_index
    call_view_5 = global_cache
    prefill_publish_inline(kv_values, slots_rows, call_view_5)
    call_view_6 = index_cache
    prefill_publish_inline(index_values, slots_rows, call_view_6)
    return global_cache, index_cache


@pl.jit.inline(auto_scope=False)
def prefill_publish_decoder_inline(
    hidden: pl.Tensor[[_PREFILL_ATTENTION_T, D], pl.FP32],
    dense_bank: pl.Tensor[[_PREFILL_ATTENTION_V], pl.FP32],
    dense_offsets: pl.Tensor[[40, 20], pl.INT32],
    compressed_cos: pl.Tensor[[_PREFILL_ATTENTION_T, ROPE_DIM // 2], pl.FP32],
    compressed_sin: pl.Tensor[[_PREFILL_ATTENTION_T, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[_PREFILL_ATTENTION_T], pl.INT32],
    global_cache: pl.Tensor[[_PREFILL_ATTENTION_BG, 128, HEAD_DIM], pl.FP32],
    index_cache: pl.Tensor[[_PREFILL_ATTENTION_BG, 128, INDEX_DIM], pl.FP32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(hidden, 0)
    hidden_rows = pl.reshape(hidden, [rows, D])
    compressed_cos_rows = pl.reshape(compressed_cos, [rows, ROPE_DIM // 2])
    compressed_sin_rows = pl.reshape(compressed_sin, [rows, ROPE_DIM // 2])
    compressed_slots_rows = pl.reshape(compressed_slots, [rows])
    weight_offset = pl.cast(pl.read(dense_offsets, [20, 13]), pl.INDEX)
    call_view_7 = dense_bank[weight_offset : weight_offset + D * HEAD_DIM]
    weight = pl.reshape(call_view_7, [D, HEAD_DIM])
    gamma_offset = pl.cast(pl.read(dense_offsets, [20, 15]), pl.INDEX)
    call_view_8 = dense_bank[gamma_offset : gamma_offset + HEAD_DIM]
    gamma = pl.reshape(call_view_8, [HEAD_DIM])
    projection = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
    prefill_linear_policy_inline(hidden_rows, weight, projection, official, 1)
    latent = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
    prefill_norm_inline(projection, gamma, latent, official)
    prefill_publish_latent_inline(
        latent,
        dense_bank,
        dense_offsets,
        20,
        compressed_cos_rows,
        compressed_sin_rows,
        compressed_slots_rows,
        global_cache,
        index_cache,
        official,
    )

    return global_cache, index_cache


_PREFILL_ATTENTION_B8 = pl.dynamic("V41_ATTN_B8")


_PREFILL_ATTENTION_BW = pl.dynamic("V41_ATTN_BW")


_PREFILL_ATTENTION_BS = pl.dynamic("V41_ATTN_BS")


@pl.jit.inline(auto_scope=False)
def prefill_attention_inline(
    hidden: pl.Tensor[[_PREFILL_ATTENTION_T, D], pl.FP32],
    bank8: pl.Tensor[[_PREFILL_ATTENTION_B8, 1024], pl.INT8],
    scales8: pl.Tensor[[_PREFILL_ATTENTION_B8], pl.UINT8],
    fp8_offsets: pl.Tensor[[40, 10], pl.INT32],
    dense_bank: pl.Tensor[[_PREFILL_ATTENTION_V], pl.FP32],
    dense_offsets: pl.Tensor[[40, 20], pl.INT32],
    layer: pl.Scalar[pl.INDEX],
    cos: pl.Tensor[[_PREFILL_ATTENTION_T, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[_PREFILL_ATTENTION_T, ROPE_DIM // 2], pl.FP32],
    compressed_cos: pl.Tensor[[_PREFILL_ATTENTION_T, ROPE_DIM // 2], pl.FP32],
    compressed_sin: pl.Tensor[[_PREFILL_ATTENTION_T, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[_PREFILL_ATTENTION_T], pl.INT32],
    window_indices: pl.Tensor[[_PREFILL_ATTENTION_T, 128], pl.INT32],
    compressed_slots: pl.Tensor[[_PREFILL_ATTENTION_T], pl.INT32],
    previous_rows: pl.Tensor[[_PREFILL_ATTENTION_T], pl.INT32],
    blocks: pl.Tensor[[_PREFILL_ATTENTION_T, _PREFILL_ATTENTION_P], pl.INT32],
    lengths: pl.Tensor[[_PREFILL_ATTENTION_T], pl.INT32],
    extents: pl.Tensor[[_PREFILL_ATTENTION_T, 2], pl.INT32],
    state_slots: pl.Tensor[[_PREFILL_ATTENTION_T], pl.INT32],
    window_cache: pl.Tensor[[_PREFILL_ATTENTION_BW, 128, HEAD_DIM], pl.FP32],
    global_cache: pl.Tensor[[_PREFILL_ATTENTION_BG, 128, HEAD_DIM], pl.FP32],
    index_cache: pl.Tensor[[_PREFILL_ATTENTION_BG, 128, INDEX_DIM], pl.FP32],
    state_cache: pl.Tensor[[_PREFILL_ATTENTION_BS, 4, 2 * HEAD_DIM], pl.FP32],
    topk: pl.Tensor[[_PREFILL_ATTENTION_T, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[_PREFILL_ATTENTION_T, D], pl.FP32],
    mode: pl.Scalar[pl.INT32],
    ratio: pl.Scalar[pl.INT32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(hidden, 0)
    hidden_rows = pl.reshape(hidden, [rows, D])
    cos_rows = pl.reshape(cos, [rows, ROPE_DIM // 2])
    sin_rows = pl.reshape(sin, [rows, ROPE_DIM // 2])
    compressed_cos_rows = pl.reshape(compressed_cos, [rows, ROPE_DIM // 2])
    compressed_sin_rows = pl.reshape(compressed_sin, [rows, ROPE_DIM // 2])
    window_slots_rows = pl.reshape(window_slots, [rows])
    window_indices_rows = pl.reshape(window_indices, [rows, 128])
    compressed_slots_rows = pl.reshape(compressed_slots, [rows])
    previous_rows_rows = pl.reshape(previous_rows, [rows])
    blocks_width_1 = pl.tensor.dim(blocks, 1)
    blocks_rows = pl.reshape(blocks, [rows, blocks_width_1])
    lengths_rows = pl.reshape(lengths, [rows])
    extents_rows = pl.reshape(extents, [rows, 2])
    state_slots_rows = pl.reshape(state_slots, [rows])
    topk_rows = pl.reshape(topk, [rows, INDEX_TOPK])
    output_rows = pl.reshape(output, [rows, D])
    global_pages = pl.tensor.dim(global_cache, 0)
    logical_pages = pl.tensor.dim(blocks_rows, 1)
    qa = pl.create_tensor([rows, Q_LORA], dtype=pl.FP32)
    prefill_packed_linear_inline(
        hidden_rows, bank8, scales8, pl.cast(pl.read(fp8_offsets, [layer, 0]), pl.INDEX), qa, official, 2
    )
    q_gamma_offset = pl.cast(pl.read(dense_offsets, [layer, 4]), pl.INDEX)
    call_view_9 = dense_bank[q_gamma_offset : q_gamma_offset + Q_LORA]
    q_gamma = pl.reshape(call_view_9, [Q_LORA])
    qr = pl.create_tensor([rows, Q_LORA], dtype=pl.FP32)
    prefill_norm_inline(qa, q_gamma, qr, official)
    qb = pl.create_tensor([rows, H * HEAD_DIM], dtype=pl.FP32)
    prefill_packed_linear_inline(
        qr, bank8, scales8, pl.cast(pl.read(fp8_offsets, [layer, 1]), pl.INDEX), qb, official, 2
    )
    query = pl.create_tensor([rows, H, HEAD_DIM], dtype=pl.FP32)
    call_view_10 = pl.reshape(qb, [rows, H, HEAD_DIM])
    prefill_rope_inline(call_view_10, cos_rows, sin_rows, query, 1.0, official)
    kv_projection = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
    prefill_packed_linear_inline(
        hidden_rows,
        bank8,
        scales8,
        pl.cast(pl.read(fp8_offsets, [layer, 2]), pl.INDEX),
        kv_projection,
        official,
        2,
    )
    kv_gamma_offset = pl.cast(pl.read(dense_offsets, [layer, 5]), pl.INDEX)
    call_view_11 = dense_bank[kv_gamma_offset : kv_gamma_offset + HEAD_DIM]
    kv_gamma = pl.reshape(call_view_11, [HEAD_DIM])
    normalized_kv = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
    prefill_norm_inline(kv_projection, kv_gamma, normalized_kv, official)
    rotated_kv = pl.create_tensor([rows, 1, HEAD_DIM], dtype=pl.FP32)
    call_view_12 = pl.reshape(normalized_kv, [rows, 1, HEAD_DIM])
    prefill_rope_inline(call_view_12, cos_rows, sin_rows, rotated_kv, 1.0, official)
    window_values = pl.reshape(rotated_kv, [rows, HEAD_DIM])
    if official == 1:
        narrowed_kv = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
        prefill_quantize_fp8_inline(window_values, narrowed_kv)
        window_values = narrowed_kv
    call_view_13 = window_cache
    prefill_publish_inline(window_values, window_slots_rows, call_view_13)
    if mode == 1:
        if ratio == 2:
            comp_wkv_offset = pl.cast(pl.read(dense_offsets, [layer, 13]), pl.INDEX)
            call_view_14 = dense_bank[comp_wkv_offset : comp_wkv_offset + D * HEAD_DIM]
            comp_wkv = pl.reshape(call_view_14, [D, HEAD_DIM])
            comp_wgate_offset = pl.cast(pl.read(dense_offsets, [layer, 14]), pl.INDEX)
            call_view_15 = dense_bank[comp_wgate_offset : comp_wgate_offset + D * HEAD_DIM]
            comp_wgate = pl.reshape(call_view_15, [D, HEAD_DIM])
            comp_kv = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
            comp_scores = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
            prefill_linear_policy_inline(hidden_rows, comp_wkv, comp_kv, official, 0)
            prefill_linear_policy_inline(hidden_rows, comp_wgate, comp_scores, official, 0)
            pooled = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
            prefill_pool_pairs_inline(comp_kv, comp_scores, previous_rows_rows, pooled)
            rounded_pool = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
            prefill_round_inline(pooled, rounded_pool, official)
            state_values = pl.create_tensor([rows, 2 * HEAD_DIM], dtype=pl.FP32)
            for state_worker in pl.spmd(32, name_hint="prefill_compressor_state"):
                for state_row in pl.range(state_worker, rows, 32):
                    state_values[state_row : state_row + 1, :HEAD_DIM] = comp_kv[state_row : state_row + 1, :]
                    state_values[state_row : state_row + 1, HEAD_DIM:] = comp_scores[
                        state_row : state_row + 1, :
                    ]
            call_view_16 = state_cache
            prefill_publish_inline(state_values, state_slots_rows, call_view_16)
            comp_gamma_offset = pl.cast(pl.read(dense_offsets, [layer, 15]), pl.INDEX)
            call_view_17 = dense_bank[comp_gamma_offset : comp_gamma_offset + HEAD_DIM]
            comp_gamma = pl.reshape(call_view_17, [HEAD_DIM])
            latent = pl.create_tensor([rows, HEAD_DIM], dtype=pl.FP32)
            prefill_norm_inline(rounded_pool, comp_gamma, latent, official)
            prefill_publish_latent_inline(
                latent,
                dense_bank,
                dense_offsets,
                layer,
                compressed_cos_rows,
                compressed_sin_rows,
                compressed_slots_rows,
                global_cache,
                index_cache,
                official,
            )
    select_indices = 0
    if mode == 1:
        select_indices = 1
    elif mode == 2:
        select_indices = 1
    if select_indices == 1:
        iq_projection = pl.create_tensor([rows, INDEX_H * INDEX_DIM], dtype=pl.FP32)
        prefill_packed_linear_inline(
            qr,
            bank8,
            scales8,
            pl.cast(pl.read(fp8_offsets, [layer, 5]), pl.INDEX),
            iq_projection,
            official,
            2,
        )
        iq_rotated = pl.create_tensor([rows, INDEX_H, INDEX_DIM], dtype=pl.FP32)
        call_view_18 = pl.reshape(iq_projection, [rows, INDEX_H, INDEX_DIM])
        prefill_rope_inline(call_view_18, cos_rows, sin_rows, iq_rotated, 1.0, official)
        iq = pl.reshape(iq_rotated, [rows * INDEX_H, INDEX_DIM])
        if official == 1:
            quantized_iq = pl.create_tensor([rows * INDEX_H, INDEX_DIM], dtype=pl.FP32)
            prefill_quantize_index_fp4_inline(iq, quantized_iq)
            iq = quantized_iq
        route_weight_offset = pl.cast(pl.read(dense_offsets, [layer, 18]), pl.INDEX)
        call_view_19 = dense_bank[route_weight_offset : route_weight_offset + D * INDEX_H]
        route_weight = pl.reshape(call_view_19, [D, INDEX_H])
        route_weights = pl.create_tensor([rows, INDEX_H], dtype=pl.FP32)
        prefill_linear_policy_inline(hidden_rows, route_weight, route_weights, official, 1)
        index_kn = pl.create_tensor([INDEX_DIM, global_pages * 128], dtype=pl.FP32)
        call_view_20 = index_cache
        prefill_transpose_inline(call_view_20, index_kn)
        dots = pl.create_tensor([rows * INDEX_H, global_pages * 128], dtype=pl.FP32)
        prefill_linear_policy_inline(iq, index_kn, dots, official, 1)
        scores = pl.create_tensor([rows, logical_pages * 128], dtype=pl.FP32)
        if official == 1:
            prefill_index_scores_official_inline(dots, route_weights, blocks_rows, lengths_rows, scores)
        else:
            prefill_index_scores_inline(dots, route_weights, blocks_rows, lengths_rows, scores)
        prefill_index_topk_inline(scores, blocks_rows, lengths_rows, topk_rows)
    elif mode == 0:
        for topk_worker in pl.spmd(32, name_hint="prefill_swa_indices"):
            for topk_row in pl.range(topk_worker, rows, 32):
                topk_rows[topk_row : topk_row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
    sink_offset = pl.cast(pl.read(dense_offsets, [layer, 6]), pl.INDEX)
    call_view_21 = dense_bank[sink_offset : sink_offset + H]
    sink = pl.reshape(call_view_21, [H])
    attended = pl.create_tensor([rows, H, HEAD_DIM], dtype=pl.FP32)
    for first in pl.range(0, rows, 1024):
        with pl.scope():
            chunk_rows = pl.min(1024, rows - first)
            if official == 1:
                call_view_22 = pl.slice(query, [chunk_rows, H, HEAD_DIM], [first, 0, 0])
                call_view_23 = pl.slice(window_indices_rows, [chunk_rows, 128], [first, 0])
                call_view_24 = pl.slice(topk_rows, [chunk_rows, INDEX_TOPK], [first, 0])
                call_view_25 = pl.slice(extents_rows, [chunk_rows, 2], [first, 0])
                call_view_26 = pl.slice(attended, [chunk_rows, H, HEAD_DIM], [first, 0, 0])
                prefill_sparse_attention_official_inline(
                    call_view_22,
                    window_cache,
                    call_view_23,
                    global_cache,
                    call_view_24,
                    sink,
                    call_view_25,
                    call_view_26,
                )
            else:
                call_view_27 = pl.slice(query, [chunk_rows, H, HEAD_DIM], [first, 0, 0])
                call_view_28 = pl.slice(window_indices_rows, [chunk_rows, 128], [first, 0])
                call_view_29 = pl.slice(topk_rows, [chunk_rows, INDEX_TOPK], [first, 0])
                call_view_30 = pl.slice(attended, [chunk_rows, H, HEAD_DIM], [first, 0, 0])
                prefill_sparse_attention_inline(
                    call_view_27, window_cache, call_view_28, global_cache, call_view_29, sink, call_view_30
                )
    rotated_output = pl.create_tensor([rows, H, HEAD_DIM], dtype=pl.FP32)
    prefill_rope_inline(attended, cos_rows, sin_rows, rotated_output, -1.0, official)
    grouped = pl.reshape(rotated_output, [rows, H * HEAD_DIM])
    projected_groups = pl.create_tensor([rows, O_GROUPS * O_LORA], dtype=pl.FP32)
    woa_base = pl.cast(pl.read(fp8_offsets, [layer, 3]), pl.INDEX)
    for group in pl.range(O_GROUPS):
        with pl.scope():
            # Column slices retain the full-row stride. Materialize each group
            # before the projection helpers reshape their input.
            group_input = pl.create_tensor([rows, O_GROUP_IN], dtype=pl.FP32)
            group_output = pl.create_tensor([rows, O_LORA], dtype=pl.FP32)
            for worker in pl.spmd(32, name_hint="prefill_woa_gather"):
                for job in pl.range(worker, ((rows + 7) // 8) * (O_GROUP_IN // 512), 32):
                    row = job // (O_GROUP_IN // 512) * 8
                    col = job % (O_GROUP_IN // 512) * 512
                    active_rows = pl.min(8, rows - row)
                    part = pl.slice(
                        grouped,
                        [8, 512],
                        [row, group * O_GROUP_IN + col],
                        valid_shape=[active_rows, 512],
                    )
                    group_input[row : row + 8, col : col + 512] = part
            prefill_packed_linear_inline(
                group_input,
                bank8,
                scales8,
                woa_base + group * (O_LORA // 32) * (O_GROUP_IN // 32),
                group_output,
                official,
                1,
            )
            # An orchestration assemble can alias group_output to a strided
            # destination view; copy explicitly to keep projection storage dense.
            for worker in pl.spmd(32, name_hint="prefill_woa_scatter"):
                for job in pl.range(worker, ((rows + 7) // 8) * (O_LORA // 512), 32):
                    row = job // (O_LORA // 512) * 8
                    col = job % (O_LORA // 512) * 512
                    active_rows = pl.min(8, rows - row)
                    part = pl.slice(group_output, [8, 512], [row, col], valid_shape=[active_rows, 512])
                    projected_groups[row : row + 8, group * O_LORA + col : group * O_LORA + col + 512] = part
    prefill_packed_linear_inline(
        projected_groups,
        bank8,
        scales8,
        pl.cast(pl.read(fp8_offsets, [layer, 4]), pl.INDEX),
        output_rows,
        official,
        2,
    )
    return (output, window_cache, global_cache, index_cache, state_cache, topk)
