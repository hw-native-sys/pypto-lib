# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C1A and C2A index-key publication, scoring, and Top-K selection."""

import pypto.language as pl
from config import INDEX_CACHE_GROUP, FLASH as M, INDEX_BLOCKS_DYN, T_DYN, B_DYN, TABLE_DYN
from hierarchical_sparse_indexer import (
    _hierarchical_sparse_indexer, make_index_query_quantizer, materialize_index_topk,
)
from rmsnorm import EPS


from quant import mxfp8_gemm, bf16_gemm_staged


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk


# tiling
M_TILE = 16
INDEX_SCORE_SCALE = INDEX_DIM ** -0.5 * INDEX_H ** -0.5
INDEX_PAGE = 128
INDEX_SCORE_TILE = 64
INDEX_QUERY_TILE = 128
INDEX_MAX_LOGITS_BYTES = 512 * 1024 * 1024
INDEX_PACKED_ELEMENTS = INDEX_SCORE_TILE * INDEX_DIM // 2
TOPK_LEAF = 8192
TOPK_PAIR_WIDTH = 2 * INDEX_TOPK
TOPK_MAX_LEAVES = (1048576 + TOPK_LEAF - 1) // TOPK_LEAF
TOPK_GROUP_LEAVES = 2
TOPK_GROUPS_PER_TOKEN = (TOPK_MAX_LEAVES + TOPK_GROUP_LEAVES - 1) // TOPK_GROUP_LEAVES


quantize_index_query = make_index_query_quantizer(dequantize=True)

INDEX_PROJ_N_TILE = 128
INDEX_WEIGHT_N_TILE = min(128, INDEX_H)
INDEX_PROJ_K_TILE = 256


@pl.jit.inline
def _index_query_rope_block(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
):
    t = block // INDEX_H
    h = block % INDEX_H
    base = h * INDEX_DIM
    output[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)]
    tail = pl.cast(x[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM], pl.FP32)
    even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
    odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
    c = cos[t : t + 1, :]
    s = pl.mul(sin[t : t + 1, :], 1.0)
    re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
    im = pl.add(pl.mul(even, s), pl.mul(odd, c))
    rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
    rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
    output[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM] = pl.cast(
        rotated, pl.BF16, mode="rint"
    )
    return output


@pl.jit.inline
def _index_key_rope_block(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
):
    t = block
    base = 0
    output[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)]
    tail = pl.cast(x[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM], pl.FP32)
    even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
    odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
    c = cos[t : t + 1, :]
    s = pl.mul(sin[t : t + 1, :], 1.0)
    re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
    im = pl.add(pl.mul(even, s), pl.mul(odd, c))
    rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
    rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
    output[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM] = pl.cast(
        rotated, pl.BF16, mode="rint"
    )
    return output


@pl.jit.inline(auto_scope=False)
def rotate_index_query(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = num_tokens * INDEX_H
    workers = pl.min(blocks, 64)
    for worker in pl.spmd(workers, name_hint='prefill_c1a_index_rope'):
        for block in pl.range(worker, blocks, workers):
            _index_query_rope_block(x, cos, sin, output, block)
    return output


@pl.jit.inline
def project_index_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project BF16 per-head weights with BF16 rounding before and after scaling."""
    tokens = pl.tensor.dim(x, 0)
    projected_weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    weight_projection_ready = pl.system.task_dummy(deps=[])
    weight_projection_result, weight_projection_tid = bf16_gemm_staged(
        x, weight,
        projected_weights,
        num_tokens, weight_projection_ready,
        width=D, output_width=INDEX_H, n_tile=INDEX_WEIGHT_N_TILE,
        k_tile=INDEX_PROJ_K_TILE, workers=2147483647, name_hint='prefill_bf16_projection_cube',
    )
    with pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="c1a_index_weights_scale") as weights_tid:
        block = pl.tile.get_block_idx()
        token = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - token)
        projected = pl.slice(projected_weights, [M_TILE, INDEX_H], [token, 0], valid_shape=[rows, INDEX_H])
        scaled = pl.mul(pl.cast(projected, pl.FP32), INDEX_SCORE_SCALE)
        rounded = pl.cast(scaled, pl.BF16, mode="rint")
        output[token:token + M_TILE, :] = pl.set_validshape(rounded, rows, INDEX_H)
    return weights_tid


@pl.jit.inline
def _merge_topk_pairs(
    arena: pl.Tensor,
    left: pl.Scalar[pl.INDEX],
    right: pl.Scalar[pl.INDEX],
    output: pl.Scalar[pl.INDEX],
):
    left_pairs = pl.load(arena, [left, 0], [1, TOPK_PAIR_WIDTH])
    right_pairs = pl.load(arena, [right, 0], [1, TOPK_PAIR_WIDTH])
    temporary = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left_pairs, right_pairs, tmp=temporary)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output, 0], arena)


@pl.jit.inline
def _merge_topk_level(
    arena: pl.Tensor,
    base: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
):
    output_count = (count + 1) // 2
    for output in pl.range(output_count):
        left = base + 2 * output
        right = left + 1
        if right < base + count:
            _merge_topk_pairs(arena, left, right, base + output)
        else:
            forwarded = pl.load(arena, [left, 0], [1, TOPK_PAIR_WIDTH])
            pl.store(forwarded, [base + output, 0], arena)
    return output_count


@pl.jit.inline
def _sort_topk_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(
        scores,
        [token, logical_begin],
        [1, TOPK_LEAF],
        valid_shape=[1, valid_count],
    )
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, TOPK_LEAF], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    indices = pl.add(pl.tile.arange(0, [1, TOPK_LEAF], dtype=pl.INT32), logical_begin_i32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


def make_paged_indexer(use_candidates=False, direct_topk=False, max_logits_bytes=INDEX_MAX_LOGITS_BYTES):
    """Specialize paged FP4 index scoring with optional candidate filtering."""

    @pl.jit.inline(auto_scope=False)
    def paged_indexer_chunk(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP],
            pl.FP8E8M0,
        ],
        index_block_table: pl.Tensor,
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[
            [Q_LORA // 32, INDEX_H * INDEX_DIM],
            pl.FP8E8M0,
            pl.MX_B_NN,
        ],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        candidate_mask: pl.Tensor,
        scores: pl.Tensor,
        topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(x, 0)
        positions = pl.tensor.dim(candidate_mask, 1)
        index_query_raw = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
        mxfp8_gemm(
            query_latent, index_wq_b, index_wq_b_scale,
            index_query_raw,
            num_tokens,
            width=Q_LORA, output_width=INDEX_H * INDEX_DIM, fp32_output=False,
            n_tile=128, name_hint="c1a_mx_projection",
        )
        index_query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
        rotate_index_query(
            index_query_raw,
            rope_cos,
            rope_sin,
            index_query,
            num_tokens,
        )
        rounded_query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
        query_scales = pl.create_tensor([tokens, INDEX_H * 4], dtype=pl.FP8E8M0)
        query_ready = quantize_index_query(index_query, rounded_query, query_scales, num_tokens, cache_ready)
        index_weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
        weights_tid = project_index_weights(x, index_weights_proj, index_weights, num_tokens)

        score_completion = pl.array.create(1, pl.TASK_ID)
        score_completion[0] = cache_ready
        if not direct_topk:
            cache_rows = pl.tensor.dim(index_cache, 0) * 128
            cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
            scale_flat = pl.reshape(index_cache_scale, [cache_rows, INDEX_DIM // INDEX_CACHE_GROUP])
            query_flat = pl.reshape(rounded_query, [tokens * INDEX_H, INDEX_DIM])
            pages = (positions + INDEX_SCORE_TILE - 1) // INDEX_SCORE_TILE
            # Read packed keys and score in one task, without a decoded-key GM arena.
            score_blocks = num_tokens * pages
            score_workers = pl.min(score_blocks, 64)
            with pl.spmd(score_workers, name_hint="c1a_index_score",
                         deps=[score_completion[0], weights_tid, query_ready], allow_early_resolve=False) as score_tid:
                worker = pl.tile.get_block_idx()
                for block in pl.range(worker, score_blocks, score_workers):
                    token = block // pages
                    page = block % pages
                    logical_begin = page * INDEX_SCORE_TILE
                    empty_score = pl.tile.full([1, INDEX_SCORE_TILE], dtype=pl.FP32, value=-1e30)
                    pl.store(empty_score, [token, logical_begin], scores)
                    visible = pl.read(compressed_lens, [token])
                    if logical_begin < visible:
                        request = pl.read(request_ids, [token])
                        logical_page = logical_begin // INDEX_PAGE
                        physical_block_i32 = pl.read(index_block_table, [request, logical_page])
                        if physical_block_i32 >= 0:
                            physical_block = pl.cast(physical_block_i32, pl.INDEX)
                            physical_row = physical_block * INDEX_PAGE + logical_begin % INDEX_PAGE
                            payload_bytes = pl.load(
                                cache_flat,
                                [physical_row, 0],
                                [INDEX_SCORE_TILE, INDEX_DIM // 2],
                            )
                            payload_signed = pl.reinterpret_view(payload_bytes, pl.INT8)
                            payload_i32 = pl.ands(pl.cast(payload_signed, pl.INT32), 255)
                            low = pl.ands(payload_i32, 15)
                            high = pl.ands(pl.shrs(payload_i32, 4), 15)
                            low = pl.reshape(low, [1, INDEX_PACKED_ELEMENTS])
                            high = pl.reshape(high, [1, INDEX_PACKED_ELEMENTS])
                            combined_codes = pl.concat(low, high)
                            output_ids = pl.tile.arange(
                                0,
                                [1, INDEX_SCORE_TILE * INDEX_DIM],
                                dtype=pl.INT32,
                            )
                            pair_ids = pl.shrs(output_ids, 1)
                            parity = pl.ands(output_ids, 1)
                            code_indices = pl.add(pair_ids, pl.mul(parity, INDEX_PACKED_ELEMENTS))
                            code_tmp = pl.create_tile([1, INDEX_SCORE_TILE * INDEX_DIM], dtype=pl.INT32)
                            payload_codes = pl.tile.gather(combined_codes, code_indices, code_tmp)
                            payload_codes = pl.reshape(
                                payload_codes,
                                [INDEX_SCORE_TILE, INDEX_DIM],
                            )
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
                            payload_fp32 = pl.reshape(
                                decoded_payload,
                                [INDEX_SCORE_TILE * (INDEX_DIM // INDEX_CACHE_GROUP), INDEX_CACHE_GROUP],
                            )
                            scale_rows = pl.load(
                                scale_flat,
                                [physical_row, 0],
                                [INDEX_SCORE_TILE, 32],
                                valid_shape=[INDEX_SCORE_TILE, INDEX_DIM // INDEX_CACHE_GROUP],
                            )
                            # Pad the encoded bytes before decoding the E8M0 scales.
                            signed_scale_codes_reinterpret_view = pl.reinterpret_view(scale_rows, pl.UINT8)
                            signed_scale_codes = pl.reinterpret_view(signed_scale_codes_reinterpret_view, pl.INT8)
                            signed_scale_codes_fillpad = pl.tile.fillpad(signed_scale_codes, pad_value=pl.PadValue.zero)
                            signed_scale_codes = pl.tile.set_validshape(signed_scale_codes_fillpad, INDEX_SCORE_TILE, 32)
                            signed_codes = pl.cast(signed_scale_codes, pl.INT32)
                            codes = pl.ands(signed_codes, 255)
                            scale_bits = pl.maximum(pl.shls(codes, 23), 4194304)
                            scale_values = pl.reinterpret_view(scale_bits, pl.FP32)
                            scale_flattened = pl.reshape(scale_values, [1, INDEX_SCORE_TILE * 32])
                            group_ids = pl.tile.arange(
                                0,
                                [1, INDEX_SCORE_TILE * (INDEX_DIM // INDEX_CACHE_GROUP)],
                                dtype=pl.INT32,
                            )
                            group_rows = pl.cast(
                                pl.div(pl.cast(group_ids, pl.FP32), INDEX_DIM // INDEX_CACHE_GROUP),
                                pl.INT32,
                                mode="trunc",
                            )
                            group_columns = pl.sub(
                                group_ids,
                                pl.mul(group_rows, INDEX_DIM // INDEX_CACHE_GROUP),
                            )
                            scale_indices = pl.add(
                                pl.mul(group_rows, 32),
                                group_columns,
                            )
                            gather_tmp = pl.create_tile([1, INDEX_SCORE_TILE * 4], dtype=pl.INT32)
                            scale_vector = pl.tile.gather(scale_flattened, scale_indices, gather_tmp)
                            payload_transposed = pl.tile.transpose(payload_fp32, 0, 1)
                            scaled_transposed = pl.col_expand_mul(payload_transposed, scale_vector)
                            key_fp32 = pl.tile.transpose(scaled_transposed, 0, 1)
                            keys = pl.cast(pl.reshape(key_fp32, [INDEX_SCORE_TILE, INDEX_DIM]), pl.BF16, mode="rint")
                            query_row = token * INDEX_H
                            query = pl.load(
                                query_flat, [query_row, 0], [INDEX_H, INDEX_DIM],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            # Materialize ND layout before the A5 vector-to-cube NZ transfer.
                            keys_transposed = pl.tile.transpose(keys, 0, 1)
                            dot = pl.matmul(query, keys_transposed, out_dtype=pl.FP32)
                            head_scores = pl.cast(dot, pl.BF16, mode="rint")
                            head_scores = pl.maximum(pl.cast(head_scores, pl.FP32), 0.0)
                            weights = pl.reshape(
                                pl.load(index_weights, [token, 0], [1, INDEX_H]),
                                [INDEX_H, 1],
                            )
                            weighted = pl.row_expand_mul(head_scores, pl.cast(weights, pl.FP32))
                            weighted = pl.cast(weighted, pl.BF16, mode="rint")
                            reduced = pl.col_sum(pl.cast(weighted, pl.FP32))
                            rounded_score = pl.cast(reduced, pl.BF16, mode="rint")
                            computed_score = pl.reshape(pl.cast(rounded_score, pl.FP32), [1, INDEX_SCORE_TILE])
                            if use_candidates:
                                candidate_u8 = pl.load(
                                    candidate_mask,
                                    [token, logical_begin],
                                    [1, INDEX_SCORE_TILE],
                                )
                                candidate_i8 = pl.reinterpret_view(candidate_u8, pl.INT8)
                                candidate = pl.cast(pl.cast(candidate_i8, pl.INT32), pl.FP32)
                                filtered_score = pl.add(
                                    computed_score,
                                    pl.mul(pl.sub(candidate, 1.0), 1e30),
                                )
                            else:
                                filtered_score = computed_score
                            valid_count = pl.min(INDEX_SCORE_TILE, visible - logical_begin)
                            valid_score = pl.fillpad(
                                pl.set_validshape(filtered_score, 1, valid_count),
                                pad_value=pl.PadValue.min,
                            )
                            stored_score = pl.maximum(
                                valid_score,
                                pl.tile.full([1, INDEX_SCORE_TILE], dtype=pl.FP32, value=-1e30),
                            )
                            pl.store(stored_score, [token, logical_begin], scores)
            score_completion[0] = score_tid

        if direct_topk:
            with pl.spmd(num_tokens, name_hint="c1a_index_topk_direct", deps=[score_completion[0]]) as direct_tid:
                token = pl.tile.get_block_idx()
                request = pl.read(request_ids, [token])
                visible_i32 = pl.read(compressed_lens, [token])
                direct_visible = pl.cast(pl.min(visible_i32, positions), pl.INDEX)
                for lane in pl.range(INDEX_TOPK):
                    physical_row_i32 = pl.cast(-1, pl.INT32)
                    if lane < direct_visible:
                        direct_page = lane // INDEX_PAGE
                        direct_offset = lane % INDEX_PAGE
                        direct_block = pl.read(index_block_table, [request, direct_page])
                        physical_row_i32 = pl.cast(
                            direct_block * INDEX_PAGE + direct_offset,
                            pl.INT32,
                        )
                    pl.write(topk_indices, [token, lane], physical_row_i32)
                direct_scores = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=0.0)
                pl.store(direct_scores, [token, 0], scores)
            score_completion[0] = direct_tid
        else:
            root_rows = tokens * TOPK_GROUPS_PER_TOKEN
            pair_arena = pl.create_tensor(
                [tokens * (TOPK_GROUPS_PER_TOKEN + TOPK_GROUP_LEAVES), TOPK_PAIR_WIDTH],
                dtype=pl.FP32,
            )
            with pl.spmd(
                num_tokens,
                name_hint="c1a_index_topk_group_wave",
                deps=[score_completion[0]],
            ) as group_tid:
                token = pl.tile.get_block_idx()
                group_visible_i32 = pl.read(compressed_lens, [token])
                group_visible = pl.cast(pl.min(group_visible_i32, positions), pl.INDEX)
                work_visible = pl.max(group_visible, 1)
                leaf_count = (work_visible + TOPK_LEAF - 1) // TOPK_LEAF
                group_count = (leaf_count + TOPK_GROUP_LEAVES - 1) // TOPK_GROUP_LEAVES
                scratch_base = root_rows + token * TOPK_GROUP_LEAVES
                for group in pl.range(group_count):
                    leaf_begin = group * TOPK_GROUP_LEAVES
                    group_leaf_count = pl.min(TOPK_GROUP_LEAVES, leaf_count - leaf_begin)
                    root_slot = token * TOPK_GROUPS_PER_TOKEN + group
                    if group_leaf_count == 1:
                        group_logical_begin = leaf_begin * TOPK_LEAF
                        group_valid_count = pl.min(
                            TOPK_LEAF,
                            work_visible - group_logical_begin,
                        )
                        _sort_topk_leaf(
                            scores,
                            pair_arena,
                            token,
                            group_logical_begin,
                            group_valid_count,
                            root_slot,
                        )
                    else:
                        for group_leaf in pl.unroll(TOPK_GROUP_LEAVES):
                            leaf = leaf_begin + group_leaf
                            scratch_logical_begin = leaf * TOPK_LEAF
                            scratch_valid_count = pl.min(
                                TOPK_LEAF,
                                work_visible - scratch_logical_begin,
                            )
                            _sort_topk_leaf(
                                scores,
                                pair_arena,
                                token,
                                scratch_logical_begin,
                                scratch_valid_count,
                                scratch_base + group_leaf,
                            )
                        _merge_topk_pairs(
                            pair_arena,
                            scratch_base,
                            scratch_base + 1,
                            root_slot,
                        )

            with pl.spmd(num_tokens, name_hint="c1a_index_topk_merge", deps=[group_tid]) as merge_tid:
                token = pl.tile.get_block_idx()
                request = pl.read(request_ids, [token])
                merge_visible_i32 = pl.read(compressed_lens, [token])
                merge_visible = pl.cast(pl.min(merge_visible_i32, positions), pl.INDEX)
                if merge_visible <= INDEX_TOPK:
                    short_indices = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
                    for lane in pl.range(INDEX_TOPK):
                        physical_row_i32 = pl.cast(-1, pl.INT32)
                        if lane < merge_visible:
                            direct_page = lane // INDEX_PAGE
                            direct_offset = lane % INDEX_PAGE
                            direct_block = pl.read(index_block_table, [request, direct_page])
                            physical_row_i32 = pl.cast(
                                direct_block * INDEX_PAGE + direct_offset,
                                pl.INT32,
                            )
                        pl.tile.write(short_indices, [0, lane], physical_row_i32)
                    pl.store(short_indices, [token, 0], topk_indices)
                else:
                    leaf_count = (merge_visible + TOPK_LEAF - 1) // TOPK_LEAF
                    count = (leaf_count + TOPK_GROUP_LEAVES - 1) // TOPK_GROUP_LEAVES
                    arena_base = token * TOPK_GROUPS_PER_TOKEN
                    if count > 1:
                        count = _merge_topk_level(pair_arena, arena_base, count)
                    if count > 1:
                        count = _merge_topk_level(pair_arena, arena_base, count)
                    if count > 1:
                        count = _merge_topk_level(pair_arena, arena_base, count)
                    if count > 1:
                        count = _merge_topk_level(pair_arena, arena_base, count)
                    if count > 1:
                        count = _merge_topk_level(pair_arena, arena_base, count)
                    if count > 1:
                        count = _merge_topk_level(pair_arena, arena_base, count)
                    root = pl.load(pair_arena, [arena_base, 0], [1, TOPK_PAIR_WIDTH])
                    root_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
                    cutoff = pl.tile.read(root_scores, [0, INDEX_TOPK - 1])
                    root_ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
                    above_flags = pl.tile.select(pl.tile.cmps(root_scores, cutoff, cmp_type=4), root_ones, 0.0)
                    sums64 = pl.col_sum(pl.reshape(above_flags, [8, 64]))
                    sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
                    above = pl.cast(0, pl.INDEX)
                    for lane in pl.unroll(8):
                        partial = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
                        above = above + pl.cast(partial, pl.INDEX)
                    materialize_index_topk(
                        scores, index_block_table, request_ids, topk_indices,
                        token, merge_visible, cutoff, INDEX_TOPK - above,
                    )

            score_completion[0] = merge_tid
        return score_completion[0]

    if max_logits_bytes < TOPK_LEAF * 4:
        raise ValueError("the score budget must hold at least one Top-K leaf")

    @pl.jit.inline(auto_scope=False)
    def paged_indexer(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0,
        ],
        index_block_table: pl.Tensor,
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[
            [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN,
        ],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        candidate_mask: pl.Tensor,
        topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
    ):
        """Score and select query chunks within the FP32 logits budget."""
        positions = pl.tensor.dim(candidate_mask, 1)
        score_width = (positions + TOPK_LEAF - 1) // TOPK_LEAF * TOPK_LEAF
        completion = pl.array.create(1, pl.TASK_ID)
        completion[0] = cache_ready
        if num_tokens > 0:
            active_tokens = pl.cast(num_tokens, pl.INDEX)
            budget_rows = pl.max(1, max_logits_bytes // (score_width * 4))
            chunk_tokens = pl.min(active_tokens, pl.min(INDEX_QUERY_TILE, budget_rows))
            scores = pl.create_tensor([chunk_tokens, score_width], dtype=pl.FP32)
            for begin in pl.range(0, active_tokens, chunk_tokens):
                with pl.scope():
                    rows = pl.min(chunk_tokens, active_tokens - begin)
                    chunk_x = pl.slice(x, [rows, D], [begin, 0])
                    chunk_latent = pl.slice(query_latent, [rows, Q_LORA], [begin, 0])
                    chunk_requests = pl.slice(request_ids, [rows], [begin])
                    chunk_lens = pl.slice(compressed_lens, [rows], [begin])
                    chunk_cos = pl.slice(rope_cos, [rows, ROPE_DIM // 2], [begin, 0])
                    chunk_sin = pl.slice(rope_sin, [rows, ROPE_DIM // 2], [begin, 0])
                    chunk_candidates = pl.slice(candidate_mask, [rows, positions], [begin, 0])
                    chunk_topk = pl.slice(topk_indices, [rows, INDEX_TOPK], [begin, 0])
                    chunk_scores = pl.slice(scores, [rows, score_width], [0, 0])
                    chunk_count = pl.cast(rows, pl.INT32)
                    topk_tid = paged_indexer_chunk(
                        chunk_x, chunk_latent, chunk_requests, chunk_lens,
                        index_cache, index_cache_scale, index_block_table,
                        chunk_cos, chunk_sin, index_wq_b, index_wq_b_scale,
                        index_weights_proj, chunk_candidates, chunk_scores, chunk_topk,
                        chunk_count, completion[0],
                    )
                    if use_candidates:
                        completion[0] = topk_tid
                    else:
                        candidate_tid = _hierarchical_sparse_indexer(chunk_scores, chunk_lens, chunk_candidates)
                        # Join both score readers before the next chunk overwrites the arena.
                        completion[0] = pl.system.task_dummy(deps=[topk_tid, candidate_tid])
        return completion[0]

    return paged_indexer


@pl.jit.inline(auto_scope=False)
def prefill_c1a_index_key_project(
    source: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    ready = pl.system.task_dummy(deps=[])
    result, completed = bf16_gemm_staged(
        source, weight, output, num_tokens, ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=2147483647, name_hint='prefill_bf16_projection_cube',
    )
    return result


@pl.jit.inline(auto_scope=False)
def prefill_c1a_index_key_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = num_tokens
    workers = pl.min(blocks, 2147483647)
    for worker in pl.spmd(workers, name_hint='attention_rope'):
        for block in pl.range(worker, blocks, workers):
            _index_key_rope_block(x, cos, sin, output, block)
    return output


prefill_c1a_index = make_paged_indexer()


prefill_c1a_reindex = make_paged_indexer(use_candidates=True)


@pl.jit.inline(auto_scope=False)
def prefill_c1a_select(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0,
    ],
    index_block_table: pl.Tensor,
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN,
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    candidate_mask: pl.Tensor,
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    use_candidates: pl.constexpr = True,
):
    if use_candidates:
        ready = prefill_c1a_reindex(x, query_latent, request_ids, compressed_lens, index_cache, index_cache_scale, index_block_table, rope_cos, rope_sin, index_wq_b, index_wq_b_scale, index_weights_proj, candidate_mask, topk_indices, num_tokens, cache_ready)
    else:
        ready = prefill_c1a_index(x, query_latent, request_ids, compressed_lens, index_cache, index_cache_scale, index_block_table, rope_cos, rope_sin, index_wq_b, index_wq_b_scale, index_weights_proj, candidate_mask, topk_indices, num_tokens, cache_ready)
    return ready


IDX_SCALES = INDEX_DIM // 32


IDX_PACKED = INDEX_DIM // 2


FP4_MAX_INV = 1.0 / 6.0


FP4_SCALE_FLOOR = 2.0**-9


E8M0_FLOOR_BITS = 4194304


SCORE_TILE = 64


LEAF = 2048


LEAF_TILES = LEAF // SCORE_TILE


PAIR_WIDTH = 2 * INDEX_TOPK


SORT_FLOOR = -3.0e38


MASK_BIAS = 1.0e30


@pl.jit.inline
def prefill_c2a_publish_index_key(
    keys: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish index keys as packed E2M1 with group-32 E8M0 scales."""
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, IDX_PACKED])
    scale_flat = pl.reshape(scales, [cache_rows, IDX_SCALES])
    with pl.spmd(num_tokens, name_hint="c2a_index_publish", deps=[cache_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(keys, [1, INDEX_DIM * 8], [t, 0], valid_shape=[1, INDEX_DIM])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, INDEX_DIM * 8)
            groups = pl.reshape(pl.cast(source, pl.FP32), [INDEX_DIM * 8 // 32, 32])
            amax = pl.row_max(pl.abs(groups))
            raw = pl.maximum(pl.mul(amax, FP4_MAX_INV), FP4_SCALE_FLOOR)
            exponent = pl.shrs(pl.add(pl.reinterpret_view(raw, pl.INT32), 8388607), 23)
            factor = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            normalized = pl.minimum(pl.maximum(pl.row_expand_div(groups, factor), -6.0), 6.0)
            magnitude = pl.abs(normalized)
            step0 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32), 1), 31)
            step1 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32), 1), 31)
            step2 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32), 1), 31)
            step3 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32), 1), 31)
            step4 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32), 1), 31)
            step5 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32), 1), 31)
            step6 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32), 1), 31)
            index = pl.add(
                pl.add(pl.add(step0, step1), pl.add(step2, step3)),
                pl.add(pl.add(step4, step5), pl.add(step6, 7)),
            )
            sign = pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 31)
            codes = pl.reshape(pl.sub(index, pl.mul(sign, 8)), [1, INDEX_DIM * 8])
            low = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
            high = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            packed = pl.add(low, pl.mul(high, 16))
            signed = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
            flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8), 1, IDX_PACKED
            )
            signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
            encoded = pl.reinterpret_view(
                pl.reinterpret_view(pl.cast(signed_exponent, pl.INT8), pl.UINT8), pl.FP8E8M0
            )
            scale_flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reshape(encoded, [1, INDEX_DIM * 8 // 32]), 1, IDX_SCALES
            )
    return publish_tid


@pl.jit.inline
def prefill_c2a_index_select(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.BF16],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    leaf_scores: pl.Tensor[[T_DYN, LEAF], pl.FP32],
    leaf_rows: pl.Tensor[[T_DYN, LEAF], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    publish_ready: pl.Scalar[pl.TASK_ID],
):
    """Select up to INDEX_TOPK compressed cache rows per token.

    A token that can reach at most INDEX_TOPK positions takes all of them and
    never scores. Longer histories are scored one LEAF-sized block at a time and
    folded into a running Top-K, so the sort width does not grow with context.
    The sort carries physical cache rows, which removes a per-entry block-table
    lookup from the result.
    """
    query_rows = pl.tensor.dim(query, 0) * INDEX_H
    query_flat = pl.reshape(query, [query_rows, INDEX_DIM])
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, IDX_PACKED])
    # Four E8M0 codes per row is below the 32-byte tile row, so the scale pool is
    # read eight cache rows at a time and reshaped back to one row per key.
    scale_wide = pl.reshape(scales, [cache_rows // 8, 32])
    with pl.spmd(num_tokens, name_hint="c2a_index_select", deps=[publish_ready]) as select_tid:
        t = pl.tile.get_block_idx()
        length = pl.read(compressed_lens, [t])
        request = pl.cast(pl.read(token_to_req_indices, [t]), pl.INDEX)
        if length <= INDEX_TOPK:
            for c in pl.range(INDEX_TOPK // SCORE_TILE):
                p0 = c * SCORE_TILE
                # Inactive tiles need no page-table entry.
                if p0 < length:
                    block_id = pl.read(block_table, [request, p0 // 128])
                    ramp = pl.arange(0, [1, SCORE_TILE], dtype=pl.INT32)
                    # Integer scalars broadcast correctly into a tensor op; an FP32 runtime
                    # scalar does not, and a one-element column is below the tile alignment.
                    slack = pl.cast(pl.add(pl.mul(ramp, -1), pl.cast(length - p0, pl.INT32)), pl.FP32)
                    visible = pl.minimum(pl.maximum(slack, 0.0), 1.0)
                    block_base = pl.cast(block_id * 128 + p0 % 128, pl.INT32)
                    physical = pl.cast(pl.add(ramp, block_base), pl.FP32)
                    masked = pl.sub(pl.mul(visible, pl.add(physical, 1.0)), 1.0)
                    topk_indices[t : t + 1, p0 : p0 + SCORE_TILE] = pl.cast(masked, pl.INT32, mode="rint")
                else:
                    topk_indices[t : t + 1, p0 : p0 + SCORE_TILE] = pl.full(
                        [1, SCORE_TILE], dtype=pl.INT32, value=-1
                    )
        else:
            running = pl.tile.full([1, PAIR_WIDTH], dtype=pl.FP32, value=SORT_FLOOR)
            weight_row = pl.reshape(pl.cast(weights[t : t + 1, :], pl.FP32), [INDEX_H, 1])
            for leaf, (running_iter,) in pl.range((length + LEAF - 1) // LEAF, init_values=(running,)):
                leaf_base = leaf * LEAF
                for c in pl.range(LEAF_TILES):
                    p0 = leaf_base + c * SCORE_TILE
                    # The last leaf may extend beyond the visible page table.
                    if p0 < length:
                        block_id = pl.read(block_table, [request, p0 // 128])
                        base = pl.cast(pl.max(block_id, 0), pl.INDEX) * 128 + p0 % 128
                        packed = flat[base : base + SCORE_TILE, :]
                        wide = pl.reinterpret_view(pl.cast(packed, target_type=pl.UINT16), pl.INT16)
                        low = pl.ands(wide, 15)
                        high = pl.ands(pl.shrs(wide, 4), 15)
                        low_sign = pl.cast(pl.shrs(low, 3), pl.FP32)
                        low_exponent = pl.cast(pl.shrs(pl.ands(low, 7), 1), pl.FP32)
                        low_fraction = pl.cast(pl.ands(low, 1), pl.FP32)
                        # Assemble the exact FP32 exponent for E2M1 exponent codes 0..3.
                        low_exponent_i32 = pl.cast(pl.shrs(pl.ands(low, 7), 1), pl.INT32)
                        low_exponent_bits = pl.shls(pl.add(low_exponent_i32, 127), 23)
                        low_pow2 = pl.reinterpret_view(low_exponent_bits, pl.FP32)
                        low_normal = pl.mul(pl.mul(low_pow2, 0.5), pl.add(pl.mul(low_fraction, 0.5), 1.0))
                        low_is_normal = pl.minimum(low_exponent, 1.0)
                        low_magnitude = pl.add(
                            pl.mul(low_is_normal, low_normal),
                            pl.mul(pl.mul(pl.sub(low_is_normal, 1.0), -1.0), pl.mul(low_fraction, 0.5)),
                        )
                        low_value = pl.mul(low_magnitude, pl.mul(pl.sub(pl.mul(low_sign, 2.0), 1.0), -1.0))
                        high_sign = pl.cast(pl.shrs(high, 3), pl.FP32)
                        high_exponent = pl.cast(pl.shrs(pl.ands(high, 7), 1), pl.FP32)
                        high_fraction = pl.cast(pl.ands(high, 1), pl.FP32)
                        # Assemble the exact FP32 exponent for E2M1 exponent codes 0..3.
                        high_exponent_i32 = pl.cast(pl.shrs(pl.ands(high, 7), 1), pl.INT32)
                        high_exponent_bits = pl.shls(pl.add(high_exponent_i32, 127), 23)
                        high_pow2 = pl.reinterpret_view(high_exponent_bits, pl.FP32)
                        high_normal = pl.mul(pl.mul(high_pow2, 0.5), pl.add(pl.mul(high_fraction, 0.5), 1.0))
                        high_is_normal = pl.minimum(high_exponent, 1.0)
                        high_magnitude = pl.add(
                            pl.mul(high_is_normal, high_normal),
                            pl.mul(pl.mul(pl.sub(high_is_normal, 1.0), -1.0), pl.mul(high_fraction, 0.5)),
                        )
                        high_value = pl.mul(high_magnitude, pl.mul(pl.sub(pl.mul(high_sign, 2.0), 1.0), -1.0))
                        scale_tile = pl.slice(scale_wide, [SCORE_TILE // 8, 32], [base // 8, 0])
                        raw_codes = pl.reinterpret_view(scale_tile, pl.UINT8)
                        signed_codes = pl.cast(pl.reinterpret_view(raw_codes, pl.INT8), pl.INT32)
                        codes = pl.ands(signed_codes, 255)
                        factors = pl.reinterpret_view(
                            pl.maximum(pl.shls(codes, 23), E8M0_FLOOR_BITS), pl.FP32
                        )
                        factor_column = pl.reshape(factors, [SCORE_TILE * IDX_SCALES, 1])
                        low_scaled = pl.reshape(
                            pl.row_expand_mul(
                                pl.reshape(low_value, [SCORE_TILE * IDX_SCALES, 16]), factor_column
                            ),
                            [SCORE_TILE, IDX_PACKED],
                        )
                        high_scaled = pl.reshape(
                            pl.row_expand_mul(
                                pl.reshape(high_value, [SCORE_TILE * IDX_SCALES, 16]), factor_column
                            ),
                            [SCORE_TILE, IDX_PACKED],
                        )
                        keys = pl.concat(
                            pl.cast(low_scaled, pl.BF16, mode="rint"),
                            pl.cast(high_scaled, pl.BF16, mode="rint"),
                        )
                        query_tile = query_flat[t * INDEX_H : t * INDEX_H + INDEX_H, :]
                        scored = pl.maximum(pl.matmul(query_tile, keys, b_trans=True), 0.0)
                        row = pl.col_sum(pl.row_expand_mul(scored, weight_row))
                        ramp = pl.arange(0, [1, SCORE_TILE], dtype=pl.INT32)
                        slack = pl.cast(pl.add(pl.mul(ramp, -1), pl.cast(length - p0, pl.INT32)), pl.FP32)
                        visible = pl.minimum(pl.maximum(slack, 0.0), 1.0)
                        row = pl.maximum(pl.add(row, pl.mul(pl.sub(visible, 1.0), MASK_BIAS)), SORT_FLOOR)
                        leaf_scores[t : t + 1, c * SCORE_TILE : c * SCORE_TILE + SCORE_TILE] = row
                        leaf_rows[t : t + 1, c * SCORE_TILE : c * SCORE_TILE + SCORE_TILE] = pl.add(
                            ramp, pl.cast(base, pl.INT32)
                        )
                    else:
                        leaf_scores[t : t + 1, c * SCORE_TILE : c * SCORE_TILE + SCORE_TILE] = pl.full(
                            [1, SCORE_TILE], dtype=pl.FP32, value=SORT_FLOOR
                        )
                        leaf_rows[t : t + 1, c * SCORE_TILE : c * SCORE_TILE + SCORE_TILE] = pl.full(
                            [1, SCORE_TILE], dtype=pl.INT32, value=-1
                        )
                score_row = pl.load(leaf_scores, [t, 0], [1, LEAF], target_memory=pl.Mem.Vec)
                row_ids = pl.load(leaf_rows, [t, 0], [1, LEAF], target_memory=pl.Mem.Vec)
                pairs = pl.sort32(score_row, pl.reinterpret_view(row_ids, pl.UINT32))
                pairs = pl.mrgsort(pairs, block_len=64)
                pairs = pl.mrgsort(pairs, block_len=256)
                pairs = pl.mrgsort(pairs, block_len=1024)
                merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
                merged = pl.tile.mrgsort(
                    running_iter, pl.tile.slice(pairs, [1, PAIR_WIDTH], [0, 0]), tmp=merge_tmp
                )
                running = pl.yield_(pl.tile.slice(merged, [1, PAIR_WIDTH], [0, 0]))
            chosen = pl.tile.gather_mask(
                running, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            ordered = pl.sort32(
                pl.mul(pl.cast(chosen, pl.FP32), -1.0), pl.reinterpret_view(chosen, pl.UINT32)
            )
            ordered = pl.mrgsort(ordered, block_len=64)
            ordered = pl.mrgsort(ordered, block_len=256)
            final = pl.tile.gather_mask(
                pl.tile.slice(ordered, [1, PAIR_WIDTH], [0, 0]),
                mask_pattern=pl.tile.MaskPattern.P1010,
                output_dtype=pl.INT32,
            )
            topk_indices = pl.store(final, [t, 0], topk_indices)
    return select_tid


@pl.jit.inline(auto_scope=False)
def prefill_c2a_project_index_q(
    x: pl.Tensor,
    weight: pl.Tensor,
    scale: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    mxfp8_gemm(
        x, weight, scale, output, num_tokens,
        width=Q_LORA, output_width=INDEX_H * INDEX_DIM, fp32_output=False,
        n_tile=128, name_hint='attention_mx_projection',
    )
    return output


@pl.jit.inline
def prefill_c2a_rotate_index_key(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = num_tokens
    workers = pl.min(blocks, 2147483647)
    for worker in pl.spmd(workers, name_hint="c2a_rope"):
        for block in pl.range(worker, blocks, workers):
            t = block
            base = 0
            output[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)]
            tail = pl.cast(x[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[t : t + 1, :]
            s = pl.mul(sin[t : t + 1, :], 1.0)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            im = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            rotated_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
            output[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM] = rotated_bf16
    return output


@pl.jit.inline
def prefill_c2a_index_key_project(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Stage the index-key projection before its normalization."""
    tokens = pl.tensor.dim(latent, 0)
    projected = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    key_projection_ready = pl.system.task_dummy(deps=[])
    key_projection_result, key_projection_tid = bf16_gemm_staged(
        latent, weight,
        projected,
        num_tokens, key_projection_ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE,
        k_tile=INDEX_PROJ_K_TILE, workers=2147483647, name_hint='prefill_bf16_projection_cube',
    )
    with pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="prefill_c2a_index_norm", deps=[pool_ready]) as norm_tid:
        t0 = pl.tile.get_block_idx() * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.cast(pl.slice(projected, [M_TILE, INDEX_DIM], [t0, 0], valid_shape=[rows, INDEX_DIM]), pl.FP32)
        square = pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / INDEX_DIM)
        inv = pl.rsqrt(pl.add(square, EPS), high_precision=True)
        gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, INDEX_DIM])
        normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, INDEX_DIM)
    return norm_tid


@pl.jit.inline
def prefill_c2a_index_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Preserve the index-weight rounding boundaries with a staged projection."""
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    weight_projection_ready = pl.system.task_dummy(deps=[])
    weight_projection_result, weight_projection_tid = bf16_gemm_staged(
        x, weight,
        projected,
        num_tokens, weight_projection_ready,
        width=D, output_width=INDEX_H, n_tile=INDEX_WEIGHT_N_TILE,
        k_tile=INDEX_PROJ_K_TILE, workers=2147483647, name_hint='prefill_bf16_projection_cube',
    )
    for block in pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="prefill_c2a_index_weight_scale"):
        t0 = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.cast(pl.slice(projected, [M_TILE, INDEX_H], [t0, 0], valid_shape=[rows, INDEX_H]), pl.FP32)
        scaled = pl.cast(pl.mul(value, INDEX_DIM**-0.5), pl.BF16, mode="rint")
        result = pl.cast(pl.mul(pl.cast(scaled, pl.FP32), INDEX_H**-0.5), pl.BF16, mode="rint")
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(result, rows, INDEX_H)
    return output


@pl.jit.inline
def prefill_c2a_index_query(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = num_tokens * INDEX_H
    workers = pl.min(blocks, 64)
    for worker in pl.spmd(workers, name_hint="c2a_rope"):
        for block in pl.range(worker, blocks, workers):
            t = block // INDEX_H
            h = block % INDEX_H
            base = h * INDEX_DIM
            tail = pl.cast(x[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[t : t + 1, :]
            s = pl.mul(sin[t : t + 1, :], 1.0)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            im = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            rotated_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
            # Preserve the RoPE BF16 rounding before the lossless lane permutation.
            source = pl.concat(x[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)], rotated_bf16)
            source_fp32 = pl.cast(source, pl.FP32)
            even_lanes = pl.gather(source_fp32, mask_pattern=pl.tile.MaskPattern.P0101)
            odd_lanes = pl.gather(source_fp32, mask_pattern=pl.tile.MaskPattern.P1010)
            output[t : t + 1, base : base + INDEX_DIM // 2] = pl.cast(even_lanes, pl.BF16, mode="rint")
            output[t : t + 1, base + INDEX_DIM // 2 : base + INDEX_DIM] = pl.cast(odd_lanes, pl.BF16, mode="rint")
    return output
