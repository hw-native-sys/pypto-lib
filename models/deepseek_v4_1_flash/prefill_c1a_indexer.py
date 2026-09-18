# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged index scoring and top-k selection for packed-prefill C1A."""

import pypto.language as pl

from models.deepseek_v4_1_flash.config import (
    D,
    INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP,
    INDEX_DIM,
    INDEX_H,
    INDEX_TOPK,
    Q_LORA,
    ROPE_DIM,
    T_DYN,
)
from models.deepseek_v4_1_flash.prefill_c1a_common import K_TILE, M_TILE, make_projection, make_rope


INDEX_SCORE_SCALE = INDEX_DIM ** -0.5 * INDEX_H ** -0.5
INDEX_PAGE = 128
INDEX_SCORE_TILE = 64
INDEX_PACKED_ELEMENTS = INDEX_SCORE_TILE * INDEX_DIM // 2
TOPK_LEAF = 8192
TOPK_PAIR_WIDTH = 2 * INDEX_TOPK
TOPK_MAX_LEAVES = (1048576 + TOPK_LEAF - 1) // TOPK_LEAF
TOPK_GROUP_LEAVES = 2
TOPK_GROUPS_PER_TOKEN = (TOPK_MAX_LEAVES + TOPK_GROUP_LEAVES - 1) // TOPK_GROUP_LEAVES


project_index_query = make_projection(Q_LORA, INDEX_H * INDEX_DIM)
rotate_index_query = make_rope(INDEX_H, head_dim=INDEX_DIM, rope_dim=ROPE_DIM)


@pl.jit.inline
def project_index_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project BF16 per-head weights with BF16 rounding before and after scaling."""
    for block in pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="c1a_index_weights"):
        token = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - token)
        accumulator = pl.create_tensor([M_TILE, INDEX_H], dtype=pl.FP32)
        for width_block in pl.range(D // K_TILE):
            offset = width_block * K_TILE
            source = pl.slice(x, [M_TILE, K_TILE], [token, offset], valid_shape=[rows, K_TILE])
            weights = weight[offset:offset + K_TILE, :]
            accumulator = pl.matmul_acc(
                accumulator,
                source,
                weights,
                init_cond=(width_block == 0),
            )
        projected = pl.cast(accumulator, pl.BF16, mode="rint")
        scaled = pl.mul(pl.cast(projected, pl.FP32), INDEX_SCORE_SCALE)
        rounded = pl.cast(scaled, pl.BF16, mode="rint")
        output[token:token + M_TILE, :] = pl.set_validshape(rounded, rows, INDEX_H)
    return output


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


def make_paged_indexer(use_candidates=False, direct_topk=False):
    """Specialize paged FP4 index scoring with optional candidate filtering."""

    @pl.jit.inline(auto_scope=False)
    def paged_indexer(
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
        project_index_query(
            query_latent,
            index_wq_b,
            index_wq_b_scale,
            index_query_raw,
            num_tokens,
        )
        index_query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
        rotate_index_query(
            index_query_raw,
            rope_cos,
            rope_sin,
            index_query,
            num_tokens,
        )
        index_weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
        project_index_weights(x, index_weights_proj, index_weights, num_tokens)

        score_completion = pl.array.create(1, pl.TASK_ID)
        score_completion[0] = cache_ready
        if not direct_topk:
            cache_rows = pl.tensor.dim(index_cache, 0) * 128
            cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
            scale_flat = pl.reshape(index_cache_scale, [cache_rows, INDEX_DIM // INDEX_CACHE_GROUP])
            query_flat = pl.reshape(index_query, [tokens * INDEX_H, INDEX_DIM])
            pages = (positions + INDEX_SCORE_TILE - 1) // INDEX_SCORE_TILE
            decoded_keys = pl.create_tensor(
                [tokens, positions, INDEX_DIM],
                dtype=pl.BF16,
            )
            with pl.spmd(num_tokens * pages, name_hint="c1a_index_decode", deps=[cache_ready]) as decode_tid:
                block = pl.tile.get_block_idx()
                token = block // pages
                page = block % pages
                logical_begin = page * INDEX_SCORE_TILE
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
                        scale_rows = pl.tile.set_validshape(
                            pl.tile.fillpad(scale_rows, pad_value=pl.PadValue.zero),
                            INDEX_SCORE_TILE,
                            32,
                        )
                        raw_codes = pl.reinterpret_view(scale_rows, pl.UINT8)
                        signed_codes = pl.cast(pl.reinterpret_view(raw_codes, pl.INT8), pl.INT32)
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
                        pl.store(keys, [token, logical_begin, 0], decoded_keys)

            with pl.spmd(num_tokens * pages, name_hint="c1a_index_score", deps=[decode_tid]) as score_tid:
                block = pl.tile.get_block_idx()
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
                        key_rows = pl.load(
                            decoded_keys,
                            [token, logical_begin, 0],
                            [1, INDEX_SCORE_TILE, INDEX_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        keys = pl.reshape(key_rows, [INDEX_SCORE_TILE, INDEX_DIM])
                        query_row = token * INDEX_H
                        query = pl.load(query_flat, [query_row, 0], [INDEX_H, INDEX_DIM])
                        dot = pl.matmul(query, pl.tile.transpose_view(keys), out_dtype=pl.FP32)
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
            with pl.spmd(num_tokens, name_hint="c1a_index_topk_direct", deps=[cache_ready]):
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

            with pl.spmd(num_tokens, name_hint="c1a_index_topk_merge", deps=[group_tid]):
                token = pl.tile.get_block_idx()
                request = pl.read(request_ids, [token])
                merge_visible_i32 = pl.read(compressed_lens, [token])
                merge_visible = pl.cast(pl.min(merge_visible_i32, positions), pl.INDEX)
                if merge_visible <= INDEX_TOPK:
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
                        pl.write(topk_indices, [token, lane], physical_row_i32)
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
                    root_indices = pl.tile.gather_mask(
                        root,
                        mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    order_scores = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=-1e30)
                    order_indices = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
                    for lane in pl.range(INDEX_TOPK):
                        score = pl.tile.read(root_scores, [0, lane])
                        if score > -1e29:
                            logical = pl.tile.read(root_indices, [0, lane])
                            pl.tile.write(order_scores, [0, lane], -pl.cast(logical, pl.FP32))
                            pl.tile.write(order_indices, [0, lane], logical)
                    ordered = pl.tile.sort32(order_scores, pl.reinterpret_view(order_indices, pl.UINT32))
                    ordered = pl.tile.mrgsort(ordered, block_len=64)
                    ordered = pl.tile.mrgsort(ordered, block_len=256)
                    logical_indices = pl.tile.gather_mask(
                        ordered,
                        mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    for lane in pl.range(INDEX_TOPK):
                        logical = pl.tile.read(logical_indices, [0, lane])
                        physical_row_i32 = pl.cast(-1, pl.INT32)
                        if logical >= 0:
                            sorted_page = logical // INDEX_PAGE
                            sorted_offset = logical % INDEX_PAGE
                            sorted_block = pl.read(index_block_table, [request, sorted_page])
                            physical_row_i32 = pl.cast(
                                sorted_block * INDEX_PAGE + sorted_offset,
                                pl.INT32,
                            )
                        pl.write(topk_indices, [token, lane], physical_row_i32)

    return paged_indexer

__all__ = ["TOPK_LEAF", "make_paged_indexer"]
