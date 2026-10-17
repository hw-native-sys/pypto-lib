# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""First-level candidate-block selection for the decoder hierarchical sparse indexer."""


import pypto.language as pl
import torch
import torch.nn.functional as F

from config import (
    FLASH as M, CMP_POSITIONS_DYN, INDEX_POOL_BLOCKS_DYN, T_DYN,
    INDEX_CACHE_GROUP, INDEX_BLOCKS_DYN,
)


from qkv_proj_rope import golden_rope_interleave
from quant import dequantize_mxfp4_cache, mxfp8_linear, quantize_mxfp4_cache, golden_mxfp8_linear
from rmsnorm import golden_rms_norm


CANDIDATE_BLOCK_SIZE = M.candidate_block_size
CANDIDATE_TOPK_BLOCKS = M.candidate_topk_blocks
CANDIDATE_PAIR_WIDTH = 2 * CANDIDATE_TOPK_BLOCKS
CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF_TILE = 8192
CANDIDATE_SHORT_LEAF_TILE = 2048
CANDIDATE_MAX_BLOCKS = M.max_position_embeddings // CANDIDATE_BLOCK_SIZE
CANDIDATE_MAX_LEAVES = (CANDIDATE_MAX_BLOCKS + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE
INDEX_SCORE_POSITIONS_DYN = pl.dynamic("V41_INDEX_SCORE_POSITIONS_DYN")


HEAD_DIM = M.head_dim

INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk
TOPK_CACHED_PAGES = (M.max_position_embeddings + 127) // 128


@pl.jit.inline
def _index_key_decode_lut():
    """Map packed FP4 nibbles to exact FP8 carrier bits."""
    codes = pl.tile.arange(0, [1, 16], dtype=pl.INT32)
    magnitude = pl.ands(codes, 7)
    values = pl.add(pl.shls(magnitude, 2), 48)
    values = pl.tile.select(pl.tile.cmps(magnitude, 0, cmp_type=0), 0, values)
    values = pl.tile.select(pl.tile.cmps(magnitude, 1, cmp_type=0), 48, values)
    values = pl.or_(values, pl.shls(pl.ands(codes, 8), 4))
    return pl.cast(values, pl.INT16)


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
            raw_scale = pl.maximum(pl.mul(maximum, 1.0 / 6.0), 2.0**-126)
            bits = pl.reinterpret_view(raw_scale, pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            normalized = pl.reshape(pl.row_expand_div(grouped, scale), [1, 8 * INDEX_CACHE_GROUP])
            padded = pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
            padded[:, :8 * INDEX_CACHE_GROUP] = normalized
            magnitude = pl.minimum(pl.abs(padded), 6.0)
            delta0 = pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32)
            delta1 = pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32)
            delta2 = pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32)
            delta3 = pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32)
            delta4 = pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32)
            delta5 = pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32)
            delta6 = pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32)
            step0 = pl.shrs(pl.sub(delta0, 1), 31)
            step1 = pl.shrs(delta1, 31)
            step2 = pl.shrs(pl.sub(delta2, 1), 31)
            step3 = pl.shrs(delta3, 31)
            step4 = pl.shrs(pl.sub(delta4, 1), 31)
            step5 = pl.shrs(delta5, 31)
            step6 = pl.shrs(pl.sub(delta6, 1), 31)
            lower = pl.add(pl.add(step0, step1), pl.add(step2, step3))
            upper = pl.add(pl.add(step4, step5), pl.add(step6, 7))
            unsigned = pl.add(lower, upper)
            sign = pl.ands(pl.shrs(pl.reinterpret_view(padded, pl.INT32), 28), 8)
            payload_codes = pl.or_(unsigned, sign)
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


def make_index_query_quantizer(*, dequantize=False):
    """Quantize group-32 index queries to E2M1/E8M0, optionally returning BF16 values."""
    heads = M.index_n_heads
    width = M.index_head_dim
    query_width = heads * width
    scale_width = query_width // 32
    output_dtype = pl.BF16 if dequantize else pl.FP8E4M3FN

    @pl.jit.inline(auto_scope=False)
    def quantize_query(
        source: pl.Tensor[[T_DYN, query_width], pl.BF16],
        output: pl.Tensor[[T_DYN, query_width], output_dtype],
        output_scale: pl.Tensor[[T_DYN, scale_width], pl.FP8E8M0],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        # Materialize the producer dependency before entering the scoped task.
        entry_deps = pl.array.create(1, pl.TASK_ID)
        entry_deps[0] = ready
        with pl.spmd(32, name_hint="c1a_index_query_fp4_as_fp8", deps=[entry_deps[0]]) as quant_tid:
            worker = pl.tile.get_block_idx()
            for t in pl.range(worker, num_tokens, 32):
                raw = pl.load(source, [t, 0], [1, query_width])
                values = pl.reshape(pl.cast(raw, pl.FP32), [scale_width, 32])
                reduce_tmp = pl.tile.create([scale_width, 32], dtype=pl.FP32)
                raw_maximum = pl.row_max(pl.abs(values), tmp_tile=reduce_tmp)
                maximum = pl.maximum(raw_maximum, 7.052966104933725e-38)
                bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 6.0), pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, scale), -6.0), 6.0)
                magnitude = pl.abs(normalized)
                rounded = pl.tile.full([scale_width, 32], dtype=pl.FP32, value=0.0)
                select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
                predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
                level = pl.tile.full([scale_width, 32], dtype=pl.FP32, value=0.5)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
                level = pl.tile.full([scale_width, 32], dtype=pl.FP32, value=1.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
                level = pl.tile.full([scale_width, 32], dtype=pl.FP32, value=1.5)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
                level = pl.tile.full([scale_width, 32], dtype=pl.FP32, value=2.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
                level = pl.tile.full([scale_width, 32], dtype=pl.FP32, value=3.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
                level = pl.tile.full([scale_width, 32], dtype=pl.FP32, value=4.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
                level = pl.tile.full([scale_width, 32], dtype=pl.FP32, value=6.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                sign = pl.ands(pl.reinterpret_view(normalized, pl.INT32), -2147483648)
                signed_bits = pl.or_(pl.reinterpret_view(rounded, pl.INT32), sign)
                exact = pl.reinterpret_view(signed_bits, pl.FP32)
                packed_input = pl.reshape(exact, [1, query_width])
                # E2M1 values use exact E4M3 carriers.
                if dequantize:
                    dequantized = pl.row_expand_mul(exact, scale)
                    stored_values = pl.reshape(dequantized, [1, query_width])
                else:
                    stored_values = packed_input
                result = pl.cast(stored_values, output_dtype, mode="rint")
                codes = pl.cast(pl.reshape(exponent, [1, scale_width]), pl.UINT8)
                pl.store(result, [t, 0], output)
                pl.store(pl.reinterpret_view(codes, pl.FP8E8M0), [t, 0], output_scale)
        return quant_tid

    return quantize_query



def make_index_topk_materializer(*, pool=False):
    """Bind logical or compact-pool score columns to physical index-cache rows."""
    @pl.jit.inline
    def materialize(
        scores: pl.Tensor,
        block_table: pl.Tensor,
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        pool_ids: pl.Tensor,
        output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        token: pl.Scalar[pl.INDEX],
        width: pl.Scalar[pl.INDEX],
        cutoff: pl.Scalar[pl.FP32],
        tie_quota: pl.Scalar[pl.INDEX],
    ):
        """Compact winners in logical order and gather physical pages from UB."""
        request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
        table_width = pl.tensor.dim(block_table, 1)
        valid_pages = pl.min(table_width, TOPK_CACHED_PAGES)
        raw_pages = pl.load(block_table, [request, 0], [1, TOPK_CACHED_PAGES], valid_shape=[1, valid_pages])
        pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
        empty_blocks = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=0)
        pool_blocks = pl.tile.fillpad(empty_blocks, pad_value=pl.PadValue.zero)
        if pool:
            pool_width = pl.tensor.dim(pool_ids, 1)
            raw_blocks = pl.load(pool_ids, [token, 0], [1, CANDIDATE_TOPK_BLOCKS], valid_shape=[1, pool_width])
            pool_blocks = pl.tile.fillpad(raw_blocks, pad_value=pl.PadValue.zero)
        pl.store(pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1), [token, 0], output)
        ones = pl.tile.full([1, 256], dtype=pl.FP32, value=1.0)
        lanes = pl.tile.arange(0, [1, 256], dtype=pl.INT32)
        lane_origin = pl.tile.full([1, 256], dtype=pl.INT32, value=256)
        # Unique integer lane keys: selected=256-lane, discarded=0.
        priority_fp32 = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
        tmp = pl.tile.create([1, 256], dtype=pl.UINT8)
        count = pl.cast(0, pl.INDEX)
        remaining = tie_quota
        for start in pl.range(0, width, 256):
            if count < INDEX_TOPK:
                valid = pl.min(256, width - start)
                raw = pl.load(scores, [token, start], [1, 256], valid_shape=[1, valid])
                values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                strict = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones, 0.0)
                ties = pl.tile.full([1, 256], dtype=pl.FP32, value=0.0)
                if remaining > 0:
                    eq = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), ones, 0.0)
                    eq_valid = pl.tile.cmps(lanes, pl.cast(valid, pl.INT32), cmp_type=2)
                    eq = pl.tile.select(eq_valid, eq, 0.0)
                    eq_sums64 = pl.col_sum(pl.reshape(eq, [4, 64]))
                    eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
                    available = pl.cast(0, pl.INDEX)
                    for eq_lane in pl.unroll(8):
                        eq_partial = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
                        available = available + pl.cast(eq_partial, pl.INDEX)
                    take = pl.min(available, remaining)
                    if take > 0:
                        if take == available:
                            ties = eq
                        else:
                            # Only the chunk crossing the tie quota needs this boundary.
                            tie_keys = pl.mul(eq, priority_fp32)
                            tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
                            tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
                            tie_left = pl.tile.slice(tie_pairs, [1, 256], [0, 0])
                            tie_right = pl.tile.slice(tie_pairs, [1, 256], [0, 256])
                            tie_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                            tie_merged = pl.tile.mrgsort(tie_left, tie_right, tmp=tie_tmp)
                            equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                            last_key = pl.cast(pl.tile.read(equal_keys, [0, take - 1]), pl.INT32)
                            boundary = pl.cast(256, pl.INT32) - last_key
                            ties = pl.tile.select(pl.tile.cmps(lanes, boundary, cmp_type=3), eq, 0.0)
                        remaining = remaining - take
                keep = pl.maximum(strict, ties)
                keep_valid = pl.tile.set_validshape(keep, 1, valid)
                keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
                keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [4, 64]))
                keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
                keep_count = pl.cast(0, pl.INDEX)
                for keep_lane in pl.unroll(8):
                    keep_partial = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
                    keep_count = keep_count + pl.cast(keep_partial, pl.INDEX)
                selected_count = pl.min(keep_count, INDEX_TOPK - count)
                if selected_count > 0:
                    order_keys = pl.mul(keep_padded, priority_fp32)
                    order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
                    order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
                    order_left = pl.tile.slice(order_pairs, [1, 256], [0, 0])
                    order_right = pl.tile.slice(order_pairs, [1, 256], [0, 256])
                    order_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                    order_merged = pl.tile.mrgsort(order_left, order_right, tmp=order_tmp)
                    order_keys = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                    offsets_i32 = pl.sub(lane_origin, pl.cast(order_keys, pl.INT32))
                    compact_valid = pl.tile.set_validshape(offsets_i32, 1, selected_count)
                    compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
                    logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
                    if pool:
                        block_indices = pl.tile.shrs(logical, 3)
                        blocks = pl.tile.gather(pool_blocks, block_indices, tmp)
                        logical = pl.add(pl.mul(blocks, CANDIDATE_BLOCK_SIZE), pl.ands(logical, 7))
                    page_indices = pl.tile.shrs(logical, 7)
                    if start < TOPK_CACHED_PAGES * 128:
                        page_ids = pl.tile.gather(pages, page_indices, tmp)
                    else:
                        # A scan chunk spans at most two pages; keep gathers within this bank.
                        first_page = start // 128
                        bank_valid = pl.min(8, table_width - first_page)
                        bank_raw = pl.load(block_table, [request, first_page], [1, 8], valid_shape=[1, bank_valid])
                        bank = pl.tile.fillpad(bank_raw, pad_value=pl.PadValue.zero)
                        local_pages = pl.sub(page_indices, pl.cast(first_page, pl.INT32))
                        page_ids = pl.tile.gather(bank, local_pages, tmp)
                    offsets = pl.sub(logical, pl.mul(page_indices, 128))
                    physical = pl.add(pl.mul(page_ids, 128), offsets)
                    pl.store(pl.tile.set_validshape(physical, 1, selected_count), [token, count], output)
                    count = count + selected_count
        return output

    return materialize


_materialize_full_index_topk = make_index_topk_materializer()


@pl.jit.inline
def materialize_index_topk(
    scores: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    cutoff: pl.Scalar[pl.FP32],
    tie_quota: pl.Scalar[pl.INDEX],
):
    """Compact full-history winners in logical order and gather physical pages."""
    return _materialize_full_index_topk(
        scores, block_table, request_ids, block_table, output, token, width, cutoff, tie_quota,
    )


def golden_index_query_fp4(value: torch.Tensor) -> torch.Tensor:
    """Round group-32 index queries to E2M1 with upward E8M0 scales."""
    payload, scales = quantize_mxfp4_cache(value, 32, "e8m0", round_to_even=True)
    return dequantize_mxfp4_cache(payload, scales, 32, "e8m0").to(value.dtype)


def golden_index_key(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Project an unrotated compressor latent into a rotated index key."""
    projected = torch.matmul(latent.float(), wk.float()).to(latent.dtype)
    key = golden_rms_norm(projected, norm_weight)
    rd = cos.shape[-1] * 2
    return torch.cat((key[..., :-rd], golden_rope_interleave(key[..., -rd:], cos, sin)), dim=-1)


def golden_paged_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    request_ids: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    compressed_lens: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    weights_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    candidates: torch.Tensor | None = None,
    topk: int = M.index_topk,
    *,
    fp4_query: bool = False,
    stable_topk: bool = False,
    sequential_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score request-local paged index keys and return scores plus physical rows."""
    if compressed_lens.ndim != 1 or compressed_lens.shape[0] != x.shape[0]:
        raise ValueError("compressed_lens must contain one causal length per query token")
    max_len = int(compressed_lens.max().item()) if compressed_lens.numel() else 0
    scores = x.new_full((x.shape[0], max_len), -torch.inf, dtype=torch.float32)
    physical = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    if max_len == 0:
        return scores, physical

    positions = torch.arange(max_len, device=x.device)
    request_rows = request_ids.to(torch.long).unsqueeze(-1)
    blocks = torch.div(positions, 128, rounding_mode="floor")
    offsets = positions.remainder(128)
    physical_rows = block_table[request_rows, blocks] * 128 + offsets
    flat_keys = index_cache.flatten(0, 1).squeeze(-2)
    keys = flat_keys[physical_rows.to(torch.long)]

    index_dim = index_cache.shape[-1]
    index_heads = weights_proj.shape[-1]
    if sequential_query and wq_b_scale is not None:
        projected = golden_mxfp8_linear(qr, wq_b, wq_b_scale)
    else:
        projected = mxfp8_linear(qr, wq_b, wq_b_scale)
    q = projected.unflatten(-1, (index_heads, index_dim))
    rd = cos.shape[-1] * 2
    q = torch.cat((q[..., :-rd], golden_rope_interleave(q[..., -rd:], cos, sin)), dim=-1)
    if fp4_query:
        q = golden_index_query_fp4(q)
    weights = torch.matmul(x.float(), weights_proj.float()).to(x.dtype)
    weights = (weights.float() * (index_dim**-0.5 * index_heads**-0.5)).to(x.dtype)
    dots = torch.einsum("thd,tkd->thk", q.float(), keys.float()).to(q.dtype)
    weighted = (dots.float().relu() * weights.float().unsqueeze(-1)).to(q.dtype)
    scores = weighted.float().sum(dim=-2).to(q.dtype).float()
    valid_positions = positions.unsqueeze(0) < compressed_lens.to(torch.long).unsqueeze(-1)
    scores = scores.masked_fill(~valid_positions, -torch.inf)
    if candidates is not None:
        scores = scores.masked_fill(~candidates[..., :max_len].to(torch.bool), -torch.inf)

    count = min(topk, max_len)
    if stable_topk:
        ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
        logical = ranked[:, :count].sort(dim=-1).values
    else:
        logical = scores.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected_scores = scores.gather(-1, logical)
    selected_rows = physical_rows.gather(-1, logical)
    for token in range(x.shape[0]):
        valid_rows = selected_rows[token, torch.isfinite(selected_scores[token])].to(torch.int32)
        physical[token, :valid_rows.numel()] = valid_rows
    return scores, physical


def select_candidate_blocks(
    logits: torch.Tensor,
    compressed_lens: torch.Tensor | int,
    topk_blocks: int = M.candidate_topk_blocks,
    block_size: int = M.candidate_block_size,
) -> torch.Tensor:
    """Return the first-level candidate mask used by late index-source layers."""
    width = logits.shape[-1]
    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last = (torch.as_tensor(compressed_lens, device=logits.device) - 1) // block_size
    while last.ndim < scores.ndim:
        last = last.unsqueeze(-1)
    block_ids = torch.arange(scores.shape[-1], device=logits.device)
    scores = scores.masked_fill(block_ids == last, torch.inf)
    top = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def index_pool_block_capacity(score_width: int) -> int:
    """Return the aligned compact-pool capacity for one Decode score row."""
    if score_width < 0:
        raise ValueError(f"score_width must be nonnegative, got {score_width}")
    blocks = max(1, (score_width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE)
    aligned = (blocks + 31) // 32 * 32
    return min(CANDIDATE_TOPK_BLOCKS, aligned)


def golden_select_candidate_block_pool(
    logits: torch.Tensor,
    compressed_lens: torch.Tensor | int,
    topk_blocks: int = M.candidate_topk_blocks,
    block_size: int = M.candidate_block_size,
    *,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted logical block IDs and counts with a causal tail and fixed selected prefix."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape {tuple(logits.shape)}")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError(f"block_size must be {CANDIDATE_BLOCK_SIZE}, got {block_size}")
    if topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError(f"topk_blocks must be {CANDIDATE_TOPK_BLOCKS}, got {topk_blocks}")

    tokens, width = logits.shape
    # Inactive tensor rows publish canonical empty pool state.
    active_tokens = tokens if num_tokens is None else max(0, min(tokens, int(num_tokens)))
    capacity = index_pool_block_capacity(width)
    ids = torch.full((tokens, capacity), -1, dtype=torch.int32, device=logits.device)
    counts = torch.zeros((tokens,), dtype=torch.int32, device=logits.device)
    if width == 0:
        return ids, counts

    lengths = torch.as_tensor(compressed_lens, device=logits.device, dtype=torch.int64)
    if lengths.ndim == 0:
        lengths = lengths.expand(tokens)
    if lengths.ndim != 1 or lengths.numel() != tokens:
        raise ValueError("compressed_lens must provide one causal length per score row")

    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    for token in range(active_tokens):
        visible = max(0, min(int(lengths[token]), width))
        visible_blocks = (visible + block_size - 1) // block_size
        if visible_blocks == 0:
            continue
        selected_count = min(visible_blocks, topk_blocks, capacity)
        # Rank invalid block maxima at the device sort floor.
        scores = block_scores[token, :visible_blocks].clone().clamp_min(-1e30)
        # Include the newest causal block independently of its score.
        scores[visible_blocks - 1] = 1e30
        # Resolve equal block maxima by stable logical order.
        selected = torch.argsort(scores, descending=True, stable=True)[:selected_count].to(torch.int32)
        ids[token, :selected_count] = selected.sort().values
        counts[token] = selected_count
    return ids, counts


def golden_hierarchical_sparse_indexer(
    index_scores: torch.Tensor,
    compressed_lens: torch.Tensor,
) -> torch.Tensor:
    return select_candidate_blocks(
        index_scores, compressed_lens, M.candidate_topk_blocks, M.candidate_block_size
    )


@pl.jit.inline
def _merge_candidate_pairs(
    arena: pl.Tensor,
    left: pl.Scalar[pl.INDEX],
    right: pl.Scalar[pl.INDEX],
    output: pl.Scalar[pl.INDEX],
):
    left_pairs = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
    right_pairs = pl.load(arena, [right, 0], [1, CANDIDATE_PAIR_WIDTH])
    temporary = pl.tile.create([1, 2 * CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left_pairs, right_pairs, tmp=temporary)
    merged = pl.tile.slice(merged_all, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output, 0], arena)


@pl.jit.inline
def _merge_candidate_level(
    arena: pl.Tensor,
    base: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
):
    output_count = (count + 1) // 2
    for output in pl.range(output_count):
        left = base + 2 * output
        right = left + 1
        if right < base + count:
            _merge_candidate_pairs(arena, left, right, base + output)
        else:
            forwarded = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
            pl.store(forwarded, [base + output, 0], arena)
    return output_count


@pl.jit.inline
def _sort_candidate_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    block_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(scores, [token, block_begin], [1, CANDIDATE_LEAF_TILE], valid_shape=[1, valid_count])
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, CANDIDATE_LEAF_TILE], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    block_begin_i32 = pl.cast(block_begin, pl.INT32)
    indices = pl.add(pl.tile.arange(0, [1, CANDIDATE_LEAF_TILE], dtype=pl.INT32), block_begin_i32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline
def _sort_candidate_short_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(scores, [token, 0], [1, CANDIDATE_SHORT_LEAF_TILE], valid_shape=[1, valid_count])
    values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    values = pl.maximum(values, -1e30)
    indices = pl.tile.arange(0, [1, CANDIDATE_SHORT_LEAF_TILE], dtype=pl.INT32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline(auto_scope=False)
def _hierarchical_sparse_indexer(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
):
    """Write the selected-block mask and return its publication TaskId."""
    tokens = pl.tensor.dim(index_scores, 0)
    positions = pl.tensor.dim(candidate_mask, 1)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    padded_block_count = (block_count + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE * CANDIDATE_LEAF_TILE
    mask_tiles = (positions + 127) // 128
    padded_positions = mask_tiles * 128
    block_scores = pl.create_tensor([tokens, padded_block_count], dtype=pl.FP32)
    aligned_mask = pl.create_tensor([tokens, padded_positions], dtype=pl.UINT8)
    score_tiles = (block_count + CANDIDATE_SCORE_TILE - 1) // CANDIDATE_SCORE_TILE
    with pl.spmd(tokens * score_tiles, name_hint="c1a_candidate_block_scores") as score_tid:
        work = pl.tile.get_block_idx()
        token = work // score_tiles
        score_tile = work % score_tiles
        block_begin = score_tile * CANDIDATE_SCORE_TILE
        position_begin = block_begin * CANDIDATE_BLOCK_SIZE
        valid_blocks = pl.min(CANDIDATE_SCORE_TILE, block_count - block_begin)
        valid_positions = pl.min(CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE, positions - position_begin)
        raw = pl.load(
            index_scores,
            [token, position_begin],
            [1, CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE],
            valid_shape=[1, valid_positions],
        )
        padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
        grouped = pl.reshape(padded, [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE])
        reduce_tmp = pl.create_tile([CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32)
        maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, CANDIDATE_SCORE_TILE])
        visible = pl.read(compressed_lens, [token])
        if visible > 0:
            last_block = (visible - 1) // CANDIDATE_BLOCK_SIZE
            if last_block >= block_begin and last_block < block_begin + valid_blocks:
                pl.tile.write(maxima, [0, last_block - block_begin], 1e30)
        maxima = pl.tile.set_validshape(maxima, 1, valid_blocks)
        pl.store(maxima, [token, block_begin], block_scores)

    mask_work = tokens * mask_tiles
    mask_workers = pl.min(mask_work, 64)
    with pl.spmd(mask_workers, name_hint="c1a_candidate_clear", allow_early_resolve=False) as clear_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, mask_work, mask_workers):
            token = work // mask_tiles
            tile = work % mask_tiles
            position = tile * 128
            empty_i32 = pl.tile.full([1, 128], dtype=pl.INT32, value=0)
            empty = pl.cast(empty_i32, pl.UINT8)
            pl.store(empty, [token, position], aligned_mask)

    arena = pl.create_tensor([tokens * CANDIDATE_MAX_LEAVES, CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(tokens, name_hint="c1a_candidate_topk", deps=[score_tid, clear_tid]) as topk_tid:
        token = pl.tile.get_block_idx()
        visible = pl.read(compressed_lens, [token])
        visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        arena_base = token * CANDIDATE_MAX_LEAVES
        count = 0
        if visible_blocks > 0 and visible_blocks <= CANDIDATE_SHORT_LEAF_TILE:
            _sort_candidate_short_leaf(block_scores, arena, token, visible_blocks, arena_base)
            count = 1
        elif visible_blocks > 0:
            leaf_count = (visible_blocks + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE
            for leaf in pl.range(leaf_count):
                block_begin = leaf * CANDIDATE_LEAF_TILE
                valid_count = pl.min(CANDIDATE_LEAF_TILE, visible_blocks - block_begin)
                _sort_candidate_leaf(block_scores, arena, token, block_begin, valid_count, arena_base + leaf)
            count = leaf_count
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)

        if visible_blocks > 0:
            root = pl.load(arena, [arena_base, 0], [1, CANDIDATE_PAIR_WIDTH])
            selected = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected_count = pl.min(visible_blocks, CANDIDATE_TOPK_BLOCKS)
            for selected_lane in pl.range(selected_count):
                selected_block = pl.tile.read(selected, [0, selected_lane])
                position_begin = selected_block * CANDIDATE_BLOCK_SIZE
                for offset in pl.unroll(CANDIDATE_BLOCK_SIZE):
                    position = position_begin + offset
                    if position < positions:
                        # Each token owns a distinct 128-byte-aligned row.
                        pl.write(aligned_mask, [token, position], pl.cast(1, pl.UINT8))

    with pl.spmd(mask_workers, name_hint="c1a_candidate_publish", deps=[topk_tid],
                 allow_early_resolve=False) as publish_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, mask_work, mask_workers):
            token = work // mask_tiles
            tile = work % mask_tiles
            position = tile * 128
            valid = pl.min(128, positions - position)
            mask = pl.load(aligned_mask, [token, position], [1, 128])
            mask = pl.tile.set_validshape(mask, 1, valid)
            pl.store(mask, [token, position], candidate_mask)
    return publish_tid


@pl.jit.inline(auto_scope=False)
def select_candidate_block_pool(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish aligned sorted logical block IDs and counts for Decode Reindex."""
    tokens = pl.tensor.dim(index_scores, 0)
    positions = pl.tensor.dim(index_scores, 1)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    padded_block_count = (block_count + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE * CANDIDATE_LEAF_TILE
    score_tiles = (block_count + CANDIDATE_SCORE_TILE - 1) // CANDIDATE_SCORE_TILE
    block_scores = pl.create_tensor([tokens, padded_block_count], dtype=pl.FP32)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > tokens:
        active_tokens = pl.cast(tokens, pl.INDEX)
    # Dispatch one guarded work item for an empty batch.
    score_task_tokens = pl.max(active_tokens, pl.cast(1, pl.INDEX))
    score_task_tiles = pl.max(score_tiles, pl.cast(1, pl.INDEX))

    with pl.spmd(
        score_task_tokens * score_task_tiles,
        name_hint="c1a_pool_block_scores",
        deps=[scores_ready],
    ) as score_tid:
        work = pl.tile.get_block_idx()
        token = work // score_task_tiles
        score_tile = work % score_task_tiles
        if token < active_tokens and score_tile < score_tiles:
            block_begin = score_tile * CANDIDATE_SCORE_TILE
            position_begin = block_begin * CANDIDATE_BLOCK_SIZE
            valid_blocks = pl.min(CANDIDATE_SCORE_TILE, block_count - block_begin)
            valid_positions = pl.min(CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE, positions - position_begin)
            raw = pl.load(
                index_scores,
                [token, position_begin],
                [1, CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE],
                valid_shape=[1, valid_positions],
            )
            padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
            grouped = pl.reshape(padded, [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE])
            reduce_tmp = pl.create_tile([CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32)
            maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, CANDIDATE_SCORE_TILE])
            visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
            if visible > 0:
                last_block = (visible - 1) // CANDIDATE_BLOCK_SIZE
                if last_block >= block_begin and last_block < block_begin + valid_blocks:
                    pl.tile.write(maxima, [0, last_block - block_begin], 1e30)
            maxima = pl.tile.set_validshape(maxima, 1, valid_blocks)
            pl.store(maxima, [token, block_begin], block_scores)

    arena = pl.create_tensor([tokens * CANDIDATE_MAX_LEAVES, CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(tokens, name_hint="c1a_pool_block_select", deps=[score_tid]) as pool_tid:
        token = pl.tile.get_block_idx()
        sorted_ids = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
        if token < active_tokens:
            visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
            visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
            arena_base = token * CANDIDATE_MAX_LEAVES
            count = 0
            if visible_blocks > 0 and visible_blocks <= CANDIDATE_SHORT_LEAF_TILE:
                _sort_candidate_short_leaf(block_scores, arena, token, visible_blocks, arena_base)
                count = 1
            elif visible_blocks > 0:
                leaf_count = (visible_blocks + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE
                for leaf in pl.range(leaf_count):
                    block_begin = leaf * CANDIDATE_LEAF_TILE
                    valid_count = pl.min(CANDIDATE_LEAF_TILE, visible_blocks - block_begin)
                    _sort_candidate_leaf(block_scores, arena, token, block_begin, valid_count, arena_base + leaf)
                count = leaf_count
            if count > 1:
                count = _merge_candidate_level(arena, arena_base, count)
            if count > 1:
                count = _merge_candidate_level(arena, arena_base, count)
            if count > 1:
                count = _merge_candidate_level(arena, arena_base, count)
            if count > 1:
                count = _merge_candidate_level(arena, arena_base, count)

            selected_count = pl.min(pl.min(visible_blocks, CANDIDATE_TOPK_BLOCKS), capacity)
            ordered_scores = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.FP32, value=-1e30)
            ordered_indices = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
            if selected_count > 0:
                root = pl.load(arena, [arena_base, 0], [1, CANDIDATE_PAIR_WIDTH])
                selected = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                for lane in pl.range(selected_count):
                    block_id = pl.tile.read(selected, [0, lane])
                    pl.tile.write(ordered_scores, [0, lane], -pl.cast(block_id, pl.FP32))
                    pl.tile.write(ordered_indices, [0, lane], block_id)
            ordered = pl.tile.sort32(ordered_scores, pl.reinterpret_view(ordered_indices, pl.UINT32))
            ordered = pl.tile.mrgsort(ordered, block_len=64)
            ordered = pl.tile.mrgsort(ordered, block_len=256)
            ordered = pl.tile.mrgsort(ordered, block_len=1024)
            sorted_ids = pl.tile.gather_mask(ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        # Publish the sorted prefix and -1 padding as one aligned pool row.
        pl.store(pl.tile.set_validshape(sorted_ids, 1, capacity), [token, 0], pool_block_ids)
    # Publish cache-line-sharing counts from one worker after the ID rows.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c1a_pool_count_publish", deps=[pool_tid]) as counts_tid:
        for publish_token in pl.range(tokens):
            published_count = pl.cast(0, pl.INT32)
            if publish_token < active_tokens:
                publish_visible = pl.min(pl.max(pl.read(compressed_lens, [publish_token]), 0), positions)
                publish_blocks = (publish_visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                published_count = pl.cast(pl.min(pl.min(publish_blocks, CANDIDATE_TOPK_BLOCKS), capacity), pl.INT32)
            pl.write(pool_block_counts, [publish_token], published_count)
    return counts_tid


@pl.jit.inline(auto_scope=False)
def hierarchical_sparse_indexer(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
):
    """Select candidate blocks from scores with optional trailing row padding."""
    _hierarchical_sparse_indexer(index_scores, compressed_lens, candidate_mask)
    return candidate_mask


if __name__ == "__main__":
    width = 2050 * 8
    scores = -torch.arange(width, dtype=torch.float32).reshape(1, width)
    candidate_mask = golden_hierarchical_sparse_indexer(scores, torch.tensor([width]))
    if not bool(candidate_mask[..., -8:].all()) or int(candidate_mask.sum()) > 2048 * 8:
        raise RuntimeError("candidate selector did not pin the newest partial block")
    print(f"[GOLDEN] PASS {golden_hierarchical_sparse_indexer.__name__} output={tuple(candidate_mask.shape)}")
