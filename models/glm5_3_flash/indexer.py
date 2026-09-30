# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared paged kpool selection stages for GLM prefill and decode.

Candidate rows are packed across requests. ``pool_slots[p, :]`` contains four
physical raw-cache slots, or -1 for an incomplete pool. Owners and last logical
positions accompany candidates through scoring and selection. Selected pool IDs
are request-local, so expansion returns logical positions consumed by MLA.

``query_source[t]`` is t for ordinary selection. Speculative rows may point to
an anchor of the same request and complete-pool count (whose source is itself); only anchors
score and sort. Expansion still uses each row's own visible length.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import (
    BLOCK_SIZE,
    INDEX_DIM,
    INDEX_H,
    INDEX_KPOOL,
    INDEX_STATE_WIDTH,
    KPOOL_SELECT_K,
    POOLS_DYN,
    TABLE_DYN,
    TOPK_INDEX_WIDTH,
    T_DYN,
)
from models.glm5_3_flash.quantization import quantize_per_token_int8

LEAF = 2048
PAIR_WIDTH = 2 * KPOOL_SELECT_K
ROTATE_TILE = 16
QUANT_TILE = 8
SCORE_TILE = 32
EXPAND_TILE = 2176
NEG_INF = -3.4028234663852886e38
SCORE_SCALE = INDEX_DIM**-0.5


def make_indexer_hadamard():
    """Return the normalized Sylvester matrix used on both sides of scoring."""
    h = torch.ones(1, 1)
    while h.shape[0] < INDEX_DIM:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / INDEX_DIM**0.5).bfloat16()


def build_indexer_metadata(block_table, kv_lens, request_ids, positions, *, share_mtp=False):
    """Lower paged cache addressing and optional speculative selection anchors.

    ``kv_lens`` includes this dispatch. A partial pool is represented explicitly
    and marked invalid by pooling. At least one dummy candidate is allocated for
    an empty batch of contexts. ``positions`` are zero-based visible query positions.
    """
    if block_table.ndim != 2 or kv_lens.numel() != block_table.shape[0]:
        raise ValueError("block_table must have one row per request length")
    if request_ids.shape != positions.shape or request_ids.ndim != 1:
        raise ValueError("request_ids and positions must be matching vectors")
    if bool((kv_lens < 0).any()):
        raise ValueError("cache lengths must be nonnegative")
    if bool(((request_ids < 0) | (request_ids >= kv_lens.numel())).any()):
        raise ValueError("query request ID is outside the batch")
    if bool(((positions < -1) | (positions >= kv_lens[request_ids.long()])).any()):
        raise ValueError("query position is outside its populated cache")
    if bool((kv_lens > block_table.shape[1] * BLOCK_SIZE).any()):
        raise ValueError("block_table does not cover the visible cache")
    device = block_table.device
    slots, owners, last = [], [], []
    for request, length in enumerate(kv_lens.tolist()):
        for pool in range((length + INDEX_KPOOL - 1) // INDEX_KPOOL):
            row = []
            for lane in range(INDEX_KPOOL):
                position = pool * INDEX_KPOOL + lane
                if position >= length:
                    row.append(-1)
                else:
                    page = int(block_table[request, position // BLOCK_SIZE])
                    if page < 0:
                        raise ValueError("a visible cache position needs a physical page")
                    row.append(page * BLOCK_SIZE + position % BLOCK_SIZE)
            slots.append(row)
            owners.append(request)
            last.append(pool * INDEX_KPOOL + INDEX_KPOOL - 1)
    if not slots:
        slots, owners, last = [[-1] * INDEX_KPOOL], [-1], [-1]
    sources = torch.arange(request_ids.numel(), device=device, dtype=torch.int32)
    if share_mtp:
        for request in torch.unique(request_ids).tolist():
            rows = torch.nonzero(request_ids == request).flatten()
            if rows.numel() > 4:
                raise ValueError("MTP sharing accepts at most four rows per request")
            complete = (positions[rows] + 1) // INDEX_KPOOL
            for count in torch.unique(complete):
                group = rows[complete == count]
                anchor = group[positions[group].argmin()]
                sources[group] = anchor.to(torch.int32)
    lengths = positions.to(torch.int32) + 1
    return {
        "pool_slots": torch.tensor(slots, device=device, dtype=torch.int32),
        "pool_request": torch.tensor(owners, device=device, dtype=torch.int32),
        "pool_last_position": torch.tensor(last, device=device, dtype=torch.int32),
        "query_request": request_ids.to(torch.int32),
        "query_position": positions.to(torch.int32),
        "query_source": sources,
        "pool_count": torch.where(
            sources == torch.arange(sources.numel(), device=device), lengths // INDEX_KPOOL, 0
        ),
        "tail_start": lengths // INDEX_KPOOL * INDEX_KPOOL,
        "tail_count": lengths % INDEX_KPOOL,
        "kv_len": lengths,
    }


def golden_indexer_kpool(packed_states, compress_ape, pool_slots=None):
    """Pool complete groups with a stable per-channel softmax over four tokens."""
    if pool_slots is None:
        rows = packed_states.shape[0]
        count = (rows + INDEX_KPOOL - 1) // INDEX_KPOOL
        pool_slots = torch.arange(count * INDEX_KPOOL, device=packed_states.device).reshape(-1, INDEX_KPOOL)
        pool_slots = pool_slots.masked_fill(pool_slots >= rows, -1)
    valid = (pool_slots >= 0).all(-1)
    keys = torch.zeros(pool_slots.shape[0], INDEX_DIM, dtype=torch.bfloat16, device=packed_states.device)
    if valid.any():
        states = packed_states[pool_slots[valid].long()].float()
        weights = (states[..., INDEX_DIM:] + compress_ape.float()).softmax(dim=1)
        keys[valid] = (weights * states[..., :INDEX_DIM]).sum(1).bfloat16()
    return keys, pool_slots, valid.to(torch.int32)


@pl.jit.inline
def indexer_kpool(
    packed_states: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_keys: pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.BF16],
    pool_valid: pl.Tensor[[POOLS_DYN], pl.INT32],
):
    pools = pl.tensor.dim(pool_keys, 0)
    valid_view = pl.reshape(pool_valid, [pools, 1])
    for p in pl.spmd(pools, name_hint="indexer_kpool"):
        valid = pl.read(pool_slots, [p, 0]) >= 0
        for lane in pl.unroll(1, INDEX_KPOOL):
            valid = valid and pl.read(pool_slots, [p, lane]) >= 0
        result = pl.tile.full([1, INDEX_DIM], dtype=pl.FP32, value=0.0)
        flag = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
        if valid:
            peak = pl.tile.full([1, INDEX_DIM], dtype=pl.FP32, value=NEG_INF)
            for lane in pl.unroll(INDEX_KPOOL):
                slot = pl.cast(pl.read(pool_slots, [p, lane]), pl.INDEX)
                gate = pl.load(packed_states, [slot, INDEX_DIM], [1, INDEX_DIM])
                ape = pl.cast(pl.load(compress_ape, [lane, 0], [1, INDEX_DIM]), pl.FP32)
                peak = pl.maximum(peak, pl.add(gate, ape))
            denom = pl.tile.full([1, INDEX_DIM], dtype=pl.FP32, value=0.0)
            for lane in pl.unroll(INDEX_KPOOL):
                slot = pl.cast(pl.read(pool_slots, [p, lane]), pl.INDEX)
                gate = pl.load(packed_states, [slot, INDEX_DIM], [1, INDEX_DIM])
                ape = pl.cast(pl.load(compress_ape, [lane, 0], [1, INDEX_DIM]), pl.FP32)
                weight = pl.exp(pl.sub(pl.add(gate, ape), peak))
                key = pl.load(packed_states, [slot, 0], [1, INDEX_DIM])
                result = pl.add(result, pl.mul(weight, key))
                denom = pl.add(denom, weight)
            result = pl.div(result, denom)
            flag = pl.tile.full([1, 8], dtype=pl.INT32, value=1)
        pl.store(pl.cast(result, pl.BF16, mode="rint"), [p, 0], pool_keys)
        pl.store(pl.set_validshape(flag, 1, 1), [p, 0], valid_view)
    return pool_keys, pool_valid


def golden_indexer_score(index_q, pool_keys, head_weights, pool_visible, hadamard=None):
    """Reference the a2a3 INT8 score path, including both dequantization scales."""
    if hadamard is None:
        hadamard = make_indexer_hadamard().to(index_q.device)
    q, qs = quantize_per_token_int8(index_q.float() @ hadamard.float())
    k, ks = quantize_per_token_int8(pool_keys.float() @ hadamard.float())
    dots = torch.einsum("thd,pd->thp", q.float(), k.float())
    dots = dots * qs * ks.flatten()[None, None, :] * SCORE_SCALE
    scores = (dots.relu() * head_weights.float().unsqueeze(-1)).sum(1)
    return scores.masked_fill(~pool_visible.bool(), NEG_INF)


@pl.jit.inline
def indexer_score(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    pool_keys: pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    pool_valid: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_last_position: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_position: pl.Tensor[[T_DYN], pl.INT32],
    pool_request: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_request: pl.Tensor[[T_DYN], pl.INT32],
    query_source: pl.Tensor[[T_DYN], pl.INT32],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
):
    tokens = pl.tensor.dim(index_q, 0)
    pools = pl.tensor.dim(pool_keys, 0)
    query_flat = pl.reshape(index_q, [tokens * INDEX_H, INDEX_DIM])
    q_i8 = pl.create_tensor([tokens * INDEX_H, INDEX_DIM], dtype=pl.INT8)
    q_scale = pl.create_tensor([tokens * INDEX_H, 1], dtype=pl.FP32)
    q_count = pl.tensor.dim(query_flat, 0)
    q_padded = (q_count + ROTATE_TILE - 1) // ROTATE_TILE * ROTATE_TILE
    q_rotated = pl.create_tensor([q_padded, INDEX_DIM], dtype=pl.FP32)
    for q_block in pl.spmd(q_padded // ROTATE_TILE, name_hint="indexer_rotate"):
        q_start = q_block * ROTATE_TILE
        q_rows = pl.min(ROTATE_TILE, q_count - q_start)
        q_tile = pl.slice(query_flat, [ROTATE_TILE, INDEX_DIM], [q_start, 0], valid_shape=[q_rows, INDEX_DIM])
        q_rotated[q_start : q_start + ROTATE_TILE, :] = pl.matmul(q_tile, hadamard, out_dtype=pl.FP32)
    for q_block in pl.spmd((q_count + QUANT_TILE - 1) // QUANT_TILE, name_hint="indexer_quantize"):
        q_start = q_block * QUANT_TILE
        q_rows = pl.min(QUANT_TILE, q_count - q_start)
        q_quant_tile = q_rotated[q_start : q_start + QUANT_TILE, :]
        q_maximum = pl.maximum(pl.reshape(pl.row_max(pl.abs(q_quant_tile)), [1, QUANT_TILE]), 1e-4)
        q_multiplier_row = pl.div(pl.full([1, QUANT_TILE], dtype=pl.FP32, value=127.0), q_maximum)
        q_multiplier = pl.reshape(q_multiplier_row, [QUANT_TILE, 1])
        q_scaled = pl.row_expand_mul(q_quant_tile, q_multiplier)
        q_ints = pl.cast(q_scaled, pl.INT32, mode="rint")
        q_halves = pl.cast(q_ints, pl.FP16, mode="round")
        q_i8 = pl.assemble(
            q_i8, pl.set_validshape(pl.cast(q_halves, pl.INT8, mode="trunc"), q_rows, INDEX_DIM), [q_start, 0]
        )
        q_scale = pl.assemble(q_scale, pl.set_validshape(pl.recip(q_multiplier), q_rows, 1), [q_start, 0])
    k_i8 = pl.create_tensor([pools, INDEX_DIM], dtype=pl.INT8)
    k_scale = pl.create_tensor([pools, 1], dtype=pl.FP32)
    k_count = pl.tensor.dim(pool_keys, 0)
    k_padded = (k_count + ROTATE_TILE - 1) // ROTATE_TILE * ROTATE_TILE
    k_rotated = pl.create_tensor([k_padded, INDEX_DIM], dtype=pl.FP32)
    for k_block in pl.spmd(k_padded // ROTATE_TILE, name_hint="indexer_rotate"):
        k_start = k_block * ROTATE_TILE
        k_rows = pl.min(ROTATE_TILE, k_count - k_start)
        k_tile = pl.slice(pool_keys, [ROTATE_TILE, INDEX_DIM], [k_start, 0], valid_shape=[k_rows, INDEX_DIM])
        k_rotated[k_start : k_start + ROTATE_TILE, :] = pl.matmul(k_tile, hadamard, out_dtype=pl.FP32)
    for k_block in pl.spmd((k_count + QUANT_TILE - 1) // QUANT_TILE, name_hint="indexer_quantize"):
        k_start = k_block * QUANT_TILE
        k_rows = pl.min(QUANT_TILE, k_count - k_start)
        k_quant_tile = k_rotated[k_start : k_start + QUANT_TILE, :]
        k_maximum = pl.maximum(pl.reshape(pl.row_max(pl.abs(k_quant_tile)), [1, QUANT_TILE]), 1e-4)
        k_multiplier_row = pl.div(pl.full([1, QUANT_TILE], dtype=pl.FP32, value=127.0), k_maximum)
        k_multiplier = pl.reshape(k_multiplier_row, [QUANT_TILE, 1])
        k_scaled = pl.row_expand_mul(k_quant_tile, k_multiplier)
        k_ints = pl.cast(k_scaled, pl.INT32, mode="rint")
        k_halves = pl.cast(k_ints, pl.FP16, mode="round")
        k_i8 = pl.assemble(
            k_i8, pl.set_validshape(pl.cast(k_halves, pl.INT8, mode="trunc"), k_rows, INDEX_DIM), [k_start, 0]
        )
        k_scale = pl.assemble(k_scale, pl.set_validshape(pl.recip(k_multiplier), k_rows, 1), [k_start, 0])
    # Each score block owns one request query and a bounded candidate tile.
    for task in pl.spmd(tokens * ((pools + SCORE_TILE - 1) // SCORE_TILE), name_hint="indexer_score"):
        t = task // ((pools + SCORE_TILE - 1) // SCORE_TILE)
        p0 = task % ((pools + SCORE_TILE - 1) // SCORE_TILE) * SCORE_TILE
        valid_rows = pl.min(SCORE_TILE, pools - p0)
        result = pl.full([1, SCORE_TILE], dtype=pl.FP32, value=NEG_INF)
        if pl.read(query_source, [t]) == t:
            keys = pl.slice(k_i8, [SCORE_TILE, INDEX_DIM], [p0, 0], valid_shape=[valid_rows, INDEX_DIM])
            query = q_i8[t * INDEX_H : (t + 1) * INDEX_H, :]
            dots = pl.cast(pl.matmul(keys, query, b_trans=True, out_dtype=pl.INT32), pl.FP32)
            qs = pl.reshape(q_scale[t * INDEX_H : (t + 1) * INDEX_H, :], [1, INDEX_H])
            ks = pl.slice(k_scale, [SCORE_TILE, 1], [p0, 0], valid_shape=[valid_rows, 1])
            dots = pl.mul(pl.col_expand_mul(pl.row_expand_mul(dots, ks), qs), SCORE_SCALE)
            scores = pl.reshape(
                pl.row_sum(pl.col_expand_mul(pl.maximum(dots, 0.0), head_weights[t : t + 1, :])),
                [1, SCORE_TILE],
            )
            request = pl.read(query_request, [t])
            position = pl.read(query_position, [t])
            for lane in pl.range(valid_rows):
                p = p0 + lane
                if (
                    pl.read(pool_valid, [p]) != 0
                    and pl.read(pool_request, [p]) == request
                    and pl.read(pool_last_position, [p]) <= position
                ):
                    pl.write(result, [0, lane], pl.read(scores, [0, lane]))
        index_scores = pl.assemble(index_scores, pl.set_validshape(result, 1, valid_rows), [t, p0])
    return index_scores


def golden_indexer_topk(index_scores, pool_count, pool_last_position=None):
    """Select up to 512 finite candidates and return request-local pool IDs."""
    tokens, pools = index_scores.shape
    selected = torch.full((tokens, KPOOL_SELECT_K), -1, dtype=torch.int32, device=index_scores.device)
    valid = torch.zeros_like(selected)
    for t in range(tokens):
        live = torch.nonzero(index_scores[t] > NEG_INF).flatten()
        count = min(KPOOL_SELECT_K, int(pool_count[t]), live.numel())
        chosen = live[index_scores[t, live].argsort(descending=True, stable=True)[:count]]
        if pool_last_position is not None:
            chosen = pool_last_position[chosen] // INDEX_KPOOL
        selected[t, :count] = chosen.to(torch.int32)
        valid[t, :count] = 1
    return selected, valid


@pl.jit.inline
def indexer_topk(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    pool_last_position: pl.Tensor[[POOLS_DYN], pl.INT32],
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
):
    tokens = pl.tensor.dim(index_scores, 0)
    pools = pl.tensor.dim(index_scores, 1)
    scratch = pl.create_tensor([tokens, PAIR_WIDTH], dtype=pl.FP32)
    for t in pl.spmd(tokens, name_hint="indexer_topk"):
        output = pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=-1)
        valid = pl.tile.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=0)
        if pl.read(pool_count, [t]) > 0:
            for leaf in pl.range((pools + LEAF - 1) // LEAF):
                first = leaf * LEAF
                count = pl.min(LEAF, pools - first)
                raw = pl.load(index_scores, [t, first], [1, LEAF], valid_shape=[1, count])
                scores = pl.maximum(pl.fillpad(raw, pad_value=pl.PadValue.min), NEG_INF)
                indices = pl.add(pl.tile.arange(0, [1, LEAF], dtype=pl.INT32), pl.cast(first, pl.INT32))
                pairs = pl.tile.sort32(scores, pl.reinterpret_view(indices, pl.UINT32))
                pairs64 = pl.tile.mrgsort(pairs, block_len=64)
                pairs256 = pl.tile.mrgsort(pairs64, block_len=256)
                pairs1024 = pl.tile.mrgsort(pairs256, block_len=1024)
                best = pl.tile.slice(pairs1024, [1, PAIR_WIDTH], [0, 0])
                if leaf == 0:
                    pl.store(best, [t, 0], scratch)
                else:
                    previous = pl.load(scratch, [t, 0], [1, PAIR_WIDTH])
                    tmp = pl.create_tile([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
                    merged = pl.tile.mrgsort(previous, best, tmp=tmp)
                    pl.store(pl.tile.slice(merged, [1, PAIR_WIDTH], [0, 0]), [t, 0], scratch)
            final_pairs = pl.load(scratch, [t, 0], [1, PAIR_WIDTH])
            final_indices = pl.tile.gather_mask(
                final_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            for lane in pl.range(pl.min(KPOOL_SELECT_K, pl.read(pool_count, [t]))):
                score = pl.tile.read(final_pairs, [0, 2 * lane])
                if score > NEG_INF:
                    candidate = pl.cast(pl.tile.read(final_indices, [0, lane]), pl.INDEX)
                    logical_pool = pl.read(pool_last_position, [candidate]) // INDEX_KPOOL
                    pl.tile.write(output, [0, lane], pl.cast(logical_pool, pl.INT32))
                    pl.tile.write(valid, [0, lane], pl.cast(1, pl.INT32))
        pl.store(output, [t, 0], selected_pools)
        pl.store(valid, [t, 0], selected_valid)
    return selected_pools, selected_valid


def golden_indexer_share_mtp(selected_pools, selected_valid, query_source):
    """Reuse each anchor's selection; ownership/causality are set by metadata."""
    return selected_pools[query_source.long()].clone(), selected_valid[query_source.long()].clone()


@pl.jit.inline
def indexer_share_mtp(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    query_source: pl.Tensor[[T_DYN], pl.INT32],
    shared_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    shared_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
):
    tokens = pl.tensor.dim(selected_pools, 0)
    for t in pl.spmd(tokens, name_hint="indexer_share_mtp"):
        source = pl.cast(pl.read(query_source, [t]), pl.INDEX)
        pl.store(pl.load(selected_pools, [source, 0], [1, KPOOL_SELECT_K]), [t, 0], shared_pools)
        pl.store(pl.load(selected_valid, [source, 0], [1, KPOOL_SELECT_K]), [t, 0], shared_valid)
    return shared_pools, shared_valid


def golden_indexer_expand(selected_pools, pool_valid, tail_start, tail_count, kv_len):
    """Front-pack complete selected groups and each query's incomplete tail."""
    out = torch.full(
        (selected_pools.shape[0], TOPK_INDEX_WIDTH), -1, dtype=torch.int32, device=selected_pools.device
    )
    for t in range(out.shape[0]):
        ids = selected_pools[t][pool_valid[t].bool()].long()
        ids = ids[(ids >= 0) & ((ids + 1) * INDEX_KPOOL <= kv_len[t])]
        positions = (ids[:, None] * INDEX_KPOOL + torch.arange(INDEX_KPOOL, device=ids.device)).flatten()
        start = int(tail_start[t])
        end = min(start + int(tail_count[t]), int(kv_len[t]))
        tail = torch.arange(start, end, device=ids.device)
        live = torch.cat((positions, tail)).to(torch.int32)
        out[t, : live.numel()] = live
    return out


@pl.jit.inline
def indexer_expand(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    pool_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
):
    tokens = pl.tensor.dim(selected_pools, 0)
    for t in pl.spmd(tokens, name_hint="indexer_expand"):
        selected = pl.load(selected_pools, [t, 0], [1, KPOOL_SELECT_K])
        valid = pl.load(pool_valid, [t, 0], [1, KPOOL_SELECT_K])
        out = pl.tile.full([1, EXPAND_TILE], dtype=pl.INT32, value=-1)
        length = pl.read(kv_len, [t])
        cursor = 0
        for lane in pl.range(KPOOL_SELECT_K):
            pool = pl.tile.read(selected, [0, lane])
            if pl.tile.read(valid, [0, lane]) != 0 and pool >= 0 and (pool + 1) * INDEX_KPOOL <= length:
                for offset in pl.unroll(INDEX_KPOOL):
                    pl.tile.write(out, [0, cursor + offset], pl.cast(pool * INDEX_KPOOL + offset, pl.INT32))
                cursor = cursor + INDEX_KPOOL
        start = pl.read(tail_start, [t])
        for lane in pl.range(pl.read(tail_count, [t])):
            if start + lane < length:
                pl.tile.write(out, [0, cursor], pl.cast(start + lane, pl.INT32))
                cursor = cursor + 1
        pl.store(pl.set_validshape(out, 1, TOPK_INDEX_WIDTH), [t, 0], topk_indices)
    return topk_indices


@pl.jit.inline
def indexer_select(
    packed_states: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    pool_last_position: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_position: pl.Tensor[[T_DYN], pl.INT32],
    pool_request: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_request: pl.Tensor[[T_DYN], pl.INT32],
    query_source: pl.Tensor[[T_DYN], pl.INT32],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
):
    """Compose paged pooling, anchor scoring/selection and per-query expansion."""
    tokens = pl.tensor.dim(index_q, 0)
    pools = pl.tensor.dim(pool_slots, 0)
    pool_keys = pl.create_tensor([pools, INDEX_DIM], dtype=pl.BF16)
    pool_valid = pl.create_tensor([pools], dtype=pl.INT32)
    pool_keys, pool_valid = indexer_kpool(packed_states, compress_ape, pool_slots, pool_keys, pool_valid)
    scores = pl.create_tensor([tokens, pools], dtype=pl.FP32)
    scores = indexer_score(
        index_q,
        pool_keys,
        head_weights,
        pool_valid,
        pool_last_position,
        query_position,
        pool_request,
        query_request,
        query_source,
        hadamard,
        scores,
    )
    selected = pl.create_tensor([tokens, KPOOL_SELECT_K], dtype=pl.INT32)
    valid = pl.create_tensor([tokens, KPOOL_SELECT_K], dtype=pl.INT32)
    selected, valid = indexer_topk(scores, pool_count, pool_last_position, selected, valid)
    shared = pl.create_tensor([tokens, KPOOL_SELECT_K], dtype=pl.INT32)
    shared_valid = pl.create_tensor([tokens, KPOOL_SELECT_K], dtype=pl.INT32)
    shared, shared_valid = indexer_share_mtp(selected, valid, query_source, shared, shared_valid)
    topk_indices = indexer_expand(shared, shared_valid, tail_start, tail_count, kv_len, topk_indices)
    return topk_indices


def _check_paged_pool_metadata_and_short_context():
    for length in [0, 1, 2, 3, 4, 7, 8, BLOCK_SIZE + 3]:
        pages = torch.tensor([[2, 0]], dtype=torch.int32)
        meta = build_indexer_metadata(
            pages, torch.tensor([length]), torch.tensor([0]), torch.tensor([length - 1])
        )
        cache = torch.zeros(3 * BLOCK_SIZE, 2 * INDEX_DIM)
        for position in range(length):
            slot = int(pages[0, position // BLOCK_SIZE]) * BLOCK_SIZE + position % BLOCK_SIZE
            cache[slot, :INDEX_DIM] = position
        (keys, slots, valid) = golden_indexer_kpool(cache, torch.zeros(4, INDEX_DIM), meta["pool_slots"])
        assert valid.sum() == length // 4
        for p in range(length // 4):
            torch.testing.assert_close(keys[p].float(), torch.full((INDEX_DIM,), p * 4 + 1.5))
            assert torch.all(slots[p] >= 0)
        selected = torch.full((1, KPOOL_SELECT_K), -1, dtype=torch.int32)
        flags = torch.zeros_like(selected)
        selected[0, : length // 4] = torch.arange(length // 4, dtype=torch.int32)
        flags[0, : length // 4] = 1
        out = golden_indexer_expand(selected, flags, meta["tail_start"], meta["tail_count"], meta["kv_len"])
        assert out[0, :length].tolist() == list(range(length))
        assert torch.all(out[0, length:] == -1)


def _check_pooling_stable_channelwise_gate_and_partial_isolation():
    g = torch.Generator().manual_seed(31)
    cache = torch.randn(8, 2 * INDEX_DIM, generator=g)
    cache[:, INDEX_DIM:] *= 1000
    ape = torch.randn(4, INDEX_DIM, generator=g)
    slots = torch.tensor([[7, 2, 5, 0], [1, 3, -1, -1]])
    (keys, _, valid) = golden_indexer_kpool(cache, ape, slots)
    states = cache[slots[0]]
    logits = states[:, INDEX_DIM:] + ape
    maximum = logits.double().max(0).values
    weights = (logits.double() - maximum).exp()
    expected = (states[:, :INDEX_DIM].double() * weights).sum(0) / weights.sum(0)
    torch.testing.assert_close(keys[0], expected.bfloat16())
    assert valid.tolist() == [1, 0]
    assert torch.count_nonzero(keys[1]) == 0


def _check_score_relu_before_head_reduction_and_visibility():
    query = torch.zeros(1, INDEX_H, INDEX_DIM, dtype=torch.bfloat16)
    query[0, 0, 0] = 1
    query[0, 1, 0] = -1
    keys = torch.zeros(3, INDEX_DIM, dtype=torch.bfloat16)
    keys[:, 0] = 1
    weights = torch.zeros(1, INDEX_H)
    weights[0, :2] = torch.tensor([2.0, 100.0])
    score = golden_indexer_score(
        query, keys, weights, torch.tensor([[True, False, True]]), torch.eye(INDEX_DIM).bfloat16()
    )
    assert abs(float(score[0, 0]) - 2 / INDEX_DIM**0.5) < 1e-06
    assert float(score[0, 1]) == NEG_INF
    torch.testing.assert_close(score[:, 0], score[:, 2])
    zero = golden_indexer_score(query * 0, keys * 0, weights, torch.ones(1, 3, dtype=torch.bool))
    assert torch.equal(zero, torch.zeros_like(zero))


def _check_hadamard_preserves_dot_products_before_quantization():
    h = make_indexer_hadamard().float()
    torch.testing.assert_close(h @ h.T, torch.eye(INDEX_DIM), atol=0.003, rtol=0)


def _check_topk_masks_nonprefix_candidates_and_returns_logical_ids():
    for count in [0, 1, 511, 512, 513, 4099]:
        scores = torch.full((1, max(count * 2, 1)), NEG_INF)
        scores[0, 1 : count * 2 : 2] = torch.arange(count).float()
        last = torch.arange(scores.shape[1], dtype=torch.int32) * 4 + 3
        (selected, valid) = golden_indexer_topk(scores, torch.tensor([count]), last)
        live = min(count, KPOOL_SELECT_K)
        assert int(valid.sum()) == live
        assert selected[0, :live].tolist() == list(range(count * 2 - 1, count * 2 - 2 * live, -2))
        assert torch.all(selected[0, live:] == -1)


def _check_expand_compacts_holes_drops_future_pool_and_fills_maximum_width():
    selected = torch.full((2, KPOOL_SELECT_K), -1, dtype=torch.int32)
    flags = torch.zeros_like(selected)
    selected[0, :4] = torch.tensor([0, -1, 2, 3])
    flags[0, :4] = torch.tensor([1, 0, 1, 1])
    selected[1] = torch.arange(KPOOL_SELECT_K, dtype=torch.int32)
    flags[1] = 1
    out = golden_indexer_expand(
        selected, flags, torch.tensor([12, 2048]), torch.tensor([1, 3]), torch.tensor([13, 2051])
    )
    assert out[0, :9].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12]
    assert torch.all(out[0, 9:] == -1)
    assert out[1].tolist() == list(range(2051))


def _check_mtp_sharing_uses_request_anchor_and_expands_each_tail():
    requests = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    positions = torch.tensor([7, 3, 8, 4, 9, 5, 10, 6])
    meta = build_indexer_metadata(
        torch.tensor([[2], [0]]), torch.tensor([7, 11]), requests, positions, share_mtp=True
    )
    assert meta["query_source"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert meta["pool_count"].tolist() == [2, 1, 0, 0, 0, 0, 0, 0]
    selected = torch.full((8, KPOOL_SELECT_K), -1, dtype=torch.int32)
    valid = torch.zeros_like(selected)
    (selected[0, :2], selected[1, :1]) = (torch.tensor([0, 1]), torch.tensor([0]))
    (valid[0, :2], valid[1, :1]) = (1, 1)
    (shared, flags) = golden_indexer_share_mtp(selected, valid, meta["query_source"])
    out = golden_indexer_expand(shared, flags, meta["tail_start"], meta["tail_count"], meta["kv_len"])
    for row, position in enumerate(positions.tolist()):
        assert out[row, : position + 1].tolist() == list(range(position + 1))
        assert torch.all(out[row, position + 1 :] == -1)


def _check_mtp_reanchors_when_a_pool_becomes_complete():
    meta = build_indexer_metadata(
        torch.tensor([[0]]),
        torch.tensor([6]),
        torch.zeros(4, dtype=torch.int32),
        torch.tensor([2, 3, 4, 5]),
        share_mtp=True,
    )
    assert meta["query_source"].tolist() == [0, 1, 1, 1]
    assert meta["pool_count"].tolist() == [0, 1, 0, 0]


def run_indexer_goldens():
    """Check reference equations and boundary cases on CPU."""
    _check_paged_pool_metadata_and_short_context()
    _check_pooling_stable_channelwise_gate_and_partial_isolation()
    _check_score_relu_before_head_reduction_and_visibility()
    _check_hadamard_preserves_dot_products_before_quantization()
    _check_topk_masks_nonprefix_candidates_and_returns_logical_ids()
    _check_expand_compacts_holes_drops_future_pool_and_fills_maximum_width()
    _check_mtp_sharing_uses_request_anchor_and_expands_each_tail()
    _check_mtp_reanchors_when_a_pool_becomes_complete()
    print("[GOLDEN] PASS indexer boundary checks")


@pl.jit
def indexer_kpool_test(
    packed_states: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_keys: pl.Out[pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.BF16]],
    pool_valid: pl.Out[pl.Tensor[[POOLS_DYN], pl.INT32]],
):
    pool_slots.bind_dynamic(0, POOLS_DYN)
    pool_keys.bind_dynamic(0, POOLS_DYN)
    pool_valid.bind_dynamic(0, POOLS_DYN)
    indexer_kpool(packed_states, compress_ape, pool_slots, pool_keys, pool_valid)
    return pool_keys, pool_valid


@pl.jit
def indexer_score_test(
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    pool_keys: pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    pool_valid: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_last_position: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_position: pl.Tensor[[T_DYN], pl.INT32],
    pool_request: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_request: pl.Tensor[[T_DYN], pl.INT32],
    query_source: pl.Tensor[[T_DYN], pl.INT32],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    index_scores: pl.Out[pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32]],
):
    index_q.bind_dynamic(0, T_DYN)
    pool_keys.bind_dynamic(0, POOLS_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    pool_valid.bind_dynamic(0, POOLS_DYN)
    pool_last_position.bind_dynamic(0, POOLS_DYN)
    query_position.bind_dynamic(0, T_DYN)
    pool_request.bind_dynamic(0, POOLS_DYN)
    query_request.bind_dynamic(0, T_DYN)
    query_source.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(0, T_DYN)
    indexer_score(
        index_q,
        pool_keys,
        head_weights,
        pool_valid,
        pool_last_position,
        query_position,
        pool_request,
        query_request,
        query_source,
        hadamard,
        index_scores,
    )
    return index_scores


@pl.jit
def indexer_topk_test(
    index_scores: pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    pool_last_position: pl.Tensor[[POOLS_DYN], pl.INT32],
    selected_pools: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
):
    index_scores.bind_dynamic(0, T_DYN)
    pool_count.bind_dynamic(0, T_DYN)
    pool_last_position.bind_dynamic(0, POOLS_DYN)
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    indexer_topk(index_scores, pool_count, pool_last_position, selected_pools, selected_valid)
    return selected_pools, selected_valid


@pl.jit
def indexer_share_mtp_test(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    query_source: pl.Tensor[[T_DYN], pl.INT32],
    shared_pools: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    shared_valid: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
):
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    query_source.bind_dynamic(0, T_DYN)
    shared_pools.bind_dynamic(0, T_DYN)
    shared_valid.bind_dynamic(0, T_DYN)
    indexer_share_mtp(selected_pools, selected_valid, query_source, shared_pools, shared_valid)
    return shared_pools, shared_valid


@pl.jit
def indexer_expand_test(
    selected_pools: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    pool_valid: pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32]],
):
    selected_pools.bind_dynamic(0, T_DYN)
    pool_valid.bind_dynamic(0, T_DYN)
    tail_start.bind_dynamic(0, T_DYN)
    tail_count.bind_dynamic(0, T_DYN)
    kv_len.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    indexer_expand(selected_pools, pool_valid, tail_start, tail_count, kv_len, topk_indices)
    return topk_indices


@pl.jit
def indexer_select_test(
    packed_states: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    pool_last_position: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_position: pl.Tensor[[T_DYN], pl.INT32],
    pool_request: pl.Tensor[[POOLS_DYN], pl.INT32],
    query_request: pl.Tensor[[T_DYN], pl.INT32],
    query_source: pl.Tensor[[T_DYN], pl.INT32],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32]],
):
    index_q.bind_dynamic(0, T_DYN)
    pool_slots.bind_dynamic(0, POOLS_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    indexer_select(
        packed_states,
        compress_ape,
        pool_slots,
        index_q,
        head_weights,
        pool_last_position,
        query_position,
        pool_request,
        query_request,
        query_source,
        hadamard,
        pool_count,
        tail_start,
        tail_count,
        kv_len,
        topk_indices,
    )
    return topk_indices


def build_fixture(share_mtp=False, long_topk=False, edge=False):
    """Exercise shuffled pages, unequal requests, partial pools and a sort tail."""
    generator = torch.Generator().manual_seed(127)
    lengths = torch.tensor([3, 19, 9], dtype=torch.int32)
    block_table = torch.tensor([[2], [0], [3]], dtype=torch.int32)
    request = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, 14, 15, 16, 17, 7, 8], dtype=torch.int32)
    if edge:
        lengths = torch.tensor([0, 1, 2, 3, 4, 7, 8, 2051], dtype=torch.int32)
        columns = (int(lengths.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_table = (
            torch.randperm(lengths.numel() * columns, generator=generator)
            .reshape(lengths.numel(), columns)
            .int()
        )
        request = torch.arange(lengths.numel(), dtype=torch.int32)
        positions = lengths - 1
    t = positions.numel()
    data = build_indexer_metadata(block_table, lengths, request, positions, share_mtp=share_mtp)
    p = data["pool_slots"].shape[0]
    data.update(
        packed_states=torch.randn(
            (int(block_table.max()) + 1) * BLOCK_SIZE, INDEX_STATE_WIDTH, generator=generator
        ),
        compress_ape=torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16(),
        index_q=torch.randn(t, INDEX_H, INDEX_DIM, generator=generator).bfloat16(),
        head_weights=torch.randn(t, INDEX_H, generator=generator),
        hadamard=make_indexer_hadamard(),
    )
    data["pool_keys"], _, data["pool_valid"] = golden_indexer_kpool(
        data["packed_states"], data["compress_ape"], data["pool_slots"]
    )
    visible = (
        (data["query_request"][:, None] == data["pool_request"][None, :])
        & (data["pool_last_position"][None, :] <= positions[:, None])
        & data["pool_valid"][None, :].bool()
    )
    visible &= (data["query_source"] == torch.arange(t))[:, None]
    data["index_scores"] = golden_indexer_score(
        data["index_q"], data["pool_keys"], data["head_weights"], visible, data["hadamard"]
    )
    if long_topk:
        p = 4099
        # Unique integer scores keep selection exact despite implementation tie order.
        scores = torch.randperm(p, generator=generator).float()[None, :].repeat(t, 1)
        scores[0] = NEG_INF
        scores[1, 1::2] = NEG_INF
        data["index_scores"] = scores
        data["pool_count"] = torch.full((t,), p, dtype=torch.int32)
        data["pool_last_position"] = torch.arange(p, dtype=torch.int32) * INDEX_KPOOL + INDEX_KPOOL - 1
    data["selected_pools"], data["selected_valid"] = golden_indexer_topk(
        data["index_scores"], data["pool_count"], data["pool_last_position"]
    )
    data["shared_pools"], data["shared_valid"] = golden_indexer_share_mtp(
        data["selected_pools"], data["selected_valid"], data["query_source"]
    )
    data["topk_indices"] = golden_indexer_expand(
        data["shared_pools"], data["shared_valid"], data["tail_start"], data["tail_count"], data["kv_len"]
    )
    return data


def golden_case(name, tensors):
    if name == "select":
        keys, _, valid = golden_indexer_kpool(
            tensors["packed_states"], tensors["compress_ape"], tensors["pool_slots"]
        )
        visible = (
            (tensors["query_request"][:, None] == tensors["pool_request"][None, :])
            & (tensors["pool_last_position"][None, :] <= tensors["query_position"][:, None])
            & valid[None, :].bool()
        )
        visible &= (tensors["query_source"] == torch.arange(tensors["query_source"].numel()))[:, None]
        scores = golden_indexer_score(
            tensors["index_q"], keys, tensors["head_weights"], visible, tensors["hadamard"]
        )
        selected, selected_valid = golden_indexer_topk(
            scores, tensors["pool_count"], tensors["pool_last_position"]
        )
        shared, shared_valid = golden_indexer_share_mtp(selected, selected_valid, tensors["query_source"])
        tensors["topk_indices"][:] = golden_indexer_expand(
            shared, shared_valid, tensors["tail_start"], tensors["tail_count"], tensors["kv_len"]
        )
    elif name == "kpool":
        keys, _, valid = golden_indexer_kpool(
            tensors["packed_states"], tensors["compress_ape"], tensors["pool_slots"]
        )
        tensors["pool_keys"][:] = keys
        tensors["pool_valid"][:] = valid
    elif name == "score":
        visible = (
            (tensors["query_request"][:, None] == tensors["pool_request"][None, :])
            & (tensors["pool_last_position"][None, :] <= tensors["query_position"][:, None])
            & tensors["pool_valid"][None, :].bool()
        )
        visible &= (tensors["query_source"] == torch.arange(tensors["query_source"].numel()))[:, None]
        tensors["index_scores"][:] = golden_indexer_score(
            tensors["index_q"], tensors["pool_keys"], tensors["head_weights"], visible, tensors["hadamard"]
        )
    elif name == "topk":
        tensors["selected_pools"][:], tensors["selected_valid"][:] = golden_indexer_topk(
            tensors["index_scores"], tensors["pool_count"], tensors["pool_last_position"]
        )
    elif name == "share_mtp":
        tensors["shared_pools"][:], tensors["shared_valid"][:] = golden_indexer_share_mtp(
            tensors["selected_pools"], tensors["selected_valid"], tensors["query_source"]
        )
    elif name == "expand":
        tensors["topk_indices"][:] = golden_indexer_expand(
            tensors["selected_pools"],
            tensors["pool_valid"],
            tensors["tail_start"],
            tensors["tail_count"],
            tensors["kv_len"],
        )


CASE_ARGS = {
    "select": (
        "packed_states",
        "compress_ape",
        "pool_slots",
        "index_q",
        "head_weights",
        "pool_last_position",
        "query_position",
        "pool_request",
        "query_request",
        "query_source",
        "hadamard",
        "pool_count",
        "tail_start",
        "tail_count",
        "kv_len",
        "topk_indices",
    ),
    "kpool": ("packed_states", "compress_ape", "pool_slots", "pool_keys", "pool_valid"),
    "score": (
        "index_q",
        "pool_keys",
        "head_weights",
        "pool_valid",
        "pool_last_position",
        "query_position",
        "pool_request",
        "query_request",
        "query_source",
        "hadamard",
        "index_scores",
    ),
    "topk": ("index_scores", "pool_count", "pool_last_position", "selected_pools", "selected_valid"),
    "share_mtp": ("selected_pools", "selected_valid", "query_source", "shared_pools", "shared_valid"),
    "expand": ("selected_pools", "pool_valid", "tail_start", "tail_count", "kv_len", "topk_indices"),
}


def compare_topk(actual, expected, *, inputs, **kwargs):
    """Accept score-equivalent tie choices, requiring unique live IDs and suffix padding."""
    for row in range(actual.shape[0]):
        count = int((expected[row] >= 0).sum())
        chosen = actual[row, :count].long()
        if bool((actual[row, count:] != -1).any()) or chosen.unique().numel() != count:
            return False, f"row {row}: invalid padding or repeated selection"
        scores = inputs["index_scores"][row]
        live = torch.nonzero(scores > NEG_INF).flatten()
        logical = inputs["pool_last_position"][live].long() // INDEX_KPOOL
        mapping = {int(key): float(value) for key, value in zip(logical, scores[live])}
        if any(int(key) not in mapping for key in chosen):
            return False, f"row {row}: selected a masked candidate"
        values = torch.tensor([mapping[int(key)] for key in chosen])
        best = scores[live].topk(count).values
        if not torch.equal(values.sort(descending=True).values, best):
            return False, f"row {row}: selected scores are not the largest {count}"
    return True, ""


def main(argv=None):
    import argparse

    from golden import TensorSpec, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=["all", *CASE_ARGS], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--share-mtp", action="store_true")
    parser.add_argument("--long-topk", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--ties", action="store_true")
    parser.add_argument("--golden-only", action="store_true")
    args = parser.parse_args(argv)
    run_indexer_goldens()
    if args.golden_only:
        return
    for name, names in CASE_ARGS.items():
        if args.case not in ("all", name):
            continue
        data = build_fixture(args.share_mtp, args.long_topk and name == "topk", args.edge)
        if args.ties and name == "topk":
            data["index_scores"].masked_fill_(data["index_scores"] > NEG_INF, 1.0)
        if name == "expand":
            data["selected_pools"], data["pool_valid"] = data["shared_pools"], data["shared_valid"]
        specs = [
            TensorSpec(key, list(data[key].shape), data[key].dtype, init_value=data[key]) for key in names
        ]
        result = run(
            fn=globals()["indexer_" + name + "_test"],
            specs=specs,
            golden_fn=lambda tensors: golden_case(name, tensors),
            config={"platform": args.platform, "device_id": args.device},
            rtol=1e-3 if name == "score" else 1.0 / 128 if name == "kpool" else 0.0,
            atol=1e-4 if name in ("score", "kpool") else 0.0,
            compare_fn={"selected_pools": compare_topk} if name == "topk" else None,
            compile_only=args.compile_only,
        )
        print(f"{name}: {result}")
        if not result.passed:
            raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
