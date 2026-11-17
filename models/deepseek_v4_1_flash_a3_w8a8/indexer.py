# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSA lightning indexer over paged index caches.

All 32 index heads are computed per rank (replicated weights), so scores are
complete without a cross-rank reduction.  The emitted compressed rows are
projected, normalized and rotated into the paged index cache; scoring walks each
request's logical pages through its block table.  Top-K outputs are physical
cache rows, ready for the packed key gather.
"""

import pypto.language as pl
import torch

from compressor import bf16_project
from config import (
    B_DYN, BLOCK_SIZE, FLASH as M, INDEX_BLOCKS_DYN, POOL_BLOCKS_DYN, SCORE_DYN,
    TABLE_DYN, T_DYN,
)
from gemm_w8a8 import w8a8_gemm
from paged_cache import publish_cache_row, snapshot_rows
from rmsnorm import rms_norm
from rope import rope_tail_bf16, rope_tail_fp32


# Dynamic score-buffer width (table_width * BLOCK_SIZE).

# model config
D = M.hidden_size
Q_LORA = M.q_lora_rank
IDX_HEADS = M.index_n_heads
IDX_DIM = M.index_head_dim
TOPK = M.index_topk
ROPE_HALF = M.qk_rope_head_dim // 2
BLOCK = M.candidate_block_size
TOP_BLOCKS = M.candidate_topk_blocks

# tiling
M_TILE = 16
T_TILE = 8
K_TILE = 256
SCORE_WORKERS = 16
SCORE_COLS = 128
LEAF = 512
PAIR_WIDTH = 2 * TOPK
BLOCK_LEAF = 512
ARENA_WIDTH = 2 * TOP_BLOCKS
MASK_FLOOR = 1.0e30


@pl.jit.inline(auto_scope=False)
def index_weights(
    x: pl.Tensor,
    weights_proj: pl.Tensor,
    weights: pl.Tensor,
):
    """Project the per-token, per-index-head score weights into ``weights``."""
    t_dim = pl.tensor.dim(x, 0)
    with pl.spmd((t_dim + M_TILE - 1) // M_TILE, name_hint="index_weights") as weights_tid:
        t0 = pl.tile.get_block_idx() * M_TILE
        rows = pl.min(M_TILE, t_dim - t0)
        acc = pl.create_tensor([M_TILE, IDX_HEADS], dtype=pl.FP32)
        for k0 in pl.pipeline(0, D, K_TILE, stage=2):
            source = pl.slice(x, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
            weight = weights_proj[k0 : k0 + K_TILE, 0:IDX_HEADS]
            acc = pl.matmul_acc(acc, source, weight, init_cond=(k0 == 0))
        rounded = pl.cast(acc, target_type=pl.BF16, mode="rint")
        rounded_fp32 = pl.cast(rounded, target_type=pl.FP32)
        weights[t0 : t0 + M_TILE, 0:IDX_HEADS] = pl.set_validshape(rounded_fp32, rows, IDX_HEADS)
    return weights, weights_tid


@pl.jit.inline(auto_scope=False)
def index_query(
    qr: pl.Tensor,
    qr_scale: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    query: pl.Tensor,
):
    """Project the quantized query latent and rotate all index heads into ``query``."""
    t_dim = pl.tensor.dim(qr, 0)
    projected = pl.create_tensor([t_dim, IDX_HEADS * IDX_DIM], dtype=pl.FP32)
    w8a8_gemm(
        qr, qr_scale, wq_b, wq_b_scale, projected,
        width=Q_LORA, output_width=IDX_HEADS * IDX_DIM, output_fp32=True,
    )
    query = rope_tail_fp32(projected, cos, sin, query, width=IDX_DIM, heads=IDX_HEADS)
    return query


@pl.jit.inline(auto_scope=False)
def index_key(
    latent: pl.Tensor,
    wk: pl.Tensor,
    k_norm: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
):
    """Project and normalize compressed rows, then rotate them into index keys."""
    c_dim = pl.tensor.dim(latent, 0)
    projected = pl.create_tensor([c_dim, IDX_DIM], dtype=pl.BF16)
    bf16_project(latent, wk, projected, width=M.head_dim, output_width=IDX_DIM, output_fp32=False)
    normalized = pl.create_tensor([c_dim, IDX_DIM], dtype=pl.BF16)
    rms_norm(projected, k_norm, normalized, width=IDX_DIM, output_fp32=False, column_tile=IDX_DIM)
    key = pl.create_tensor([c_dim, IDX_DIM], dtype=pl.BF16)
    key = rope_tail_bf16(normalized, cos, sin, key, width=IDX_DIM, heads=1)
    return key


@pl.jit.inline(auto_scope=False)
def index_scores(
    query: pl.Tensor,
    weights: pl.Tensor,
    cache: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    compressed_lens: pl.Tensor,
    scores: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Score every visible compressed page of each request from the index cache."""
    cache_rows = pl.tensor.dim(cache, 0) * BLOCK_SIZE
    flat = pl.reshape(cache, [cache_rows, IDX_DIM])
    tokens = pl.tensor.dim(query, 0)
    query_flat = pl.reshape(query, [tokens * IDX_HEADS, IDX_DIM])
    table_width = pl.tensor.dim(block_table, 1)
    for token in pl.spmd(tokens, name_hint="index_scores"):
        request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
        length = pl.read(compressed_lens, [token])
        pages = (pl.max(length, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE
        for page in pl.range(pages):
            physical = pl.cast(pl.read(block_table, [request, page]), pl.INDEX)
            base = physical * BLOCK_SIZE
            key = pl.slice(flat, [BLOCK_SIZE, IDX_DIM], [base, 0])
            accumulated = pl.full([1, BLOCK_SIZE], dtype=pl.FP32, value=0.0)
            # Static count: unroll so the loop carries no tile across the
            # ForStmt boundary (composes with the inline MoE).
            for head in pl.unroll(IDX_HEADS):
                query_row = pl.slice(query_flat, [1, IDX_DIM], [token * IDX_HEADS + head, 0])
                dots = pl.matmul(query_row, key, b_trans=True, out_dtype=pl.FP32)
                head_scores = pl.maximum(dots, 0.0)
                head_weight = pl.read(weights, [token, head])
                accumulated = pl.add(accumulated, pl.mul(head_scores, head_weight))
            lane_limit = length - pl.cast(page * BLOCK_SIZE, target_type=pl.INT32)
            mask_row = pl.full([1, BLOCK_SIZE], dtype=pl.FP32, value=0.0)
            for lane in pl.range(BLOCK_SIZE):
                if lane < lane_limit:
                    pl.write(mask_row, [0, lane], 1.0)
            masked = pl.mul(accumulated, mask_row)
            scores[token : token + 1, page * BLOCK_SIZE : (page + 1) * BLOCK_SIZE] = masked
        for tail_page in pl.range(pages, table_width):
            zero_page = pl.full([1, BLOCK_SIZE], dtype=pl.FP32, value=0.0)
            scores[token : token + 1, tail_page * BLOCK_SIZE : (tail_page + 1) * BLOCK_SIZE] = zero_page
    return scores


@pl.jit.inline(auto_scope=False)
def filter_candidate_scores(
    scores: pl.Tensor,
    compressed_lens: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    filtered: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Mask logical rows outside the counted source candidate blocks."""
    width = pl.tensor.dim(scores, 1)
    capacity = pl.tensor.dim(pool_ids, 1)
    with pl.spmd(num_tokens, name_hint="index_candidate_filter", deps=[ready]) as filtered_tid:
        token = pl.tile.get_block_idx()
        for begin in pl.range(0, width, BLOCK_SIZE):
            masked = pl.tile.full([1, BLOCK_SIZE], dtype=pl.FP32, value=-MASK_FLOOR)
            pl.store(masked, [token, begin], filtered)
        visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), width)
        count = pl.min(pl.max(pl.read(pool_counts, [token]), 0), capacity)
        for candidate in pl.range(count):
            block = pl.read(pool_ids, [token, candidate])
            begin = block * BLOCK
            if block >= 0 and begin < visible:
                valid = pl.min(BLOCK, visible - begin)
                raw = pl.load(scores, [token, begin], [1, BLOCK], valid_shape=[1, valid])
                row = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                pl.store(row, [token, begin], filtered)
    return filtered, filtered_tid


@pl.jit.inline(auto_scope=False)
def index_topk(
    scores: pl.Tensor,
    compressed_lens: pl.Tensor,
    request_ids: pl.Tensor,
    block_table: pl.Tensor,
    indices: pl.Tensor,
    values: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    masked: pl.constexpr,
):
    """Keep the best 512 visible rows and map them to physical cache rows."""
    t_dim = pl.tensor.dim(scores, 0)
    width = pl.tensor.dim(scores, 1)
    arena = pl.create_tensor([t_dim, PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(pl.min(t_dim, num_tokens), name_hint="index_topk", deps=[ready]) as selected_tid:
        token = pl.tile.get_block_idx()
        pl.store(pl.tile.full([1, TOPK], dtype=pl.INT32, value=-1), [token, 0], indices)
        pl.store(pl.tile.full([1, TOPK], dtype=pl.FP32, value=0.0), [token, 0], values)
        visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), width)
        if visible > 0:
            leaf_count = (visible + LEAF - 1) // LEAF
            for leaf in pl.range(leaf_count):
                begin = leaf * LEAF
                valid = pl.min(LEAF, visible - begin)
                raw = pl.load(scores, [token, begin], [1, LEAF], valid_shape=[1, valid])
                row = pl.maximum(pl.tile.fillpad(raw, pad_value=pl.PadValue.min), -MASK_FLOOR)
                lane_ids = pl.tile.arange(0, [1, LEAF], dtype=pl.INT32)
                lane_ids = pl.add(lane_ids, pl.cast(begin, pl.INT32))
                pairs = pl.tile.sort32(row, pl.reinterpret_view(lane_ids, pl.UINT32))
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
                leaf_pairs = pl.tile.slice(pairs, [1, PAIR_WIDTH], [0, 0])
                if leaf == 0:
                    pl.store(leaf_pairs, [token, 0], arena)
                else:
                    running = pl.load(arena, [token, 0], [1, PAIR_WIDTH])
                    temporary = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
                    merged = pl.tile.mrgsort(running, leaf_pairs, tmp=temporary)
                    pl.store(pl.tile.slice(merged, [1, PAIR_WIDTH], [0, 0]), [token, 0], arena)
            top_pairs = pl.load(arena, [token, 0], [1, PAIR_WIDTH])
            selected = pl.tile.gather_mask(top_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected_values = pl.tile.gather_mask(top_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
            output_indices = pl.tile.full([1, TOPK], dtype=pl.INT32, value=-1)
            output_values = pl.tile.full([1, TOPK], dtype=pl.FP32, value=0.0)
            request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
            for position in pl.range(TOPK):
                if position < visible:
                    logical = pl.tile.read(selected, [0, position])
                    selected_value = pl.tile.read(selected_values, [0, position])
                    if logical >= 0 and (not masked or selected_value > -MASK_FLOOR):
                        page = logical // BLOCK_SIZE
                        offset = logical % BLOCK_SIZE
                        physical = pl.read(block_table, [request, page])
                        if physical >= 0:
                            pl.tile.write(output_indices, [0, position], pl.cast(physical * BLOCK_SIZE + offset, target_type=pl.INT32))
                            pl.tile.write(output_values, [0, position], selected_value)
            pl.store(output_indices, [token, 0], indices)
            pl.store(output_values, [token, 0], values)
    return indices, values, selected_tid


@pl.jit.inline(auto_scope=False)
def select_candidate_block_pool(
    scores: pl.Tensor,
    compressed_lens: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    width: pl.Scalar[pl.INT32],
    rows: pl.Scalar[pl.INT32],
    topk_blocks: pl.constexpr,
    block_size: pl.constexpr,
    ready: pl.Scalar[pl.TASK_ID],
):
    """Publish the highest logical block maxima as the source candidate pool.

    ``width`` is the score width passed in by the caller.  Reading it back with
    ``pl.tensor.dim(scores, 1)`` loses the value when the scores buffer is a
    chain-local ``create_tensor`` inside the composed fwd program (the
    publisher then publishes zero blocks); the caller's score-width scalar is
    a host-derived runtime value and stays valid at every inline depth.
    """
    t_dim = pl.cast(rows, pl.INDEX)
    block_width = (width + block_size - 1) // block_size
    arena = pl.create_tensor([t_dim, ARENA_WIDTH], dtype=pl.FP32)
    with pl.spmd(t_dim, name_hint="index_block_pool", deps=[ready]) as selected_tid:
        token = pl.tile.get_block_idx()
        visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), width)
        visible_blocks = (visible + block_size - 1) // block_size
        leaf_count = (visible_blocks + BLOCK_LEAF - 1) // BLOCK_LEAF
        for leaf in pl.range(leaf_count):
            begin = leaf * BLOCK_LEAF
            valid_blocks = pl.min(BLOCK_LEAF, visible_blocks - begin)
            maxima = pl.tile.full([1, BLOCK_LEAF], dtype=pl.FP32, value=-MASK_FLOOR)
            for block_tile in pl.range(0, valid_blocks, 64):
                count = pl.min(64, valid_blocks - block_tile)
                position = (begin + block_tile) * block_size
                valid_positions = pl.min(count * block_size, pl.max(visible - position, 0))
                raw = pl.load(scores, [token, position], [1, 64 * block_size], valid_shape=[1, valid_positions])
                grouped = pl.reshape(pl.tile.fillpad(raw, pad_value=pl.PadValue.min), [64, block_size])
                reduce_tmp = pl.create_tile([64, block_size], dtype=pl.FP32)
                reduced = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, 64])
                maxima[0 : 1, block_tile : block_tile + 64] = pl.set_validshape(reduced, 1, count)
            if leaf == (visible_blocks - 1) // BLOCK_LEAF:
                pl.tile.write(maxima, [0, visible_blocks - 1 - begin], MASK_FLOOR)
            lane_ids = pl.tile.arange(0, [1, BLOCK_LEAF], dtype=pl.INT32)
            lane_ids = pl.add(lane_ids, pl.cast(begin, pl.INT32))
            pairs = pl.tile.sort32(maxima, pl.reinterpret_view(lane_ids, pl.UINT32))
            pairs = pl.tile.mrgsort(pairs, block_len=64)
            pairs = pl.tile.mrgsort(pairs, block_len=256)
            leaf_pairs = pl.tile.assemble(
                pl.tile.full([1, ARENA_WIDTH], dtype=pl.FP32, value=-MASK_FLOOR),
                pl.tile.slice(pairs, [1, 2 * BLOCK_LEAF], [0, 0]),
                [0, 0],
            )
            if leaf == 0:
                pl.store(leaf_pairs, [token, 0], arena)
            else:
                running = pl.load(arena, [token, 0], [1, ARENA_WIDTH])
                temporary = pl.tile.create([1, 2 * ARENA_WIDTH], dtype=pl.FP32)
                merged = pl.tile.mrgsort(running, leaf_pairs, tmp=temporary)
                pl.store(pl.tile.slice(merged, [1, ARENA_WIDTH], [0, 0]), [token, 0], arena)
        selected_count = pl.min(pl.min(visible_blocks, topk_blocks), block_width)
        selected_ids = pl.tile.full([1, topk_blocks], dtype=pl.INT32, value=-1)
        if selected_count > 0:
            root = pl.load(arena, [token, 0], [1, ARENA_WIDTH])
            selected = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            for position in pl.range(selected_count):
                pl.tile.write(selected_ids, [0, position], pl.tile.read(selected, [0, position]))
        pl.store(selected_ids, [token, 0], pool_ids)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="index_pool_counts", deps=[selected_tid]) as counts_tid:
        for publish_token in pl.range(t_dim):
            publish_visible = pl.min(pl.max(pl.read(compressed_lens, [publish_token]), 0), width)
            publish_blocks = (publish_visible + block_size - 1) // block_size
            published = pl.min(pl.min(publish_blocks, topk_blocks), block_width)
            pl.write(pool_counts, [publish_token], pl.cast(published, pl.INT32))
    return pool_ids, pool_counts, counts_tid


@pl.jit.inline(auto_scope=False)
def lightning_indexer(
    x: pl.Tensor,
    qr: pl.Tensor,
    qr_scale: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    weights_proj: pl.Tensor,
    wk: pl.Tensor,
    k_norm: pl.Tensor,
    latent: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    compressed_cos: pl.Tensor,
    compressed_sin: pl.Tensor,
    compressed_slots: pl.Tensor,
    position_ids: pl.Tensor,
    request_ids: pl.Tensor,
    compressed_lens: pl.Tensor,
    cache: pl.Tensor,
    block_table: pl.Tensor,
    scores: pl.Tensor,
    indices: pl.Tensor,
    values: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    topk_blocks: pl.constexpr,
):
    """Publish index keys, score the paged cache, and select Top-K plus the pool."""
    t_dim = pl.tensor.dim(x, 0)
    query = pl.create_tensor([t_dim, IDX_HEADS * IDX_DIM], dtype=pl.BF16)
    weights = pl.create_tensor([t_dim, IDX_HEADS], dtype=pl.FP32)
    query_out = index_query(qr, qr_scale, wq_b, wq_b_scale, cos, sin, query)
    weights_out, weights_tid = index_weights(x, weights_proj, weights)
    key = index_key(latent, wk, k_norm, compressed_cos, compressed_sin)
    key_snapshot, snapshot_tid = snapshot_rows(key, width=IDX_DIM)
    cache_out, publish_tid = publish_cache_row(
        key_snapshot, compressed_slots, cache, num_tokens, width=IDX_DIM, ready=snapshot_tid,
    )
    scoring_ready = pl.system.task_dummy(deps=[publish_tid, weights_tid])
    scores_out = index_scores(
        query_out, weights_out, cache_out, block_table, request_ids, compressed_lens,
        scores, num_tokens, scoring_ready,
    )
    indices_out, values_out, topk_tid = index_topk(
        scores_out, compressed_lens, request_ids, block_table, indices, values,
        num_tokens, scoring_ready, masked=False,
    )
    pool_ids_out, pool_counts_out, pool_tid = select_candidate_block_pool(
        scores_out, compressed_lens, pool_ids, pool_counts,
        width=pl.tensor.dim(block_table, 1) * BLOCK_SIZE,
        rows=num_tokens,
        topk_blocks=topk_blocks, block_size=BLOCK, ready=scoring_ready,
    )
    completion = pl.system.task_dummy(deps=[topk_tid, pool_tid])
    return scores, completion


# ---------------------------------------------------------------------------
# Torch references
# ---------------------------------------------------------------------------


def _rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved rotary embedding of the trailing lanes covered by cos/sin."""
    half = cos.shape[-1]
    pairs = value[..., -2 * half :].unflatten(-1, (half, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    cosine = cos.float()
    sine = sin.float()
    while cosine.ndim < even.ndim:
        cosine = cosine.unsqueeze(-2)
        sine = sine.unsqueeze(-2)
    real = (even * cosine - odd * sine).to(torch.bfloat16)
    imag = (even * sine + odd * cosine).to(torch.bfloat16)
    rotated = torch.stack([real, imag], dim=-1).flatten(-2)
    return torch.cat([value[..., : value.shape[-1] - 2 * half], rotated], dim=-1)


def _rms(value: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    fp32 = value.float()
    inverse = torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
    return (fp32 * inverse * gamma.float()).to(torch.bfloat16)


def golden_index_keys(tensors) -> torch.Tensor:
    """Project/normalize/rotate the emitted latent rows."""
    latent = tensors["latent"]
    key = (latent.float() @ tensors["wk"].float().t()).to(torch.bfloat16)
    key = _rms(key, tensors["k_norm"])
    return _rope(key.unsqueeze(-2), tensors["compressed_cos"], tensors["compressed_sin"]).squeeze(-2).to(torch.bfloat16)


def _publish(cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    flat = cache.reshape(-1, cache.shape[-1])
    valid = slots >= 0
    flat[slots[valid].to(torch.long)] = rows[valid].to(cache.dtype)
    return cache


def golden_index_scores(tensors):
    """Torch reference for paged index scoring (all 32 heads)."""
    from utils import gather_cache_rows, unpack_nz

    cache = tensors["index_cache"].clone()
    keys = golden_index_keys(tensors)
    _publish(cache, tensors["compressed_slots"], keys)
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = unpack_nz(tensors["wq_b"]).float()
    query = (torch.matmul(qr.to(torch.int32), wq_b.to(torch.int32)).float() * qr_scale * tensors["wq_b_scale"].float().reshape(1, -1))
    query = _rope(query.view(query.shape[0], IDX_HEADS, IDX_DIM), tensors["cos"], tensors["sin"]).to(torch.bfloat16).float()
    weights = (tensors["x"].float() @ tensors["weights_proj"].float()).to(torch.bfloat16).float()
    table = tensors["index_block_table"]
    request_ids = tensors["request_ids"]
    positions = tensors["position_ids"]
    lens = tensors["compressed_lens"]
    scores = torch.zeros(tensors["scores"].shape, dtype=torch.float32)
    for token in range(query.shape[0]):
        length = max(0, min(int(lens[token]), scores.shape[1]))
        if length == 0:
            continue
        logical = torch.arange(length)
        page = torch.div(logical, BLOCK_SIZE, rounding_mode="floor")
        offset = logical % BLOCK_SIZE
        request = int(request_ids[token])
        physical = table[request, page].to(torch.long)
        slots = physical * BLOCK_SIZE + offset
        keys_row = gather_cache_rows(cache, slots).float()
        dots = torch.einsum("hd,cd->hc", query[token], keys_row)
        scores[token, :length] = (torch.relu(dots) * weights[token].unsqueeze(-1)).sum(dim=0)
    tensors["index_cache"][:] = cache
    tensors["scores"][:] = scores
    return scores


def index_topk_compare(values_name: str):
    """Compare paged Top-K indices with exact invalid lanes and zero padding."""
    from golden import topk_pair_compare

    gate = topk_pair_compare(values_name)

    def compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        valid = actual >= 0
        expected_valid = expected >= 0
        if actual.shape != expected.shape or not torch.equal(valid, expected_valid):
            return False, "Top-K valid-lane mask differs from the independent reference"
        if torch.any(actual[~valid] != -1) or torch.any(actual_outputs[values_name][~valid] != 0):
            return False, "Top-K invalid lanes must contain index -1 and value zero"
        values = actual_outputs[values_name].masked_fill(~valid, float("-inf"))
        outputs = {**actual_outputs, values_name: values}
        return gate(
            actual, expected, actual_outputs=outputs, expected_outputs=expected_outputs,
            inputs=inputs, rtol=rtol, atol=atol,
        )

    compare.__name__ = "index_topk_compare"
    return compare


def candidate_pool_fixture(lengths, capacity: int = TOP_BLOCKS):
    """Restricted logical block pools with empty rows and unused valid IDs."""
    ids = torch.full((len(lengths), capacity), -1, dtype=torch.int32)
    counts = torch.zeros(len(lengths), dtype=torch.int32)
    for token, length in enumerate(lengths):
        blocks = (int(length) + BLOCK - 1) // BLOCK
        if blocks == 0:
            continue
        selected = sorted({0, blocks // 2, blocks - 1}, reverse=True)
        count = 0 if token % 4 == 3 else min(len(selected), token % 3 + 1, capacity)
        ids[token, :len(selected)] = torch.tensor(selected, dtype=torch.int32)
        counts[token] = count
    return ids, counts


def golden_reindex_select(tensors, scores):
    """Select physical rows whose logical blocks belong to the source pool."""
    active = int(tensors.get("num_tokens", scores.shape[0]))
    tensors["indices"][:active].fill_(-1)
    tensors["values"][:active].zero_()
    for token in range(active):
        length = max(0, min(int(tensors["compressed_lens"][token]), scores.shape[1]))
        logical = torch.arange(length)
        count = max(0, min(int(tensors["pool_counts"][token]), tensors["pool_ids"].shape[1]))
        pool = tensors["pool_ids"][token, :count].long()
        eligible = logical[torch.isin(logical // BLOCK, pool)]
        order = torch.argsort(scores[token, eligible], descending=True, stable=True)[:TOPK]
        chosen = eligible[order]
        request = int(tensors["request_ids"][token])
        pages = tensors["index_block_table"][request, chosen // BLOCK_SIZE].long()
        tensors["indices"][token, :len(chosen)] = (pages * BLOCK_SIZE + chosen % BLOCK_SIZE).int()
        tensors["values"][token, :len(chosen)] = scores[token, chosen]
    return tensors["indices"], tensors["values"]


def golden_index_select(tensors, scores, topk_blocks: int = TOP_BLOCKS):
    """Torch reference for Top-K (physical rows) and the logical candidate pool."""
    tokens, width = scores.shape
    indices = torch.full((tokens, TOPK), -1, dtype=torch.int32)
    values = torch.zeros((tokens, TOPK), dtype=torch.float32)
    pool_ids = torch.full((tokens, topk_blocks), -1, dtype=torch.int32)
    pool_counts = torch.zeros((tokens,), dtype=torch.int32)
    table = tensors["index_block_table"]
    request_ids = tensors["request_ids"]
    for token in range(tokens):
        visible = max(0, min(int(tensors["compressed_lens"][token]), width))
        if visible == 0:
            continue
        row = scores[token, :visible]
        keep = min(TOPK, visible)
        chosen = torch.argsort(row, descending=True, stable=True)[:keep]
        logical = chosen.to(torch.long)
        page = torch.div(logical, BLOCK_SIZE, rounding_mode="floor")
        offset = logical % BLOCK_SIZE
        request = int(request_ids[token])
        physical = table[request, page].to(torch.long)
        indices[token, :keep] = (physical * BLOCK_SIZE + offset).to(torch.int32)
        values[token, :keep] = row[chosen]
        blocks = (visible + BLOCK - 1) // BLOCK
        padded = torch.nn.functional.pad(row, (0, blocks * BLOCK - visible), value=float("-inf"))
        maxima = padded.view(blocks, BLOCK).max(dim=-1).values.clone()
        maxima[blocks - 1] = MASK_FLOOR
        keep_blocks = min(blocks, topk_blocks)
        selected = torch.argsort(maxima, descending=True, stable=True)[:keep_blocks]
        pool_ids[token, :keep_blocks] = selected.to(torch.int32)
        pool_counts[token] = keep_blocks
    tensors["indices"][:] = indices
    tensors["values"][:] = values
    tensors["pool_ids"][:] = pool_ids
    tensors["pool_counts"][:] = pool_counts
    return indices, values, pool_ids, pool_counts


INDEX_SCORE_REL_L2 = 1e-2


def index_score_compare(
    actual, expected, *,
    actual_outputs=None, expected_outputs=None, inputs=None,
    rtol=None, atol=None, **_kwargs,
):
    """Gate composed-chain index scores on the accepted child arithmetic floor."""
    from golden import ratio_allclose

    ok, detail = ratio_allclose(atol=5e-2, rtol=2e-2, max_error_ratio=5e-2)(
        actual, expected,
        actual_outputs=actual_outputs, expected_outputs=expected_outputs,
        inputs=inputs, rtol=rtol, atol=atol,
    )
    a = actual.float()
    e = expected.float()
    denom = e.norm()
    if denom.item() == 0.0:
        rel_l2 = 0.0 if a.norm().item() == 0.0 else float("inf")
    else:
        rel_l2 = ((a - e).norm() / denom).item()
    ok = ok and rel_l2 <= INDEX_SCORE_REL_L2
    return ok, f"{detail}; rel_l2={rel_l2:.4%}"


def pool_set_compare():
    """Pool IDs carry set semantics; tie ordering within the selected set is legal."""

    def compare(actual, expected, **kwargs):
        selected_actual = [sorted(int(v) for v in row if v >= 0) for row in actual.cpu()]
        selected_expected = [sorted(int(v) for v in row if v >= 0) for row in expected.cpu()]
        if selected_actual != selected_expected:
            return False, "selected block pool sets differ"
        return True, "pool sets match"

    return compare


def staged_pool_compare(ratio: int):
    """Gate the candidate pool with set semantics on the device's own scores."""

    def compare(actual, expected, *, actual_outputs=None, inputs=None, rtol=None, atol=None, **_kwargs):
        scores = None if actual_outputs is None else actual_outputs.get("index_scores")
        if scores is None:
            return pool_set_compare()(actual, expected)
        positions = inputs["position_ids"]
        tokens = scores.shape[0]
        selection = {
            "compressed_lens": ((positions.long() + 1) // ratio).to(torch.int32),
            "indices": torch.full((tokens, TOPK), -1, dtype=torch.int32),
            "values": torch.zeros((tokens, TOPK), dtype=torch.float32),
            "pool_ids": torch.full_like(actual, -1),
            "pool_counts": torch.zeros((tokens,), dtype=torch.int32),
            "index_block_table": inputs["index_block_table"],
            "request_ids": inputs["request_ids"],
        }
        golden_index_select(selection, scores, actual.shape[-1])
        return pool_set_compare()(
            actual.reshape(-1, actual.shape[-1]),
            selection["pool_ids"].reshape(-1, actual.shape[-1]),
            actual_outputs=actual_outputs, expected_outputs={**selection},
            inputs=inputs, rtol=rtol, atol=atol,
        )

    compare.__name__ = "staged_pool_compare"
    return compare


def golden_lightning_indexer(tensors):
    """Torch reference for the paged indexer (scores, Top-K, block pool)."""
    scores = golden_index_scores(tensors)
    golden_index_select(tensors, scores)




def _rope_tables(count: int, start: int = 0, theta: float | None = None):
    half = ROPE_HALF
    theta = M.rope_theta if theta is None else theta
    inv_freq = 1.0 / torch.pow(
        torch.tensor(theta, dtype=torch.float32),
        torch.arange(0, half, dtype=torch.float32) * 2.0 / (2 * half),
    )
    angle = torch.arange(start, start + count, dtype=torch.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
    return torch.cos(angle), torch.sin(angle)


def build_tensor_specs(tokens: int = 8, seed: int = 0, requests=None):
    from golden import TensorSpec
    from utils import build_request_plan, pack_nz, quant_w_per_channel

    def quantized(output_width):
        return pack_nz(
            quant_w_per_channel(torch.randn(output_width, Q_LORA) * 0.05)[0].transpose(0, 1).contiguous()
        )

    def scale_of(output_width):
        return quant_w_per_channel(torch.randn(output_width, Q_LORA) * 0.05)[1]

    if requests is None:
        requests = [(tokens + 2, tokens // 2), (tokens + 5, tokens - tokens // 2)]
    plan = build_request_plan(requests, ratio=1)
    if plan["tokens"] != tokens:
        raise ValueError("request lengths must sum to tokens")
    pages = plan["pages"]
    generator = torch.Generator().manual_seed(seed)
    cos, sin = _rope_tables(max(64, int(plan["position_ids"].max()) + 1))
    compressed_cos = cos[plan["compressed_lens"].long() - 1]
    compressed_sin = sin[plan["compressed_lens"].long() - 1]
    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=lambda: torch.randn(tokens, D) * 0.1),
        TensorSpec("qr", [tokens, Q_LORA], torch.int8, init_value=lambda: torch.randint(-127, 128, (tokens, Q_LORA))),
        TensorSpec("qr_scale", [tokens, 1], torch.float32, init_value=lambda: torch.rand(tokens, 1) * 0.02 + 1e-4),
        TensorSpec("wq_b", [Q_LORA, IDX_HEADS * IDX_DIM], torch.int8, init_value=lambda: quantized(IDX_HEADS * IDX_DIM)),
        TensorSpec("wq_b_scale", [IDX_HEADS * IDX_DIM], torch.float32, init_value=lambda: scale_of(IDX_HEADS * IDX_DIM)),
        TensorSpec("weights_proj", [D, IDX_HEADS], torch.bfloat16, init_value=lambda: torch.randn(D, IDX_HEADS) * 0.02),
        TensorSpec("wk", [IDX_DIM, M.head_dim], torch.bfloat16, init_value=lambda: torch.randn(IDX_DIM, M.head_dim) * 0.02),
        TensorSpec("k_norm", [IDX_DIM], torch.bfloat16, init_value=lambda: torch.randn(IDX_DIM) * 0.05 + 1.0),
        TensorSpec("latent", [tokens, M.head_dim], torch.bfloat16, init_value=lambda: torch.randn(tokens, M.head_dim)),
        TensorSpec("cos", [tokens, ROPE_HALF], torch.float32, init_value=cos[plan["position_ids"].long()]),
        TensorSpec("sin", [tokens, ROPE_HALF], torch.float32, init_value=sin[plan["position_ids"].long()]),
        TensorSpec("compressed_cos", [tokens, ROPE_HALF], torch.float32, init_value=compressed_cos),
        TensorSpec("compressed_sin", [tokens, ROPE_HALF], torch.float32, init_value=compressed_sin),
        TensorSpec("compressed_slots", [tokens], torch.int64, init_value=plan["compressed_slots"]),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=plan["position_ids"]),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=plan["request_ids"]),
        TensorSpec("compressed_lens", [tokens], torch.int32, init_value=plan["compressed_lens"]),
        TensorSpec("index_cache", [pages, BLOCK_SIZE, 1, IDX_DIM], torch.bfloat16,
                   init_value=lambda: torch.randn(pages, BLOCK_SIZE, 1, IDX_DIM, generator=generator) * 0.1),
        TensorSpec("index_block_table", [plan["index_block_table"].shape[0], plan["index_block_table"].shape[1]],
                   torch.int32, init_value=plan["index_block_table"]),
        TensorSpec("scores", [tokens, plan["table_width"] * BLOCK_SIZE], torch.float32),
        TensorSpec("indices", [tokens, TOPK], torch.int32),
        TensorSpec("values", [tokens, TOPK], torch.float32),
        TensorSpec("pool_ids", [tokens, TOP_BLOCKS], torch.int32),
        TensorSpec("pool_counts", [tokens], torch.int32),
    ]


@pl.jit
def lightning_indexer_test(
    x: pl.Tensor[[T_DYN, 5120], pl.BF16],
    qr: pl.Tensor[[T_DYN, 1280], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[1280, 4096], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[4096], pl.FP32],
    weights_proj: pl.Tensor[[5120, 32], pl.BF16],
    wk: pl.Tensor[[128, 512], pl.BF16],
    k_norm: pl.Tensor[[128], pl.BF16],
    latent: pl.Tensor[[T_DYN, 512], pl.BF16],
    cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, 128], pl.BF16]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    scores: pl.Out[pl.Tensor[[T_DYN, SCORE_DYN], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32]],
    values: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32]],
    pool_ids: pl.Out[pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
    pool_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
):
    x.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    latent.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    compressed_cos.bind_dynamic(0, T_DYN)
    compressed_sin.bind_dynamic(0, T_DYN)
    compressed_slots.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    scores.bind_dynamic(0, T_DYN)
    scores.bind_dynamic(1, SCORE_DYN)
    indices.bind_dynamic(0, T_DYN)
    values.bind_dynamic(0, T_DYN)
    pool_ids.bind_dynamic(0, T_DYN)
    pool_ids.bind_dynamic(1, POOL_BLOCKS_DYN)
    pool_counts.bind_dynamic(0, T_DYN)
    lightning_indexer(
        x, qr, qr_scale, wq_b, wq_b_scale, weights_proj, wk, k_norm, latent,
        cos, sin, compressed_cos, compressed_sin, compressed_slots, position_ids,
        request_ids, compressed_lens, index_cache, index_block_table,
        scores, indices, values, pool_ids, pool_counts,
        pl.tensor.dim(x, 0), topk_blocks=TOP_BLOCKS,
    )
    return scores, indices, values, pool_ids, pool_counts




@pl.jit
def reindex_select_test(
    scores: pl.Tensor[[T_DYN, SCORE_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_ids: pl.InOut[pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
    pool_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    indices: pl.InOut[pl.Tensor[[T_DYN, TOPK], pl.INT32]],
    values: pl.InOut[pl.Tensor[[T_DYN, TOPK], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    scores.bind_dynamic(0, T_DYN)
    scores.bind_dynamic(1, SCORE_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    pool_ids.bind_dynamic(0, T_DYN)
    pool_ids.bind_dynamic(1, POOL_BLOCKS_DYN)
    pool_counts.bind_dynamic(0, T_DYN)
    indices.bind_dynamic(0, T_DYN)
    values.bind_dynamic(0, T_DYN)
    filtered = pl.create_tensor([pl.tensor.dim(scores, 0), pl.tensor.dim(scores, 1)], dtype=pl.FP32)
    ready = pl.system.task_dummy(deps=[])
    filtered_out, filtered_tid = filter_candidate_scores(
        scores, compressed_lens, pool_ids, pool_counts, filtered, num_tokens, ready,
    )
    index_topk(
        filtered_out, compressed_lens, request_ids, index_block_table,
        indices, values, num_tokens, filtered_tid, masked=True,
    )
    return indices, values, pool_ids, pool_counts


def build_reindex_specs(active: int = 5):
    """Ordered scores, permuted pages, partial/empty pools and inactive rows."""
    from golden import ScalarSpec, TensorSpec

    scores = torch.arange(1, 257, dtype=torch.float32).repeat(5, 1)
    pools = torch.tensor([[0, 16, 1, -1], [0, 2, 5, 7], [2, 0, 1, -1], [3, 4, 0, 1], [0, 1, 2, 3]])
    return [
        TensorSpec("scores", [5, 256], torch.float32, init_value=scores),
        TensorSpec("compressed_lens", [5], torch.int32, init_value=torch.tensor([133, 256, 17, 128, 0])),
        TensorSpec("request_ids", [5], torch.int32, init_value=torch.tensor([1, 0, 1, 0, 1])),
        TensorSpec("index_block_table", [2, 2], torch.int32, init_value=torch.tensor([[2, 0], [3, 1]])),
        TensorSpec("pool_ids", [5, 4], torch.int32, init_value=pools),
        TensorSpec("pool_counts", [5], torch.int32, init_value=torch.tensor([2, 3, 1, 0, 1])),
        TensorSpec("indices", [5, TOPK], torch.int32, init_value=-777),
        TensorSpec("values", [5, TOPK], torch.float32, init_value=777.0),
        ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True),
    ]


def golden_reindex_test(tensors):
    golden_reindex_select(tensors, tensors["scores"])


@pl.jit
def candidate_pool_test(
    scores: pl.Tensor[[T_DYN, SCORE_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    pool_ids: pl.Out[pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
    pool_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
):
    scores.bind_dynamic(0, T_DYN)
    scores.bind_dynamic(1, SCORE_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    pool_ids.bind_dynamic(0, T_DYN)
    pool_ids.bind_dynamic(1, POOL_BLOCKS_DYN)
    pool_counts.bind_dynamic(0, T_DYN)
    ready = pl.system.task_dummy(deps=[])
    ids, counts, published = select_candidate_block_pool(scores, compressed_lens, pool_ids, pool_counts, pl.tensor.dim(scores, 1), pl.tensor.dim(scores, 0), topk_blocks=TOP_BLOCKS, block_size=BLOCK, ready=ready)
    return ids, counts


def build_pool_specs(blocks):
    """Descending distinct block maxima with a forced low-scoring latest block."""
    from golden import TensorSpec

    scores = -torch.arange(blocks, dtype=torch.float32).repeat_interleave(BLOCK).reshape(1, -1)
    return [
        TensorSpec("scores", list(scores.shape), torch.float32, init_value=scores),
        TensorSpec("compressed_lens", [1], torch.int32, init_value=torch.tensor([blocks * BLOCK], dtype=torch.int32)),
        TensorSpec("pool_ids", [1, TOP_BLOCKS], torch.int32),
        TensorSpec("pool_counts", [1], torch.int32),
    ]


def golden_pool_boundary(tensors):
    """Closed-form block ordering: latest first, then descending distinct maxima."""
    blocks = int(tensors["compressed_lens"][0]) // BLOCK
    count = min(blocks, TOP_BLOCKS)
    tensors["pool_ids"].fill_(-1)
    tensors["pool_ids"][0, 0] = blocks - 1
    tensors["pool_ids"][0, 1:count] = torch.arange(count - 1, dtype=torch.int32)
    tensors["pool_counts"][0] = count


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    cases = [
        ("tokens=8 cap2048", 8, None),
        ("tokens=16 cap2048", 16, None),
        ("page-boundary tokens=8 cap2048", 8, [(125, 4), (129, 4)]),
    ]
    if args.tokens is not None:
        cases = [(f"tokens={args.tokens}", args.tokens, None)]
    failures = []
    for label, tokens, requests in cases:
        entry = lightning_indexer_test
        golden_fn = golden_lightning_indexer
        print(f"[IDX] {label}", flush=True)
        torch.manual_seed(args.seed)
        result = run(
            fn=entry,
            specs=build_tensor_specs(tokens, seed=args.seed, requests=requests),
            golden_fn=golden_fn,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=2e-2,
            atol=2e-2,
            compare_fn={
                "scores": ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2),
                "indices": index_topk_compare("values"),
                "pool_ids": ratio_allclose(atol=0.0, rtol=0.0),
                "pool_counts": ratio_allclose(atol=0.0, rtol=0.0),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            failures.append(f"{label}: {result.error or 'validation failed'}")
    for blocks in (65, 2048, 2049, 4097):
        print(f"[IDX] candidate publication blocks={blocks} cap={TOP_BLOCKS}", flush=True)
        result = run(
            fn=candidate_pool_test, specs=build_pool_specs(blocks), golden_fn=golden_pool_boundary,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compare_fn={name: ratio_allclose(atol=0, rtol=0, max_error_ratio=0) for name in ("pool_ids", "pool_counts")},
            compile_only=args.compile_only,
        )
        if not result.passed:
            failures.append(f"candidate blocks={blocks}: {result.error or 'validation failed'}")
    reindex_work_dir = None
    for active in (5, 3):
        print(f"[IDX] restricted-pool active={active}", flush=True)
        result = run(
            fn=reindex_select_test,
            specs=build_reindex_specs(active),
            golden_fn=golden_reindex_test,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compare_fn={name: ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0) for name in (
                "indices", "values", "pool_ids", "pool_counts",
            )},
            compile_only=args.compile_only,
            runtime_dir=reindex_work_dir,
        )
        if not result.passed:
            failures.append(f"restricted-pool active={active}: {result.error or 'validation failed'}")
        elif not args.compile_only:
            reindex_work_dir = str(result.work_dir)
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
