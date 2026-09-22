# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C2A full attention."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult, golden_compressed_attention
from models.deepseek_v4_1_flash.attention_tp import prefill_tp_output_all_reduce
from models.deepseek_v4_1_flash.config import (
    D, TP_SIZE, AttentionMode, T_DYN, Q_LORA, LOCAL_H, HEAD_DIM, INDEX_H, INDEX_DIM, ROPE_DIM,
)
from models.deepseek_v4_1_flash.decode_attn_c2a_full import COMPRESSOR_K_TILE, make_attend_sparse, make_c2a_full_partial, run_c2a
from models.deepseek_v4_1_flash.attention_ops import EPS, M_TILE, N_TILE, make_bf16_projection_staged, make_rope
from models.deepseek_v4_1_flash.o_proj import prefill_o_proj
from models.deepseek_v4_1_flash.qkv_proj_rope import prefill_q_proj_qr, prefill_q_proj_rope, kv_proj_rope


PREFILL_COMPRESSOR_SEGMENTS = 16
PREFILL_COMPRESSOR_K_CHUNK = D // PREFILL_COMPRESSOR_SEGMENTS
PREFILL_C2A_FULL_RING_HEAP = (2 * 1024 * 1024 * 1024,) * 4


@pl.jit.inline(auto_scope=False)
def prefill_c2a_qkv(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Use group-32 Q-A accumulation and bounded prefill head-RoPE workers."""
    prefill_q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
    prefill_q_proj_rope(query_latent, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens)
    kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens)
    return query_latent, query, window_kv


@pl.jit.inline
def prefill_c2a_permute_index_query(
    x: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Split each index-query head into even and odd lanes.

    Packed E2M1 keys decode into even and odd lanes rather than interleaved
    order. A dot product is invariant under a shared permutation, so the query
    is permuted once per token instead of re-interleaving every gathered key.
    """
    half = INDEX_DIM // 2
    blocks = num_tokens * INDEX_H
    workers = pl.min(blocks, 64)
    for worker in pl.spmd(workers, name_hint="prefill_c2a_index_query_permute"):
        for block in pl.range(worker, blocks, workers):
            t = block // INDEX_H
            h = block % INDEX_H
            base = h * INDEX_DIM
            row = pl.cast(x[t : t + 1, base : base + INDEX_DIM], pl.FP32)
            even = pl.gather(row, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(row, mask_pattern=pl.tile.MaskPattern.P1010)
            output[t : t + 1, base : base + half] = pl.cast(even, pl.BF16, mode="rint")
            output[t : t + 1, base + half : base + INDEX_DIM] = pl.cast(odd, pl.BF16, mode="rint")
    return output


@pl.jit.inline
def prefill_compressor_projection(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    output: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Retain FP32 weight residuals through high and residual accumulators.

    Three BF16 weight terms preserve FP32 weights. Bound each Cube reduction
    to one K chunk, then compensate the Vector sum of the staged GM products
    to reduce accumulation error near pooled BF16 rounding boundaries.
    Mixed producers use Vector startup and FP32 Vector stores.
    """
    tokens = pl.tensor.dim(x, 0)
    high_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    low_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    tail_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    high_product = pl.create_tensor([PREFILL_COMPRESSOR_SEGMENTS * tokens, HEAD_DIM], dtype=pl.FP32)
    residual_product = pl.create_tensor([PREFILL_COMPRESSOR_SEGMENTS * tokens, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(D // COMPRESSOR_K_TILE * (HEAD_DIM // N_TILE), name_hint="prefill_compressor_split") as split_tid:
        block = pl.tile.get_block_idx()
        k0 = block // (HEAD_DIM // N_TILE) * COMPRESSOR_K_TILE
        n0 = block % (HEAD_DIM // N_TILE) * N_TILE
        value = weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
        high = pl.cast(value, pl.BF16, mode="rint")
        residual = pl.sub(value, pl.cast(high, pl.FP32))
        low = pl.cast(residual, pl.BF16, mode="rint")
        tail = pl.cast(pl.sub(residual, pl.cast(low, pl.FP32)), pl.BF16, mode="rint")
        high_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = high
        low_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = low
        tail_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = tail
    output_blocks = (num_tokens + M_TILE - 1) // M_TILE * (HEAD_DIM // N_TILE)
    blocks = output_blocks * PREFILL_COMPRESSOR_SEGMENTS
    # The synchronous mixed cohort must fit the target A5 core count.
    workers = pl.min(blocks, 16)
    with pl.spmd(workers, name_hint="prefill_compressor_cube", deps=[split_tid], sync_start=True) as cube_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            segment = block % PREFILL_COMPRESSOR_SEGMENTS
            tile_block = block // PREFILL_COMPRESSOR_SEGMENTS
            t0 = tile_block // (HEAD_DIM // N_TILE) * M_TILE
            n0 = tile_block % (HEAD_DIM // N_TILE) * N_TILE
            rows = pl.min(M_TILE, num_tokens - t0)
            high_acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
            residual_acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
            for k0 in pl.range(segment * PREFILL_COMPRESSOR_K_CHUNK, (segment + 1) * PREFILL_COMPRESSOR_K_CHUNK, COMPRESSOR_K_TILE):
                a = pl.slice(x, [M_TILE, COMPRESSOR_K_TILE], [t0, k0], valid_shape=[rows, COMPRESSOR_K_TILE])
                a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                high = high_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
                low = low_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
                tail = tail_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
                high_acc = pl.matmul_acc(high_acc, a, high, init_cond=(k0 == segment * PREFILL_COMPRESSOR_K_CHUNK))
                residual_acc = pl.matmul_acc(residual_acc, a, low, init_cond=(k0 == segment * PREFILL_COMPRESSOR_K_CHUNK))
                residual_acc = pl.matmul_acc(residual_acc, a, tail)
            high_product[segment * tokens + t0 : segment * tokens + t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(pl.mul(high_acc, 1.0), rows, N_TILE)
            residual_product[segment * tokens + t0 : segment * tokens + t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(pl.mul(residual_acc, 1.0), rows, N_TILE)
    with pl.spmd(pl.min(output_blocks, 16), name_hint="prefill_compressor_sum", deps=[cube_tid]):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, output_blocks, pl.min(output_blocks, 16)):
            t0 = block // (HEAD_DIM // N_TILE) * M_TILE
            n0 = block % (HEAD_DIM // N_TILE) * N_TILE
            rows = pl.min(M_TILE, num_tokens - t0)
            high_sum = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            low_sum = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            high_error = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            low_error = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            for segment in pl.range(PREFILL_COMPRESSOR_SEGMENTS):
                high_part = pl.slice(high_product, [M_TILE, N_TILE], [segment * tokens + t0, n0], valid_shape=[rows, N_TILE])
                low_part = pl.slice(residual_product, [M_TILE, N_TILE], [segment * tokens + t0, n0], valid_shape=[rows, N_TILE])
                high_delta = pl.sub(high_part, high_error)
                high_next = pl.add(high_sum, high_delta)
                high_error = pl.sub(pl.sub(high_next, high_sum), high_delta)
                high_sum = high_next
                low_delta = pl.sub(low_part, low_error)
                low_next = pl.add(low_sum, low_delta)
                low_error = pl.sub(pl.sub(low_next, low_sum), low_delta)
                low_sum = low_next
            output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(pl.add(high_sum, low_sum), rows, N_TILE)
    return output


@pl.jit.inline
def prefill_compressor_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    kv_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    prefill_compressor_projection(x, wkv, kv_out, num_tokens)
    prefill_compressor_projection(x, wgate, score_out, num_tokens)
    return kv_out, score_out


_prefill_index_key_projection = make_bf16_projection_staged(HEAD_DIM, INDEX_DIM)
_prefill_index_weight_projection = make_bf16_projection_staged(D, INDEX_H)


@pl.jit.inline
def prefill_index_key_project(
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
    _prefill_index_key_projection(latent, weight, projected, num_tokens)
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
def prefill_indexer_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Preserve the index-weight rounding boundaries with a staged projection."""
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    _prefill_index_weight_projection(x, weight, projected, num_tokens)
    for block in pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="prefill_c2a_index_weight_scale"):
        t0 = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.cast(pl.slice(projected, [M_TILE, INDEX_H], [t0, 0], valid_shape=[rows, INDEX_H]), pl.FP32)
        scaled = pl.cast(pl.mul(value, INDEX_DIM**-0.5), pl.BF16, mode="rint")
        result = pl.cast(pl.mul(pl.cast(scaled, pl.FP32), INDEX_H**-0.5), pl.BF16, mode="rint")
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(result, rows, INDEX_H)
    return output


prefill_attend_sparse = make_attend_sparse(vector_query=True)
_prefill_index_rope = make_rope(INDEX_H, head_dim=INDEX_DIM, max_workers=64)
c2a_full_partial = make_c2a_full_partial(
    prefill_c2a_qkv, prefill_o_proj, _prefill_index_rope, prefill_c2a_permute_index_query,
    prefill_compressor_project, prefill_index_key_project, prefill_indexer_weights, prefill_attend_sparse,
)


def golden_prefill_attn_c2a_full(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    position_ids: torch.Tensor,
    compressed_rope_cos: torch.Tensor,
    compressed_rope_sin: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_block_table: torch.Tensor,
    state_cache: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    compressed_slots: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Reference the C2A Full leaf: attention plus every cache and state it publishes."""
    return golden_compressed_attention(
        mode=AttentionMode.FULL,
        ratio=2,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=None,
        compressor_wkv=compressor_wkv,
        compressor_wgate=compressor_wgate,
        compressor_norm_weight=compressor_norm_weight,
        query_start_loc=query_start_loc,
        state_block_table=state_block_table,
        state_cache=state_cache,
        compressed_slots=compressed_slots,
        position_ids=position_ids,
        compressed_lens=compressed_lens,
        compressed_rope_cos=compressed_rope_cos,
        compressed_rope_sin=compressed_rope_sin,
        index_wk=index_wk,
        index_norm_weight=index_norm_weight,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_cache=index_cache,
        index_cache_scale=index_cache_scale,
        index_block_table=index_block_table,
        request_ids=token_to_req_indices,
        candidate_mask=None,
    )


@pl.jit.inline(auto_scope=False)
def prefill_attn_c2a_full_partial(
    x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
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
    window_cache_scale: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    token_to_req_indices: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[C.Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[C.B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[C.STATE_BLOCKS_DYN, C.STATE_CAPACITY, C.STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[C.T_DYN, C.D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write FP32 head-TP partial output using zero-initialized windows and consecutive 1-based epochs."""
    # A later epoch must not overwrite cache or transport storage still being read.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="c2a_previous_epoch", allow_early_resolve=False
    ) as cache_ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                cmp=pld.WaitCmp.Ge,
            )
    c2a_full_partial(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
        attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices,
        compressed_lens, index_cache, index_cache_scale, index_block_table, position_ids,
        compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate,
        query_start_loc, state_block_table, state_cache, compressor_norm_weight, compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, partial, num_tokens, cache_ready,
    )
    return partial


@pl.jit.inline(auto_scope=False)
def prefill_attn_c2a_full(
    x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
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
    window_cache_scale: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    token_to_req_indices: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[C.Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[C.B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[C.STATE_BLOCKS_DYN, C.STATE_CAPACITY, C.STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Return replicated output for standalone C2A attention callers."""
    tokens = pl.tensor.dim(x, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    prefill_attn_c2a_full_partial(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b,
        wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, token_to_req_indices, compressed_lens, index_cache, index_cache_scale,
        index_block_table, position_ids, compressed_rope_cos, compressed_rope_sin, compressor_wkv,
        compressor_wgate, query_start_loc, state_block_table, state_cache, compressor_norm_weight,
        compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
        index_weights_proj, topk_indices, output_arrived, partial, num_tokens,
        attention_epoch,
    )
    prefill_tp_output_all_reduce(
        partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens,
        attention_epoch,
    )
    return output


__all__ = ["golden_prefill_attn_c2a_full", "prefill_attn_c2a_full", "prefill_attn_c2a_full_partial"]


def validate(argv=None):
    """Validate the Prefill C2A Full leaf operator on A5."""
    return run_c2a(prefill_attn_c2a_full, "prefill", argv=argv, ring_heap=PREFILL_C2A_FULL_RING_HEAP)


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,dp", [(1, 1), (2, 2), (4, 1)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
