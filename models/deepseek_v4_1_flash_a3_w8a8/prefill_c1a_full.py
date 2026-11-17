# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill C1A full attention on the paged dynamic-batch interface.

Projection, window/compressed/index cache publication, paged index selection,
per-token packed key gather and the MLA core, then the output projection.  The
chain is layer-level (it owns hc_pre, RMSNorm and hc_post) and the bounded entry
validates the pre-TP-reduce output on one card.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from attention import attention_core
from compressor import c1a_compressor
from config import (
    B_DYN, BLOCK_SIZE, CMP_BLOCKS_DYN, FLASH as M, INDEX_BLOCKS_DYN, LOCAL_HEADS,
    ORI_BLOCKS_DYN, POOL_BLOCKS_DYN, PREFILL_MAX_TOKENS, SELECTED_KEYS, TABLE_DYN, T_DYN,
    TP_SIZE, WINDOW_KEYS,
)
from hc import hc_post, hc_pre
from indexer import (
    IDX_DIM, IDX_HEADS, index_key, index_query, index_scores, index_topk, index_weights,
    select_candidate_block_pool,
)
from mla_proj import mla_proj
from o_proj import o_proj
from rope import golden_rope_tail_bf16, rope_tail_bf16
from paged_cache import compare_published_source_cache
from paged_cache import gather_selected_keys, publish_cache_row, snapshot_rows
from rmsnorm import rms_norm
from tp_reduce import tp_output_reduce


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
LOCAL_Q = LOCAL_HEADS * HEAD_DIM
O_GROUPS = M.o_groups // (M.num_attention_heads // LOCAL_HEADS)
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
O_LORA = M.o_lora_rank
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
MIX_PAD = 32
HC_DIM = M.hc_dim
INDEX_TOPK = M.index_topk
CANDIDATE_BLOCK = M.candidate_block_size
POOL_BLOCKS = M.candidate_topk_blocks

# tiling
QUERY_CHUNK = 16


@pl.jit.inline(auto_scope=False)
def prefill_c1a_full_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressor_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
    index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    topk_values: pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """One TP rank's paged C1A attention, publishing its caches on the way."""
    tokens = pl.tensor.dim(x, 0)
    q = pl.create_tensor([tokens, LOCAL_HEADS, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([tokens, 1], dtype=pl.FP32)
    mla_proj(
        x, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        rope_cos, rope_sin, q, kv, qr, qr_scale,
    )
    # Snapshot every early intermediate the later stages consume: the fused
    # planner may otherwise hand a dead InOut buffer to a later temporary.
    q_rows = pl.reshape(q, [tokens * LOCAL_HEADS, HEAD_DIM])
    q_flat, q_tid = snapshot_rows(q_rows, width=HEAD_DIM)
    kv_snapshot, kv_tid = snapshot_rows(kv, width=HEAD_DIM)
    query = pl.create_tensor([tokens, IDX_HEADS * IDX_DIM], dtype=pl.BF16)
    weights = pl.create_tensor([tokens, IDX_HEADS], dtype=pl.FP32)
    query_out = index_query(qr, qr_scale, index_wq_b, index_wq_b_scale, rope_cos, rope_sin, query)
    weights_out, weights_tid = index_weights(x, index_weights_proj, weights)
    window_ready = pl.system.task_dummy(deps=[kv_tid])
    window_cache_out, window_tid = publish_cache_row(kv_snapshot, window_slots, window_cache, num_tokens, width=HEAD_DIM, ready=window_ready)

    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    latent_out = c1a_compressor(x, compressor_wkv, compressor_norm, latent)
    attention_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    attention_latent = rope_tail_bf16(latent_out, compressed_cos, compressed_sin, attention_latent, width=HEAD_DIM, heads=1)
    attention_snapshot, attention_tid = snapshot_rows(attention_latent, width=HEAD_DIM)
    compressed_cache_out, compressed_tid = publish_cache_row(
        attention_snapshot, compressed_slots, compressed_cache, num_tokens, width=HEAD_DIM, ready=attention_tid,
    )

    key = index_key(latent_out, index_wk, index_norm_w, compressed_cos, compressed_sin)
    key_snapshot, key_snapshot_tid = snapshot_rows(key, width=IDX_DIM)
    index_cache_out, key_tid = publish_cache_row(
        key_snapshot, compressed_slots, index_cache, num_tokens, width=IDX_DIM, ready=key_snapshot_tid,
    )

    score_width = pl.tensor.dim(index_block_table, 1) * BLOCK_SIZE
    scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    scoring_ready = pl.system.task_dummy(deps=[key_tid, weights_tid])
    scores_out = index_scores(
        query_out, weights_out, index_cache, index_block_table, request_ids, compressed_lens,
        scores, num_tokens, scoring_ready,
    )
    topk_idx_out, topk_val_out, topk_tid = index_topk(
        scores_out, compressed_lens, request_ids, index_block_table, topk_indices, topk_values,
        num_tokens, scoring_ready, masked=False,
    )
    pool_ids_out, pool_counts_out, pool_tid = select_candidate_block_pool(
        scores_out, compressed_lens, pool_block_ids, pool_block_counts,
        width=score_width,
        rows=num_tokens,
        topk_blocks=POOL_BLOCKS, block_size=CANDIDATE_BLOCK,
        ready=scoring_ready,
    )

    attended = pl.create_tensor([tokens, LOCAL_Q], dtype=pl.BF16)
    gather_ready = pl.system.task_dummy(deps=[topk_tid, compressed_tid, window_tid])
    for begin in pl.range(0, num_tokens, QUERY_CHUNK):
        active = pl.min(QUERY_CHUNK, num_tokens - begin)
        window_chunk = pl.slice(window_indices, [active, WINDOW_KEYS], [begin, 0])
        topk_chunk = pl.slice(topk_indices, [active, INDEX_TOPK], [begin, 0])
        selected = pl.create_tensor([active, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
        combined = pl.create_tensor([active, SELECTED_KEYS], dtype=pl.INT32)
        selected_gathered, combined_gathered, gathered = gather_selected_keys(
            window_cache, window_chunk, compressed_cache, topk_chunk,
            selected, combined, active, gather_ready,
        )
        q_chunk = pl.slice(q_flat, [active * LOCAL_HEADS, HEAD_DIM], [begin * LOCAL_HEADS, 0])
        attended_chunk = pl.create_tensor([active, LOCAL_Q], dtype=pl.BF16)
        attended_gathered = attention_core(
            q_chunk, selected_gathered, combined_gathered, sink, attended_chunk,
            keys=SELECTED_KEYS, ready=gathered,
        )
        attended[begin : begin + active, 0:LOCAL_Q] = attended_gathered
    partial = o_proj(
        attended, wo_a, wo_b, rope_cos, rope_sin, partial, groups=O_GROUPS, group_in=O_GROUP_IN, o_lora=O_LORA,
    )
    return partial, topk_indices, topk_values, pool_block_ids, pool_block_counts


@pl.jit.inline(auto_scope=False)
def prefill_c1a_full(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressor_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
    index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    topk_values: pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Layer-level C1A full block: HC pre, norm, attention, TP reduce, HC post."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hidden = hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hidden, next_pre_mix, post_mix, residual_mix,
    )
    _attn_input_ret = rms_norm(hidden, attn_norm_weight, attn_input, width=D, output_fp32=False, column_tile=256)
    partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
    prefill_c1a_full_attention(
        attn_input, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
        compressed_cos, compressed_sin, compressed_slots, compressor_wkv, compressor_norm,
        index_wq_b, index_wq_b_scale, index_weights_proj, index_wk, index_norm_w,
        topk_indices, topk_values, pool_block_ids, pool_block_counts, partial, num_tokens,
    )
    attn_out = tp_output_reduce(partial, output_window, output_arrived, attn_out, group_base, tp_rank, tokens, attention_epoch)
    _output_ret = hc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output, topk_indices, topk_values, pool_block_ids, pool_block_counts, attn_input


# ---------------------------------------------------------------------------
# Bounded validation entry (one card, pre-TP-reduce output)
# ---------------------------------------------------------------------------


def _rope_tables(count: int, start: int = 0):
    half = 32
    inv_freq = 1.0 / torch.pow(
        torch.tensor(M.rope_theta, dtype=torch.float32),
        torch.arange(0, half, dtype=torch.float32) * 2.0 / (2 * half),
    )
    angle = torch.arange(start, start + count, dtype=torch.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
    return torch.cos(angle), torch.sin(angle)


def _fixture(tokens: int = 8, seed: int = 0):
    from utils import build_request_plan

    plan = build_request_plan([(tokens + 2, tokens // 2), (tokens + 5, tokens - tokens // 2)], ratio=1)
    cos, sin = _rope_tables(256)
    positions = plan["position_ids"].long()
    compressed_positions = plan["compressed_lens"].long() - 1
    return plan, cos, sin, positions, compressed_positions


def build_tensor_specs(tokens: int = 8, seed: int = 0):
    from golden import TensorSpec
    from utils import pack_nz, quant_w_per_channel

    def quantized(output_width, input_width):
        return pack_nz(
            quant_w_per_channel(torch.randn(output_width, input_width) * 0.05)[0].transpose(0, 1).contiguous()
        )

    def scale_of(output_width, input_width):
        return quant_w_per_channel(torch.randn(output_width, input_width) * 0.05)[1]

    plan, cos, sin, positions, compressed_positions = _fixture(tokens, seed)
    pages = plan["pages"]
    generator = torch.Generator().manual_seed(seed)
    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=lambda: torch.randn(tokens, D) * 0.1),
        TensorSpec("wq_a", [D, Q_LORA], torch.int8, init_value=lambda: quantized(Q_LORA, D)),
        TensorSpec("wq_a_scale", [Q_LORA], torch.float32, init_value=lambda: scale_of(Q_LORA, D)),
        TensorSpec("q_norm_w", [Q_LORA], torch.bfloat16, init_value=lambda: torch.randn(Q_LORA) * 0.1 + 1.0),
        TensorSpec("wq_b", [Q_LORA, LOCAL_Q], torch.int8, init_value=lambda: quantized(LOCAL_Q, Q_LORA)),
        TensorSpec("wq_b_scale", [LOCAL_Q], torch.float32, init_value=lambda: scale_of(LOCAL_Q, Q_LORA)),
        TensorSpec("wkv", [D, HEAD_DIM], torch.int8, init_value=lambda: quantized(HEAD_DIM, D)),
        TensorSpec("wkv_scale", [HEAD_DIM], torch.float32, init_value=lambda: scale_of(HEAD_DIM, D)),
        TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM) * 0.1 + 1.0),
        TensorSpec("sink", [LOCAL_HEADS], torch.float32, init_value=lambda: torch.randn(LOCAL_HEADS) * 0.5),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16,
                   init_value=lambda: torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) * 0.02),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.bfloat16,
                   init_value=lambda: torch.randn(D, O_GROUPS * O_LORA) * 0.02),
        TensorSpec("rope_cos", [tokens, 32], torch.float32, init_value=cos[positions]),
        TensorSpec("rope_sin", [tokens, 32], torch.float32, init_value=sin[positions]),
        TensorSpec("window_slots", [tokens], torch.int64, init_value=plan["window_slots"]),
        TensorSpec("window_indices", [tokens, WINDOW_KEYS], torch.int32, init_value=plan["window_indices"]),
        TensorSpec("window_cache", [pages, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: torch.randn(pages, BLOCK_SIZE, 1, HEAD_DIM, generator=generator) * 0.1),
        TensorSpec("compressed_cache", [pages, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: torch.randn(pages, BLOCK_SIZE, 1, HEAD_DIM, generator=generator) * 0.1),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=plan["request_ids"]),
        TensorSpec("compressed_lens", [tokens], torch.int32, init_value=plan["compressed_lens"]),
        TensorSpec("index_cache", [pages, BLOCK_SIZE, 1, IDX_DIM], torch.bfloat16,
                   init_value=lambda: torch.randn(pages, BLOCK_SIZE, 1, IDX_DIM, generator=generator) * 0.1),
        TensorSpec("index_block_table", [plan["index_block_table"].shape[0], plan["index_block_table"].shape[1]],
                   torch.int32, init_value=plan["index_block_table"]),
        TensorSpec("compressed_cos", [tokens, 32], torch.float32, init_value=cos[compressed_positions]),
        TensorSpec("compressed_sin", [tokens, 32], torch.float32, init_value=sin[compressed_positions]),
        TensorSpec("compressed_slots", [tokens], torch.int64, init_value=plan["compressed_slots"]),
        TensorSpec("compressor_wkv", [HEAD_DIM, D], torch.bfloat16,
                   init_value=lambda: torch.randn(HEAD_DIM, D) * 0.02),
        TensorSpec("compressor_norm", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM) * 0.05 + 1.0),
        TensorSpec("index_wq_b", [Q_LORA, IDX_HEADS * IDX_DIM], torch.int8,
                   init_value=lambda: quantized(IDX_HEADS * IDX_DIM, Q_LORA)),
        TensorSpec("index_wq_b_scale", [IDX_HEADS * IDX_DIM], torch.float32,
                   init_value=lambda: scale_of(IDX_HEADS * IDX_DIM, Q_LORA)),
        TensorSpec("index_weights_proj", [D, IDX_HEADS], torch.bfloat16,
                   init_value=lambda: torch.randn(D, IDX_HEADS) * 0.02),
        TensorSpec("index_wk", [IDX_DIM, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: torch.randn(IDX_DIM, HEAD_DIM) * 0.02),
        TensorSpec("index_norm_w", [IDX_DIM], torch.bfloat16, init_value=lambda: torch.randn(IDX_DIM) * 0.05 + 1.0),
        TensorSpec("topk_indices", [tokens, INDEX_TOPK], torch.int32),
        TensorSpec("topk_values", [tokens, INDEX_TOPK], torch.float32),
        TensorSpec("pool_block_ids", [tokens, POOL_BLOCKS], torch.int32),
        TensorSpec("pool_block_counts", [tokens], torch.int32),
        TensorSpec("partial", [tokens, D], torch.bfloat16),
    ]


def golden_c1a_attention(tensors):
    """Compose the per-operator references on the paged fixtures."""
    from attention import golden_attention_core
    from compressor import golden_c1a_compressor
    from indexer import golden_index_scores, golden_index_select
    from mla_proj import _golden_mla_proj
    from o_proj import golden_o_proj
    from utils import scatter_cache_rows

    mla = {name: tensors[name] for name in (
        "x", "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
        "kv_norm_w", "rope_cos", "rope_sin")}
    mla["q"] = torch.zeros(tensors["x"].shape[0], LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    mla["kv"] = torch.zeros(tensors["x"].shape[0], HEAD_DIM, dtype=torch.bfloat16)
    mla["qr"] = torch.zeros(tensors["x"].shape[0], Q_LORA, dtype=torch.int8)
    mla["qr_scale"] = torch.zeros(tensors["x"].shape[0], 1, dtype=torch.float32)
    _golden_mla_proj(mla)
    scatter_cache_rows(tensors["window_cache"], tensors["window_slots"], mla["kv"].reshape(-1, HEAD_DIM))

    compressed = {
        "x": tensors["x"], "wkv": tensors["compressor_wkv"], "norm_w": tensors["compressor_norm"],
        "output": torch.zeros(tensors["x"].shape[0], HEAD_DIM, dtype=torch.bfloat16),
    }
    golden_c1a_compressor(compressed)
    attention_latent = golden_rope_tail_bf16(compressed["output"], tensors["compressed_cos"], tensors["compressed_sin"])
    scatter_cache_rows(tensors["compressed_cache"], tensors["compressed_slots"], attention_latent)

    index = {
        "latent": compressed["output"], "wk": tensors["index_wk"], "k_norm": tensors["index_norm_w"],
        "compressed_cos": tensors["compressed_cos"], "compressed_sin": tensors["compressed_sin"],
        "compressed_slots": tensors["compressed_slots"], "index_cache": tensors["index_cache"],
        "qr": mla["qr"], "qr_scale": mla["qr_scale"], "wq_b": tensors["index_wq_b"],
        "wq_b_scale": tensors["index_wq_b_scale"], "x": tensors["x"],
        "weights_proj": tensors["index_weights_proj"], "cos": tensors["rope_cos"],
        "sin": tensors["rope_sin"], "position_ids": torch.zeros_like(tensors["compressed_lens"]),
        "compressed_lens": tensors["compressed_lens"], "index_block_table": tensors["index_block_table"],
        "request_ids": tensors["request_ids"],
        "scores": torch.zeros(tensors["topk_values"].shape[0], tensors["index_block_table"].shape[1] * BLOCK_SIZE, dtype=torch.float32),
        "indices": torch.full((tensors["topk_indices"].shape[0], INDEX_TOPK), -1, dtype=torch.int32),
        "values": torch.zeros(tensors["topk_values"].shape, dtype=torch.float32),
        "pool_ids": torch.full_like(tensors["pool_block_ids"], -1),
        "pool_counts": torch.zeros_like(tensors["pool_block_counts"]),
    }
    scores = golden_index_scores(index)
    golden_index_select(index, scores, topk_blocks=index["pool_ids"].shape[1])
    tensors["topk_indices"][:] = index["indices"]
    tensors["topk_values"][:] = index["values"]
    tensors["pool_block_ids"][:] = index["pool_ids"]
    tensors["pool_block_counts"][:] = index["pool_counts"]

    from paged_cache import golden_gather_selected_keys

    gather = {
        "window_cache": tensors["window_cache"], "window_indices": tensors["window_indices"],
        "compressed_cache": tensors["compressed_cache"], "topk_indices": index["indices"],
        "selected": torch.zeros(tensors["x"].shape[0], SELECTED_KEYS, HEAD_DIM, dtype=torch.bfloat16),
        "combined": torch.zeros(tensors["x"].shape[0], SELECTED_KEYS, dtype=torch.int32),
    }
    golden_gather_selected_keys(gather)
    attention = {
        "q": mla["q"], "selected": gather["selected"], "combined": gather["combined"],
        "sink": tensors["sink"],
        "output": torch.zeros(tensors["x"].shape[0], LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16),
    }
    golden_attention_core(attention)
    out = {
        "attended": attention["output"].reshape(tensors["x"].shape[0], LOCAL_Q).float(),
        "wo_a": tensors["wo_a"], "wo_b": tensors["wo_b"],
        "output": torch.zeros(tensors["x"].shape[0], D, dtype=torch.bfloat16),
    }
    out["cos"] = tensors["rope_cos"]
    out["sin"] = tensors["rope_sin"]
    golden_o_proj(out)
    tensors["partial"][:] = out["output"]


@pl.jit
def prefill_attention_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressor_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
    index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    topk_values: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32]],
    pool_block_ids: pl.Out[pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
    partial: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    compressed_cos.bind_dynamic(0, T_DYN)
    compressed_sin.bind_dynamic(0, T_DYN)
    compressed_slots.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    topk_values.bind_dynamic(0, T_DYN)
    pool_block_ids.bind_dynamic(0, T_DYN)
    pool_block_counts.bind_dynamic(0, T_DYN)
    partial.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    prefill_c1a_full_attention(
        x, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
        compressed_cos, compressed_sin, compressed_slots, compressor_wkv, compressor_norm,
        index_wq_b, index_wq_b_scale, index_weights_proj, index_wk, index_norm_w,
        topk_indices, topk_values, pool_block_ids, pool_block_counts, partial,
        pl.tensor.dim(x, 0),
    )
    return topk_indices, topk_values, pool_block_ids, pool_block_counts, partial


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    from indexer import index_score_compare, index_topk_compare

    compare = {
        "partial": ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2),
        "topk_values": index_score_compare,
        "topk_indices": index_topk_compare("topk_values"),
        "pool_block_ids": ratio_allclose(atol=0.0, rtol=0.0),
        "pool_block_counts": ratio_allclose(atol=0.0, rtol=0.0),
        # Cache contents are device-computed projections, so they carry the
        # accepted child arithmetic band; addressing faults still fail loudly.
        "window_cache": ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2),
        "compressed_cache": compare_published_source_cache,
        "index_cache": compare_published_source_cache,
    }
    result = run(
        fn=prefill_attention_test,
        specs=build_tensor_specs(args.tokens, seed=args.seed),
        golden_fn=golden_c1a_attention,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
        compare_fn=compare,
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)
