# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill C1A attention with source-pool-filtered reindex selection."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from attention import attention_core
from config import (
    B_DYN, BLOCK_SIZE, CMP_BLOCKS_DYN, FLASH as M, INDEX_BLOCKS_DYN, LOCAL_HEADS, ORI_BLOCKS_DYN,
    POOL_BLOCKS_DYN, PREFILL_MAX_TOKENS, SELECTED_KEYS, TABLE_DYN, T_DYN, TP_SIZE, WINDOW_KEYS,
)
from hc import hc_post, hc_pre
from indexer import (
    IDX_DIM, IDX_HEADS, filter_candidate_scores, index_query, index_scores, index_topk, index_weights,
)
from mla_proj import mla_proj
from o_proj import o_proj
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
MIX_PAD = 32  # hc_mixes reads the padded [MIX_PAD, HC_DIM] coefficient table
HC_DIM = M.hc_dim
INDEX_TOPK = M.index_topk
POOL_BLOCKS = M.candidate_topk_blocks

# tiling
QUERY_CHUNK = 16


@pl.jit.inline(auto_scope=False)
def prefill_c1a_reindex_attention(
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
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    topk_values: pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """One TP rank's reindex attention; it reads the source index cache and publishes no keys."""
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
    window_ready = pl.system.task_dummy(deps=[kv_tid])
    window_cache_out, window_tid = publish_cache_row(
        kv_snapshot, window_slots, window_cache, num_tokens, width=HEAD_DIM, ready=window_ready,
    )

    query = pl.create_tensor([tokens, IDX_HEADS * IDX_DIM], dtype=pl.BF16)
    weights = pl.create_tensor([tokens, IDX_HEADS], dtype=pl.FP32)
    query_out = index_query(qr, qr_scale, index_wq_b, index_wq_b_scale, rope_cos, rope_sin, query)
    weights_out, weights_tid = index_weights(x, index_weights_proj, weights)
    score_width = pl.tensor.dim(index_block_table, 1) * BLOCK_SIZE
    scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    scoring_ready = pl.system.task_dummy(deps=[weights_tid])
    scores_out = index_scores(
        query_out, weights_out, index_cache, index_block_table, request_ids, compressed_lens,
        scores, num_tokens, scoring_ready,
    )
    filtered = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    filtered_out, filtered_tid = filter_candidate_scores(
        scores_out, compressed_lens, pool_block_ids, pool_block_counts,
        filtered, num_tokens, scoring_ready,
    )
    topk_idx_out, topk_val_out, topk_tid = index_topk(
        filtered_out, compressed_lens, request_ids, index_block_table, topk_indices, topk_values,
        num_tokens, filtered_tid, masked=True,
    )

    attended = pl.create_tensor([tokens, LOCAL_Q], dtype=pl.BF16)
    gather_ready = pl.system.task_dummy(deps=[topk_tid, window_tid, q_tid])
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
def prefill_c1a_reindex(
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
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
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
    """Layer-level reindex block: HC pre, norm, attention, TP reduce, HC post."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hidden = hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hidden, next_pre_mix, post_mix, residual_mix,
    )
    _attn_input_ret = rms_norm(hidden, attn_norm_weight, attn_input, width=D, output_fp32=False, column_tile=256)
    partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
    prefill_c1a_reindex_attention(
        attn_input, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
        index_wq_b, index_wq_b_scale, index_weights_proj,
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
    return plan, cos, sin, positions


def build_tensor_specs(tokens: int = 8, seed: int = 0):
    from golden import TensorSpec
    from utils import pack_nz, quant_w_per_channel

    def quantized(output_width, input_width):
        return pack_nz(
            quant_w_per_channel(torch.randn(output_width, input_width) * 0.05)[0].transpose(0, 1).contiguous()
        )

    def scale_of(output_width, input_width):
        return quant_w_per_channel(torch.randn(output_width, input_width) * 0.05)[1]

    plan, cos, sin, positions = _fixture(tokens, seed)
    pages = plan["pages"]
    generator = torch.Generator().manual_seed(seed)
    from indexer import candidate_pool_fixture

    pool_ids, pool_counts = candidate_pool_fixture(plan["compressed_lens"], POOL_BLOCKS)
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
        TensorSpec("index_wq_b", [Q_LORA, IDX_HEADS * IDX_DIM], torch.int8,
                   init_value=lambda: quantized(IDX_HEADS * IDX_DIM, Q_LORA)),
        TensorSpec("index_wq_b_scale", [IDX_HEADS * IDX_DIM], torch.float32,
                   init_value=lambda: scale_of(IDX_HEADS * IDX_DIM, Q_LORA)),
        TensorSpec("index_weights_proj", [D, IDX_HEADS], torch.bfloat16,
                   init_value=lambda: torch.randn(D, IDX_HEADS) * 0.02),
        TensorSpec("topk_indices", [tokens, INDEX_TOPK], torch.int32),
        TensorSpec("topk_values", [tokens, INDEX_TOPK], torch.float32),
        TensorSpec("pool_block_ids", [tokens, POOL_BLOCKS], torch.int32, init_value=pool_ids),
        TensorSpec("pool_block_counts", [tokens], torch.int32, init_value=pool_counts),
        TensorSpec("partial", [tokens, D], torch.bfloat16),
    ]


def golden_c1a_reindex_attention(tensors):
    """Compose the per-operator references; the source index cache is given, not published."""
    from attention import golden_attention_core
    from indexer import golden_index_scores, golden_reindex_select
    from mla_proj import _golden_mla_proj
    from o_proj import golden_o_proj
    from paged_cache import golden_gather_selected_keys
    from utils import scatter_cache_rows

    mla = {name: tensors[name] for name in (
        "x", "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
        "kv_norm_w", "rope_cos", "rope_sin")}
    mla["q"] = torch.zeros(tensors["x"].shape[0], LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    mla["kv"] = torch.zeros(tensors["x"].shape[0], HEAD_DIM, dtype=torch.bfloat16)
    mla["qr"] = torch.zeros(tensors["x"].shape[0], Q_LORA, dtype=torch.int8)
    mla["qr_scale"] = torch.zeros(tensors["x"].shape[0], 1, dtype=torch.float32)
    _golden_mla_proj(mla)
    scatter_cache_rows(tensors["window_cache"], tensors["window_slots"], mla["kv"])

    index = {
        # The source layer published these rows; the reindex layer only reads
        # them, so the reference is passed the same cache and publishes nothing.
        "latent": torch.zeros(tensors["x"].shape[0], HEAD_DIM, dtype=torch.bfloat16),
        "wk": torch.zeros(IDX_DIM, HEAD_DIM, dtype=torch.bfloat16),
        "k_norm": torch.ones(IDX_DIM, dtype=torch.bfloat16),
        "compressed_cos": torch.zeros(tensors["x"].shape[0], 32, dtype=torch.float32),
        "compressed_sin": torch.zeros(tensors["x"].shape[0], 32, dtype=torch.float32),
        "compressed_slots": torch.full((tensors["x"].shape[0],), -1, dtype=torch.int64),
        "index_cache": tensors["index_cache"].clone(),
        "qr": mla["qr"], "qr_scale": mla["qr_scale"], "wq_b": tensors["index_wq_b"],
        "wq_b_scale": tensors["index_wq_b_scale"], "x": tensors["x"],
        "weights_proj": tensors["index_weights_proj"], "cos": tensors["rope_cos"],
        "sin": tensors["rope_sin"], "position_ids": torch.zeros_like(tensors["request_ids"]),
        "compressed_lens": tensors["compressed_lens"], "index_block_table": tensors["index_block_table"],
        "request_ids": tensors["request_ids"],
        "scores": torch.zeros(tensors["x"].shape[0], tensors["index_block_table"].shape[1] * BLOCK_SIZE,
                              dtype=torch.float32),
        "indices": torch.full((tensors["topk_indices"].shape[0], INDEX_TOPK), -1, dtype=torch.int32),
        "values": torch.zeros(tensors["topk_values"].shape, dtype=torch.float32),
        "pool_ids": tensors["pool_block_ids"],
        "pool_counts": tensors["pool_block_counts"],
    }
    scores = golden_index_scores(index)
    golden_reindex_select(index, scores)
    tensors["topk_indices"][:] = index["indices"]
    tensors["topk_values"][:] = index["values"]

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
def prefill_c1a_reindex_test(
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
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    topk_values: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32]],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    partial: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
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
    prefill_c1a_reindex_attention(
        x, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, topk_values, pool_block_ids, pool_block_counts, partial,
        pl.tensor.dim(x, 0),
    )
    return (window_cache, topk_indices, topk_values, pool_block_ids, pool_block_counts, partial)


# ---------------------------------------------------------------------------
# Layer-level host-compile entry (compile-only; the fwd driver owns device runs)
# ---------------------------------------------------------------------------

LAYER_INPUT_NAMES = (
    "x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
    "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale", "kv_norm_w",
    "sink", "wo_a", "wo_b", "rope_cos", "rope_sin", "window_slots", "window_indices", "window_cache",
    "compressed_cache", "request_ids", "compressed_lens", "index_cache", "index_block_table",
    "index_wq_b", "index_wq_b_scale", "index_weights_proj",
    "topk_indices", "topk_values", "pool_block_ids", "pool_block_counts",
)

ATTENTION_INPUT_NAMES = LAYER_INPUT_NAMES[6:]


def _layer_specs(tokens: int = 8, seed: int = 0):
    from golden import ScalarSpec, TensorSpec

    base = {spec.name: spec for spec in build_tensor_specs(tokens, seed)}
    generator = torch.Generator().manual_seed(seed + 100)
    specs = [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, D, generator=generator)),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32,
                   init_value=lambda: torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))),
        TensorSpec("hc_attn_fn", [MIX_PAD, HC_DIM], torch.float32,
                   init_value=lambda: torch.randn(MIX_PAD, HC_DIM, generator=generator) / HC_DIM**0.5),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=lambda: torch.randn(3, generator=generator)),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32,
                   init_value=lambda: torch.randn(MIX_HC, generator=generator)),
        TensorSpec("attn_norm_weight", [D], torch.bfloat16,
                   init_value=lambda: torch.randn(D, generator=generator) * 0.1 + 1.0),
    ]
    specs.extend(base[name] for name in ATTENTION_INPUT_NAMES)
    specs.extend([
        TensorSpec("hidden", [tokens, D], torch.bfloat16),
        TensorSpec("attn_input", [tokens, D], torch.bfloat16),
        TensorSpec("attn_out", [tokens, D], torch.bfloat16),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
        TensorSpec("next_pre_mix", [tokens, HC_MULT], torch.float32),
        ScalarSpec("num_tokens", torch.int32, tokens),
    ])
    return specs


@pl.jit
def prefill_c1a_reindex_layer_rank(
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
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    topk_values: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32]],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one TP rank of the reindex layer chain (host-compile harness)."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    topk_values.bind_dynamic(0, T_DYN)
    pool_block_ids.bind_dynamic(0, T_DYN)
    pool_block_counts.bind_dynamic(0, T_DYN)
    hidden.bind_dynamic(0, T_DYN)
    attn_input.bind_dynamic(0, T_DYN)
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    next_pre_mix.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    prefill_c1a_reindex(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, topk_values, pool_block_ids, pool_block_counts,
        output_window, output_arrived, hidden, attn_input, attn_out, output, next_pre_mix,
        0, 0, num_tokens, 1,
    )
    return output, attn_input


@pl.jit.host
def prefill_c1a_reindex_layer_entry(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    # W8A8 weights are NZ-packed on device; the host ABI carries the packed
    # buffer under its logical shape, as the forward driver does.
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8],
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
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    topk_values: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32]],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Allocate one TP window and host-compile the reindex layer chain."""
    data_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    output_window = pld.window(data_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    output_arrived = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
    prefill_c1a_reindex_layer_rank(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, request_ids, compressed_lens, index_cache, index_block_table,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, topk_values, pool_block_ids, pool_block_counts,
        output_window, output_arrived, hidden, attn_input, attn_out, output, next_pre_mix,
        num_tokens, device=0,
    )


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
    config = dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device)
    if args.compile_only:
        print("[C1A-REINDEX] host compile: operator chain", flush=True)
        result = run(fn=prefill_c1a_reindex_test, specs=build_tensor_specs(args.tokens, seed=args.seed),
                     golden_fn=None, config=config, compare_fn={}, compile_only=True)
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)
        print("[C1A-REINDEX] host compile: layer chain", flush=True)
        result = run(fn=prefill_c1a_reindex_layer_entry, specs=_layer_specs(args.tokens, seed=args.seed),
                     golden_fn=None, config=config, compare_fn={}, compile_only=True)
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)
        raise SystemExit(0)

    from indexer import index_score_compare, index_topk_compare

    compare = {
        "partial": ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2),
        "topk_values": index_score_compare,
        "topk_indices": index_topk_compare("topk_values"),
        "pool_block_ids": ratio_allclose(atol=0.0, rtol=0.0),
        "pool_block_counts": ratio_allclose(atol=0.0, rtol=0.0),
        "window_cache": ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2),
    }
    result = run(
        fn=prefill_c1a_reindex_test,
        specs=build_tensor_specs(args.tokens, seed=args.seed),
        golden_fn=golden_c1a_reindex_attention,
        config=config,
        rtol=2e-2,
        atol=2e-2,
        compare_fn=compare,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)
