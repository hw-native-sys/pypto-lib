# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C1A attention leaves: full, reindex and reuse."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch


from config import (
    BLOCK_SIZE,
    FLASH as M,
    AttentionMode,
    T_DYN,
    B_DYN,
    TABLE_DYN,
    ORI_BLOCKS_DYN,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    CMP_POSITIONS_DYN,
    TP_SIZE,
    COMPRESSED_CACHE_GROUP,
    INDEX_CACHE_GROUP,
    PREFILL_MAX_TOKENS,
    WINDOW_CACHE_GROUP,
)

from qkv_proj_rope import kv_proj_rope, q_proj_rope, prefill_c1a_publish_window
from sparse_attention import attend_packed
from o_proj import prefill_o_proj
from compressor import publish_compressed_cache
from hierarchical_sparse_indexer import publish_index_cache
from qkv_proj_rope import (
    make_bf16_projection_staged,
    make_rope,
)
from rmsnorm import rms_norm
from qkv_proj_rope import q_proj_qr
from prefill_c1a_indexer import make_paged_indexer
from utils import AttentionGoldenResult
from sparse_attention import golden_compressed_attention
from golden import mapped_pool_ratio_allclose, ratio_allclose
from o_proj import prefill_tp_output_all_reduce
from utils import (
    CASE_DEFAULT,
    CASE_MAX_TOKENS,
    CASE_NAMES,
    CASE_TOKENS,
    COMMON_INPUT_NAMES,
    apply_distributed_golden,
    fixture_token_count,
    make_tensor_specs,
    topk_indices_compare,
)
from sparse_attention import (
    golden_prefill_c1a_attention,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk


PREFILL_MAX_TOKENS = PREFILL_MAX_TOKENS

FULL_INPUT_NAMES = COMMON_INPUT_NAMES + (
    "request_ids",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "compressed_rope_cos",
    "compressed_rope_sin",
    "compressor_wkv",
    "compressor_norm_weight",
    "compressed_slots",
    "index_wk",
    "index_norm_weight",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
)

REINDEX_INPUT_NAMES = COMMON_INPUT_NAMES + (
    "request_ids",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "candidate_mask",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
)

REUSE_INPUT_NAMES = COMMON_INPUT_NAMES + ("compressed_indices",)


if TP_SIZE not in (1, 2, 4):
    raise ValueError("Prefill C1A currently supports TP1, TP2, and TP4; TP8 requires head-tile padding")

project_compressed = make_bf16_projection_staged(D, HEAD_DIM)
rotate_compressed = make_rope(1)
project_index_key = make_bf16_projection_staged(HEAD_DIM, INDEX_DIM)
rotate_index_key = make_rope(1, head_dim=INDEX_DIM, rope_dim=ROPE_DIM)
full_paged_indexer = make_paged_indexer()
reindex_paged_indexer = make_paged_indexer(use_candidates=True)


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
        pl.BF16,
    ],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        pl.BF16,
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
    prefill_c1a_publish_window(
        window_kv,
        window_slots,
        window_cache,
        window_cache_scale,
        num_tokens,
        cache_ready,
    )

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attend_packed(
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






def make_prefill_publish_decoder():
    """Reuse the C1A source stages over the complete encoder output."""
    from prefill_attn_c1a import (
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
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
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
        norm_blocks = (num_tokens + 7) // 8
        for norm_worker in pl.spmd(norm_blocks, name_hint="attention_rmsnorm"):
            rms_norm(compressed_projection, compressor_norm_weight, compressed_latent, norm_worker, num_tokens, width=HEAD_DIM)
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
        norm_blocks = (num_tokens + 7) // 8
        for norm_worker in pl.spmd(norm_blocks, name_hint="attention_rmsnorm"):
            rms_norm(index_projection, index_norm_weight, index_normalized, norm_worker, num_tokens, width=INDEX_DIM)
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



def golden_prefill_attn_c1a_full(
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
    request_ids: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    compressed_rope_cos: torch.Tensor,
    compressed_rope_sin: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    compressed_slots: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Return one rank's FP32 partial output and reference cache state."""
    return golden_compressed_attention(
        mode=AttentionMode.FULL,
        ratio=1,
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
        compressor_wgate=None,
        compressor_norm_weight=compressor_norm_weight,
        state_block_table=None,
        state_cache=None,
        compressed_slots=compressed_slots,
        position_ids=None,
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
        request_ids=request_ids,
        candidate_mask=None,
        attention_fn=golden_prefill_c1a_attention,
        output_dtype=torch.float32,
    )


def make_prefill_attn_c1a_full(indexer, reducer=prefill_tp_output_all_reduce, output_tokens=T_DYN):
    @pl.jit.inline(auto_scope=False)
    def prefill_attn_c1a_full_impl(
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
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
        ],
        index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
        compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
        output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[output_tokens, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Publish ratio-1 caches, select sparse rows, and compute packed C1A."""
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        if num_tokens > 0:
            compressed_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
            project_compressed(x, compressor_wkv, compressed_projection, num_tokens)
            compressed_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
            norm_blocks = (num_tokens + 7) // 8
            for norm_worker in pl.spmd(norm_blocks, name_hint="attention_rmsnorm"):
                rms_norm(compressed_projection, compressor_norm_weight, compressed_latent, norm_worker, num_tokens, width=HEAD_DIM)
            compressed_rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
            rotate_compressed(
                compressed_latent,
                compressed_rope_cos,
                compressed_rope_sin,
                compressed_rotated,
                num_tokens,
            )
            publish_compressed_cache(
                compressed_rotated,
                compressed_slots,
                compressed_cache,
                compressed_cache_scale,
                num_tokens,
            )

            index_projection = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
            project_index_key(compressed_latent, index_wk, index_projection, num_tokens)
            index_normalized = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
            norm_blocks = (num_tokens + 7) // 8
            for norm_worker in pl.spmd(norm_blocks, name_hint="attention_rmsnorm"):
                rms_norm(index_projection, index_norm_weight, index_normalized, norm_worker, num_tokens, width=INDEX_DIM)
            index_rotated = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
            rotate_index_key(
                index_normalized,
                compressed_rope_cos,
                compressed_rope_sin,
                index_rotated,
                num_tokens,
            )
            index_ready = publish_index_cache(
                index_rotated,
                compressed_slots,
                index_cache,
                index_cache_scale,
                num_tokens,
            )

            query_latent = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
            q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
            indexer_completion = indexer(
                x,
                query_latent,
                request_ids,
                compressed_lens,
                index_cache,
                index_cache_scale,
                index_block_table,
                rope_cos,
                rope_sin,
                index_wq_b,
                index_wq_b_scale,
                index_weights_proj,
                candidate_mask,
                topk_indices,
                num_tokens,
                index_ready,
            )
            prefill_c1a_partial(
                x,
                query_latent,
                wq_b,
                wq_b_scale,
                wkv,
                wkv_scale,
                kv_norm_weight,
                attn_sink,
                wo_a,
                wo_b,
                wo_b_scale,
                rope_cos,
                rope_sin,
                window_slots,
                window_indices,
                window_cache,
                window_cache_scale,
                compressed_cache,
                compressed_cache_scale,
                topk_indices,
                partial,
                num_tokens,
            )
        reducer(
            partial,
            output_window,
            output_arrived,
            output,
            group_base,
            tp_rank,
            num_tokens,
            attention_epoch,
        )
        return output

    return prefill_attn_c1a_full_impl


prefill_attn_c1a_full = make_prefill_attn_c1a_full(full_paged_indexer)


def golden_prefill_attn_c1a_reuse(
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
    compressed_indices: torch.Tensor,
) -> AttentionGoldenResult:
    """Return one rank's FP32 partial output and reference cache state."""
    return golden_compressed_attention(
        mode=AttentionMode.REUSE,
        ratio=1,
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
        compressed_indices=compressed_indices,
        compressor_wkv=None,
        compressor_wgate=None,
        compressor_norm_weight=None,
        state_block_table=None,
        state_cache=None,
        compressed_slots=None,
        position_ids=None,
        compressed_lens=None,
        compressed_rope_cos=None,
        compressed_rope_sin=None,
        index_wk=None,
        index_norm_weight=None,
        index_wq_b=None,
        index_wq_b_scale=None,
        index_weights_proj=None,
        index_cache=None,
        index_cache_scale=None,
        index_block_table=None,
        request_ids=None,
        candidate_mask=None,
        attention_fn=golden_prefill_c1a_attention,
        output_dtype=torch.float32,
    )


def make_prefill_attn_c1a_reuse(reducer=prefill_tp_output_all_reduce, output_tokens=T_DYN):
    @pl.jit.inline(auto_scope=False)
    def prefill_attn_c1a_reuse(
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
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ],
        compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[output_tokens, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Read published ratio-1 Top-K rows and compute packed-prefill C1A."""
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        if num_tokens > 0:
            query_latent = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
            q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
            prefill_c1a_partial(
                x,
                query_latent,
                wq_b,
                wq_b_scale,
                wkv,
                wkv_scale,
                kv_norm_weight,
                attn_sink,
                wo_a,
                wo_b,
                wo_b_scale,
                rope_cos,
                rope_sin,
                window_slots,
                window_indices,
                window_cache,
                window_cache_scale,
                compressed_cache,
                compressed_cache_scale,
                compressed_indices,
                partial,
                num_tokens,
            )
        reducer(
            partial,
            output_window,
            output_arrived,
            output,
            group_base,
            tp_rank,
            num_tokens,
            attention_epoch,
        )
        return output

    return prefill_attn_c1a_reuse


prefill_attn_c1a_reuse = make_prefill_attn_c1a_reuse()


def golden_prefill_attn_c1a_reindex(
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
    request_ids: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_mask: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Return one rank's FP32 partial output and reference cache state."""
    return golden_compressed_attention(
        mode=AttentionMode.REINDEX,
        ratio=1,
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
        compressor_wkv=None,
        compressor_wgate=None,
        compressor_norm_weight=None,
        state_block_table=None,
        state_cache=None,
        compressed_slots=None,
        position_ids=None,
        compressed_lens=compressed_lens,
        compressed_rope_cos=None,
        compressed_rope_sin=None,
        index_wk=None,
        index_norm_weight=None,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_cache=index_cache,
        index_cache_scale=index_cache_scale,
        index_block_table=index_block_table,
        request_ids=request_ids,
        candidate_mask=candidate_mask,
        attention_fn=golden_prefill_c1a_attention,
        output_dtype=torch.float32,
    )


def make_prefill_attn_c1a_reindex(indexer, reducer=prefill_tp_output_all_reduce, output_tokens=T_DYN):
    @pl.jit.inline(auto_scope=False)
    def prefill_attn_c1a_reindex_impl(
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
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
        ],
        index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[output_tokens, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Refresh ratio-1 Top-K rows inside a supplied candidate set."""
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        if num_tokens > 0:
            query_latent = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
            q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
            cache_ready = pl.system.task_dummy(deps=[])
            indexer_completion = indexer(
                x,
                query_latent,
                request_ids,
                compressed_lens,
                index_cache,
                index_cache_scale,
                index_block_table,
                rope_cos,
                rope_sin,
                index_wq_b,
                index_wq_b_scale,
                index_weights_proj,
                candidate_mask,
                topk_indices,
                num_tokens,
                cache_ready,
            )
            prefill_c1a_partial(
                x,
                query_latent,
                wq_b,
                wq_b_scale,
                wkv,
                wkv_scale,
                kv_norm_weight,
                attn_sink,
                wo_a,
                wo_b,
                wo_b_scale,
                rope_cos,
                rope_sin,
                window_slots,
                window_indices,
                window_cache,
                window_cache_scale,
                compressed_cache,
                compressed_cache_scale,
                topk_indices,
                partial,
                num_tokens,
            )
        reducer(
            partial,
            output_window,
            output_arrived,
            output,
            group_base,
            tp_rank,
            num_tokens,
            attention_epoch,
        )
        return output

    return prefill_attn_c1a_reindex_impl


prefill_attn_c1a_reindex = make_prefill_attn_c1a_reindex(reindex_paged_indexer)


@pl.jit
def prefill_attn_c1a_full_test(
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
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            pl.BF16,
        ]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale: pl.InOut[
        pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP],
            pl.FP8E8M0,
        ]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.Out[pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8]],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    return prefill_attn_c1a_full(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos,
        compressed_rope_sin, compressor_wkv, compressor_norm_weight,
        compressed_slots, index_wk, index_norm_weight, index_wq_b,
        index_wq_b_scale, index_weights_proj, topk_indices, candidate_mask,
        output_window, output_arrived, output, 0, tp_rank, num_tokens, 1,
    )


@pl.jit.host
def l3_prefill_attn_c1a_full_test(
    x: pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TP_SIZE, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[
        [TP_SIZE, Q_LORA // 32, LOCAL_H * HEAD_DIM],
        pl.FP8E8M0,
    ],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TP_SIZE, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[
        [TP_SIZE, LOCAL_O_WIDTH // 32, D],
        pl.FP8E8M0,
    ],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[
            [TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            pl.BF16,
        ]
    ],
    compressed_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[
            [TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            pl.BF16,
        ]
    ],
    request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale: pl.InOut[
        pl.Tensor[
            [TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP],
            pl.FP8E8M0,
        ]
    ],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM],
        pl.FP8E8M0,
    ],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.Out[
        pl.Tensor[[TP_SIZE, T_DYN, CMP_POSITIONS_DYN], pl.UINT8]
    ],
    output: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output_window_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
    output_arrived_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        output_window = pld.window(output_window_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        output_arrived = pld.window(output_arrived_buf, [TP_SIZE, 1], dtype=pl.INT32)
        # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = index_wq_b_scale[rank]
        prefill_attn_c1a_full_test(
            x[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank],
            wq_b[rank], wq_b_scale_r, wkv[rank], wkv_scale_r,
            kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank],
            wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank],
            window_indices[rank], window_cache[rank], window_cache_scale[rank],
            compressed_cache[rank], compressed_cache_scale[rank], request_ids[rank],
            compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
            index_block_table[rank], compressed_rope_cos[rank], compressed_rope_sin[rank],
            compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank],
            index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank],
            candidate_mask[rank], output_window, output_arrived, output[rank], rank,
            num_tokens, device=rank,
        )


def build_tensor_specs(token_count=CASE_TOKENS, case_name=CASE_DEFAULT):
    return make_tensor_specs(
        FULL_INPUT_NAMES,
        ("topk_indices", "candidate_mask", "output"),
        token_count,
        case_name,
    )


def golden_prefill_attn_c1a_full_case(tensors):
    apply_distributed_golden("full", golden_prefill_attn_c1a_full, tensors)


# Keep the pre-split names available to the HC orchestration wrapper.
golden_prefill_c1a_full = golden_prefill_attn_c1a_full
prefill_c1a_full = prefill_attn_c1a_full
golden_prefill_c1a_reindex = golden_prefill_attn_c1a_reindex
prefill_c1a_reindex = prefill_attn_c1a_reindex
golden_prefill_c1a_reuse = golden_prefill_attn_c1a_reuse
prefill_c1a_reuse = prefill_attn_c1a_reuse


CACHE_SLOTS = {
    "window_cache": "window_slots",
    "window_cache_scale": "window_slots",
    "compressed_cache": "compressed_slots",
    "compressed_cache_scale": "compressed_slots",
    "index_cache": "compressed_slots",
    "index_cache_scale": "compressed_slots",
}


def cache_compare(world_size, tokens, names=None):
    """Mapped rows must match byte for byte and every other cache row must stay unchanged."""
    options = dict(
        mapping_shape=(world_size, tokens),
        block_size=BLOCK_SIZE,
        leading_rank_axis=True,
        atol=0.0,
        rtol=0.0,
        max_error_ratio=0.0,
    )
    return {
        name: mapped_pool_ratio_allclose(CACHE_SLOTS[name], pool_name=name, **options)
        for name in (CACHE_SLOTS if names is None else names)
    }


def validate(argv=None):
    import argparse

    from pypto.ir import DistributedConfig

    from golden import run
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C1A full validation")
    parser.add_argument("-p", "--platform", default="a5", choices=("a5", "a5sim"))
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(TP_SIZE)))
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=(1, 2, 4))
    parser.add_argument("--dp", type=int, default=1, choices=(1,))
    parser.add_argument("--tokens", type=int, default=CASE_TOKENS, help="token count for the causal case")
    parser.add_argument("--case", default=CASE_DEFAULT, choices=CASE_NAMES)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args(argv)

    if args.tp != TP_SIZE:
        parser.error(f"--tp was parsed as TP{TP_SIZE}, got --tp {args.tp}")
    if args.case == "causal" and not 1 <= args.tokens <= CASE_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {CASE_MAX_TOKENS}]")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")


    result = run(
        fn=l3_prefill_attn_c1a_full_test,
        specs=build_tensor_specs(args.tokens, args.case),
        golden_fn=golden_prefill_attn_c1a_full_case,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config={
            "platform": args.platform,
            "distributed_config": DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            "dump_passes": args.dump_passes,
            "enable_chip_swimlane": args.enable_chip_swimlane,
        },
        compare_fn={
            "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            **cache_compare(len(device_ids), fixture_token_count(args.tokens, args.case)),
            "topk_indices": topk_indices_compare("full"),
        },
    )
    return result


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
