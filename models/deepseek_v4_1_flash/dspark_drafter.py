# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 DSpark projection, metadata, and three-layer drafter."""

from dataclasses import dataclass
import pypto.language as pl
import pypto.language.distributed as pld
import torch

import config as C
from config import BLOCK_SIZE, B_DYN, FLASH, TP_SIZE, T_DYN, TABLE_DYN
from o_proj import prefill_tp_output_all_reduce
from rmsnorm import golden_rms_norm_tiled, rms_norm_tiled
from hc import hc_head, hc_post, hc_pre, zero_bf16_padding
from expert_routed import (
    MX_PACKED_LANE_COLS,
    MX_W1_PACKED_ROWS,
    MX_W2_PACKED_ROWS,
    MX_W3_PACKED_ROWS,
)
from input_pack import pack_x_hc
from utils import paged_slots
from prefill_moe import AUX_WIDTH, N_LOCAL_EXPERTS, RECV_MAX, ROUTE_WIDTH, SIGNAL_PAD, prefill_moe

from dspark_attention import dspark_attention
from dspark_context_kv import dspark_context_kv
from config import (
    DSPARK_DRAFT_LAYERS,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_WIDTH,
    DSPARK_SWA_INDEX_WIDTH,
)


D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
HC_DIM = FLASH.hc_dim
MIX_HC = FLASH.mix_hc
HEAD_DIM = FLASH.head_dim
ROPE_DIM = FLASH.qk_rope_head_dim
Q_LORA = FLASH.q_lora_rank
LOCAL_H = FLASH.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = FLASH.o_groups // TP_SIZE
O_LORA = FLASH.o_lora_rank
O_GROUP_IN = FLASH.num_attention_heads * HEAD_DIM // FLASH.o_groups
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA
VOCAB = FLASH.vocab_size
N_EXPERTS = FLASH.n_routed_experts
MOE_INTER = FLASH.moe_intermediate_size
MX_GROUP = FLASH.weight_block_size[0]
TOPK = FLASH.num_experts_per_tok

MAIN_IN = DSPARK_DRAFT_LAYERS * D
DSPARK_WINDOW = FLASH.sliding_window
DSPARK_QUERY_T_DYN = pl.dynamic("V41_DSPARK_QUERY_T_DYN")
DSPARK_CHUNK_REQUESTS = C.MOE_TOKENS // DSPARK_QUERY_WIDTH
DSPARK_CHUNK_TOKENS = DSPARK_CHUNK_REQUESTS * DSPARK_QUERY_WIDTH
assert DSPARK_CHUNK_REQUESTS > 0
def golden_main_projection(
    target_hidden: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Combine checkpoint-selected target layers in their declared order."""
    if target_hidden.ndim != 3 or target_hidden.shape[1] != DSPARK_DRAFT_LAYERS:
        raise ValueError("target_hidden must be [tokens, three target layers, hidden]")
    flat = target_hidden.flatten(-2)
    if weight.shape != (flat.shape[-1], target_hidden.shape[-1]):
        raise ValueError("main projection weight must be [3 * hidden, hidden]")
    projected = torch.matmul(flat.float(), weight.float()).to(flat.dtype)
    return golden_rms_norm_tiled(projected, norm_weight)


def golden_query_block(
    anchor_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    noise_token_id: int,
    num_speculative_tokens: int = DSPARK_QUERY_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor-first block: every query row predicts one draft token."""
    if anchor_ids.ndim != 1 or anchor_positions.shape != anchor_ids.shape:
        raise ValueError("anchors and positions must be one row per request")
    if num_speculative_tokens <= 0:
        raise ValueError("num_speculative_tokens must be positive")
    ids = anchor_ids[:, None].expand(-1, num_speculative_tokens).clone()
    ids[:, 1:] = noise_token_id
    positions = anchor_positions[:, None] + torch.arange(
        num_speculative_tokens, device=anchor_positions.device
    )
    return ids, positions


def golden_anchor_ids(
    num_sampled: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
) -> torch.Tensor:
    if (
        num_sampled.ndim != 1
        or last_sampled.shape != num_sampled.shape
        or next_prefill_tokens.shape != num_sampled.shape
    ):
        raise ValueError("draft state must have one entry per request")
    return torch.where(num_sampled > 0, last_sampled, next_prefill_tokens)


@dataclass(frozen=True)
class DSparkQueryMetadata:
    token_ids: torch.Tensor
    positions: torch.Tensor
    query_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lens: torch.Tensor


def golden_query_metadata(
    anchor_ids: torch.Tensor,
    first_query_positions: torch.Tensor,
    block_tables: torch.Tensor,
    noise_token_id: int,
    window_size: int = FLASH.sliding_window,
) -> DSparkQueryMetadata:
    """Build per-layer paged slots and non-causal visibility for one draft block."""
    if block_tables.ndim != 3 or block_tables.shape[:2] != (DSPARK_DRAFT_LAYERS, anchor_ids.numel()):
        raise ValueError("block tables must be [three draft layers, batch, pages]")
    if first_query_positions.shape != anchor_ids.shape or bool((first_query_positions < 0).any()):
        raise ValueError("first query positions must be nonnegative and match anchors")
    if window_size <= 0:
        raise ValueError("window size must be positive")
    token_ids, positions = golden_query_block(anchor_ids, first_query_positions, noise_token_id)
    batch, width = positions.shape
    index_width = (window_size + width + 63) // 64 * 64
    query_slots = torch.empty(DSPARK_DRAFT_LAYERS, batch, width, dtype=torch.int64, device=positions.device)
    window_indices = torch.full(
        (DSPARK_DRAFT_LAYERS, batch, width, index_width),
        -1,
        dtype=torch.int32,
        device=positions.device,
    )
    window_lens = torch.empty(DSPARK_DRAFT_LAYERS, batch, dtype=torch.int32, device=positions.device)
    for layer in range(DSPARK_DRAFT_LAYERS):
        table = block_tables[layer]
        for request in range(batch):
            prefix_end = int(first_query_positions[request])
            context_positions = torch.arange(
                max(0, prefix_end - window_size), prefix_end, device=positions.device
            )
            visible_positions = torch.cat((context_positions, positions[request]))
            token_to_req_indices = torch.full_like(visible_positions, request)
            visible_slots = paged_slots(visible_positions, token_to_req_indices, table)
            window_lens[layer, request] = visible_slots.numel()
            window_indices[layer, request, :, : visible_slots.numel()] = visible_slots.to(torch.int32)
            query_slots[layer, request] = visible_slots[-width:]
    return DSparkQueryMetadata(token_ids, positions, query_slots, window_indices, window_lens)


@pl.jit.inline(auto_scope=False)
def dspark_proj(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project the ordered target taps and normalize only active token rows."""
    tokens = pl.tensor.dim(target_hidden, 0)
    projected = pl.create_tensor([tokens, D], dtype=pl.BF16)
    for block in pl.spmd((num_tokens + 15) // 16 * (D // 128), name_hint="dspark_main_proj"):
        row = block // (D // 128) * 16
        col = block % (D // 128) * 128
        rows = pl.min(16, num_tokens - row)
        acc = pl.create_tensor([16, 128], dtype=pl.FP32)
        for k in pl.range(0, MAIN_IN, 256):
            source = pl.slice(target_hidden, [16, 256], [row, k], valid_shape=[rows, 256])
            weight = pl.slice(main_proj_weight, [256, 128], [k, col])
            acc = pl.matmul_acc(acc, source, weight, init_cond=(k == 0))
        projected[row : row + 16, col : col + 128] = pl.set_validshape(
            pl.cast(acc, pl.BF16, mode="rint"), rows, 128
        )
    for block in pl.spmd((num_tokens + 7) // 8, name_hint="dspark_main_norm"):
        rms_norm_tiled(projected, main_norm_weight, main_x, block, num_tokens, output_fp32=False)
    return main_x


@pl.jit
def dspark_proj_test(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    main_x: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    target_hidden.bind_dynamic(0, T_DYN)
    main_x.bind_dynamic(0, T_DYN)
    return dspark_proj(target_hidden, main_proj_weight, main_norm_weight, main_x, num_tokens)


@pl.jit.inline(auto_scope=False)
def dspark_metadata(
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    query_ids: pl.Tensor[[T_DYN], pl.INT64],
    query_positions: pl.Tensor[[T_DYN], pl.INT32],
    query_slots: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_lens: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN], pl.INT32],
):
    batch = pl.tensor.dim(anchor_ids, 0)
    for worker in pl.spmd(1, name_hint="dspark_query_metadata"):
        for request_offset in pl.range(batch):
            request = worker + request_offset
            first = pl.read(first_query_positions, [request])
            anchor = pl.read(anchor_ids, [request])
            prefix_start = pl.max(first - DSPARK_WINDOW, 0)
            context_len = first - prefix_start
            visible_len = context_len + DSPARK_QUERY_WIDTH
            for offset in pl.range(DSPARK_QUERY_WIDTH):
                token = request * DSPARK_QUERY_WIDTH + offset
                token_id = pl.cast(DSPARK_NOISE_TOKEN_ID, pl.INT64)
                if offset == 0:
                    token_id = anchor
                pl.write(query_ids, [token], token_id)
                pl.write(query_positions, [token], pl.cast(first + offset, pl.INT32))
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(window_lens, [layer, request], pl.cast(visible_len, pl.INT32))
                for offset in pl.range(DSPARK_SWA_INDEX_WIDTH):
                    slot = pl.cast(-1, pl.INT32)
                    if offset < visible_len:
                        position = prefix_start + offset
                        logical_block = position // BLOCK_SIZE
                        physical_block = pl.read(
                            block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)]
                        )
                        slot = pl.cast(physical_block * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT32)
                    for query in pl.range(DSPARK_QUERY_WIDTH):
                        token = request * DSPARK_QUERY_WIDTH + query
                        pl.write(window_indices, [layer, token, offset], slot)
                        if offset == context_len + query:
                            pl.write(query_slots, [layer, token], pl.cast(slot, pl.INT64))
    return query_ids, query_positions, query_slots, window_indices, window_lens


@pl.jit.inline
def select_dspark_anchor_ids(
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
):
    batch = pl.tensor.dim(num_sampled, 0)
    for request in pl.spmd(batch, name_hint="dspark_select_anchor"):
        anchor = pl.read(next_prefill_tokens, [request])
        if pl.read(num_sampled, [request]) > 0:
            anchor = pl.read(last_sampled, [request])
        pl.write(anchor_ids, [request], anchor)
    return anchor_ids


@pl.jit.inline(auto_scope=False)
def draft_layer(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    hc_attn_fn: pl.Tensor,
    hc_attn_scale: pl.Tensor,
    hc_attn_base: pl.Tensor,
    attn_norm_weight: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor,
    kv_norm_weight: pl.Tensor,
    attn_sink: pl.Tensor,
    wo_a: pl.Tensor,
    wo_a_scale: pl.Tensor,
    wo_b: pl.Tensor,
    wo_b_scale: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor,
    window_cache_scale: pl.Tensor,
    hc_ffn_fn: pl.Tensor,
    hc_ffn_scale: pl.Tensor,
    hc_ffn_base: pl.Tensor,
    ffn_norm_weight: pl.Tensor,
    gate_weight: pl.Tensor,
    correction_bias: pl.Tensor,
    routed_w1: pl.Tensor,
    routed_w1_scale: pl.Tensor,
    routed_w2: pl.Tensor,
    routed_w2_scale: pl.Tensor,
    routed_w3: pl.Tensor,
    routed_w3_scale: pl.Tensor,
    mxfp4_pair_lut: pl.Tensor,
    shared_w1: pl.Tensor,
    shared_w1_scale: pl.Tensor,
    shared_w2: pl.Tensor,
    shared_w2_scale: pl.Tensor,
    shared_w3: pl.Tensor,
    shared_w3_scale: pl.Tensor,
    next_pre_mix: pl.Tensor,
    x_mixed: pl.Tensor,
    x_next: pl.Tensor,
    attention_output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    attention_output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[C.EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one V4.1 draft decoder layer with delayed mHC boundaries."""
    tokens = pl.tensor.dim(x_hc, 0)
    attention_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normalized_attention = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attention_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, incoming_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        attention_input, attention_pre_mix, post_mix, residual_mix,
    )
    for block in pl.spmd((tokens + 7) // 8, name_hint="dspark_attn_norm"):
        rms_norm_tiled(attention_input, attn_norm_weight, normalized_attention, block, tokens, output_fp32=False)
    zero_bf16_padding(normalized_attention, num_tokens)
    attention_partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
    dspark_attention(
        normalized_attention,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_a_scale,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        attention_partial,
        num_tokens,
    )
    attention_output = pl.create_tensor([tokens, D], dtype=pl.BF16)
    group_base = rank // TP_SIZE * TP_SIZE
    tp_rank = rank % TP_SIZE
    prefill_tp_output_all_reduce(
        attention_partial,
        attention_output_window,
        attention_output_arrived,
        attention_output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    zero_bf16_padding(attention_output, num_tokens)
    attention_hidden = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden, deps=hc_post_deps)
    prefill_moe(
        attention_hidden,
        attention_pre_mix,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        ffn_norm_weight,
        gate_weight,
        correction_bias,
        routed_w1,
        routed_w1_scale,
        routed_w2,
        routed_w2_scale,
        routed_w3,
        routed_w3_scale,
        mxfp4_pair_lut,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_w3,
        shared_w3_scale,
        next_pre_mix,
        x_mixed,
        x_next,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        num_tokens,
        rank,
        moe_epoch,
    )
    return x_next, next_pre_mix


def _dspark_drafter(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    context_slots: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    context_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    context_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_cos: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN // MX_GROUP, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    window_cache: pl.InOut[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[
            [DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // MX_GROUP], pl.BF16
        ]
    ],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w1_scale: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0
    ],
    routed_w2: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w2_scale: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0
    ],
    routed_w3: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w3_scale: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0
    ],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    head_hidden: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    attention_output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    attention_output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[C.EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Compose three draft layers and return five head rows per request.

    The caller supplies all weights and scales with a leading draft-layer axis,
    plus per-layer paged caches and target context taps.
    Complete requests are processed in physical MoE-sized chunks.
    """
    target_hidden.bind_dynamic(0, T_DYN)
    context_slots.bind_dynamic(1, T_DYN)
    context_rope_cos.bind_dynamic(0, T_DYN)
    context_rope_sin.bind_dynamic(0, T_DYN)
    query_rope_cos.bind_dynamic(0, DSPARK_QUERY_T_DYN)
    query_rope_sin.bind_dynamic(0, DSPARK_QUERY_T_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    first_query_positions.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, TABLE_DYN)
    window_cache.bind_dynamic(1, C.ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(1, C.ORI_BLOCKS_DYN)
    head_hidden.bind_dynamic(0, B_DYN)
    context_tokens = pl.tensor.dim(target_hidden, 0)
    batch = pl.tensor.dim(num_sampled, 0)
    query_tokens = batch * DSPARK_QUERY_WIDTH
    main_x = pl.create_tensor([context_tokens, D], dtype=pl.BF16)
    dspark_proj(target_hidden, main_proj_weight, main_norm_weight, main_x, context_tokens)

    anchor_ids = pl.create_tensor([batch], dtype=pl.INT64)
    select_dspark_anchor_ids(num_sampled, last_sampled, next_prefill_tokens, anchor_ids)
    query_ids = pl.create_tensor([query_tokens], dtype=pl.INT64)
    query_positions = pl.create_tensor([query_tokens], dtype=pl.INT32)
    flat_head_hidden = pl.reshape(head_hidden, [query_tokens, D])
    query_slots = pl.create_tensor([DSPARK_DRAFT_LAYERS, query_tokens], dtype=pl.INT64)
    window_indices = pl.create_tensor(
        [DSPARK_DRAFT_LAYERS, query_tokens, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32
    )
    window_lens = pl.create_tensor([DSPARK_DRAFT_LAYERS, batch], dtype=pl.INT32)
    dspark_metadata(
        anchor_ids,
        first_query_positions,
        block_tables,
        query_ids,
        query_positions,
        query_slots,
        window_indices,
        window_lens,
    )

    for layer in pl.unroll(DSPARK_DRAFT_LAYERS):
        layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[layer]
        dspark_context_kv(
            main_x,
            wkv[layer],
            layer_wkv_scale,
            kv_norm_weight[layer],
            context_rope_cos,
            context_rope_sin,
            context_slots[layer],
            window_cache[layer],
            window_cache_scale[layer],
            context_tokens,
        )

    for chunk in pl.range((batch + DSPARK_CHUNK_REQUESTS - 1) // DSPARK_CHUNK_REQUESTS):
        first_token = chunk * DSPARK_CHUNK_TOKENS
        active_tokens = pl.min(DSPARK_CHUNK_TOKENS, query_tokens - first_token)
        padded_ids = pl.create_tensor([C.MOE_TOKENS], dtype=pl.INT64)
        padded_slots = pl.create_tensor([DSPARK_DRAFT_LAYERS, C.MOE_TOKENS], dtype=pl.INT64)
        padded_indices = pl.create_tensor(
            [DSPARK_DRAFT_LAYERS, C.MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32
        )
        rope_cos = pl.create_tensor([C.MOE_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
        rope_sin = pl.create_tensor([C.MOE_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
        for token in pl.spmd(C.MOE_TOKENS, name_hint="dspark_pad_queries"):
            token_id = pl.cast(0, pl.INT64)
            if token < active_tokens:
                source = first_token + token
                token_id = pl.read(query_ids, [source])
                rope_cos[token : token + 1, :] = query_rope_cos[source : source + 1, :]
                rope_sin[token : token + 1, :] = query_rope_sin[source : source + 1, :]
            pl.write(padded_ids, [token], token_id)
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                slot = pl.cast(-1, pl.INT64)
                if token < active_tokens:
                    slot = pl.read(query_slots, [layer, first_token + token])
                pl.write(padded_slots, [layer, token], slot)
                for offset in pl.range(DSPARK_SWA_INDEX_WIDTH):
                    index = pl.cast(-1, pl.INT32)
                    if token < active_tokens:
                        index = pl.read(window_indices, [layer, first_token + token, offset])
                    pl.write(padded_indices, [layer, token, offset], index)

        x_hc = pl.create_tensor([C.MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        pack_x_hc(padded_ids, embedding_weight, x_hc)
        pre_mix = pl.create_tensor([C.MOE_TOKENS, HC_MULT], dtype=pl.FP32)
        for token in pl.spmd(C.MOE_TOKENS, name_hint="dspark_initial_pre_mix"):
            for lane in pl.range(HC_MULT):
                value = pl.cast(0.0, pl.FP32)
                if lane == 0:
                    value = pl.cast(1.0, pl.FP32)
                pl.write(pre_mix, [token, lane], value)

        for layer in pl.unroll(DSPARK_DRAFT_LAYERS):
            layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[
                layer
            ]
            layer_wq_b_scale: pl.Tensor[
                [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
            ] = wq_b_scale[layer]
            layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[
                layer
            ]
            layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = (
                wo_b_scale[layer]
            )
            layer_routed_w1_scale: pl.Tensor[
                [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w1_scale[layer]
            layer_routed_w2_scale: pl.Tensor[
                [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w2_scale[layer]
            layer_routed_w3_scale: pl.Tensor[
                [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w3_scale[layer]
            layer_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = (
                shared_w1_scale[layer]
            )
            layer_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = (
                shared_w2_scale[layer]
            )
            layer_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = (
                shared_w3_scale[layer]
            )
            next_pre_mix = pl.create_tensor([C.MOE_TOKENS, HC_MULT], dtype=pl.FP32)
            x_mixed = pl.create_tensor([C.MOE_TOKENS, D], dtype=pl.BF16)
            x_next = pl.create_tensor([C.MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
            draft_layer(
                x_hc,
                pre_mix,
                hc_attn_fn[layer],
                hc_attn_scale[layer],
                hc_attn_base[layer],
                attn_norm_weight[layer],
                wq_a[layer],
                layer_wq_a_scale,
                q_norm_weight[layer],
                wq_b[layer],
                layer_wq_b_scale,
                wkv[layer],
                layer_wkv_scale,
                kv_norm_weight[layer],
                attn_sink[layer],
                wo_a[layer],
                wo_a_scale[layer],
                wo_b[layer],
                layer_wo_b_scale,
                rope_cos,
                rope_sin,
                padded_slots[layer],
                padded_indices[layer],
                window_cache[layer],
                window_cache_scale[layer],
                hc_ffn_fn[layer],
                hc_ffn_scale[layer],
                hc_ffn_base[layer],
                ffn_norm_weight[layer],
                gate_weight[layer],
                correction_bias[layer],
                routed_w1[layer],
                layer_routed_w1_scale,
                routed_w2[layer],
                layer_routed_w2_scale,
                routed_w3[layer],
                layer_routed_w3_scale,
                mxfp4_pair_lut,
                shared_w1[layer],
                layer_shared_w1_scale,
                shared_w2[layer],
                layer_shared_w2_scale,
                shared_w3[layer],
                layer_shared_w3_scale,
                next_pre_mix,
                x_mixed,
                x_next,
                attention_output_window,
                attention_output_arrived,
                recv_meta,
                recv_x,
                recv_scale,
                recv_weights,
                recv_routes,
                arrived,
                data_arrived,
                routed_output,
                combine_arrived,
                active_tokens,
                rank,
                attention_epoch + chunk * DSPARK_DRAFT_LAYERS + layer,
                moe_epoch + chunk * DSPARK_DRAFT_LAYERS + layer,
            )
            x_hc = x_next
            pre_mix = next_pre_mix

        padded_head = pl.create_tensor([C.MOE_TOKENS, D], dtype=pl.BF16)
        hc_head(x_hc, pre_mix, padded_head)
        for row in pl.spmd(active_tokens, name_hint="dspark_head_unpad"):
            flat_head_hidden[first_token + row : first_token + row + 1, :] = padded_head[row : row + 1, :]
    return head_hidden, window_cache, window_cache_scale


dspark_drafter = pl.jit.inline(auto_scope=False)(_dspark_drafter)
l2_dspark_drafter = pl.jit(auto_scope=False)(_dspark_drafter)


@pl.jit.host
def l3_dspark_drafter(
    target_hidden: pl.Tensor[[C.EP_SIZE, T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[C.EP_SIZE, MAIN_IN, D], pl.BF16],
    main_norm_weight: pl.Tensor[[C.EP_SIZE, D], pl.BF16],
    num_sampled: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT32],
    block_tables: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    context_slots: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    context_rope_cos: pl.Tensor[[C.EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    context_rope_sin: pl.Tensor[[C.EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_cos: pl.Tensor[[C.EP_SIZE, DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[C.EP_SIZE, DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    embedding_weight: pl.Tensor[[C.EP_SIZE, VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[
        [C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0
    ],
    wkv: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[
        [C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN // MX_GROUP, O_LORA], pl.FP8E8M0
    ],
    wo_b: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    window_cache: pl.InOut[
        pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[
            [C.EP_SIZE, DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // MX_GROUP],
            pl.BF16,
        ]
    ],
    hc_ffn_fn: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[
        [C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w1_scale: pl.Tensor[
        [C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0
    ],
    routed_w2: pl.Tensor[
        [C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w2_scale: pl.Tensor[
        [C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0
    ],
    routed_w3: pl.Tensor[
        [C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w3_scale: pl.Tensor[
        [C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0
    ],
    mxfp4_pair_lut: pl.Tensor[[C.EP_SIZE, 2, 256], pl.INT16],
    shared_w1: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    head_hidden: pl.Out[pl.Tensor[[C.EP_SIZE, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    attention_epoch: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    target_hidden.bind_dynamic(1, T_DYN)
    context_slots.bind_dynamic(2, T_DYN)
    context_rope_cos.bind_dynamic(1, T_DYN)
    context_rope_sin.bind_dynamic(1, T_DYN)
    query_rope_cos.bind_dynamic(1, DSPARK_QUERY_T_DYN)
    query_rope_sin.bind_dynamic(1, DSPARK_QUERY_T_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    first_query_positions.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, B_DYN)
    block_tables.bind_dynamic(3, TABLE_DYN)
    window_cache.bind_dynamic(2, C.ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(2, C.ORI_BLOCKS_DYN)
    head_hidden.bind_dynamic(1, B_DYN)

    attention_output_buf = pld.alloc_window_buffer([C.PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    attention_arrived_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([C.EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
    recv_weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([C.EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
    routed_output_buf = pld.alloc_window_buffer([C.MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        attention_output = pld.window(attention_output_buf, [C.PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        attention_arrived = pld.window(attention_arrived_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [C.EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [C.EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buf, [C.MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(
            combine_arrived_buf, [C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32
        )
        l2_dspark_drafter(
            target_hidden[rank],
            main_proj_weight[rank],
            main_norm_weight[rank],
            num_sampled[rank],
            last_sampled[rank],
            next_prefill_tokens[rank],
            first_query_positions[rank],
            block_tables[rank],
            context_slots[rank],
            context_rope_cos[rank],
            context_rope_sin[rank],
            query_rope_cos[rank],
            query_rope_sin[rank],
            embedding_weight[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank],
            wq_a_scale[rank],
            q_norm_weight[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            wkv_scale[rank],
            kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_a_scale[rank],
            wo_b[rank],
            wo_b_scale[rank],
            window_cache[rank],
            window_cache_scale[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            ffn_norm_weight[rank],
            gate_weight[rank],
            correction_bias[rank],
            routed_w1[rank],
            routed_w1_scale[rank],
            routed_w2[rank],
            routed_w2_scale[rank],
            routed_w3[rank],
            routed_w3_scale[rank],
            mxfp4_pair_lut[rank],
            shared_w1[rank],
            shared_w1_scale[rank],
            shared_w2[rank],
            shared_w2_scale[rank],
            shared_w3[rank],
            shared_w3_scale[rank],
            head_hidden[rank],
            attention_output,
            attention_arrived,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            rank,
            attention_epoch,
            moe_epoch,
            device=rank,
        )


@pl.jit
def dspark_metadata_test(
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    query_ids: pl.Out[pl.Tensor[[T_DYN], pl.INT64]],
    query_positions: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
    query_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN, DSPARK_SWA_INDEX_WIDTH], pl.INT32]],
    window_lens: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN], pl.INT32]],
):
    anchor_ids.bind_dynamic(0, B_DYN)
    first_query_positions.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, TABLE_DYN)
    query_ids.bind_dynamic(0, T_DYN)
    query_positions.bind_dynamic(0, T_DYN)
    query_slots.bind_dynamic(1, T_DYN)
    window_indices.bind_dynamic(1, T_DYN)
    window_lens.bind_dynamic(1, B_DYN)
    return dspark_metadata(
        anchor_ids,
        first_query_positions,
        block_tables,
        query_ids,
        query_positions,
        query_slots,
        window_indices,
        window_lens,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark drafter validation")
    parser.add_argument("--mode", choices=("projection", "metadata", "all"), default="all")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.mode in ("projection", "all"):
        import torch

        from golden import ScalarSpec, TensorSpec, run

        torch.manual_seed(41)
        tokens = 6
        hidden = torch.randn(tokens, MAIN_IN, dtype=torch.bfloat16) * 0.1
        weight = (torch.randn(MAIN_IN, D) * 0.01).to(torch.bfloat16)

        def golden(values):
            values["main_x"].copy_(
                golden_main_projection(
                    values["target_hidden"].reshape(tokens, DSPARK_DRAFT_LAYERS, D),
                    values["main_proj_weight"],
                    values["main_norm_weight"],
                )
            )

        specs = [
            TensorSpec("target_hidden", [tokens, MAIN_IN], torch.bfloat16, init_value=hidden),
            TensorSpec("main_proj_weight", [MAIN_IN, D], torch.bfloat16, init_value=weight),
            TensorSpec("main_norm_weight", [D], torch.bfloat16, init_value=1.0),
            TensorSpec("main_x", [tokens, D], torch.bfloat16),
            ScalarSpec("num_tokens", torch.int32, tokens, compile_runtime=True),
        ]
        result = run(
            fn=dspark_proj_test,
            specs=specs,
            golden_fn=golden,
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode,
                "device_id": args.device,
            },
            rtol=0.05,
            atol=0.05,
            compile_only=args.compile_only,
        )
        print(result)
        if not result.passed:
            raise SystemExit(1)

    if args.mode in ("metadata", "all"):
        import torch

        from golden import TensorSpec, run

        batch = 1
        tokens = batch * DSPARK_QUERY_WIDTH
        anchors = torch.tensor([7], dtype=torch.int64)
        positions = torch.tensor([125], dtype=torch.int32)
        tables = torch.tensor(
            [[[4, 5]], [[14, 15]], [[24, 25]]],
            dtype=torch.int32,
        )

        def golden(values):
            expected = golden_query_metadata(
                values["anchor_ids"],
                values["first_query_positions"],
                values["block_tables"],
                DSPARK_NOISE_TOKEN_ID,
            )
            values["query_ids"].copy_(expected.token_ids.reshape(-1))
            values["query_positions"].copy_(expected.positions.reshape(-1))
            values["query_slots"].copy_(expected.query_slots.reshape(DSPARK_DRAFT_LAYERS, tokens))
            values["window_indices"].copy_(
                expected.window_indices.reshape(DSPARK_DRAFT_LAYERS, tokens, DSPARK_SWA_INDEX_WIDTH)
            )
            values["window_lens"].copy_(expected.window_lens)

        specs = [
            TensorSpec("anchor_ids", [batch], torch.int64, init_value=anchors),
            TensorSpec("first_query_positions", [batch], torch.int32, init_value=positions),
            TensorSpec("block_tables", [DSPARK_DRAFT_LAYERS, batch, 2], torch.int32, init_value=tables),
            TensorSpec("query_ids", [tokens], torch.int64),
            TensorSpec("query_positions", [tokens], torch.int32),
            TensorSpec("query_slots", [DSPARK_DRAFT_LAYERS, tokens], torch.int64),
            TensorSpec("window_indices", [DSPARK_DRAFT_LAYERS, tokens, DSPARK_SWA_INDEX_WIDTH], torch.int32),
            TensorSpec("window_lens", [DSPARK_DRAFT_LAYERS, batch], torch.int32),
        ]
        result = run(
            fn=dspark_metadata_test,
            specs=specs,
            golden_fn=golden,
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode,
                "device_id": args.device,
            },
            compile_only=args.compile_only,
        )
        print(result)
        if not result.passed:
            raise SystemExit(1)
