# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
"""Compose the three-layer DeepSeek-V4-Flash DSpark drafter with DSA-CP attention."""

import sys

import config


def _parse_parallel_arg(name: str, default: int) -> tuple[int, bool]:
    flag = f"--{name}"
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1]), True
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1]), True
    return default, False


config.TP, _tp_explicit = _parse_parallel_arg("tp", 4)
config.EP, _ep_explicit = _parse_parallel_arg("ep", 4)
_caller_argv_length = len(sys.argv)
if not _tp_explicit:
    sys.argv.extend(("--tp", str(config.TP)))
if not _ep_explicit:
    sys.argv.extend(("--ep", str(config.EP)))

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import (
    BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FLASH as M,
    KV_ORI_BLOCK_NUM,
    MOE_TOKENS,
    PREFILL_TOKENS,
    TP,
)
from dspark_attention import dspark_attention
from dspark_context_kv import dspark_context_kv
from dspark_proj import dspark_proj
from decode_o_proj import (
    ATTENTION_PUBLISH_WORKERS,
    ATTENTION_WINDOW_ROWS,
    COMM_ROW_TILE,
    GROUP_T_PAD,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    LOCAL_T_PAD,
    O_WINDOW_ROWS,
    TP_SIZE,
    o_group_a2a,
    o_proj_reduce_scatter,
)
from hc_head import hc_head
from hc_post import hc_post_prefill
from hc_pre import hc_pre
from lookup_embedding import lookup_embedding
from rmsnorm import rms_norm
from moe import (
    AUX_PAD,
    IDX_PAD,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    TOPK,
    VOCAB,
    clear_moe_signals,
    moe,
)
from prefill_cp_token_allgather import (
    PREFILL_GROUP_CAP,
    TP_SIZE as PREFILL_CP_SIZE,
    prefill_cp_token_allgather_step,
)
del sys.argv[_caller_argv_length:]

T_MAIN_DYN = pl.dynamic("DSPARK_BACKBONE_T_MAIN_DYN")
CP_CONTEXT_T_DYN = pl.dynamic("DSPARK_BACKBONE_CP_CONTEXT_T_DYN")
B_DYN = pl.dynamic("DSPARK_BACKBONE_B_DYN")
PREPARE_T_MAIN_DYN = pl.dynamic("DSPARK_PREPARE_T_MAIN_DYN")
PREPARE_B_DYN = pl.dynamic("DSPARK_PREPARE_B_DYN")
METADATA_B_DYN = pl.dynamic("DSPARK_METADATA_B_DYN")

# DSpark program contract.
DSPARK_DRAFT_LAYERS = 3
DSPARK_QUERY_WIDTH = 7
DSPARK_QUERY_PAD = 8
DSPARK_NOISE_TOKEN_ID = 128799
DSPARK_SUPPORTED_BATCHES = (4, 8, 12, 16)
DSPARK_MAX_BATCH = max(DSPARK_SUPPORTED_BATCHES)
# Each rank owns complete seven-row request blocks; a CP group therefore carries
# exactly DSPARK_CP_SIZE times one supported local batch.
DSPARK_QUERY_TOKENS = DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH
DSPARK_MOE_TOKENS = DSPARK_MAX_BATCH * DSPARK_QUERY_PAD
DSPARK_SWA_INDEX_WIDTH = (M.sliding_window + DSPARK_QUERY_WIDTH + 63) // 64 * 64
DSPARK_CP_SIZE = PREFILL_CP_SIZE

assert DECODE_SEQ == 1 + DSPARK_QUERY_WIDTH
assert DSPARK_QUERY_WIDTH < DSPARK_QUERY_PAD
assert DSPARK_MAX_BATCH == DECODE_BATCH // TP
assert DSPARK_MOE_TOKENS == MOE_TOKENS
assert DSPARK_NOISE_TOKEN_ID < M.vocab_size
assert DSPARK_SWA_INDEX_WIDTH >= M.sliding_window + DSPARK_QUERY_WIDTH
assert DSPARK_QUERY_TOKENS <= LOCAL_T_PAD
assert TP_SIZE == TP
assert DSPARK_CP_SIZE == TP_SIZE
assert N_RANKS % DSPARK_CP_SIZE == 0
assert PREFILL_TOKENS % DSPARK_CP_SIZE == 0
assert DSPARK_CP_SIZE * DSPARK_QUERY_TOKENS <= PREFILL_GROUP_CAP
assert (
    DSPARK_CP_SIZE
    * DSPARK_MAX_BATCH
    * ((M.sliding_window + DSPARK_QUERY_WIDTH + BLOCK_SIZE - 1) // BLOCK_SIZE)
    <= KV_ORI_BLOCK_NUM
)

# model config
T = DSPARK_MOE_TOKENS
T_QUERY = DSPARK_QUERY_TOKENS
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
MAIN_IN = DSPARK_DRAFT_LAYERS * D

WIN = M.sliding_window
PAD_D_TILE = 512

# Three draft layers plus their MoE communication graph exceed the runtime's
# default per-ring heap. Match the established large-model harness allocation.
_DSPARK_RING_HEAP = (4 * 1024 * 1024 * 1024,) * 4


@pl.jit.inline
def prepare_dspark_inputs(
    target_hidden: pl.Tensor[[PREPARE_T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[PREPARE_B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[PREPARE_B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[PREPARE_B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    main_x: pl.Tensor[[PREPARE_T_MAIN_DYN, D], pl.BF16],
    query_token_ids: pl.Tensor[[DSPARK_MOE_TOKENS], pl.INT64],
    query_hc_flat: pl.Tensor[[DSPARK_MOE_TOKENS, HC_MULT * D], pl.FP32],
):
    dspark_proj(target_hidden, main_proj_weight, main_norm_weight, main_x)

    batch = pl.tensor.dim(num_sampled, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_query_ids"):
        for token in pl.range(DSPARK_MOE_TOKENS):
            token_id = pl.cast(0, pl.INT64)
            if token < active_tokens:
                request = token // DSPARK_QUERY_WIDTH
                query_offset = token % DSPARK_QUERY_WIDTH
                token_id = pl.cast(DSPARK_NOISE_TOKEN_ID, pl.INT64)
                if query_offset == 0:
                    token_id = pl.read(next_prefill_tokens, [request])
                    if pl.read(num_sampled, [request]) > 0:
                        token_id = pl.read(last_sampled, [request])
            pl.write(query_token_ids, [token], token_id)

    lookup_hidden = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
    query_hc = pl.reshape(query_hc_flat, [DSPARK_MOE_TOKENS, HC_MULT, D])
    lookup_embedding(
        query_token_ids,
        embedding_weight,
        lookup_hidden,
        query_hc,
    )
    return main_x, query_token_ids, query_hc_flat

@pl.jit.inline
def build_dspark_metadata(
    anchor_positions: pl.Tensor[[METADATA_B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, METADATA_B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    swa_indices: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH], pl.INT32],
    query_positions: pl.Tensor[[DSPARK_QUERY_TOKENS], pl.INT32],
):
    batch = pl.tensor.dim(anchor_positions, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    for metadata_core in pl.spmd(1, name_hint="dspark_query_metadata"):
        for token in pl.range(metadata_core, DSPARK_QUERY_TOKENS):
            pl.write(query_positions, [token], pl.cast(0, pl.INT32))
            if token < active_tokens:
                request = token // DSPARK_QUERY_WIDTH
                query_offset = token % DSPARK_QUERY_WIDTH
                anchor_position = pl.read(anchor_positions, [request])
                query_position = anchor_position + 1 + query_offset
                pl.write(query_positions, [token], pl.cast(query_position, pl.INT32))

    for request in pl.spmd(DSPARK_MAX_BATCH, name_hint="dspark_visible_metadata"):
        start_position = pl.cast(0, pl.INT32)
        visible_len = pl.cast(0, pl.INT32)
        if request < batch:
            anchor_position = pl.read(anchor_positions, [request])
            prefix_len = anchor_position + 1
            start_position = pl.cast(pl.max(prefix_len - WIN, 0), pl.INT32)
            visible_len = pl.cast(
                prefix_len + DSPARK_QUERY_WIDTH - start_position,
                pl.INT32,
            )
        for layer in pl.range(DSPARK_DRAFT_LAYERS):
            pl.write(swa_lens, [layer, request], visible_len)
            for visible_offset in pl.range(DSPARK_SWA_INDEX_WIDTH):
                visible_slot = pl.cast(-1, pl.INT32)
                if visible_offset < visible_len:
                    visible_position = start_position + visible_offset
                    logical_block = visible_position // BLOCK_SIZE
                    block_offset = visible_position % BLOCK_SIZE
                    physical_block = pl.read(
                        block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)]
                    )
                    visible_slot = pl.cast(
                        physical_block * BLOCK_SIZE + block_offset,
                        pl.INT32,
                    )
                pl.write(swa_indices, [layer, request, visible_offset], visible_slot)
    return (
        swa_indices,
        swa_lens,
        query_positions,
    )


@pl.jit.inline(auto_scope=False)
def draft_layer(
    query_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    draft_layer_index: pl.Scalar[pl.INT32],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    query_positions: pl.Tensor[[T_QUERY], pl.INT32],
    query_group_positions: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT32],
    kv_cache: pl.Tensor[[KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_group_slot_mapping: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT64],
    swa_indices: pl.Tensor[[DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[DSPARK_MAX_BATCH], pl.INT32],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    query_token_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    output_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hidden_gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    hidden_gather_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    dsa_cp_rank: pl.Scalar[pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    active_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    layer_hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [draft_layer_index * MIX_HC, 0])
    layer_hc_attn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [draft_layer_index * 3])
    layer_hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [draft_layer_index * MIX_HC])
    layer_attn_norm_w: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [draft_layer_index * D])
    layer_wq_a: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [draft_layer_index * D, 0])
    layer_wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [draft_layer_index * Q_LORA, 0])
    layer_wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [draft_layer_index * H * HEAD_DIM])
    layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [draft_layer_index * D, 0])
    layer_gamma_cq: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [draft_layer_index * Q_LORA])
    layer_gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [draft_layer_index * HEAD_DIM])
    layer_attn_sink: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [draft_layer_index * H])
    layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [draft_layer_index * LOCAL_O_GROUPS, 0, 0])
    layer_wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8] = pl.slice(wo_b, [D, LOCAL_O_WIDTH], [draft_layer_index * D, 0])
    layer_wo_b_scale: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [draft_layer_index * D])
    layer_hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [draft_layer_index * MIX_HC, 0])
    layer_hc_ffn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [draft_layer_index * 3])
    layer_hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [draft_layer_index * MIX_HC])
    layer_ffn_norm_w: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_w, [D], [draft_layer_index * D])
    layer_gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [draft_layer_index * N_EXPERTS_GLOBAL, 0])
    layer_gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [draft_layer_index * N_EXPERTS_GLOBAL])
    layer_tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [draft_layer_index * VOCAB, 0])
    layer_routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [draft_layer_index * N_LOCAL, 0, 0])
    layer_routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [draft_layer_index * N_LOCAL, 0])
    layer_routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [draft_layer_index * N_LOCAL, 0, 0])
    layer_routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [draft_layer_index * N_LOCAL, 0])
    layer_routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [draft_layer_index * N_LOCAL, 0, 0])
    layer_routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [draft_layer_index * N_LOCAL, 0])
    layer_shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [draft_layer_index * MOE_INTER, 0])
    layer_shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [draft_layer_index * MOE_INTER])
    layer_shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [draft_layer_index * MOE_INTER, 0])
    layer_shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [draft_layer_index * MOE_INTER])
    layer_shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [draft_layer_index * D, 0])
    layer_shared_w2_scale: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [draft_layer_index * D])

    query_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    combine = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        query_hc,
        layer_hc_attn_fn,
        layer_hc_attn_scale,
        layer_hc_attn_base,
        query_mixed,
        post,
        combine,
    )

    query_normed = pl.create_tensor([T_QUERY, D], dtype=pl.BF16)
    query_mixed_active: pl.Tensor[[T_QUERY, D], pl.BF16] = pl.slice(
        query_mixed,
        [T_QUERY, D],
        [0, 0],
    )
    rms_norm(query_mixed_active, layer_attn_norm_w, query_normed)
    query_group = pl.create_tensor(
        [DSPARK_CP_SIZE * T_QUERY, D], dtype=pl.BF16
    )
    with pl.scope():
        _gathered_query, hidden_gather_signal = prefill_cp_token_allgather_step(
            query_normed,
            query_group,
            hidden_gather_window,
            hidden_gather_signal,
            group_base,
            dsa_cp_rank,
        )
    o_packed_heads = pl.create_tensor(
        [O_GROUPS * LOCAL_T_PAD * (H // O_GROUPS), HEAD_DIM],
        dtype=pl.BF16,
    )
    dspark_attention(
        query_normed,
        query_group,
        layer_wq_a, layer_wq_b, layer_wq_b_scale, layer_wkv, layer_gamma_cq, layer_gamma_ckv,
        freqs_cos, freqs_sin, query_positions, query_group_positions,
        kv_cache, query_group_slot_mapping, swa_indices, swa_lens,
        layer_attn_sink, o_packed_heads,
    )
    attention_grouped = pl.reshape(
        o_packed_heads,
        [O_GROUPS * LOCAL_T_PAD, O_GROUP_IN],
    )
    attention_local_flat = pl.create_tensor(
        [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
    )
    # DSA-CP reuses the same physical four-rank group as TP-sharded O projection.
    tp_rank = dsa_cp_rank
    with pl.spmd(
        ATTENTION_PUBLISH_WORKERS,
        name_hint="dspark_o_group_publish",
    ) as attention_publish_tid:
        worker = pl.tile.get_block_idx()
        for global_group in pl.range(worker, O_GROUPS, ATTENTION_PUBLISH_WORKERS):
            destination_rank = global_group // LOCAL_O_GROUPS
            local_group = global_group - destination_rank * LOCAL_O_GROUPS
            source_row = global_group * LOCAL_T_PAD
            target_row = local_group * GROUP_T_PAD + tp_rank * active_tokens
            pld.tensor.put(
                dst=attention_window,
                peer=group_base + destination_rank,
                src=attention_grouped,
                dst_offsets=[target_row, 0],
                src_offsets=[source_row, 0],
                shape=[active_tokens, O_GROUP_IN],
                chunk_rows=COMM_ROW_TILE,
                chunk_cols=O_GROUP_IN,
            )

        for destination_rank in pl.range(TP_SIZE):
            if destination_rank != tp_rank:
                pld.system.notify(
                    target=attention_signal,
                    peer=group_base + destination_rank,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    attention_local_flat, _attention_signal = o_group_a2a(
        attention_local_flat,
        attention_window,
        attention_signal,
        group_base,
        tp_rank,
        active_tokens,
        attention_publish_tid,
        ATTENTION_PUBLISH_WORKERS,
    )
    attention_local_groups = pl.reshape(
        attention_local_flat,
        [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN],
    )
    o_local = pl.create_tensor([LOCAL_T_PAD, D], dtype=pl.BF16)
    o_local, _o_signal = o_proj_reduce_scatter(
        attention_local_groups,
        layer_wo_a,
        layer_wo_b,
        layer_wo_b_scale,
        active_tokens,
        o_local,
        o_window,
        o_signal,
        group_base,
        tp_rank,
    )

    padded_attention = pl.create_tensor([T, D], dtype=pl.BF16)
    for pad_idx in pl.spmd(T * (D // PAD_D_TILE), name_hint="dspark_attention_pad"):
        pad_token = pad_idx // (D // PAD_D_TILE)
        pad_col = (pad_idx % (D // PAD_D_TILE)) * PAD_D_TILE
        output_tile = pl.full([1, PAD_D_TILE], dtype=pl.BF16, value=0.0)
        if pad_token < active_tokens:
            output_tile = o_local[pad_token : pad_token + 1, pad_col : pad_col + PAD_D_TILE]
        padded_attention[pad_token : pad_token + 1, pad_col : pad_col + PAD_D_TILE] = output_tile

    attention_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    hc_post_prefill(
        padded_attention,
        query_hc,
        post,
        combine,
        attention_hc,
        active_tokens,
    )

    moe(
        attention_hc,
        layer_hc_ffn_fn, layer_hc_ffn_scale, layer_hc_ffn_base,
        layer_ffn_norm_w, layer_gate_w, layer_gate_bias, layer_tid2eid, query_token_ids,
        layer_routed_w1, layer_routed_w1_scale, layer_routed_w3, layer_routed_w3_scale, layer_routed_w2, layer_routed_w2_scale,
        layer_shared_w1, layer_shared_w1_scale, layer_shared_w3, layer_shared_w3_scale, layer_shared_w2, layer_shared_w2_scale,
        output_hc,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        layer_id, active_tokens, my_rank, moe_epoch,
    )
    return output_hc, hidden_gather_signal

@pl.jit
def dspark_drafter(
    target_hidden: pl.Tensor[[T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    context_group_position_ids: pl.Tensor[[CP_CONTEXT_T_DYN], pl.INT32],
    context_group_slot_mapping: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, CP_CONTEXT_T_DYN], pl.INT64
    ],
    anchor_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS],
        pl.INT32,
    ],
    query_group_position_ids: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT32],
    query_group_slot_mapping: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * T_QUERY], pl.INT64
    ],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    initial_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    intermediate_hidden: pl.Out[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, T, HC_MULT, D], pl.FP32]
    ],
    head_hidden: pl.Out[pl.Tensor[[B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    hidden_gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    hidden_gather_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    dsa_cp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run one local-query shard of the DSA-CP drafter.

    Every rank in the four-rank group must provide the same number of local
    ``target_hidden`` rows. Group position and slot metadata must contain exactly
    four times that row count in rank-major order; padded rows use slot ``-1``.
    Physical slots use one group-global block namespace replicated on all ranks.
    The group row count must not exceed ``PREFILL_GROUP_CAP`` (8192). The runtime
    does not validate relationships between dynamic tensor axes, so the
    dispatching host must reject inputs that violate this contract.
    """
    target_hidden.bind_dynamic(0, T_MAIN_DYN)
    context_group_position_ids.bind_dynamic(0, CP_CONTEXT_T_DYN)
    context_group_slot_mapping.bind_dynamic(1, CP_CONTEXT_T_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    anchor_positions.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    head_hidden.bind_dynamic(0, B_DYN)
    batch = pl.tensor.dim(num_sampled, 0)
    target_tokens = pl.tensor.dim(target_hidden, 0)
    active_tokens = pl.cast(batch * DSPARK_QUERY_WIDTH, pl.INT32)

    kv_cache_0 = kv_caches[0]
    kv_cache_1 = kv_caches[1]
    kv_cache_2 = kv_caches[2]
    wkv_0 = pl.slice(wkv, [D, HEAD_DIM], [0, 0])
    wkv_1 = pl.slice(wkv, [D, HEAD_DIM], [D, 0])
    wkv_2 = pl.slice(wkv, [D, HEAD_DIM], [2 * D, 0])
    gamma_ckv_0 = pl.slice(gamma_ckv, [HEAD_DIM], [0])
    gamma_ckv_1 = pl.slice(gamma_ckv, [HEAD_DIM], [HEAD_DIM])
    gamma_ckv_2 = pl.slice(gamma_ckv, [HEAD_DIM], [2 * HEAD_DIM])
    main_x = pl.create_tensor([target_tokens, D], dtype=pl.BF16)
    query_token_ids = pl.create_tensor([T], dtype=pl.INT64)
    hidden_0_flat = pl.reshape(initial_hidden, [T, HC_MULT * D])
    main_x, query_token_ids, hidden_0_flat = prepare_dspark_inputs(
        target_hidden,
        main_proj_weight,
        main_norm_weight,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        embedding_weight,
        main_x,
        query_token_ids,
        hidden_0_flat,
    )
    group_context_tokens = pl.tensor.dim(context_group_position_ids, 0)
    group_main_x = pl.create_tensor([group_context_tokens, D], dtype=pl.BF16)
    _gathered_context, hidden_gather_signal = prefill_cp_token_allgather_step(
        main_x,
        group_main_x,
        hidden_gather_window,
        hidden_gather_signal,
        group_base,
        dsa_cp_rank,
    )
    dspark_context_kv(
        group_main_x, wkv_0, gamma_ckv_0, freqs_cos, freqs_sin,
        context_group_position_ids, context_group_slot_mapping, pl.const(0, pl.INT32), kv_cache_0,
    )
    dspark_context_kv(
        group_main_x, wkv_1, gamma_ckv_1, freqs_cos, freqs_sin,
        context_group_position_ids, context_group_slot_mapping, pl.const(1, pl.INT32), kv_cache_1,
    )
    dspark_context_kv(
        group_main_x, wkv_2, gamma_ckv_2, freqs_cos, freqs_sin,
        context_group_position_ids, context_group_slot_mapping, pl.const(2, pl.INT32), kv_cache_2,
    )

    swa_indices = pl.create_tensor(
        [DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH],
        dtype=pl.INT32,
    )
    swa_lens = pl.create_tensor([DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH], dtype=pl.INT32)
    query_positions = pl.create_tensor([T_QUERY], dtype=pl.INT32)
    (
        swa_indices,
        swa_lens,
        query_positions,
    ) = build_dspark_metadata(
        anchor_positions,
        block_tables,
        swa_indices,
        swa_lens,
        query_positions,
    )
    swa_indices_0 = swa_indices[0]
    swa_indices_1 = swa_indices[1]
    swa_indices_2 = swa_indices[2]
    swa_lens_0 = swa_lens[0]
    swa_lens_1 = swa_lens[1]
    swa_lens_2 = swa_lens[2]
    hidden_1 = intermediate_hidden[0]
    hidden_1, hidden_gather_signal = draft_layer(
        initial_hidden, pl.const(0, pl.INT32),
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, query_positions, query_group_position_ids,
        kv_cache_0, query_group_slot_mapping[0], swa_indices_0, swa_lens_0,
        attn_sink, wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
        gate_w, gate_bias, tid2eid, query_token_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        hidden_1, hidden_gather_window, hidden_gather_signal,
        attention_window, attention_signal, o_window, o_signal, group_base, dsa_cp_rank,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(40, pl.INT32), active_tokens, my_rank, pl.const(1, pl.INT32),
    )

    hidden_2 = intermediate_hidden[1]
    hidden_2, hidden_gather_signal = draft_layer(
        hidden_1, pl.const(1, pl.INT32),
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, query_positions, query_group_position_ids,
        kv_cache_1, query_group_slot_mapping[1], swa_indices_1, swa_lens_1,
        attn_sink, wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
        gate_w, gate_bias, tid2eid, query_token_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        hidden_2, hidden_gather_window, hidden_gather_signal,
        attention_window, attention_signal, o_window, o_signal, group_base, dsa_cp_rank,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(41, pl.INT32), active_tokens, my_rank, pl.const(2, pl.INT32),
    )

    hidden_3 = intermediate_hidden[2]
    hidden_3, hidden_gather_signal = draft_layer(
        hidden_2, pl.const(2, pl.INT32),
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, query_positions, query_group_position_ids,
        kv_cache_2, query_group_slot_mapping[2], swa_indices_2, swa_lens_2,
        attn_sink, wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
        gate_w, gate_bias, tid2eid, query_token_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        hidden_3, hidden_gather_window, hidden_gather_signal,
        attention_window, attention_signal, o_window, o_signal, group_base, dsa_cp_rank,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
        pl.const(42, pl.INT32), active_tokens, my_rank, pl.const(3, pl.INT32),
    )
    clear_moe_signals(hidden_3, arrived, data_arrived, combine_arrived)

    padded_head_hidden = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_head(hidden_3, hc_head_fn, hc_head_scale, hc_head_base, padded_head_hidden)
    head_hidden_flat = pl.reshape(head_hidden, [batch * DSPARK_QUERY_WIDTH, D])
    for token in pl.spmd(T, name_hint="dspark_head_unpad"):
        if token < active_tokens:
            head_hidden_flat[token : token + 1, :] = padded_head_hidden[
                token : token + 1,
                :,
            ]
    return head_hidden

@pl.jit.host
def l3_dspark_drafter(
    target_hidden: pl.Tensor[[N_RANKS, T_MAIN_DYN, MAIN_IN], pl.BF16],
    initial_hidden: pl.Out[
        pl.Tensor[[N_RANKS, DSPARK_MAX_BATCH * DSPARK_QUERY_PAD, HC_MULT, D], pl.FP32]
    ],
    intermediate_hidden: pl.Out[
        pl.Tensor[
            [N_RANKS, DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH * DSPARK_QUERY_PAD, HC_MULT, D],
            pl.FP32,
        ]
    ],
    main_proj_weight: pl.Tensor[[N_RANKS, D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[N_RANKS, D], pl.BF16],
    num_sampled: pl.Tensor[[N_RANKS, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[N_RANKS, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[N_RANKS, B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[N_RANKS, VOCAB, D], pl.BF16],
    context_group_position_ids: pl.Tensor[[N_RANKS, CP_CONTEXT_T_DYN], pl.INT32],
    context_group_slot_mapping: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS, CP_CONTEXT_T_DYN], pl.INT64
    ],
    anchor_positions: pl.Tensor[[N_RANKS, B_DYN], pl.INT32],
    block_tables: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    query_group_position_ids: pl.Tensor[
        [N_RANKS, DSPARK_CP_SIZE * T_QUERY], pl.INT32
    ],
    query_group_slot_mapping: pl.Tensor[
        [N_RANKS, DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * T_QUERY], pl.INT64
    ],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[
        pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    head_hidden: pl.Out[pl.Tensor[[N_RANKS, B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
):
    target_hidden.bind_dynamic(1, T_MAIN_DYN)
    context_group_position_ids.bind_dynamic(1, CP_CONTEXT_T_DYN)
    context_group_slot_mapping.bind_dynamic(2, CP_CONTEXT_T_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    anchor_positions.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, B_DYN)
    head_hidden.bind_dynamic(1, B_DYN)

    hidden_gather_window_buf = pld.alloc_window_buffer(
        [PREFILL_GROUP_CAP, D], dtype=pl.BF16
    )
    hidden_gather_signal_buf = pld.alloc_window_buffer(
        [DSPARK_CP_SIZE, 1], dtype=pl.INT32
    )
    attention_window_buf = pld.alloc_window_buffer(
        [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
    )
    attention_signal_buf = pld.alloc_window_buffer(
        [TP_SIZE, 1], dtype=pl.INT32
    )
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        hidden_gather_window = pld.window(
            hidden_gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16
        )
        hidden_gather_signal = pld.window(
            hidden_gather_signal_buf, [DSPARK_CP_SIZE, 1], dtype=pl.INT32
        )
        attention_window = pld.window(
            attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
        )
        attention_signal = pld.window(
            attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.BF16)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        dspark_drafter(
            target_hidden[rank], main_proj_weight[rank], main_norm_weight[rank],
            num_sampled[rank], last_sampled[rank], next_prefill_tokens[rank],
            embedding_weight[rank],
            context_group_position_ids[rank], context_group_slot_mapping[rank],
            anchor_positions[rank], block_tables[rank],
            query_group_position_ids[rank], query_group_slot_mapping[rank],
            freqs_cos[rank], freqs_sin[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_w[rank],
            wq_a[rank], wq_b[rank], wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            kv_caches[rank], attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], ffn_norm_w[rank],
            gate_w[rank], gate_bias[rank], tid2eid[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank], initial_hidden[rank],
            intermediate_hidden[rank], head_hidden[rank],
            hidden_gather_window, hidden_gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
            rank // DSPARK_CP_SIZE * DSPARK_CP_SIZE, rank % DSPARK_CP_SIZE,
            rank,
            device=rank,
        )


def _anchor_position_set(batch):
    import torch

    cases = torch.tensor(
        [
            1,
            BLOCK_SIZE - 1,
            M.sliding_window - 1,
            M.sliding_window + BLOCK_SIZE,
            2 * BLOCK_SIZE - 1,
            4 * BLOCK_SIZE + 3,
            8 * BLOCK_SIZE - 1,
            MAX_SEQ_LEN - DSPARK_QUERY_WIDTH - 1,
        ],
        dtype=torch.int32,
    )
    repeats = (batch + cases.numel() - 1) // cases.numel()
    return cases.repeat(repeats)[:batch].contiguous()


def _block_tables(batch):
    import torch

    logical = torch.arange(ORI_MAX_BLOCKS, dtype=torch.int32)
    tables = torch.empty(N_RANKS, DSPARK_DRAFT_LAYERS, batch, ORI_MAX_BLOCKS, dtype=torch.int32)
    blocks_per_request = ORI_BLOCK_NUM // (DSPARK_CP_SIZE * batch)
    required_blocks = (WIN + DSPARK_QUERY_WIDTH + BLOCK_SIZE - 1) // BLOCK_SIZE
    if blocks_per_request < required_blocks:
        raise ValueError(
            "DSpark fixture KV cache cannot assign a sliding-window ring to every CP request"
        )
    for rank in range(N_RANKS):
        cp_rank = rank % DSPARK_CP_SIZE
        for layer in range(DSPARK_DRAFT_LAYERS):
            for request in range(batch):
                group_request = cp_rank * batch + request
                request_base = group_request * blocks_per_request
                ring_offset = layer * 7
                request_block = (logical + ring_offset) % blocks_per_request
                tables[rank, layer, request] = request_base + request_block
    return tables


def _context_slots(
    tables,
    positions,
    request_ids,
    *,
    valid_mask=None,
    first_cached_positions=None,
):
    import torch

    slots = torch.full(
        (N_RANKS, DSPARK_DRAFT_LAYERS, positions.shape[1]),
        -1,
        dtype=torch.int64,
    )
    for rank in range(N_RANKS):
        for layer in range(DSPARK_DRAFT_LAYERS):
            for token in range(positions.shape[1]):
                request = int(request_ids[token])
                if valid_mask is not None and not bool(valid_mask[token]):
                    continue
                position = int(positions[rank, token])
                if (
                    first_cached_positions is not None
                    and position < int(first_cached_positions[request])
                ):
                    continue
                physical_block = int(tables[rank, layer, request, position // BLOCK_SIZE])
                slots[rank, layer, token] = physical_block * BLOCK_SIZE + position % BLOCK_SIZE
    return slots


def _balanced_routes():
    import torch

    token_ids = torch.arange(VOCAB, dtype=torch.int64).unsqueeze(1)
    route_ids = torch.arange(TOPK, dtype=torch.int64).unsqueeze(0)
    routes = ((token_ids * TOPK + route_ids) % N_EXPERTS_GLOBAL).to(torch.int32)
    return routes.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()


def build_tensor_specs(batch, *, mode="decode"):
    import torch
    from golden import TensorSpec

    if batch not in DSPARK_SUPPORTED_BATCHES:
        raise ValueError(f"unsupported DSpark batch {batch}; expected one of {DSPARK_SUPPORTED_BATCHES}")
    if mode not in ("decode", "prefill"):
        raise ValueError(f"unsupported DSpark mode {mode!r}; expected 'decode' or 'prefill'")

    if mode == "decode":
        local_context_tokens = batch * DECODE_SEQ
        positions = _anchor_position_set(batch).unsqueeze(0).expand(N_RANKS, -1).contiguous()
        valid_pattern = torch.tensor([1, 4, 7, 2, 5, 8, 3, 6], dtype=torch.int32)
        valid_counts = valid_pattern.repeat((batch + valid_pattern.numel() - 1) // valid_pattern.numel())[:batch]
        context_request_ids = torch.arange(batch, dtype=torch.int64).repeat_interleave(DECODE_SEQ)
        context_offsets = torch.arange(DECODE_SEQ, dtype=torch.int32).repeat(batch)
        valid_mask = context_offsets < valid_counts.repeat_interleave(DECODE_SEQ)
        context_positions = torch.zeros(N_RANKS, local_context_tokens, dtype=torch.int32)
        for request in range(batch):
            valid_count = int(valid_counts[request])
            context_start = int(positions[0, request]) - valid_count + 1
            context_positions[
                :,
                request * DECODE_SEQ : request * DECODE_SEQ + valid_count,
            ] = torch.arange(context_start, context_start + valid_count, dtype=torch.int32)
        first_cached_positions = None
    else:
        local_context_tokens = PREFILL_TOKENS // DSPARK_CP_SIZE
        local_rows = torch.arange(local_context_tokens, dtype=torch.int32)
        cp_ranks = torch.arange(N_RANKS, dtype=torch.int32) % DSPARK_CP_SIZE
        context_positions = (
            cp_ranks[:, None] * local_context_tokens + local_rows[None, :]
        )
        context_request_ids = torch.zeros(local_context_tokens, dtype=torch.int64)
        context_offsets = local_rows
        positions = torch.full(
            (N_RANKS, batch), WIN - 1, dtype=torch.int32
        )
        positions[::DSPARK_CP_SIZE, 0] = PREFILL_TOKENS - 1
        valid_mask = None
        first_cached_positions = torch.full(
            (batch,), max(PREFILL_TOKENS - WIN, 0), dtype=torch.int32
        )
    if int(context_positions.min()) < 0 or int(context_positions.max()) >= ORI_MAX_BLOCKS * BLOCK_SIZE:
        raise ValueError("DSpark context positions exceed the logical KV block table")
    if int((positions + DSPARK_QUERY_WIDTH).max()) >= ORI_MAX_BLOCKS * BLOCK_SIZE:
        raise ValueError("DSpark query positions exceed the logical KV block table")
    tables = _block_tables(batch)
    if int(tables.min()) < 0 or int(tables.max()) >= ORI_BLOCK_NUM:
        raise ValueError("DSpark block table references a physical KV block outside the cache")
    if mode == "decode":
        context_slots = _context_slots(
            tables,
            context_positions,
            context_request_ids,
            valid_mask=valid_mask,
            first_cached_positions=first_cached_positions,
        )
    else:
        # Model one real prompt split over the four rank-major token shards. Its
        # query is owned by request zero on the group leader; other requests are
        # valid computational fillers required by the B4/B8/B12/B16 ABI.
        context_tables = tables.clone()
        for rank in range(N_RANKS):
            group_base = rank // DSPARK_CP_SIZE * DSPARK_CP_SIZE
            context_tables[rank, :, 0] = tables[group_base, :, 0]
        context_slots = _context_slots(
            context_tables,
            context_positions,
            context_request_ids,
            valid_mask=valid_mask,
            first_cached_positions=first_cached_positions,
        )
    context_group_positions = torch.empty(
        N_RANKS, DSPARK_CP_SIZE * local_context_tokens, dtype=torch.int32
    )
    context_group_slots = torch.empty(
        N_RANKS,
        DSPARK_DRAFT_LAYERS,
        DSPARK_CP_SIZE * local_context_tokens,
        dtype=torch.int64,
    )
    query_positions_local = torch.zeros(N_RANKS, T_QUERY, dtype=torch.int32)
    query_slots_local = torch.full(
        (N_RANKS, DSPARK_DRAFT_LAYERS, T_QUERY), -1, dtype=torch.int64
    )
    for rank in range(N_RANKS):
        group_base = rank // DSPARK_CP_SIZE * DSPARK_CP_SIZE
        group_slice = slice(group_base, group_base + DSPARK_CP_SIZE)
        context_group_positions[rank] = context_positions[group_slice].reshape(-1)
        context_group_slots[rank] = context_slots[group_slice].permute(1, 0, 2).reshape(
            DSPARK_DRAFT_LAYERS, -1
        )
        for request in range(batch):
            anchor = int(positions[rank, request])
            for offset in range(DSPARK_QUERY_WIDTH):
                token = request * DSPARK_QUERY_WIDTH + offset
                query_position = anchor + 1 + offset
                query_positions_local[rank, token] = query_position
                for layer in range(DSPARK_DRAFT_LAYERS):
                    physical_block = int(
                        tables[rank, layer, request, query_position // BLOCK_SIZE]
                    )
                    query_slots_local[rank, layer, token] = (
                        physical_block * BLOCK_SIZE + query_position % BLOCK_SIZE
                    )
    query_group_positions = torch.empty(
        N_RANKS, DSPARK_CP_SIZE * T_QUERY, dtype=torch.int32
    )
    query_group_slots = torch.empty(
        N_RANKS,
        DSPARK_DRAFT_LAYERS,
        DSPARK_CP_SIZE * T_QUERY,
        dtype=torch.int64,
    )
    for rank in range(N_RANKS):
        group_base = rank // DSPARK_CP_SIZE * DSPARK_CP_SIZE
        group_slice = slice(group_base, group_base + DSPARK_CP_SIZE)
        query_group_positions[rank] = query_positions_local[group_slice].reshape(-1)
        query_group_slots[rank] = query_slots_local[group_slice].permute(1, 0, 2).reshape(
            DSPARK_DRAFT_LAYERS, -1
        )
    request_ids = torch.arange(batch, dtype=torch.int64).unsqueeze(0)
    owner_ids = torch.arange(N_RANKS, dtype=torch.int64).unsqueeze(1)
    last_sampled = owner_ids * batch + request_ids + 1
    next_prefill_tokens = N_RANKS * batch + owner_ids * batch + request_ids + 1
    num_sampled = torch.full(
        (N_RANKS, batch),
        1 if mode == "decode" else 0,
        dtype=torch.int32,
    )
    routes = _balanced_routes().unsqueeze(1)
    routes = routes.expand(-1, DSPARK_DRAFT_LAYERS, -1, -1)
    routes = routes.reshape(N_RANKS, DSPARK_DRAFT_LAYERS * VOCAB, TOPK).contiguous()

    def init_target_hidden():
        values = torch.zeros(N_RANKS, local_context_tokens, MAIN_IN, dtype=torch.bfloat16)
        columns = torch.arange(D, dtype=torch.float32)
        base = ((columns % 31) - 15) * 0.002
        for rank in range(N_RANKS):
            for row in range(local_context_tokens):
                if mode == "prefill":
                    group = rank // DSPARK_CP_SIZE
                    global_position = int(context_positions[rank, row])
                    row_value = base + 0.01 * (group + 1) + 0.001 * global_position
                else:
                    request = int(context_request_ids[row])
                    offset = int(context_offsets[row])
                    row_value = base + 0.01 * (rank + request + 1) + 0.001 * offset
                values[rank, row, :D] = row_value.to(torch.bfloat16)
        return values

    def init_main_proj_weight():
        weight = torch.zeros(N_RANKS, D, MAIN_IN, dtype=torch.bfloat16)
        diagonal = torch.arange(D)
        weight[:, diagonal, diagonal] = 1
        return weight

    def init_embedding_weight():
        weight = torch.zeros(N_RANKS, VOCAB, D, dtype=torch.bfloat16)
        columns = torch.arange(D, dtype=torch.float32)
        base = ((columns % 29) - 14) * 0.002
        for rank in range(N_RANKS):
            weight[rank, 0] = base.to(torch.bfloat16)
            weight[rank, DSPARK_NOISE_TOKEN_ID] = (base + 0.03).to(torch.bfloat16)
            for owner_rank in range(N_RANKS):
                for request in range(batch):
                    owner_offset = 0.005 * owner_rank
                    last_token = int(last_sampled[owner_rank, request])
                    next_token = int(next_prefill_tokens[owner_rank, request])
                    weight[rank, last_token] = (
                        base + 0.01 * (request + 1) + owner_offset
                    ).to(torch.bfloat16)
                    weight[rank, next_token] = (
                        base + 0.02 * (request + 1) + owner_offset
                    ).to(torch.bfloat16)
        return weight

    def init_wkv():
        weight = torch.zeros(N_RANKS, DSPARK_DRAFT_LAYERS * D, HEAD_DIM, dtype=torch.bfloat16)
        diagonal = torch.arange(HEAD_DIM)
        for layer in range(DSPARK_DRAFT_LAYERS):
            weight[:, layer * D + diagonal, diagonal] = 1
        return weight

    def ranked(name, shape, dtype, init_value=0, *, resident=False):
        spec = TensorSpec(name, [N_RANKS, *shape], dtype, init_value=init_value)
        if resident:
            spec.resident = "stacked"
        return spec

    def init_kv_caches():
        cache = torch.empty(
            N_RANKS,
            DSPARK_DRAFT_LAYERS,
            ORI_BLOCK_NUM,
            BLOCK_SIZE,
            1,
            HEAD_DIM,
            dtype=torch.bfloat16,
        )
        for rank in range(N_RANKS):
            for layer in range(DSPARK_DRAFT_LAYERS):
                cache[rank, layer].fill_(
                    layer + 1 + (rank // DSPARK_CP_SIZE) * 0.25
                )
        return cache

    specs = [
        ranked("target_hidden", [local_context_tokens, MAIN_IN], torch.bfloat16, init_value=init_target_hidden),
        TensorSpec(
            "initial_hidden",
            [N_RANKS, DSPARK_MAX_BATCH * DSPARK_QUERY_PAD, HC_MULT, D],
            torch.float32,
        ),
        TensorSpec(
            "intermediate_hidden",
            [N_RANKS, DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH * DSPARK_QUERY_PAD, HC_MULT, D],
            torch.float32,
        ),
        ranked("main_proj_weight", [D, MAIN_IN], torch.bfloat16, init_value=init_main_proj_weight, resident=True),
        ranked("main_norm_weight", [D], torch.bfloat16, init_value=1, resident=True),
        ranked("num_sampled", [batch], torch.int32, init_value=lambda: num_sampled),
        ranked("last_sampled", [batch], torch.int64, init_value=lambda: last_sampled),
        ranked(
            "next_prefill_tokens",
            [batch],
            torch.int64,
            init_value=lambda: next_prefill_tokens,
        ),
        ranked("embedding_weight", [VOCAB, D], torch.bfloat16, init_value=init_embedding_weight, resident=True),
        ranked(
            "context_group_position_ids",
            [DSPARK_CP_SIZE * local_context_tokens],
            torch.int32,
            init_value=lambda: context_group_positions,
        ),
        ranked(
            "context_group_slot_mapping",
            [DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * local_context_tokens],
            torch.int64,
            init_value=lambda: context_group_slots,
        ),
        ranked("anchor_positions", [batch], torch.int32, init_value=lambda: positions),
        ranked("block_tables", [DSPARK_DRAFT_LAYERS, batch, ORI_MAX_BLOCKS], torch.int32, init_value=lambda: tables),
        ranked(
            "query_group_position_ids",
            [DSPARK_CP_SIZE * T_QUERY],
            torch.int32,
            init_value=lambda: query_group_positions,
        ),
        ranked(
            "query_group_slot_mapping",
            [DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * T_QUERY],
            torch.int64,
            init_value=lambda: query_group_slots,
        ),
        ranked("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=1, resident=True),
        ranked("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, resident=True),
    ]

    specs.extend(
        [
            ranked("hc_attn_fn", [DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], torch.float32, resident=True),
            ranked("hc_attn_scale", [DSPARK_DRAFT_LAYERS * 3], torch.float32, resident=True),
            ranked("hc_attn_base", [DSPARK_DRAFT_LAYERS * MIX_HC], torch.float32, resident=True),
            ranked("attn_norm_w", [DSPARK_DRAFT_LAYERS * D], torch.bfloat16, init_value=1, resident=True),
            ranked("wq_a", [DSPARK_DRAFT_LAYERS * D, Q_LORA], torch.bfloat16, resident=True),
            ranked("wq_b", [DSPARK_DRAFT_LAYERS * Q_LORA, H * HEAD_DIM], torch.int8, resident=True),
            ranked("wq_b_scale", [DSPARK_DRAFT_LAYERS * H * HEAD_DIM], torch.float32, resident=True),
            ranked("wkv", [DSPARK_DRAFT_LAYERS * D, HEAD_DIM], torch.bfloat16, init_value=init_wkv, resident=True),
            ranked("gamma_cq", [DSPARK_DRAFT_LAYERS * Q_LORA], torch.bfloat16, init_value=1, resident=True),
            ranked("gamma_ckv", [DSPARK_DRAFT_LAYERS * HEAD_DIM], torch.bfloat16, init_value=1, resident=True),
            ranked(
                "kv_caches",
                [DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
                torch.bfloat16,
                init_value=init_kv_caches,
            ),
            ranked("attn_sink", [DSPARK_DRAFT_LAYERS * H], torch.float32, resident=True),
            ranked("wo_a", [DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, resident=True),
            ranked("wo_b", [DSPARK_DRAFT_LAYERS * D, LOCAL_O_WIDTH], torch.int8, resident=True),
            ranked("wo_b_scale", [DSPARK_DRAFT_LAYERS * D], torch.float32, resident=True),
            ranked("hc_ffn_fn", [DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], torch.float32, resident=True),
            ranked("hc_ffn_scale", [DSPARK_DRAFT_LAYERS * 3], torch.float32, resident=True),
            ranked("hc_ffn_base", [DSPARK_DRAFT_LAYERS * MIX_HC], torch.float32, resident=True),
            ranked("ffn_norm_w", [DSPARK_DRAFT_LAYERS * D], torch.bfloat16, init_value=1, resident=True),
            ranked("gate_w", [DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], torch.float32, resident=True),
            ranked("gate_bias", [DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], torch.float32, resident=True),
            ranked(
                "tid2eid",
                [DSPARK_DRAFT_LAYERS * VOCAB, TOPK],
                torch.int32,
                init_value=lambda: routes,
                resident=True,
            ),
            ranked("routed_w1", [DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], torch.int8, resident=True),
            ranked("routed_w1_scale", [DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], torch.float32, resident=True),
            ranked("routed_w3", [DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], torch.int8, resident=True),
            ranked("routed_w3_scale", [DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], torch.float32, resident=True),
            ranked("routed_w2", [DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], torch.int8, resident=True),
            ranked("routed_w2_scale", [DSPARK_DRAFT_LAYERS * N_LOCAL, D], torch.float32, resident=True),
            ranked("shared_w1", [DSPARK_DRAFT_LAYERS * MOE_INTER, D], torch.int8, resident=True),
            ranked("shared_w1_scale", [DSPARK_DRAFT_LAYERS * MOE_INTER], torch.float32, resident=True),
            ranked("shared_w3", [DSPARK_DRAFT_LAYERS * MOE_INTER, D], torch.int8, resident=True),
            ranked("shared_w3_scale", [DSPARK_DRAFT_LAYERS * MOE_INTER], torch.float32, resident=True),
            ranked("shared_w2", [DSPARK_DRAFT_LAYERS * D, MOE_INTER], torch.int8, resident=True),
            ranked("shared_w2_scale", [DSPARK_DRAFT_LAYERS * D], torch.float32, resident=True),
        ]
    )

    specs.extend(
        [
            ranked("hc_head_fn", [HC_MULT, HC_DIM], torch.float32, resident=True),
            ranked("hc_head_scale", [1], torch.float32, resident=True),
            ranked("hc_head_base", [HC_MULT], torch.float32, resident=True),
            TensorSpec(
                "head_hidden",
                [N_RANKS, batch, DSPARK_QUERY_WIDTH, D],
                torch.bfloat16,
            ),
        ]
    )
    return specs


def _dspark_kv_cache_compare():
    """Compare the KV golden and every replica in each DSA-CP group."""
    import torch

    from golden import ratio_allclose

    strict_compare = ratio_allclose(
        atol=1e-4,
        rtol=1.0 / 128,
        max_error_ratio=0.0,
    )

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, f"    KV shape mismatch: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
        if actual.shape[0] != N_RANKS:
            return False, f"    expected {N_RANKS} rank replicas, got {actual.shape[0]}"

        for group_base in range(0, N_RANKS, DSPARK_CP_SIZE):
            reference = actual[group_base]
            for peer_rank in range(group_base + 1, group_base + DSPARK_CP_SIZE):
                if not torch.equal(actual[peer_rank], reference):
                    return False, (
                        f"    DSA-CP KV replica mismatch: rank {peer_rank} "
                        f"!= rank {group_base}"
                    )

        return strict_compare(actual, expected, **kwargs)

    compare.__name__ = "dspark_kv_cache_compare"
    return compare


def golden_dspark_drafter(tensors):
    import torch

    def rms_norm(hidden):
        hidden_fp32 = hidden.float()
        inv_rms = torch.rsqrt(hidden_fp32.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
        return (hidden_fp32 * inv_rms).to(torch.bfloat16)

    def project_kv(hidden):
        projected = hidden.float()[..., :HEAD_DIM]
        inv_rms = torch.rsqrt(projected.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
        return (projected * inv_rms).to(torch.bfloat16)

    def hc_coefficients(token_count):
        pre = torch.full((token_count, HC_MULT), 0.5 + M.hc_eps, dtype=torch.float32)
        post = torch.ones(token_count, HC_MULT, dtype=torch.float32)
        combine = torch.full((token_count, HC_MULT, HC_MULT), 0.25 + M.hc_eps, dtype=torch.float32)
        combine = combine / (combine.sum(-2, keepdim=True) + M.hc_eps)
        for _ in range(M.hc_sinkhorn_iters - 1):
            combine = combine / (combine.sum(-1, keepdim=True) + M.hc_eps)
            combine = combine / (combine.sum(-2, keepdim=True) + M.hc_eps)
        return pre, post, combine

    def hc_pre_zero_function(hidden):
        pre, post, combine = hc_coefficients(hidden.shape[0])
        mixed = hidden[:, 0] * pre[:, 0:1]
        for lane in range(1, HC_MULT):
            mixed = mixed + hidden[:, lane] * pre[:, lane : lane + 1]
        return mixed.to(torch.bfloat16), post, combine

    def hc_post_zero_output(residual, post, combine):
        output = torch.empty_like(residual)
        zero = torch.zeros(residual.shape[0], D, dtype=torch.float32)
        for output_lane in range(HC_MULT):
            row = zero * post[:, output_lane : output_lane + 1]
            for input_lane in range(HC_MULT):
                row = row + residual[:, input_lane] * combine[:, input_lane, output_lane : output_lane + 1]
            output[:, output_lane] = row
        return output

    tensors["head_hidden"].zero_()
    tensors["initial_hidden"].zero_()
    tensors["intermediate_hidden"].zero_()
    positions = tensors["anchor_positions"]
    group_context_slots = tensors["context_group_slot_mapping"]
    group_query_slots = tensors["query_group_slot_mapping"]
    local_context_tokens = tensors["target_hidden"].shape[1]
    selected_anchor_ids = torch.where(
        tensors["num_sampled"] > 0,
        tensors["last_sampled"],
        tensors["next_prefill_tokens"],
    )
    main_x_by_rank = []
    hidden_by_rank = []
    for rank in range(N_RANKS):
        main_x_by_rank.append(rms_norm(tensors["target_hidden"][rank, :, :D]))
        query_ids = torch.zeros(DSPARK_MAX_BATCH * DSPARK_QUERY_PAD, dtype=torch.int64)
        for request in range(positions.shape[1]):
            row = request * DSPARK_QUERY_WIDTH
            query_ids[row] = selected_anchor_ids[rank, request]
            query_ids[row + 1 : row + DSPARK_QUERY_WIDTH] = DSPARK_NOISE_TOKEN_ID
        query_hidden = tensors["embedding_weight"][rank].index_select(0, query_ids)
        hidden = query_hidden.float().unsqueeze(1).expand(-1, HC_MULT, -1).contiguous()
        tensors["initial_hidden"][rank] = hidden
        hidden_by_rank.append(hidden)

    for layer in range(DSPARK_DRAFT_LAYERS):
        layer_state = []
        for rank in range(N_RANKS):
            mixed, post, combine = hc_pre_zero_function(hidden_by_rank[rank])
            query_normed = rms_norm(mixed[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH])
            layer_state.append((post, combine, query_normed))

        for rank in range(N_RANKS):
            cache = tensors["kv_caches"][rank, layer].view(-1, HEAD_DIM)
            group_base = rank // DSPARK_CP_SIZE * DSPARK_CP_SIZE
            for source_cp, source_rank in enumerate(
                range(group_base, group_base + DSPARK_CP_SIZE)
            ):
                context_start = source_cp * local_context_tokens
                layer_context_slots = group_context_slots[
                    rank,
                    layer,
                    context_start : context_start + local_context_tokens,
                ]
                valid_context = layer_context_slots >= 0
                valid_context_slots = layer_context_slots[valid_context].long()
                cache[valid_context_slots] = project_kv(
                    main_x_by_rank[source_rank][valid_context]
                )
                query_start = source_cp * T_QUERY
                layer_query_slots = group_query_slots[
                    rank,
                    layer,
                    query_start : query_start + T_QUERY,
                ]
                valid_query = layer_query_slots >= 0
                valid_query_slots = layer_query_slots[valid_query].long()
                query_normed = layer_state[source_rank][2]
                cache[valid_query_slots] = project_kv(
                    query_normed[valid_query]
                )

        next_hidden_by_rank = []
        for rank in range(N_RANKS):
            hidden = hidden_by_rank[rank]
            post, combine, _ = layer_state[rank]
            attention_hidden = hc_post_zero_output(
                hidden[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
                post[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
                combine[: DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH],
            )
            padded_attention = torch.zeros_like(hidden)
            padded_attention[: positions.shape[1] * DSPARK_QUERY_WIDTH] = attention_hidden[
                : positions.shape[1] * DSPARK_QUERY_WIDTH
            ]
            _, moe_post, moe_combine = hc_pre_zero_function(padded_attention)
            hidden = hc_post_zero_output(padded_attention, moe_post, moe_combine)
            tensors["intermediate_hidden"][rank, layer] = hidden
            next_hidden_by_rank.append(hidden)
        hidden_by_rank = next_hidden_by_rank

    for rank in range(N_RANKS):
        hidden = hidden_by_rank[rank]
        active = hidden[: positions.shape[1] * DSPARK_QUERY_WIDTH]
        head_pre = torch.full((active.shape[0], HC_MULT), 0.5 + M.hc_eps, dtype=torch.float32)
        head = active[:, 0] * head_pre[:, 0:1]
        head = head + active[:, 1] * head_pre[:, 1:2]
        tail = active[:, 2] * head_pre[:, 2:3]
        tail = tail + active[:, 3] * head_pre[:, 3:4]
        head = (head + tail).to(torch.bfloat16)
        tensors["head_hidden"][rank] = head.view(positions.shape[1], DSPARK_QUERY_WIDTH, D)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="Validate the multi-rank DeepSeek V4 DSpark drafter.")
    parser.add_argument("--batch", type=int, choices=DSPARK_SUPPORTED_BATCHES, default=4)
    parser.add_argument("--tp", type=int, choices=(4,), default=TP_SIZE)
    parser.add_argument("--ep", type=int, choices=(4, 8, 16), default=N_RANKS)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    assert args.tp == TP_SIZE
    assert args.ep == N_RANKS
    assert len(device_ids) >= N_RANKS
    result = run(
        fn=l3_dspark_drafter,
        specs=build_tensor_specs(args.batch),
        golden_fn=golden_dspark_drafter,
        compile_only=args.compile_only,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            ring_heap=_DSPARK_RING_HEAP,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv_caches": _dspark_kv_cache_compare(),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
