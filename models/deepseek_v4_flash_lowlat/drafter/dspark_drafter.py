# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys

from . import config

from .history_gather import history_ring_to_window

_TP_CHOICES = (2, 4)

_EP_CHOICES = (2, 4, 8, 16)

_TP_DEFAULT = 2

_EP_DEFAULT = 2


def _parse_parallel_arg(name: str, default: int) -> tuple[int, bool]:
    flag = f"--{name}"
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1]), True
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1]), True
    return default, False


config.TP, _tp_explicit = _parse_parallel_arg("tp", _TP_DEFAULT)

config.EP, _ep_explicit = _parse_parallel_arg("ep", _EP_DEFAULT)

if config.TP not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {config.TP})")

if config.EP not in _EP_CHOICES:
    raise ValueError(f"--ep must be one of {_EP_CHOICES} (got {config.EP})")

if not _tp_explicit:
    sys.argv.extend(("--tp", str(config.TP)))

if not _ep_explicit:
    sys.argv.extend(("--ep", str(config.EP)))

import pypto.language as pl

import pypto.language.distributed as pld

from .config import BLOCK_SIZE, FLASH as M, KV_ORI_BLOCK_NUM, MOE_TOKENS, PREFILL_TOKENS

from .dspark_attention import T as DSPARK_ATTENTION_T, dspark_attention

from .dspark_context_kv import dspark_context_kv

from .decode_o_proj import (
    ATTENTION_PUBLISH_WORKERS,
    ATTENTION_WINDOW_ROWS,
    COMM_ROW_TILE,
    GROUP_T_PAD,
    LOCAL_O_GROUPS,
    LOCAL_T_PAD,
    O_WINDOW_ROWS,
    TP_SIZE,
    o_group_a2a,
    o_proj_reduce_scatter,
)

from .hc_head import hc_head

from .hc_post import hc_post_prefill

from .hc_pre import hc_pre

from .lookup_embedding import lookup_embedding

from .rmsnorm import rms_norm

from .moe import (
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

from .prefill_cp_token_allgather import (
    PREFILL_GROUP_CAP,
    TP_SIZE as PREFILL_CP_SIZE,
    prefill_cp_token_allgather_step,
)

T_MAIN_DYN = pl.dynamic("NATIVE_D_DSPARK_BACKBONE_T_MAIN_DYN")

CP_CONTEXT_T_DYN = pl.dynamic("NATIVE_D_DSPARK_BACKBONE_CP_CONTEXT_T_DYN")

B_DYN = pl.dynamic("NATIVE_D_DSPARK_BACKBONE_B_DYN")

METADATA_B_DYN = pl.dynamic("NATIVE_D_DSPARK_METADATA_B_DYN")

DSPARK_DRAFT_LAYERS = 3

DSPARK_QUERY_WIDTH = 5

DSPARK_QUERY_PAD = 8

DSPARK_NOISE_TOKEN_ID = 128799

DSPARK_SUPPORTED_BATCHES = (1, 4)

DSPARK_MAX_BATCH = max(DSPARK_SUPPORTED_BATCHES)

DSPARK_QUERY_TOKENS = DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH

DSPARK_MOE_TOKENS = DSPARK_MAX_BATCH * DSPARK_QUERY_PAD

DSPARK_SWA_INDEX_WIDTH = (M.sliding_window + DSPARK_QUERY_WIDTH + 63) // 64 * 64

DSPARK_CP_SIZE = PREFILL_CP_SIZE

assert DSPARK_QUERY_TOKENS == DSPARK_ATTENTION_T

assert DSPARK_MOE_TOKENS == MOE_TOKENS

assert DSPARK_SWA_INDEX_WIDTH >= M.sliding_window + DSPARK_QUERY_WIDTH

assert DSPARK_QUERY_TOKENS <= LOCAL_T_PAD

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

MAIN_PROJ_T_TILE = 128  # cube M-tile; matmul rows must be a multiple of 16

MAIN_PROJ_N_TILE = 128  # 32 output blocks over D

MAIN_PROJ_K_TILE = 512  # 24 reduction steps over MAIN_IN


@pl.jit.inline
def build_dspark_metadata(
    anchor_positions: pl.Tensor[[METADATA_B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, METADATA_B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    swa_indices: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH], pl.INT32],
    query_positions: pl.Tensor[[DSPARK_QUERY_TOKENS], pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    batch = pl.tensor.dim(anchor_positions, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    for metadata_core in pl.spmd(1, name_hint="dspark_query_metadata", deps=[prior_cycle_gate]):
        for token in pl.range(metadata_core, DSPARK_QUERY_TOKENS):
            pl.write(query_positions, [token], pl.cast(0, pl.INT32))
            if token < active_tokens:
                request = token // DSPARK_QUERY_WIDTH
                query_offset = token % DSPARK_QUERY_WIDTH
                anchor_position = pl.read(anchor_positions, [request])
                query_position = anchor_position + 1 + query_offset
                pl.write(query_positions, [token], pl.cast(query_position, pl.INT32))

        for lens_request in pl.range(metadata_core, DSPARK_MAX_BATCH):
            lens_visible_len = pl.cast(0, pl.INT32)
            if lens_request < batch:
                lens_anchor_position = pl.read(anchor_positions, [lens_request])
                lens_prefix_len = lens_anchor_position + 1
                lens_start_position = pl.cast(pl.max(lens_prefix_len - WIN, 0), pl.INT32)
                lens_visible_len = pl.cast(
                    lens_prefix_len + DSPARK_QUERY_WIDTH - lens_start_position, pl.INT32
                )
            for lens_layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(swa_lens, [lens_layer, lens_request], lens_visible_len)

    for request in pl.spmd(DSPARK_MAX_BATCH, name_hint="dspark_visible_metadata", deps=[prior_cycle_gate]):
        start_position = pl.cast(0, pl.INT32)
        visible_len = pl.cast(0, pl.INT32)
        if request < batch:
            anchor_position = pl.read(anchor_positions, [request])
            prefix_len = anchor_position + 1
            start_position = pl.cast(pl.max(prefix_len - WIN, 0), pl.INT32)
            visible_len = pl.cast(prefix_len + DSPARK_QUERY_WIDTH - start_position, pl.INT32)
        for layer in pl.range(DSPARK_DRAFT_LAYERS):
            for visible_offset in pl.range(DSPARK_SWA_INDEX_WIDTH):
                visible_slot = pl.cast(-1, pl.INT32)
                if visible_offset < visible_len:
                    visible_position = start_position + visible_offset
                    logical_block = visible_position // BLOCK_SIZE
                    block_offset = visible_position % BLOCK_SIZE
                    physical_block = pl.read(block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)])
                    visible_slot = pl.cast(physical_block * BLOCK_SIZE + block_offset, pl.INT32)
                pl.write(swa_indices, [layer, request, visible_offset], visible_slot)
    return (swa_indices, swa_lens, query_positions)


@pl.jit.inline(auto_scope=False)
def draft_layer(
    query_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    draft_layer_index: pl.Scalar[pl.INT32],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    query_freqs_cos: pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16],
    query_freqs_sin: pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16],
    query_group_freqs_cos: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16],
    query_group_freqs_sin: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16],
    query_positions: pl.Tensor[[T_QUERY], pl.INT32],
    query_group_positions: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT32],
    kv_cache: pl.Tensor[[KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_group_slot_mapping: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT64],
    swa_indices: pl.Tensor[[DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[DSPARK_MAX_BATCH], pl.INT32],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    query_token_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.INT8, pl.NZ],
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
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    layer_hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [draft_layer_index * MIX_HC, 0]
    )
    layer_hc_attn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [draft_layer_index * 3])
    layer_hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_attn_base, [MIX_HC], [draft_layer_index * MIX_HC]
    )
    layer_attn_norm_w: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [draft_layer_index * D])
    layer_wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ] = wq_a[draft_layer_index]
    layer_wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ] = wq_b[draft_layer_index]
    layer_wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
        wq_b_scale, [H * HEAD_DIM], [draft_layer_index * H * HEAD_DIM]
    )
    layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [draft_layer_index * D, 0])
    layer_gamma_cq: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [draft_layer_index * Q_LORA])
    layer_gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        gamma_ckv, [HEAD_DIM], [draft_layer_index * HEAD_DIM]
    )
    layer_attn_sink: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [draft_layer_index * H])
    layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [draft_layer_index * LOCAL_O_GROUPS, 0, 0]
    )
    layer_wo_b: pl.Tensor[[LOCAL_O_GROUPS, D, O_LORA], pl.INT8] = pl.slice(
        wo_b, [LOCAL_O_GROUPS, D, O_LORA], [draft_layer_index * LOCAL_O_GROUPS, 0, 0]
    )
    layer_wo_b_scale: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [draft_layer_index * D])
    layer_hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [draft_layer_index * MIX_HC, 0]
    )
    layer_hc_ffn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [draft_layer_index * 3])
    layer_hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_ffn_base, [MIX_HC], [draft_layer_index * MIX_HC]
    )
    layer_ffn_norm_w: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_w, [D], [draft_layer_index * D])
    layer_gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(
        gate_w, [N_EXPERTS_GLOBAL, D], [draft_layer_index * N_EXPERTS_GLOBAL, 0]
    )
    layer_gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(
        gate_bias, [N_EXPERTS_GLOBAL], [draft_layer_index * N_EXPERTS_GLOBAL]
    )
    layer_tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
        tid2eid, [VOCAB, TOPK], [draft_layer_index * VOCAB, 0]
    )
    layer_routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ] = pl.slice(
        routed_w1, [N_LOCAL, MOE_INTER, D], [draft_layer_index * N_LOCAL, 0, 0]
    )
    layer_routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
        routed_w1_scale, [N_LOCAL, MOE_INTER], [draft_layer_index * N_LOCAL, 0]
    )
    layer_routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ] = pl.slice(
        routed_w3, [N_LOCAL, MOE_INTER, D], [draft_layer_index * N_LOCAL, 0, 0]
    )
    layer_routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
        routed_w3_scale, [N_LOCAL, MOE_INTER], [draft_layer_index * N_LOCAL, 0]
    )
    layer_routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ] = pl.slice(
        routed_w2, [N_LOCAL, D, MOE_INTER], [draft_layer_index * N_LOCAL, 0, 0]
    )
    layer_routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
        routed_w2_scale, [N_LOCAL, D], [draft_layer_index * N_LOCAL, 0]
    )
    layer_shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ] = shared_w1[draft_layer_index]
    layer_shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
        shared_w1_scale, [MOE_INTER], [draft_layer_index * MOE_INTER]
    )
    layer_shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ] = shared_w3[draft_layer_index]
    layer_shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
        shared_w3_scale, [MOE_INTER], [draft_layer_index * MOE_INTER]
    )
    layer_shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ] = shared_w2[draft_layer_index]
    layer_shared_w2_scale: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [draft_layer_index * D])

    query_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    combine = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(query_hc, layer_hc_attn_fn, layer_hc_attn_scale, layer_hc_attn_base, query_mixed, post, combine)

    query_normed = pl.create_tensor([T_QUERY, D], dtype=pl.BF16)
    query_mixed_active: pl.Tensor[[T_QUERY, D], pl.BF16] = pl.slice(query_mixed, [T_QUERY, D], [0, 0])
    rms_norm(query_mixed_active, layer_attn_norm_w, query_normed)
    query_group = pl.create_tensor([DSPARK_CP_SIZE * T_QUERY, D], dtype=pl.BF16)
    with pl.scope():
        _gathered_query, hidden_gather_signal = prefill_cp_token_allgather_step(
            query_normed,
            query_group,
            hidden_gather_window,
            hidden_gather_signal,
            group_base,
            dsa_cp_rank,
            prior_cycle_gate,
        )
    o_packed_heads = pl.create_tensor([O_GROUPS * LOCAL_T_PAD * (H // O_GROUPS), HEAD_DIM], dtype=pl.BF16)
    dspark_attention(
        query_normed,
        query_group,
        layer_wq_a,
        layer_wq_b,
        layer_wq_b_scale,
        layer_wkv,
        layer_gamma_cq,
        layer_gamma_ckv,
        query_freqs_cos,
        query_freqs_sin,
        query_group_freqs_cos,
        query_group_freqs_sin,
        query_positions,
        query_group_positions,
        kv_cache,
        query_group_slot_mapping,
        swa_indices,
        swa_lens,
        layer_attn_sink,
        o_packed_heads,
    )
    attention_grouped = pl.reshape(o_packed_heads, [O_GROUPS * LOCAL_T_PAD, O_GROUP_IN])
    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
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
    attention_local_groups = pl.reshape(attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN])
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
    for pad_token in pl.spmd(T, name_hint="dspark_attention_pad"):
        output_tile = pl.full([1, D], dtype=pl.BF16, value=0.0)
        if pad_token < active_tokens:
            output_tile = o_local[pad_token : pad_token + 1, 0:D]
        padded_attention[pad_token : pad_token + 1, 0:D] = output_tile

    attention_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    hc_post_prefill(padded_attention, query_hc, post, combine, attention_hc, active_tokens)

    moe(
        attention_hc,
        layer_hc_ffn_fn,
        layer_hc_ffn_scale,
        layer_hc_ffn_base,
        layer_ffn_norm_w,
        layer_gate_w,
        layer_gate_bias,
        layer_tid2eid,
        query_token_ids,
        layer_routed_w1,
        layer_routed_w1_scale,
        layer_routed_w3,
        layer_routed_w3_scale,
        layer_routed_w2,
        layer_routed_w2_scale,
        layer_shared_w1,
        layer_shared_w1_scale,
        layer_shared_w3,
        layer_shared_w3_scale,
        layer_shared_w2,
        layer_shared_w2_scale,
        output_hc,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        data_arrived,
        routed_y_buf,
        combine_arrived,
        layer_id,
        active_tokens,
        my_rank,
        moe_epoch,
        prior_cycle_gate,
    )
    return output_hc, hidden_gather_signal


def _dspark_drafter(
    target_hidden: pl.Tensor[[T_MAIN_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    context_group_position_ids: pl.Tensor[[CP_CONTEXT_T_DYN], pl.INT32],
    context_group_slot_mapping: pl.Tensor[[DSPARK_DRAFT_LAYERS, CP_CONTEXT_T_DYN], pl.INT64],
    anchor_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, B_DYN, ORI_MAX_BLOCKS],
        pl.INT32,
    ],
    query_group_position_ids: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT32],
    query_group_slot_mapping: pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * T_QUERY], pl.INT64],
    context_group_freqs_cos: pl.Tensor[[CP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16],
    context_group_freqs_sin: pl.Tensor[[CP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16],
    query_freqs_cos: pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16],
    query_freqs_sin: pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16],
    query_group_freqs_cos: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16],
    query_group_freqs_sin: pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[pl.Tensor[[DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS * LOCAL_O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    initial_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    intermediate_hidden: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, T, HC_MULT, D], pl.FP32]],
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
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    """Run one local-query shard of the DSA-CP drafter.

    Every rank in the four-rank group must provide the same number of local
    ``target_hidden`` rows. Group position and slot metadata must contain exactly
    four times that row count in rank-major order; padded rows use slot ``-1``.
    The context and group-query RoPE rows use the same rank-major order as their
    position metadata; local-query RoPE rows follow the local query order.
    Physical slots use one group-global block namespace replicated on all ranks.
    The group row count must not exceed ``PREFILL_GROUP_CAP`` (8192). The runtime
    does not validate relationships between dynamic tensor axes, so the
    dispatching host must reject inputs that violate this contract.
    """
    target_hidden.bind_dynamic(0, T_MAIN_DYN)
    context_group_position_ids.bind_dynamic(0, CP_CONTEXT_T_DYN)
    context_group_slot_mapping.bind_dynamic(1, CP_CONTEXT_T_DYN)
    context_group_freqs_cos.bind_dynamic(0, CP_CONTEXT_T_DYN)
    context_group_freqs_sin.bind_dynamic(0, CP_CONTEXT_T_DYN)
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
    query_token_ids = pl.create_tensor([T], dtype=pl.INT64)
    hidden_0_flat = pl.reshape(initial_hidden, [T, HC_MULT * D])
    target_tokens_padded = (target_tokens + MAIN_PROJ_T_TILE - 1) // MAIN_PROJ_T_TILE * MAIN_PROJ_T_TILE
    projected = pl.create_tensor([target_tokens, D], dtype=pl.BF16)
    main_x = pl.create_tensor([target_tokens, D], dtype=pl.BF16)
    ordered_target_hidden = pl.create_tensor([target_tokens, MAIN_IN], dtype=pl.BF16)
    ordered_target_hidden, history_ready_tid = history_ring_to_window(
        target_hidden, anchor_positions, ordered_target_hidden, prior_cycle_gate
    )
    for proj_idx in pl.spmd(
        target_tokens_padded // MAIN_PROJ_T_TILE * (D // MAIN_PROJ_N_TILE),
        name_hint="dspark_main_proj",
        allow_early_resolve=True,
        deps=[history_ready_tid],
    ):
        token_begin = proj_idx // (D // MAIN_PROJ_N_TILE) * MAIN_PROJ_T_TILE
        hidden_begin = proj_idx % (D // MAIN_PROJ_N_TILE) * MAIN_PROJ_N_TILE
        projection_rows = pl.min(MAIN_PROJ_T_TILE, target_tokens - token_begin)
        hidden_k0 = pl.slice(
            ordered_target_hidden,
            [MAIN_PROJ_T_TILE, MAIN_PROJ_K_TILE],
            [token_begin, 0],
            valid_shape=[projection_rows, MAIN_PROJ_K_TILE],
        )
        weight_k0 = main_proj_weight[hidden_begin : hidden_begin + MAIN_PROJ_N_TILE, 0:MAIN_PROJ_K_TILE]
        projection = pl.matmul(hidden_k0, weight_k0, b_trans=True, out_dtype=pl.FP32)
        for reduce_begin in pl.pipeline(MAIN_PROJ_K_TILE, MAIN_IN, MAIN_PROJ_K_TILE, stage=2):
            hidden_k = pl.slice(
                ordered_target_hidden,
                [MAIN_PROJ_T_TILE, MAIN_PROJ_K_TILE],
                [token_begin, reduce_begin],
                valid_shape=[projection_rows, MAIN_PROJ_K_TILE],
            )
            weight_k = main_proj_weight[
                hidden_begin : hidden_begin + MAIN_PROJ_N_TILE, reduce_begin : reduce_begin + MAIN_PROJ_K_TILE
            ]
            projection = pl.matmul_acc(projection, hidden_k, weight_k, b_trans=True)
        projection_bf16 = pl.cast(projection, target_type=pl.BF16, mode="rint")
        projection_valid = pl.set_validshape(projection_bf16, projection_rows, MAIN_PROJ_N_TILE)
        projected[
            token_begin : token_begin + MAIN_PROJ_T_TILE, hidden_begin : hidden_begin + MAIN_PROJ_N_TILE
        ] = projection_valid
    rms_norm(projected, main_norm_weight, main_x)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_query_ids", deps=[prior_cycle_gate]):
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
    query_hc = pl.reshape(hidden_0_flat, [DSPARK_MOE_TOKENS, HC_MULT, D])
    lookup_embedding(query_token_ids, embedding_weight, lookup_hidden, query_hc)
    group_context_tokens = pl.tensor.dim(context_group_position_ids, 0)
    group_main_x = pl.create_tensor([group_context_tokens, D], dtype=pl.BF16)
    _gathered_context, hidden_gather_signal = prefill_cp_token_allgather_step(
        main_x,
        group_main_x,
        hidden_gather_window,
        hidden_gather_signal,
        group_base,
        dsa_cp_rank,
        prior_cycle_gate,
    )
    dspark_context_kv(
        group_main_x,
        wkv_0,
        gamma_ckv_0,
        context_group_freqs_cos,
        context_group_freqs_sin,
        context_group_position_ids,
        context_group_slot_mapping,
        pl.const(0, pl.INT32),
        kv_cache_0,
    )
    dspark_context_kv(
        group_main_x,
        wkv_1,
        gamma_ckv_1,
        context_group_freqs_cos,
        context_group_freqs_sin,
        context_group_position_ids,
        context_group_slot_mapping,
        pl.const(1, pl.INT32),
        kv_cache_1,
    )
    dspark_context_kv(
        group_main_x,
        wkv_2,
        gamma_ckv_2,
        context_group_freqs_cos,
        context_group_freqs_sin,
        context_group_position_ids,
        context_group_slot_mapping,
        pl.const(2, pl.INT32),
        kv_cache_2,
    )

    swa_indices = pl.create_tensor(
        [DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32
    )
    swa_lens = pl.create_tensor([DSPARK_DRAFT_LAYERS, DSPARK_MAX_BATCH], dtype=pl.INT32)
    query_positions = pl.create_tensor([T_QUERY], dtype=pl.INT32)
    (
        swa_indices,
        swa_lens,
        query_positions,
    ) = build_dspark_metadata(
        anchor_positions, block_tables, swa_indices, swa_lens, query_positions, prior_cycle_gate
    )
    swa_indices_0 = swa_indices[0]
    swa_indices_1 = swa_indices[1]
    swa_indices_2 = swa_indices[2]
    swa_lens_0 = swa_lens[0]
    swa_lens_1 = swa_lens[1]
    swa_lens_2 = swa_lens[2]
    hidden_1 = intermediate_hidden[0]
    hidden_1, hidden_gather_signal = draft_layer(
        initial_hidden,
        pl.const(0, pl.INT32),
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        query_freqs_cos,
        query_freqs_sin,
        query_group_freqs_cos,
        query_group_freqs_sin,
        query_positions,
        query_group_position_ids,
        kv_cache_0,
        query_group_slot_mapping[0],
        swa_indices_0,
        swa_lens_0,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        ffn_norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        query_token_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        hidden_1,
        hidden_gather_window,
        hidden_gather_signal,
        attention_window,
        attention_signal,
        o_window,
        o_signal,
        group_base,
        dsa_cp_rank,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        data_arrived,
        routed_y_buf,
        combine_arrived,
        pl.const(40, pl.INT32),
        active_tokens,
        my_rank,
        pl.const(1, pl.INT32),
        prior_cycle_gate,
    )

    hidden_2 = intermediate_hidden[1]
    hidden_2, hidden_gather_signal = draft_layer(
        hidden_1,
        pl.const(1, pl.INT32),
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        query_freqs_cos,
        query_freqs_sin,
        query_group_freqs_cos,
        query_group_freqs_sin,
        query_positions,
        query_group_position_ids,
        kv_cache_1,
        query_group_slot_mapping[1],
        swa_indices_1,
        swa_lens_1,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        ffn_norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        query_token_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        hidden_2,
        hidden_gather_window,
        hidden_gather_signal,
        attention_window,
        attention_signal,
        o_window,
        o_signal,
        group_base,
        dsa_cp_rank,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        data_arrived,
        routed_y_buf,
        combine_arrived,
        pl.const(41, pl.INT32),
        active_tokens,
        my_rank,
        pl.const(2, pl.INT32),
        prior_cycle_gate,
    )

    hidden_3 = intermediate_hidden[2]
    hidden_3, hidden_gather_signal = draft_layer(
        hidden_2,
        pl.const(2, pl.INT32),
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        query_freqs_cos,
        query_freqs_sin,
        query_group_freqs_cos,
        query_group_freqs_sin,
        query_positions,
        query_group_position_ids,
        kv_cache_2,
        query_group_slot_mapping[2],
        swa_indices_2,
        swa_lens_2,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        ffn_norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        query_token_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        hidden_3,
        hidden_gather_window,
        hidden_gather_signal,
        attention_window,
        attention_signal,
        o_window,
        o_signal,
        group_base,
        dsa_cp_rank,
        recv_meta,
        recv_x,
        recv_aux,
        recv_route,
        arrived,
        data_arrived,
        routed_y_buf,
        combine_arrived,
        pl.const(42, pl.INT32),
        active_tokens,
        my_rank,
        pl.const(3, pl.INT32),
        prior_cycle_gate,
    )
    cleanup_tid = clear_moe_signals(hidden_3, arrived, data_arrived, combine_arrived)

    padded_head_hidden = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_head(hidden_3, hc_head_fn, hc_head_scale, hc_head_base, padded_head_hidden)
    head_hidden_flat = pl.reshape(head_hidden, [batch * DSPARK_QUERY_WIDTH, D])
    with pl.spmd(batch * DSPARK_QUERY_WIDTH, name_hint="dspark_head_unpad") as head_hidden_ready_tid:
        token = pl.tile.get_block_idx()
        head_hidden_flat[token : token + 1, :] = padded_head_hidden[
            token : token + 1,
            :,
        ]
    return head_hidden, head_hidden_ready_tid, cleanup_tid


dspark_drafter = pl.jit.inline(auto_scope=False)(_dspark_drafter)
