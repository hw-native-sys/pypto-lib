# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2 # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 Flash single-request prefill layer with MoE EP2.

The layer intentionally supports only one fresh 128-token prompt
(``B=1, S=128``). Multi-request packing, chunked prefill, and long-sequence
continuation are outside this standalone layer's contract.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

# The prefill path routes PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe (which freezes recv shapes and derives RECV_MAX = EP * MOE_TOKENS at import).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS
# Import moe first. It applies the EP2 FLASH override before dependent
# modules bake config-derived MoE shapes.
from moe import (
    AUX_PAD,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    clear_moe_signals,
    golden_moe,
    moe,
)
from config import FLASH as MODEL_CONFIG, PREFILL_BATCH, PREFILL_SEQ
from prefill_attention_swa import (
    BLOCK_NUM as SWA_ORI_BLOCK_NUM,
    BLOCK_SIZE as SWA_BLOCK_SIZE,
    build_tensor_specs as build_swa_attention_tensor_specs,
    golden_prefill_attention_swa,
    prefill_attention_swa,
)
from prefill_attention_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    HCA_CMP_BLOCK_NUM,
    HCA_ORI_BLOCK_NUM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    build_tensor_specs as build_hca_attention_tensor_specs,
    golden_prefill_attention_hca,
    prefill_attention_hca,
)
from prefill_attention_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    PREFILL_IDX_BLOCK_NUM,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    build_tensor_specs as build_csa_attention_tensor_specs,
    golden_prefill_attention_csa,
    prefill_attention_csa,
)
assert SWA_BLOCK_SIZE == BLOCK_SIZE, "SWA/HCA/CSA must share the PyPTO block size"
assert SWA_ORI_BLOCK_NUM == HCA_ORI_BLOCK_NUM == CSA_ORI_BLOCK_NUM
assert HCA_CMP_BLOCK_NUM == CSA_CMP_BLOCK_NUM

# The standalone layer is deliberately fixed to one full child-kernel tile.
TOK_TILE = T
USER_BATCH = 1

assert PREFILL_BATCH == USER_BATCH, "prefill_layer requires B=1"
assert PREFILL_SEQ == TOK_TILE == 128, "prefill_layer requires S=128"

# Fixed cache/state/table capacities for the one supported request.
ORI_CACHE_BLOCKS = CSA_ORI_BLOCK_NUM
CMP_CACHE_BLOCKS = CSA_CMP_BLOCK_NUM
IDX_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
ORI_TABLE_BLOCKS = SPARSE_ORI_MAX_BLOCKS
CMP_TABLE_BLOCKS = SPARSE_CMP_MAX_BLOCKS
IDX_TABLE_BLOCKS = IDX_CACHE_MAX_BLOCKS

@pl.jit
def prefill_layer_core(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[
        [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, 2 * HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, 2 * CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, 2 * INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[ORI_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[CMP_TABLE_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCKS, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCKS, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_TABLE_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    for request_id in pl.range(USER_BATCH):
        ridx = pl.cast(request_id, pl.INDEX)

        # The single request addresses the fixed-capacity physical pools.
        kv_cache_req = kv_cache
        ori_block_table_req = pl.slice(ori_block_table, [ORI_TABLE_BLOCKS], [ridx * ORI_TABLE_BLOCKS])
        cmp_kv_req = cmp_kv
        cmp_block_table_req = pl.slice(cmp_block_table, [CMP_TABLE_BLOCKS], [ridx * CMP_TABLE_BLOCKS])
        idx_kv_cache_req = idx_kv_cache
        idx_kv_scale_req = idx_kv_scale
        idx_block_table_req = pl.slice(idx_block_table, [IDX_TABLE_BLOCKS], [ridx * IDX_TABLE_BLOCKS])
        hca_compress_state_req = hca_compress_state
        hca_state_table_req = pl.slice(hca_compress_state_block_table, [HCA_STATE_MAX_BLOCKS],
                                       [ridx * HCA_STATE_MAX_BLOCKS])
        csa_compress_state_req = csa_compress_state
        csa_state_table_req = pl.slice(csa_compress_state_block_table, [CSA_STATE_MAX_BLOCKS],
                                       [ridx * CSA_STATE_MAX_BLOCKS])
        csa_inner_compress_state_req = csa_inner_compress_state
        csa_inner_state_table_req = pl.slice(csa_inner_compress_state_block_table, [INNER_STATE_MAX_BLOCKS],
                                             [ridx * INNER_STATE_MAX_BLOCKS])

        for tile_id in pl.range(1):
            tile_base = tile_id * TOK_TILE
            valid_n = pl.cast(TOK_TILE, pl.INT32)
            moe_epoch = pl.cast(1, pl.INT32)

            # The only supported request is exactly one full child-kernel tile.
            x_hc_tile = pl.slice(x_hc, [TOK_TILE, HC_MULT, D], [tile_base, 0, 0])
            ori_slot_tile = pl.slice(ori_slot_mapping, [TOK_TILE], [tile_base])
            position_ids_tile = pl.slice(position_ids, [TOK_TILE], [tile_base])
            hca_cmp_slot_tile = pl.slice(hca_cmp_slot_mapping, [TOK_TILE], [tile_base])
            hca_state_slot_tile = pl.slice(hca_state_slot_mapping, [TOK_TILE], [tile_base])
            csa_cmp_slot_tile = pl.slice(csa_cmp_slot_mapping, [TOK_TILE], [tile_base])
            csa_idx_slot_tile = pl.slice(csa_idx_slot_mapping, [TOK_TILE], [tile_base])
            csa_state_slot_tile = pl.slice(csa_state_slot_mapping, [TOK_TILE], [tile_base])
            csa_inner_state_slot_tile = pl.slice(csa_inner_state_slot_mapping, [TOK_TILE], [tile_base])
            input_ids_tile = pl.slice(input_ids, [TOK_TILE], [tile_base])

            x_attn_tile = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.FP32)
            if layer_id < 2:
                prefill_attention_swa(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    kv_cache_req, ori_block_table_req, ori_slot_tile,
                    position_ids_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )
            elif layer_id % 2 == 1:
                prefill_attention_hca(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
                    hca_compress_state_req, hca_state_table_req,
                    kv_cache_req, ori_slot_tile, ori_block_table_req,
                    cmp_kv_req, cmp_block_table_req,
                    position_ids_tile, hca_cmp_slot_tile, hca_state_slot_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )
            else:
                prefill_attention_csa(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
                    csa_compress_state_req, csa_state_table_req,
                    csa_hadamard_idx,
                    csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj,
                    csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
                    csa_inner_compress_state_req, csa_inner_state_table_req,
                    kv_cache_req, ori_block_table_req, ori_slot_tile,
                    cmp_kv_req, cmp_block_table_req, idx_kv_cache_req, idx_kv_scale_req, idx_block_table_req,
                    position_ids_tile, csa_cmp_slot_tile, csa_idx_slot_tile,
                    csa_state_slot_tile, csa_inner_state_slot_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )

            x_next_tile = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.FP32)
            moe(
                x_attn_tile,
                hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                norm_w, gate_w, gate_bias, tid2eid, input_ids_tile,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                x_next_tile,
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived,
                layer_id, valid_n, my_rank, moe_epoch,
            )

            # Write the one full tile to the fixed output.
            x_next = pl.assemble(x_next, x_next_tile, [tile_base, 0, 0])
    clear_moe_signals(x_next, arrived, data_arrived, combine_arrived)
    return x_next


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[
        [N_RANKS, HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, 2 * HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[[N_RANKS, CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, 2 * CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[[N_RANKS, INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, 2 * INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, ORI_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, CMP_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, CMP_TABLE_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, IDX_CACHE_BLOCKS, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, IDX_CACHE_BLOCKS, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[N_RANKS, IDX_TABLE_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        prefill_layer_core(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank], hca_cmp_norm_w[rank],
            hca_compress_state[rank], hca_compress_state_block_table[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank],
            csa_compress_state[rank], csa_compress_state_block_table[rank],
            csa_hadamard_idx[rank],
            csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank], csa_weights_proj[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_compress_state[rank],
            csa_inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping[rank],
            cmp_kv[rank], cmp_block_table[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            position_ids[rank],
            hca_cmp_slot_mapping[rank], hca_state_slot_mapping[rank],
            csa_cmp_slot_mapping[rank], csa_idx_slot_mapping[rank],
            csa_state_slot_mapping[rank], csa_inner_state_slot_mapping[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_next[rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            layer_id, rank,
            device=rank,
        )


HOST_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "csa_inner_compress_state_block_table",
    "kv_cache",
    "ori_block_table",
    "ori_slot_mapping",
    "cmp_kv",
    "cmp_block_table",
    "idx_kv_cache",
    "idx_kv_scale",
    "idx_block_table",
    "position_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "input_ids",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "x_next",
)


# ---------------------------------------------------------------------------
# Host-side fixed metadata builder, tensor specs, and golden reference.
# ---------------------------------------------------------------------------

_KIND_BUILDER = {
    "swa": build_swa_attention_tensor_specs,
    "hca": build_hca_attention_tensor_specs,
    "csa": build_csa_attention_tensor_specs,
}

# Child-local token-metadata tensors gathered for the fixed tile.
_TOKEN_META_NAMES = {
    "position_ids", "ori_slot_mapping",
    "cmp_slot_mapping", "state_slot_mapping", "idx_slot_mapping", "inner_state_slot_mapping",
}
# Child cache/state pools plus request-local table views (persist across tiles).
_CACHE_STATE_NAMES = {
    "kv_cache", "block_table", "ori_block_table", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "compress_state", "compress_state_block_table",
    "inner_compress_state", "inner_compress_state_block_table",
}

# Global cache/state pools and child-local mappings.
_PACKED_CACHE_SPECS = {
    "kv_cache": "kv_cache",
    "ori_block_table": "ori_block_table",
    "cmp_kv": "cmp_kv",
    "cmp_block_table": "cmp_block_table",
    "idx_kv_cache": "idx_kv_cache",
    "idx_kv_scale": "idx_kv_scale",
    "idx_block_table": "idx_block_table",
    "hca_compress_state": ("hca", "compress_state"),
    "hca_compress_state_block_table": ("hca", "compress_state_block_table"),
    "csa_compress_state": ("csa", "compress_state"),
    "csa_compress_state_block_table": ("csa", "compress_state_block_table"),
    "csa_inner_compress_state": ("csa", "inner_compress_state"),
    "csa_inner_compress_state_block_table": ("csa", "inner_compress_state_block_table"),
}

_HISTORY_CACHE_NAMES = {
    "kv_cache", "cmp_kv", "idx_kv_cache", "idx_kv_scale",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state",
}


def _req_block_count(kind, child_name):
    """Per-request dim0 of a child-local cache/state/table tensor."""
    if child_name == "kv_cache":
        return ORI_CACHE_BLOCKS
    if child_name in ("block_table", "ori_block_table"):
        return ORI_TABLE_BLOCKS
    if child_name == "cmp_kv":
        return CMP_CACHE_BLOCKS
    if child_name == "cmp_block_table":
        return CMP_TABLE_BLOCKS
    if child_name in ("idx_kv_cache", "idx_kv_scale"):
        return IDX_CACHE_BLOCKS
    if child_name == "idx_block_table":
        return IDX_TABLE_BLOCKS
    if child_name == "compress_state":
        return HCA_STATE_BLOCK_NUM if kind == "hca" else CSA_STATE_BLOCK_NUM
    if child_name == "compress_state_block_table":
        return HCA_STATE_MAX_BLOCKS if kind == "hca" else CSA_STATE_MAX_BLOCKS
    if child_name == "inner_compress_state":
        return INNER_STATE_BLOCK_NUM
    if child_name == "inner_compress_state_block_table":
        return INNER_STATE_MAX_BLOCKS
    raise KeyError(child_name)


def _child_to_packed(kind, child_name):
    """Map a child-local cache/state name to the layer-level tensor name."""
    if child_name in ("block_table", "ori_block_table"):
        return "ori_block_table"
    if child_name in ("kv_cache", "cmp_kv", "cmp_block_table", "idx_kv_cache", "idx_kv_scale", "idx_block_table"):
        return child_name
    prefix = "hca_" if kind == "hca" else "csa_"
    return prefix + child_name


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def _attention_kind_for_layer(layer_id):
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def _tile_token_meta(kind, context_len, valid_tok, torch):
    """Child-local [T] token metadata for one tile, via the fixed-T child builder.

    Reuses the existing single-tile builders, which already encode the
    absolute-position paged-cache/state coordinate logic. ``context_len``
    is the tile's absolute start position; ``valid_tok`` its active token count.
    """
    from golden import TensorSpec

    specs = {s.name: s for s in _KIND_BUILDER[kind](start_pos=context_len, num_tokens=valid_tok)
             if isinstance(s, TensorSpec)}
    meta = {name: _spec_value(specs[name], torch) for name in specs if name in _TOKEN_META_NAMES}
    return meta


def _fixed_token_metadata(kind, torch):
    """Build rank-shared metadata for one fresh 128-token prompt."""
    meta = _tile_token_meta(kind, context_len=0, valid_tok=T, torch=torch)
    pos = meta["position_ids"][:T]
    ori_slot = meta["ori_slot_mapping"][:T]
    hca_cmp = torch.full((T,), -1, dtype=torch.int64)
    hca_state = torch.full((T,), -1, dtype=torch.int64)
    csa_cmp = torch.full((T,), -1, dtype=torch.int64)
    csa_idx = torch.full((T,), -1, dtype=torch.int64)
    csa_state = torch.full((T,), -1, dtype=torch.int64)
    csa_inner = torch.full((T,), -1, dtype=torch.int64)

    if kind == "hca":
        hca_cmp = meta["cmp_slot_mapping"][:T]
        hca_state = meta["state_slot_mapping"][:T]
    elif kind == "csa":
        csa_cmp = meta["cmp_slot_mapping"][:T]
        csa_idx = meta["idx_slot_mapping"][:T]
        csa_state = meta["state_slot_mapping"][:T]
        csa_inner = meta["inner_state_slot_mapping"][:T]

    return {
        "position_ids": pos,
        "ori_slot_mapping": ori_slot,
        "hca_cmp_slot_mapping": hca_cmp,
        "hca_state_slot_mapping": hca_state,
        "csa_cmp_slot_mapping": csa_cmp,
        "csa_idx_slot_mapping": csa_idx,
        "csa_state_slot_mapping": csa_state,
        "csa_inner_state_slot_mapping": csa_inner,
    }


def build_tensor_specs(layer_id=2):
    """Build tensor specs for one fresh 128-token request."""
    import torch
    from golden import ScalarSpec, TensorSpec

    kind = _attention_kind_for_layer(layer_id)
    total_tokens = T

    def kind_specs(build_fn):
        return {s.name: s for s in build_fn(start_pos=0, num_tokens=T) if isinstance(s, TensorSpec)}

    swa = kind_specs(build_swa_attention_tensor_specs)
    hca = kind_specs(build_hca_attention_tensor_specs)
    csa = kind_specs(build_csa_attention_tensor_specs)
    active = {"swa": swa, "hca": hca, "csa": csa}[kind]
    src_by_kind = {"swa": swa, "hca": hca, "csa": csa}

    def ranked_init(src):
        def init():
            return torch.stack([_spec_value(src, torch) for _ in range(N_RANKS)], dim=0).contiguous()
        return init

    def replicate(values):
        def init():
            return torch.stack([values.clone() for _ in range(N_RANKS)], dim=0).contiguous()
        return init

    # Per-rank weight tensors (same selection as prefill_layer.py minus token +
    # cache/state tensors, which are rebuilt for the standalone layer below).
    weight_specs = [
        ("hc_attn_fn", active["hc_attn_fn"]),
        ("hc_attn_scale", active["hc_attn_scale"]),
        ("hc_attn_base", active["hc_attn_base"]),
        ("attn_norm_w", active["attn_norm_w"]),
        ("wq_a", active["wq_a"]),
        ("wq_b", active["wq_b"]),
        ("wq_b_scale", active["wq_b_scale"]),
        ("wkv", active["wkv"]),
        ("gamma_cq", active["gamma_cq"]),
        ("gamma_ckv", active["gamma_ckv"]),
        ("freqs_cos", active["freqs_cos"]),
        ("freqs_sin", active["freqs_sin"]),
        ("hca_cmp_wkv", hca["cmp_wkv"]),
        ("hca_cmp_wgate", hca["cmp_wgate"]),
        ("hca_cmp_ape", hca["cmp_ape"]),
        ("hca_cmp_norm_w", hca["cmp_norm_w"]),
        ("csa_cmp_wkv", csa["cmp_wkv"]),
        ("csa_cmp_wgate", csa["cmp_wgate"]),
        ("csa_cmp_ape", csa["cmp_ape"]),
        ("csa_cmp_norm_w", csa["cmp_norm_w"]),
        ("csa_hadamard_idx", csa["hadamard_idx"]),
        ("csa_idx_wq_b", csa["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa["idx_wq_b_scale"]),
        ("csa_weights_proj", csa["idx_weights_proj"]),
        ("csa_inner_wkv", csa["inner_wkv"]),
        ("csa_inner_wgate", csa["inner_wgate"]),
        ("csa_inner_ape", csa["inner_ape"]),
        ("csa_inner_norm_w", csa["inner_norm_w"]),
        ("attn_sink", active["attn_sink"]),
        ("wo_a", active["wo_a"]),
        ("wo_b", active["wo_b"]),
        ("wo_b_scale", active["wo_b_scale"]),
    ]

    tensor_specs = [TensorSpec(name, [N_RANKS, *src.shape], src.dtype, init_value=ranked_init(src))
                    for name, src in weight_specs]

    # Token metadata is rank-shared; x_hc/input_ids carry per-rank data.
    meta = _fixed_token_metadata(kind, torch)

    def init_x_hc():
        return (torch.rand(N_RANKS, T, HC_MULT, D, dtype=torch.float32) - 0.5) / 10.0

    def init_input_ids():
        ids = [((torch.arange(T, dtype=torch.int64) + rank) % VOCAB) for rank in range(N_RANKS)]
        return torch.stack(ids, dim=0).contiguous()

    tensor_specs.append(TensorSpec("x_hc", [N_RANKS, total_tokens, HC_MULT, D], torch.float32, init_value=init_x_hc))
    tensor_specs.append(TensorSpec("input_ids", [N_RANKS, total_tokens], torch.int64, init_value=init_input_ids))
    tensor_specs.append(TensorSpec("position_ids", [N_RANKS, total_tokens], torch.int32,
                                   init_value=replicate(meta["position_ids"])))
    tensor_specs.append(TensorSpec("ori_slot_mapping", [N_RANKS, total_tokens], torch.int64,
                                   init_value=replicate(meta["ori_slot_mapping"])))
    for name in ("hca_cmp_slot_mapping", "hca_state_slot_mapping", "csa_cmp_slot_mapping",
                 "csa_idx_slot_mapping", "csa_state_slot_mapping", "csa_inner_state_slot_mapping"):
        tensor_specs.append(TensorSpec(name, [N_RANKS, total_tokens], torch.int64, init_value=replicate(meta[name])))

    def resolve_cache_src(packed_name, info):
        """Resolve (source spec, source kind, child-local name) for a layer cache."""
        if isinstance(info, tuple):
            sk, cn = info
            return src_by_kind[sk][cn], sk, cn
        cn = info
        if cn == "ori_block_table":
            return (active.get("ori_block_table") or swa["block_table"]), kind, cn
        if cn in ("cmp_kv", "cmp_block_table"):
            return (active.get(cn) or csa[cn]), kind, cn
        if cn in ("idx_kv_cache", "idx_kv_scale", "idx_block_table"):
            return csa[cn], kind, cn
        return active[cn], kind, cn  # kv_cache

    # Fixed-capacity cache/state pools and tables for the fresh request.
    for packed_name, info in _PACKED_CACHE_SPECS.items():
        src, _, _ = resolve_cache_src(packed_name, info)
        value = _spec_value(src, torch)

        def make_init(value=value):
            def init():
                return torch.stack([value.clone() for _ in range(N_RANKS)], dim=0).contiguous()

            return init

        tensor_specs.append(
            TensorSpec(
                packed_name,
                [N_RANKS, *src.shape],
                src.dtype,
                init_value=make_init(),
                is_output=packed_name in _HISTORY_CACHE_NAMES,
            )
        )

    # MoE weight tensors (per rank). tid2eid keeps its hash-table init.
    for spec in build_moe_tensor_specs(layer_id=layer_id):
        if not isinstance(spec, TensorSpec) or spec.name in {"x_hc", "x_next", "input_ids"}:
            continue
        if spec.name == "tid2eid":
            def init_tid2eid(spec=spec):
                _, vocab, topk = spec.shape
                ids = torch.arange(vocab, dtype=torch.int64).view(vocab, 1)
                ks = torch.arange(topk, dtype=torch.int64).view(1, topk)
                table = ((ids * topk + ks) % N_EXPERTS_GLOBAL).to(dtype=spec.dtype)
                return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

            tensor_specs.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_tid2eid))
        else:
            tensor_specs.append(spec)

    tensor_specs.append(TensorSpec("x_next", [N_RANKS, total_tokens, HC_MULT, D], torch.float32, is_output=True))

    # Keep static weight parameters device-resident (child_memory), sharded per
    # rank. Cache/state/table tensors remain host tensors for output validation.
    RESIDENT_WEIGHT_NAMES = frozenset([
        # Attention core weights + RoPE tables
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin",
        # HCA / CSA compressor + indexer weights (states/block tables excluded)
        "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
        "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
        "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
        "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
        # Attention output projection
        "attn_sink", "wo_a", "wo_b", "wo_b_scale",
        # MoE FFN / gate / experts + static route table
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
    ])
    for spec in tensor_specs:
        if spec.name in RESIDENT_WEIGHT_NAMES:
            spec.resident = "stacked"

    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing prefill layer tensor specs: {missing}")
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + [
        ScalarSpec("layer_id", torch.int32, layer_id),
    ]


def golden_prefill_layer(tensors):
    """Reference for the fixed one-request, one-tile prefill layer."""
    import torch
    from golden import TensorSpec

    layer_id = int(tensors["layer_id"])
    kind = _attention_kind_for_layer(layer_id)

    # Map child-local attention tensor names to layer-level names.
    mapped = dict(tensors)
    if kind == "swa":
        mapped["block_table"] = tensors["ori_block_table"]
        attention_golden = golden_prefill_attention_swa
    elif kind == "hca":
        mapped.update({
            "cmp_wkv": tensors["hca_cmp_wkv"], "cmp_wgate": tensors["hca_cmp_wgate"],
            "cmp_ape": tensors["hca_cmp_ape"], "cmp_norm_w": tensors["hca_cmp_norm_w"],
            "compress_state": tensors["hca_compress_state"],
            "compress_state_block_table": tensors["hca_compress_state_block_table"],
            "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"], "state_slot_mapping": tensors["hca_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_hca
    else:
        mapped.update({
            "cmp_wkv": tensors["csa_cmp_wkv"], "cmp_wgate": tensors["csa_cmp_wgate"],
            "cmp_ape": tensors["csa_cmp_ape"], "cmp_norm_w": tensors["csa_cmp_norm_w"],
            "compress_state": tensors["csa_compress_state"],
            "compress_state_block_table": tensors["csa_compress_state_block_table"],
            "hadamard_idx": tensors["csa_hadamard_idx"], "idx_wq_b": tensors["csa_idx_wq_b"],
            "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"], "idx_weights_proj": tensors["csa_weights_proj"],
            "inner_wkv": tensors["csa_inner_wkv"], "inner_wgate": tensors["csa_inner_wgate"],
            "inner_ape": tensors["csa_inner_ape"], "inner_norm_w": tensors["csa_inner_norm_w"],
            "inner_compress_state": tensors["csa_inner_compress_state"],
            "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"],
            "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"], "idx_slot_mapping": tensors["csa_idx_slot_mapping"],
            "state_slot_mapping": tensors["csa_state_slot_mapping"],
            "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_csa

    attn_specs = _KIND_BUILDER[kind](start_pos=0, num_tokens=T)
    x_next = tensors["x_next"]

    def tile_buffer(packed_per_rank, rank, base, _valid, feature_shape, dtype):
        buf = torch.zeros((T, *feature_shape), dtype=dtype)
        buf[:] = packed_per_rank[rank, base:base + T]
        return buf

    for request_id in range(USER_BATCH):
        # Global mutable pool views plus request-local table views.
        req_views = {}
        for packed_name, info in _PACKED_CACHE_SPECS.items():
            if packed_name in _HISTORY_CACHE_NAMES:
                req_views[packed_name] = tensors[packed_name]
                continue
            child_name = info[1] if isinstance(info, tuple) else info
            cnt = _req_block_count(kind, child_name)
            req_views[packed_name] = tensors[packed_name][:, request_id * cnt:(request_id + 1) * cnt]

        for tile_id in range(1):
            valid = T
            base = tile_id * T

            x_attn_tile = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
            for rank in range(N_RANKS):
                attn_tensors = {}
                for spec in attn_specs:
                    if not isinstance(spec, TensorSpec):
                        continue  # scalar (num_tokens) set explicitly below
                    name = spec.name
                    if name == "x_out":
                        attn_tensors[name] = x_attn_tile[rank]
                    elif name == "x_hc":
                        attn_tensors[name] = tile_buffer(tensors["x_hc"], rank, base, valid, (HC_MULT, D), torch.float32)
                    elif name in _TOKEN_META_NAMES:
                        packed = mapped[name]
                        attn_tensors[name] = tile_buffer(packed, rank, base, valid, tuple(packed.shape[2:]), packed.dtype)
                    elif name in _CACHE_STATE_NAMES:
                        attn_tensors[name] = req_views[_child_to_packed(kind, name)][rank]
                    else:
                        attn_tensors[name] = mapped[name][rank]
                attn_tensors["num_tokens"] = valid
                attention_golden(attn_tensors)
                x_attn_tile[rank] = attn_tensors["x_out"]

            moe_tensors = dict(tensors)
            moe_tensors["x_hc"] = x_attn_tile
            input_ids_tile = torch.zeros(N_RANKS, T, dtype=torch.int64)
            input_ids_tile[:, :valid] = tensors["input_ids"][:, base:base + valid]
            moe_tensors["input_ids"] = input_ids_tile
            moe_tensors["num_tokens"] = valid
            x_next_tile = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
            moe_tensors["x_next"] = x_next_tile
            golden_moe(moe_tensors)

            x_next[:, base:base + valid] = x_next_tile[:, :valid]


def valid_ratio_reldiff(diff_thd, pct_thd):
    """Relative-diff comparator for the fixed 128 logical token rows."""
    from golden import ratio_reldiff

    return ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd)


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe)")
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--layer-id", type=int, default=2,
                        help="Layer id selects attention by MODEL_CONFIG.compress_ratios[layer_id].")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_prefill_layer,
        specs=build_tensor_specs(layer_id=args.layer_id),
        golden_fn=golden_prefill_layer,
        compile_only=args.compile_only,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Real-weight x_next over-thd fractions (frac>5e-3 / frac>1e-2):
            # B=1, S=128 single-request prefill.
            "x_next": valid_ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            # CSA runs preserve the standalone child-kernel precision
            # contracts: sparse BF16 cache rows may differ by one ULP, C8 rows
            # by one LSB, and only a small fraction of recurrent-state values
            # cross the strict pointwise FP32 threshold.
            "csa_compress_state": ratio_allclose(
                atol=1e-3, rtol=1e-3, max_error_ratio=0.005
            ),
            "csa_inner_compress_state": ratio_allclose(
                atol=1e-3, rtol=1e-3, max_error_ratio=0.005
            ),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005),
            "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
