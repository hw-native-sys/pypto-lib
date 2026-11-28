# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Preserve actual owner-zero selection in the eight-rank candidate."""

import pypto.language as pl
import pypto.language.distributed as pld
from .draft_entry_tp8 import draft_tp8_b1
from .entry import (
    BANK_COUNT,
    BLOCK_SIZE,
    D,
    DSPARK_DRAFT_LAYERS,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    MAIN_IN,
    METADATA_B_DYN,
    MIX_HC,
    N_EXPERTS_GLOBAL,
    ORI_BLOCK_NUM,
    Q_LORA,
    T,
    TOPK,
    VOCAB,
    build_dspark_metadata,
)


@pl.jit.inline(auto_scope=False)
def select_owner_zero_metadata(
    request_state: pl.Tensor[[4], pl.INT64],
    context_pos_bank: pl.Tensor[[BANK_COUNT,512],pl.INT32],
    context_slots_bank: pl.Tensor[[BANK_COUNT,3,512],pl.INT64],
    query_pos_bank: pl.Tensor[[BANK_COUNT,80],pl.INT32],
    query_slots_bank: pl.Tensor[[BANK_COUNT,3,80],pl.INT64],
    context_cos_bank: pl.Tensor[[BANK_COUNT,512,64],pl.BF16],
    context_sin_bank: pl.Tensor[[BANK_COUNT,512,64],pl.BF16],
    query_cos_bank: pl.Tensor[[BANK_COUNT,20,64],pl.BF16],
    query_sin_bank: pl.Tensor[[BANK_COUNT,20,64],pl.BF16],
    num_sampled: pl.Tensor[[1],pl.INT32],
    last_sampled: pl.Tensor[[1],pl.INT64],
    next_prefill_tokens: pl.Tensor[[1],pl.INT64],
    anchor_positions: pl.Tensor[[1],pl.INT32],
    context_pos: pl.Tensor[[128],pl.INT32],
    context_slots: pl.Tensor[[3,512],pl.INT64],
    query_pos: pl.Tensor[[8],pl.INT32],
    query_slots: pl.Tensor[[3,80],pl.INT64],
    context_cos: pl.Tensor[[128,64],pl.BF16],
    context_sin: pl.Tensor[[128,64],pl.BF16],
    query_cos: pl.Tensor[[8,64],pl.BF16],
    query_sin: pl.Tensor[[8,64],pl.BF16],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    """Select the owner-zero prefix on-device, preserving the public bank strides."""
    first_anchor = pl.cast(pl.read(query_pos_bank,[0,0]),pl.INDEX)-1
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="drafter_state_scalars",deps=[prior_cycle_gate]):
        pl.write(num_sampled,[0],pl.cast(1,pl.INT32))
        pl.write(last_sampled,[0],pl.read(request_state,[3]))
        pl.write(next_prefill_tokens,[0],pl.cast(0,pl.INT64))
        pl.write(anchor_positions,[0],pl.cast(pl.read(request_state,[1]),pl.INT32))
    flat_ctx=pl.reshape(context_pos,[1,128])
    flat_query=pl.reshape(query_pos,[1,8])
    with pl.spmd(8,name_hint="draft_tp8_select_context_metadata",deps=[prior_cycle_gate],
                 allow_early_resolve=True):
        block=pl.tile.get_block_idx()
        row=block*16
        bank=pl.cast(pl.read(request_state,[1]),pl.INDEX)-first_anchor
        flat_ctx[0:1,row:row+16]=context_pos_bank[bank:bank+1,row:row+16]
        context_cos[row:row+16,:]=pl.reshape(context_cos_bank[bank:bank+1,row:row+16,:],[16,64])
        context_sin[row:row+16,:]=pl.reshape(context_sin_bank[bank:bank+1,row:row+16,:],[16,64])
        for layer in pl.unroll(3):
            context_slot_panel=pl.reshape(context_slots_bank[bank:bank+1,layer:layer+1,row:row+16],[1,16])
            context_slots[layer:layer+1,row:row+16]=context_slot_panel
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="draft_tp8_select_query_metadata",deps=[prior_cycle_gate],
                 allow_early_resolve=True):
        bank=pl.cast(pl.read(request_state,[1]),pl.INDEX)-first_anchor
        flat_query[0:1,0:8]=query_pos_bank[bank:bank+1,0:8]
        query_cos[0:8,:]=pl.reshape(query_cos_bank[bank:bank+1,0:8,:],[8,64])
        query_sin[0:8,:]=pl.reshape(query_sin_bank[bank:bank+1,0:8,:],[8,64])
        for layer in pl.unroll(3):
            query_slot_panel=pl.reshape(query_slots_bank[bank:bank+1,layer:layer+1,0:8],[1,8])
            query_slots[layer:layer+1,0:8]=query_slot_panel


@pl.jit.inline(auto_scope=False)
def device_drafter_tp8(
    target_hidden: pl.Tensor[[128, 12288], pl.BF16],
    main_proj_weight: pl.Tensor[[D, MAIN_IN], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    request_state: pl.Tensor[[4], pl.INT64],
    last_sampled: pl.Out[pl.Tensor[[1], pl.INT64]],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    context_group_position_ids_bank: pl.Tensor[[BANK_COUNT, 512], pl.INT32],
    context_group_slot_mapping_bank: pl.Tensor[[BANK_COUNT, 3, 512], pl.INT64],
    block_tables: pl.Tensor[[3, 1, 32768], pl.INT32],
    query_group_position_ids_bank: pl.Tensor[[BANK_COUNT, 80], pl.INT32],
    query_group_slot_mapping_bank: pl.Tensor[[BANK_COUNT, 3, 80], pl.INT64],
    context_group_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 512, 64], pl.BF16],
    context_group_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 512, 64], pl.BF16],
    query_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 20, 64], pl.BF16],
    query_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 20, 64], pl.BF16],
    query_group_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 80, 64], pl.BF16],
    query_group_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 80, 64], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[3, 1024, 4096], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[3, 4096], pl.FP32],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[DSPARK_DRAFT_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[DSPARK_DRAFT_LAYERS * HEAD_DIM], pl.BF16],
    kv_caches: pl.InOut[pl.Tensor[[DSPARK_DRAFT_LAYERS, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[3, 8], pl.FP32],
    wo_a: pl.Tensor[[3, 1024, 4096], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[3, 4096, 1024], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[DSPARK_DRAFT_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[3, 257, 256, 4096], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[3, 257, 256], pl.FP32],
    routed_w3: pl.Tensor[[3, 257, 256, 4096], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[3, 257, 256], pl.FP32],
    routed_w2: pl.Tensor[[3, 257, 512, 2048], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[3, 257, 512], pl.FP32],
    shared_w1: pl.Tensor[[3, 256, 4096], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[3, 256], pl.FP32],
    shared_w3: pl.Tensor[[3, 256, 4096], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[3, 256], pl.FP32],
    shared_w2: pl.Tensor[[3, 512, 2048], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[3, 512], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    initial_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    intermediate_hidden: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, T, HC_MULT, D], pl.FP32]],
    head_hidden: pl.Out[pl.Tensor[[1, 5, 4096], pl.BF16]],
    d_tp8_o_window: pld.DistributedTensor[[80, 4096], pl.FP32],
    d_tp8_o_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    d_tp8_max_window: pld.DistributedTensor[[640, 8], pl.FP32],
    d_tp8_max_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    d_tp8_int8_window: pld.DistributedTensor[[640, 256], pl.INT8],
    d_tp8_int8_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    d_tp8_output_window: pld.DistributedTensor[[80, 512], pl.BF16],
    d_tp8_output_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor[[1,5,4096],pl.BF16],pl.Scalar[pl.TASK_ID],pl.Scalar[pl.TASK_ID]]:
    num_sampled = pl.create_tensor([1], dtype=pl.INT32)
    next_prefill_tokens = pl.create_tensor([1], dtype=pl.INT64)
    anchor_positions = pl.create_tensor([1], dtype=pl.INT32)
    context_group_position_ids = pl.create_tensor([128], dtype=pl.INT32)
    context_group_slot_mapping = pl.create_tensor([3,512], dtype=pl.INT64)
    query_group_position_ids = pl.create_tensor([8], dtype=pl.INT32)
    query_group_slot_mapping = pl.create_tensor([3,80], dtype=pl.INT64)
    context_group_freqs_cos = pl.create_tensor([128,64], dtype=pl.BF16)
    context_group_freqs_sin = pl.create_tensor([128,64], dtype=pl.BF16)
    query_freqs_cos = pl.create_tensor([8,64], dtype=pl.BF16)
    query_freqs_sin = pl.create_tensor([8,64], dtype=pl.BF16)
    select_owner_zero_metadata(
        request_state,context_group_position_ids_bank,context_group_slot_mapping_bank,
        query_group_position_ids_bank,query_group_slot_mapping_bank,
        context_group_freqs_cos_bank,context_group_freqs_sin_bank,query_freqs_cos_bank,query_freqs_sin_bank,
        num_sampled,last_sampled,next_prefill_tokens,anchor_positions,
        context_group_position_ids,context_group_slot_mapping,query_group_position_ids,
        query_group_slot_mapping,context_group_freqs_cos,context_group_freqs_sin,query_freqs_cos,query_freqs_sin,
        prior_cycle_gate,
    )
    tables=block_tables
    tables.bind_dynamic(1,METADATA_B_DYN)
    selected_swa=pl.create_tensor([3,4,192],dtype=pl.INT32)
    selected_lens=pl.create_tensor([3,4],dtype=pl.INT32)
    selected_pos=pl.create_tensor([20],dtype=pl.INT32)
    build_dspark_metadata(anchor_positions,tables,selected_swa,selected_lens,selected_pos,prior_cycle_gate)
    ctx_pos: pl.Tensor[[128],pl.INT32] = pl.slice(context_group_position_ids,[128],[0])
    ctx_cos: pl.Tensor[[128,64],pl.BF16] = pl.slice(context_group_freqs_cos,[128,64],[0,0])
    ctx_sin: pl.Tensor[[128,64],pl.BF16] = pl.slice(context_group_freqs_sin,[128,64],[0,0])
    q_cos: pl.Tensor[[5,64],pl.BF16] = pl.slice(query_freqs_cos,[5,64],[0,0])
    q_sin: pl.Tensor[[5,64],pl.BF16] = pl.slice(query_freqs_sin,[5,64],[0,0])
    q_pos: pl.Tensor[[5],pl.INT32] = pl.slice(query_group_position_ids,[5],[0])
    q_slot: pl.Tensor[[3,5],pl.INT64] = pl.slice(query_group_slot_mapping,[3,5],[0,0])
    typed_swa_indices: pl.Tensor[[3, 1, 192],pl.INT32] = pl.slice(selected_swa,[3,1,192],[0,0,0])
    typed_swa_lens: pl.Tensor[[3, 1],pl.INT32] = pl.slice(selected_lens,[3,1],[0,0])
    typed_attention_hc_fn: pl.Tensor[[3, 24, 16384],pl.FP32] = pl.reshape(hc_attn_fn,[3,24,16384])
    typed_attention_hc_scale: pl.Tensor[[3, 3],pl.FP32] = pl.reshape(hc_attn_scale,[3,3])
    typed_attention_hc_base: pl.Tensor[[3, 24],pl.FP32] = pl.reshape(hc_attn_base,[3,24])
    typed_attention_norm: pl.Tensor[[3, 4096],pl.BF16] = pl.reshape(attn_norm_w,[3,4096])
    typed_wkv: pl.Tensor[[3, 4096, 512],pl.BF16] = pl.reshape(wkv,[3,4096,512])
    typed_gamma_cq: pl.Tensor[[3, 1024],pl.BF16] = pl.reshape(gamma_cq,[3,1024])
    typed_gamma_ckv: pl.Tensor[[3, 512],pl.BF16] = pl.reshape(gamma_ckv,[3,512])
    typed_wo_scale: pl.Tensor[[3, 4096],pl.FP32] = pl.reshape(wo_b_scale,[3,4096])
    typed_ffn_hc_fn: pl.Tensor[[3, 24, 16384],pl.FP32] = pl.reshape(hc_ffn_fn,[3,24,16384])
    typed_ffn_hc_scale: pl.Tensor[[3, 3],pl.FP32] = pl.reshape(hc_ffn_scale,[3,3])
    typed_ffn_hc_base: pl.Tensor[[3, 24],pl.FP32] = pl.reshape(hc_ffn_base,[3,24])
    typed_ffn_norm: pl.Tensor[[3, 4096],pl.BF16] = pl.reshape(ffn_norm_w,[3,4096])
    typed_gate_w: pl.Tensor[[3, 256, 4096],pl.FP32] = pl.reshape(gate_w,[3,256,4096])
    typed_gate_bias: pl.Tensor[[3, 256],pl.FP32] = pl.reshape(gate_bias,[3,256])
    typed_hash_routes: pl.Tensor[[3, 129280, 6],pl.INT32] = pl.reshape(tid2eid,[3,129280,6])
    full_head,ready=draft_tp8_b1(
        target_hidden,
        main_proj_weight,
        main_norm_weight,
        num_sampled,
        last_sampled,
        next_prefill_tokens,
        embedding_weight,
        anchor_positions,
        ctx_pos,
        ctx_cos,
        ctx_sin,
        context_group_slot_mapping,
        q_cos,
        q_sin,
        q_pos,
        q_slot,
        kv_caches,
        typed_swa_indices,
        typed_swa_lens,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        head_hidden,
        typed_attention_hc_fn,
        typed_attention_hc_scale,
        typed_attention_hc_base,
        typed_attention_norm,
        wq_a,
        wq_b,
        wq_b_scale,
        typed_wkv,
        typed_gamma_cq,
        typed_gamma_ckv,
        attn_sink,
        wo_a,
        wo_b,
        typed_wo_scale,
        typed_ffn_hc_fn,
        typed_ffn_hc_scale,
        typed_ffn_hc_base,
        typed_ffn_norm,
        typed_gate_w,
        typed_gate_bias,
        typed_hash_routes,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        d_tp8_o_window,
        d_tp8_o_flags,
        d_tp8_max_window,
        d_tp8_max_flags,
        d_tp8_int8_window,
        d_tp8_int8_flags,
        d_tp8_output_window,
        d_tp8_output_flags,
        initial_hidden,
        intermediate_hidden,
        my_rank,
        pl.const(0,pl.INT32),
        prior_cycle_gate,
    )
    with pl.at(level=pl.Level.CORE_GROUP,name_hint="d_tp8_clear_signals",deps=[ready]) as clean:
        for owner in pl.range(8):
            pl.write(d_tp8_o_flags,[owner,0],pl.const(0,pl.INT32))
            pl.write(d_tp8_max_flags,[owner,0],pl.const(0,pl.INT32))
            pl.write(d_tp8_int8_flags,[owner,0],pl.const(0,pl.INT32))
            pl.write(d_tp8_output_flags,[owner,0],pl.const(0,pl.INT32))
    return full_head,ready,clean
