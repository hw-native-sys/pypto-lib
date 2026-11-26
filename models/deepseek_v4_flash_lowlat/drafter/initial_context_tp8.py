# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real logical context and five initial query positions for TP8 Drafter."""

import pypto.language as pl
from .device_metadata import history_ring_to_window
from .qkv_proj_rope import rope_prepare
from .kv_tp8 import draft_kv_projection_tp8,draft_kv_norm_tp8
from .entry import (
    D,MAIN_IN,MAIN_PROJ_T_TILE,MAIN_PROJ_N_TILE,MAIN_PROJ_K_TILE,
    DSPARK_QUERY_WIDTH,DSPARK_MOE_TOKENS,DSPARK_NOISE_TOKEN_ID,
    HC_MULT,lookup_embedding,rms_norm,
)

T=DSPARK_MOE_TOKENS
ROWS=128


@pl.jit.inline(auto_scope=False)
def prepare_context_tp8(
    target_hidden: pl.Tensor[[ROWS,MAIN_IN],pl.BF16],
    main_proj_weight: pl.Tensor[[D,MAIN_IN],pl.BF16],
    main_norm_weight: pl.Tensor[[D],pl.BF16],
    num_sampled: pl.Tensor[[1],pl.INT32],
    last_sampled: pl.Tensor[[1],pl.INT64],
    next_prefill_tokens: pl.Tensor[[1],pl.INT64],
    embedding_weight: pl.Tensor[[129280,D],pl.BF16],
    anchor_positions: pl.Tensor[[1],pl.INT32],
    context_positions: pl.Tensor[[ROWS],pl.INT32],
    context_cos: pl.Tensor[[ROWS,64],pl.BF16],
    context_sin: pl.Tensor[[ROWS,64],pl.BF16],
    context_slots: pl.Tensor[[3,512],pl.INT64],
    wkv: pl.Tensor[[3,D,512],pl.BF16],
    gamma_ckv: pl.Tensor[[3,512],pl.BF16],
    kv_caches: pl.InOut[pl.Tensor[[3,512,32,1,512],pl.BF16]],
    initial_hidden: pl.Out[pl.Tensor[[T,HC_MULT,D],pl.FP32]],
    query_token_ids: pl.Out[pl.Tensor[[T],pl.INT64]],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    # All eight ranks consume the original actual owner and query history.
    target_tokens=pl.const(128,pl.INDEX)
    active_tokens=pl.const(5,pl.INT32)
    hidden_0_flat=pl.reshape(initial_hidden,[T,HC_MULT*D])
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

    context_cos_il=pl.create_tensor([128,64],dtype=pl.FP32)
    context_sin_signed=pl.create_tensor([128,64],dtype=pl.FP32)
    context_swap=pl.create_tensor([128,64],dtype=pl.INT32)
    rope_prepare(context_cos,context_sin,context_cos_il,context_sin_signed,context_swap)
    ctx_start=pl.const(0,pl.INDEX)
    ctx_rows=pl.const(128,pl.INDEX)
    ctx_seed=pl.system.task_dummy(deps=[])
    for layer in pl.unroll(3):
        output_cache=kv_caches[layer]
        wkv_layer: pl.Tensor[[4096,512],pl.BF16]=wkv[layer]
        g_ckv_layer: pl.Tensor[[512],pl.BF16]=gamma_ckv[layer]
        context_raw=pl.create_tensor([128,512],dtype=pl.FP32)
        context_kv=pl.create_tensor([128,512],dtype=pl.BF16)
        ctx_projected=draft_kv_projection_tp8(main_x,wkv_layer,context_raw,ctx_rows,ctx_start,ctx_rows,ctx_seed)
        draft_kv_norm_tp8(context_raw,g_ckv_layer,context_cos_il,context_sin_signed,
                          context_swap,context_kv,ctx_rows,ctx_start,ctx_rows,ctx_projected)
        flat_cache=pl.reshape(output_cache,[512*32,512])
        with pl.at(level=pl.Level.CORE_GROUP,name_hint="draft_tp8_context_kv_commit"):
            for ctx_row in pl.range(128):
                address=pl.read(context_slots,[layer,ctx_row])
                if address>=0:
                    dest=pl.cast(address,pl.INDEX)
                    flat_cache[dest:dest+1,:]=context_kv[ctx_row:ctx_row+1,:]
    return initial_hidden,query_token_ids
