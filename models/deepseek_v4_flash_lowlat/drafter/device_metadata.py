# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

BANK_FIRST_ANCHOR = 0

BANK_COUNT = pl.dynamic("D_METADATA_BANK_COUNT")


@pl.jit.inline(auto_scope=False)
def select_drafter_metadata(
    request_state: pl.Tensor[[4], pl.INT64],
    context_group_position_ids_bank: pl.Tensor[[BANK_COUNT, 512], pl.INT32],
    context_group_slot_mapping_bank: pl.Tensor[[BANK_COUNT, 3, 512], pl.INT64],
    query_group_position_ids_bank: pl.Tensor[[BANK_COUNT, 80], pl.INT32],
    query_group_slot_mapping_bank: pl.Tensor[[BANK_COUNT, 3, 80], pl.INT64],
    context_group_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 512, 64], pl.BF16],
    context_group_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 512, 64], pl.BF16],
    query_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 20, 64], pl.BF16],
    query_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 20, 64], pl.BF16],
    query_group_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 80, 64], pl.BF16],
    query_group_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 80, 64], pl.BF16],
    num_sampled: pl.Tensor[[1], pl.INT32],
    last_sampled: pl.Tensor[[1], pl.INT64],
    next_prefill_tokens: pl.Tensor[[1], pl.INT64],
    anchor_positions: pl.Tensor[[1], pl.INT32],
    context_group_position_ids: pl.Tensor[[512], pl.INT32],
    context_group_slot_mapping: pl.Tensor[[3, 512], pl.INT64],
    query_group_position_ids: pl.Tensor[[80], pl.INT32],
    query_group_slot_mapping: pl.Tensor[[3, 80], pl.INT64],
    context_group_freqs_cos: pl.Tensor[[512, 64], pl.BF16],
    context_group_freqs_sin: pl.Tensor[[512, 64], pl.BF16],
    query_freqs_cos: pl.Tensor[[20, 64], pl.BF16],
    query_freqs_sin: pl.Tensor[[20, 64], pl.BF16],
    query_group_freqs_cos: pl.Tensor[[80, 64], pl.BF16],
    query_group_freqs_sin: pl.Tensor[[80, 64], pl.BF16],
    my_rank: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="drafter_state_scalars", deps=[prior_cycle_gate]):
        anchor = pl.cast(0, pl.INT32)
        token = pl.cast(0, pl.INT64)
        if my_rank == 0:
            anchor = pl.cast(pl.read(request_state, [1]), pl.INT32)
            token = pl.read(request_state, [3])
        pl.write(num_sampled, [0], pl.cast(1, pl.INT32))
        pl.write(last_sampled, [0], token)
        pl.write(next_prefill_tokens, [0], pl.cast(0, pl.INT64))
        pl.write(anchor_positions, [0], anchor)
    cp_pos_output_view = pl.reshape(context_group_position_ids, [1, 512])
    for cp_pos_block in pl.spmd(4, name_hint="select_context_group_position_ids", deps=[prior_cycle_gate]):
        cp_pos_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        cp_pos_bank = cp_pos_anchor - BANK_FIRST_ANCHOR
        cp_pos_row = cp_pos_block * 128
        cp_pos_selected = context_group_position_ids_bank[
            cp_pos_bank : cp_pos_bank + 1, cp_pos_row : cp_pos_row + 128
        ]
        cp_pos_output_view[0:1, cp_pos_row : cp_pos_row + 128] = cp_pos_selected
    for cp_slot_block in pl.spmd(12, name_hint="select_context_group_slot_mapping", deps=[prior_cycle_gate]):
        cp_slot_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        cp_slot_bank = cp_slot_anchor - BANK_FIRST_ANCHOR
        cp_slot_layer = cp_slot_block // 4
        cp_slot_row = cp_slot_block % 4 * 128
        cp_slot_selected = context_group_slot_mapping_bank[
            cp_slot_bank : cp_slot_bank + 1,
            cp_slot_layer : cp_slot_layer + 1,
            cp_slot_row : cp_slot_row + 128,
        ]
        cp_slot_selected_row = pl.reshape(cp_slot_selected, [1, 128])
        context_group_slot_mapping[cp_slot_layer : cp_slot_layer + 1, cp_slot_row : cp_slot_row + 128] = (
            cp_slot_selected_row
        )
    qg_pos_output_view = pl.reshape(query_group_position_ids, [1, 80])
    for qg_pos_block in pl.spmd(1, name_hint="select_query_group_position_ids", deps=[prior_cycle_gate]):
        qg_pos_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        qg_pos_bank = qg_pos_anchor - BANK_FIRST_ANCHOR
        qg_pos_row = qg_pos_block * 80
        qg_pos_selected = query_group_position_ids_bank[
            qg_pos_bank : qg_pos_bank + 1, qg_pos_row : qg_pos_row + 80
        ]
        qg_pos_output_view[0:1, qg_pos_row : qg_pos_row + 80] = qg_pos_selected
    for qg_slot_block in pl.spmd(3, name_hint="select_query_group_slot_mapping", deps=[prior_cycle_gate]):
        qg_slot_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        qg_slot_bank = qg_slot_anchor - BANK_FIRST_ANCHOR
        qg_slot_layer = qg_slot_block // 1
        qg_slot_row = qg_slot_block % 1 * 80
        qg_slot_selected = query_group_slot_mapping_bank[
            qg_slot_bank : qg_slot_bank + 1, qg_slot_layer : qg_slot_layer + 1, qg_slot_row : qg_slot_row + 80
        ]
        qg_slot_selected_row = pl.reshape(qg_slot_selected, [1, 80])
        query_group_slot_mapping[qg_slot_layer : qg_slot_layer + 1, qg_slot_row : qg_slot_row + 80] = (
            qg_slot_selected_row
        )
    for cp_cos_block in pl.spmd(128, name_hint="select_context_group_freqs_cos", deps=[prior_cycle_gate]):
        cp_cos_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        cp_cos_bank = cp_cos_anchor - BANK_FIRST_ANCHOR
        cp_cos_row = cp_cos_block * 4
        cp_cos_selected = context_group_freqs_cos_bank[
            cp_cos_bank : cp_cos_bank + 1, cp_cos_row : cp_cos_row + 4, :
        ]
        cp_cos_selected_rows = pl.reshape(cp_cos_selected, [4, 64])
        context_group_freqs_cos[cp_cos_row : cp_cos_row + 4, :] = cp_cos_selected_rows
    for cp_sin_block in pl.spmd(128, name_hint="select_context_group_freqs_sin", deps=[prior_cycle_gate]):
        cp_sin_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        cp_sin_bank = cp_sin_anchor - BANK_FIRST_ANCHOR
        cp_sin_row = cp_sin_block * 4
        cp_sin_selected = context_group_freqs_sin_bank[
            cp_sin_bank : cp_sin_bank + 1, cp_sin_row : cp_sin_row + 4, :
        ]
        cp_sin_selected_rows = pl.reshape(cp_sin_selected, [4, 64])
        context_group_freqs_sin[cp_sin_row : cp_sin_row + 4, :] = cp_sin_selected_rows
    for q_cos_block in pl.spmd(5, name_hint="select_query_freqs_cos", deps=[prior_cycle_gate]):
        q_cos_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        q_cos_bank = q_cos_anchor - BANK_FIRST_ANCHOR
        q_cos_row = q_cos_block * 4
        q_cos_selected = query_freqs_cos_bank[q_cos_bank : q_cos_bank + 1, q_cos_row : q_cos_row + 4, :]
        q_cos_selected_rows = pl.reshape(q_cos_selected, [4, 64])
        query_freqs_cos[q_cos_row : q_cos_row + 4, :] = q_cos_selected_rows
    for q_sin_block in pl.spmd(5, name_hint="select_query_freqs_sin", deps=[prior_cycle_gate]):
        q_sin_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        q_sin_bank = q_sin_anchor - BANK_FIRST_ANCHOR
        q_sin_row = q_sin_block * 4
        q_sin_selected = query_freqs_sin_bank[q_sin_bank : q_sin_bank + 1, q_sin_row : q_sin_row + 4, :]
        q_sin_selected_rows = pl.reshape(q_sin_selected, [4, 64])
        query_freqs_sin[q_sin_row : q_sin_row + 4, :] = q_sin_selected_rows
    for qg_cos_block in pl.spmd(20, name_hint="select_query_group_freqs_cos", deps=[prior_cycle_gate]):
        qg_cos_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        qg_cos_bank = qg_cos_anchor - BANK_FIRST_ANCHOR
        qg_cos_row = qg_cos_block * 4
        qg_cos_selected = query_group_freqs_cos_bank[
            qg_cos_bank : qg_cos_bank + 1, qg_cos_row : qg_cos_row + 4, :
        ]
        qg_cos_selected_rows = pl.reshape(qg_cos_selected, [4, 64])
        query_group_freqs_cos[qg_cos_row : qg_cos_row + 4, :] = qg_cos_selected_rows
    for qg_sin_block in pl.spmd(20, name_hint="select_query_group_freqs_sin", deps=[prior_cycle_gate]):
        qg_sin_anchor = pl.cast(pl.read(request_state, [1]), pl.INDEX)
        qg_sin_bank = qg_sin_anchor - BANK_FIRST_ANCHOR
        qg_sin_row = qg_sin_block * 4
        qg_sin_selected = query_group_freqs_sin_bank[
            qg_sin_bank : qg_sin_bank + 1, qg_sin_row : qg_sin_row + 4, :
        ]
        qg_sin_selected_rows = pl.reshape(qg_sin_selected, [4, 64])
        query_group_freqs_sin[qg_sin_row : qg_sin_row + 4, :] = qg_sin_selected_rows
