# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build absolute-anchor metadata banks for the batch-one drafter."""

import torch


def build_drafter_metadata_banks(block_tables, rope_cos, rope_sin, *, max_anchor, shared=False):
    """Build the metadata selected by ``drafter.device_metadata`` on ranks 0..3.

    Banks are indexed directly by absolute anchor, so every anchor from zero to
    ``max_anchor`` is present. Tables use the drafter's 32-token pages.
    """
    if (not isinstance(block_tables, torch.Tensor) or block_tables.dtype != torch.int32
            or tuple(block_tables.shape) != (4, 3, 1, 32768) or block_tables.device.type != "cpu"):
        raise ValueError("block_tables must be CPU INT32 [4, 3, 1, 32768]")
    if (not isinstance(rope_cos, torch.Tensor) or not isinstance(rope_sin, torch.Tensor)
            or rope_cos.dtype != torch.bfloat16 or rope_sin.dtype != torch.bfloat16
            or rope_cos.shape != rope_sin.shape or rope_cos.ndim != 2 or rope_cos.shape[1] != 64
            or rope_cos.device.type != "cpu" or rope_sin.device.type != "cpu"):
        raise ValueError("RoPE tables must be matching CPU BF16 [positions, 64] tensors")
    if type(max_anchor) is not int or not 0 <= max_anchor < 32768 * 32 - 5:
        raise ValueError("max_anchor and its five query positions must fit the drafter tables")
    if max_anchor + 5 >= rope_cos.shape[0]:
        raise ValueError("RoPE tables do not cover the final query positions")
    anchors = torch.arange(max_anchor + 1, dtype=torch.int64)
    count = (anchors + 1).clamp(max=128)
    first = anchors + 1 - count
    columns = torch.arange(128, dtype=torch.int64)
    local = torch.full((anchors.numel(), 4, 128), -1, dtype=torch.int32)
    local[:, 0, :] = torch.where(columns[None, :] < count[:, None], first[:, None] + columns, -1).int()
    local[:, 1:, 0] = 0
    context_slots = torch.full((anchors.numel(), 3, 4, 128), -1, dtype=torch.int64)
    query = torch.zeros((anchors.numel(), 4, 20), dtype=torch.int32)
    query_slots = torch.full((anchors.numel(), 3, 4, 20), -1, dtype=torch.int64)
    for owner in range(4):
        context_pos = local[:, owner, :].to(torch.int64)
        safe_context = context_pos.clamp(min=0)
        owner_anchor = anchors if owner == 0 else torch.zeros_like(anchors)
        query[:, owner, :5] = (owner_anchor[:, None] + torch.arange(1, 6, dtype=torch.int64)).int()
        query_pos = query[:, owner, :5].to(torch.int64)
        for layer in range(3):
            table = block_tables[owner, layer, 0].to(torch.int64)
            context_page = table[safe_context // 32]
            query_page = table[query_pos // 32]
            if bool((context_page[context_pos >= 0] < 0).any()) or bool((query_page < 0).any()):
                raise ValueError("The drafter block tables do not cover the metadata banks")
            context_slots[:, layer, owner, :] = torch.where(
                context_pos >= 0, context_page * 32 + safe_context % 32, -1
            )
            query_slots[:, layer, owner, :5] = query_page * 32 + query_pos % 32

    def replicated(value):
        return value.unsqueeze(0).expand(4, *value.shape).contiguous()

    def rope(table, positions):
        return table.index_select(0, positions.clamp(min=0).long().reshape(-1)).reshape(*positions.shape, 64)

    group_positions = local.reshape(anchors.numel(), 512)
    group_query = query.reshape(anchors.numel(), 80)
    result = {
        "context_group_position_ids_bank": replicated(group_positions),
        "context_group_slot_mapping_bank": replicated(context_slots.reshape(anchors.numel(), 3, 512)),
        "query_group_position_ids_bank": replicated(group_query),
        "query_group_slot_mapping_bank": replicated(query_slots.reshape(anchors.numel(), 3, 80)),
    }
    for suffix, table in (("cos", rope_cos), ("sin", rope_sin)):
        result["context_group_freqs_" + suffix + "_bank"] = replicated(rope(table, group_positions))
        result["query_group_freqs_" + suffix + "_bank"] = replicated(rope(table, group_query))
        result["query_freqs_" + suffix + "_bank"] = rope(table, query).permute(1, 0, 2, 3).contiguous()
    if shared:
        for value in result.values():
            value.share_memory_()
    return result
