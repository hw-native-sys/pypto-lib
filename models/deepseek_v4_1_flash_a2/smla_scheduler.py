# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 single-layer dependencies extracted from the shared model operators."""

from .scheduler_utils import require
from .swa_scheduler import HEAD_DIM, HEADS, PAGE_ROWS, WINDOW, build as build_window


TOPK = 512


def from_controls(starts, seq_lens, ori_table, cmp_table, selected, *,
                  ori_blocks, cmp_blocks, capacity):
    require(type(starts) is list and len(starts) >= 2 and type(starts[0]) is int
            and starts[0] == 0 and all(type(v) is int and 0 <= v < (1 << 31) for v in starts)
            and type(seq_lens) is list and len(seq_lens) == len(starts) - 1,
            'Own cumulative query starts and ratio1 cache lengths required')
    require(all(type(v) is int and 0 < v < (1 << 31) for v in seq_lens),
            'Own ratio1 sequence lengths must be positive INT32')
    lengths = [b - a for a, b in zip(starts[:-1], starts[1:], strict=True)]
    prefixes = [end - count for end, count in zip(seq_lens, lengths, strict=True)]
    ori = build_window(lengths, prefixes, ori_table, cache_blocks=ori_blocks, capacity=capacity)
    cmp = build_window(lengths, prefixes, cmp_table, cache_blocks=cmp_blocks, capacity=capacity)
    require(type(selected) is list and len(selected) == ori['active'],
            'Full own logical compressed selection required for every active query')
    physical, counts = [], []
    for request, (left, right) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        for token in range(left, right):
            row, position = selected[token], ori['positions'][token]
            require(type(row) is list and len(row) == TOPK
                    and all(type(value) is int and -1 <= value <= position for value in row),
                    'Own selected logical IDs must be causal with -1 padding')
            live = [value for value in row if value >= 0]
            require(live == sorted(live) and len(set(live)) == len(live)
                    and row == live + [-1] * (TOPK - len(live)),
                    'Own distinct sorted logical IDs followed by -1 padding required')
            table = cmp_table[request]
            physical.append([table[value // PAGE_ROWS] * PAGE_ROWS + value % PAGE_ROWS
                             for value in live] + [-1] * (TOPK - len(live)))
            counts.append(len(live))
    return dict(active=ori['active'], capacity=capacity, num_reqs=len(lengths),
        query_lengths=lengths, prefix_lengths=prefixes, query_start_loc=list(starts),
        seq_lens=list(seq_lens), cmp_seq_lens=list(seq_lens), positions=ori['positions'],
        ori_block_table=[list(v) for v in ori_table], cmp_block_table=[list(v) for v in cmp_table],
        ori_blocks=ori_blocks, cmp_blocks=cmp_blocks,
        ori_slots=ori['slot_mapping'], cmp_slots=cmp['slot_mapping'],
        ori_flat_slots=ori['flat_slot_mapping'], cmp_flat_slots=cmp['flat_slot_mapping'],
        logical_window_indices=ori['logical_window_indices'],
        physical_window_indices=ori['physical_window_indices'],
        selected_logical=[list(v) for v in selected], physical_compressed_indices=physical,
        extents=[[sum(v >= 0 for v in row), count] for row, count in
                 zip(ori['physical_window_indices'], counts, strict=True)],
        max_query_len=max(lengths), max_seq_len=max(seq_lens),
        heads=HEADS, head_dim=HEAD_DIM, ori_topk=0, cmp_topk=TOPK, cmp_ratio=1,
        ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=WINDOW - 1, ori_win_right=0,
        cmp_residual=None, layout_q='TND', layout_kv='PA_BBND', topk_value_mode=1,
        return_softmax_lse=False, lanes_retained_separately=True)


def canonical_selection(raw, positions):
    require(type(raw) is list and type(positions) is list and len(raw) == len(positions),
            'Own QLI raw selection must cover every actual query')
    result = []
    for row, position in zip(raw, positions, strict=True):
        require(type(position) is int and position >= 0 and type(row) is list
                and len(row) == TOPK and all(type(v) is int and -(1 << 31) <= v < (1 << 31) for v in row),
                'Complete own INT32 QLI selection required')
        live = sorted(value for value in row if 0 <= value <= position)
        require(len(set(live)) == len(live), 'Own causal QLI IDs must be distinct')
        result.append(live + [-1] * (TOPK - len(live)))
    return result
