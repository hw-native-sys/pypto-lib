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


HEADS = 64


HEAD_DIM = 512


PAGE_ROWS = WINDOW = 128


INT32_MAX = (1 << 31) - 1


def integer(value, *, minimum=0):
    require(type(value) is int and minimum <= value <= INT32_MAX,
            'A bounded exact integer scheduler control is required')
    return value


def build(query_lengths, prefix_lengths, block_table, *, cache_blocks, capacity):
    """Derive physical windows and publications for every request and active row."""
    integer(cache_blocks, minimum=1)
    require(cache_blocks * PAGE_ROWS <= INT32_MAX, 'Physical window indices must fit the original INT32 ABI')
    integer(capacity, minimum=1)
    require(type(query_lengths) is list and type(prefix_lengths) is list
            and type(block_table) is list and query_lengths
            and len(query_lengths) == len(prefix_lengths) == len(block_table),
            'Explicit own request lengths/prefixes/page rows are required')
    requests = len(query_lengths)
    require(all(type(row) is list and row for row in block_table)
            and len({len(row) for row in block_table}) == 1,
            'A rectangular nonempty own block table is required')
    starts, seq_lens, positions, flat_slots, slots, indices, logical = [0], [], [], [], [], [], []
    allocated = set()
    for query_len, prefix, pages in zip(query_lengths, prefix_lengths, block_table, strict=True):
        integer(query_len, minimum=1)
        integer(prefix)
        end = integer(prefix + query_len, minimum=1)
        required_pages = (end + PAGE_ROWS - 1) // PAGE_ROWS
        require(required_pages <= len(pages), 'Own cache pages must cover the complete sequence')
        require(all(type(page) is int and -1 <= page < cache_blocks for page in pages),
                'Own page IDs must be valid physical pages or unused -1 padding')
        used = pages[:required_pages]
        require(all(page >= 0 for page in used) and len(set(used)) == len(used)
                and not allocated.intersection(used), 'Live own request pages must be distinct')
        allocated.update(used)

        def physical(position, current_pages=pages):
            return current_pages[position // PAGE_ROWS] * PAGE_ROWS + position % PAGE_ROWS

        for position in range(prefix, end):
            address = physical(position)
            visible = list(range(max(0, position - WINDOW + 1), position + 1))
            padding = [-1] * (WINDOW - len(visible))
            positions.append(position)
            flat_slots.append(address)
            slots.append([address // PAGE_ROWS, address % PAGE_ROWS])
            logical.append(visible + padding)
            indices.append([physical(token) for token in visible] + padding)
        starts.append(starts[-1] + query_len)
        seq_lens.append(end)
    require(starts[-1] <= capacity and len(set(flat_slots)) == len(flat_slots),
            'Own active rows must fit capacity with unique cache publications')
    return dict(active=starts[-1], capacity=capacity, cache_blocks=cache_blocks,
                num_reqs=requests, query_lengths=list(query_lengths), prefix_lengths=list(prefix_lengths),
                query_start_loc=starts, seq_lens=seq_lens,
                block_table=[list(row) for row in block_table], positions=positions,
                flat_slot_mapping=flat_slots, slot_mapping=slots,
                physical_window_indices=indices, logical_window_indices=logical,
                max_query_len=max(query_lengths), max_seq_len=max(seq_lens),
                cache_shape=[cache_blocks, PAGE_ROWS, 1, HEAD_DIM], cache_dtype='BF16',
                heads=HEADS, head_dim=HEAD_DIM, sliding_window=WINDOW,
                cmp_ratio=0, causal=True, ori_mask_mode=4, cmp_mask_mode=0,
                ori_win_left=127, ori_win_right=0, ori_topk=0, cmp_topk=0,
                layout_q='TND', layout_kv='PA_BBND', topk_value_mode=1,
                return_softmax_lse=False)


def from_controls(starts, seq_lens, block_table, *, cache_blocks, capacity):
    require(type(starts) is list and len(starts) >= 2 and starts[0] == 0
            and type(starts[0]) is int and type(seq_lens) is list
            and len(seq_lens) == len(starts) - 1, 'Own cumulative query starts must partition requests')
    for value in starts:
        integer(value)
    for value in seq_lens:
        integer(value, minimum=1)
    lengths = [right - left for left, right in zip(starts[:-1], starts[1:], strict=True)]
    prefixes = [end - count for end, count in zip(seq_lens, lengths, strict=True)]
    return build(lengths, prefixes, block_table, cache_blocks=cache_blocks, capacity=capacity)
