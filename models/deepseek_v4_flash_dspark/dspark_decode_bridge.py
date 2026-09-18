# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-side target-to-drafter metadata bridge for fused DSpark decode."""

import pypto.language as pl
import pypto.language.distributed as pld

from dspark_device_state import LOCAL_BATCH, S, STATE_META_WIDTH
from dspark_drafter import (
    BLOCK_SIZE,
    DSPARK_CP_SIZE,
    DSPARK_DRAFT_LAYERS,
    DSPARK_QUERY_WIDTH,
    ORI_MAX_BLOCKS,
    ROPE_DIM,
    T_QUERY,
)
from lm_head import MAX_LOGIT_ROWS


# The bridge consumes only S + K = 15 rows, but a BF16 GM tile with a
# 15-row axis is not 32-byte aligned.  Keep one unused padding row so device
# preparation can publish the candidate slab without a Host-side gather.
ROPE_CANDIDATE_ROWS = 16
CONTEXT_T = LOCAL_BATCH * S
LOCAL_METADATA_ROWS = CONTEXT_T + T_QUERY
GROUP_METADATA_ROWS = DSPARK_CP_SIZE * LOCAL_METADATA_ROWS
METADATA_WIDTH = 1 + DSPARK_DRAFT_LAYERS
META_COMM_ROWS = 16
ROPE_COMM_ROWS = 8
BRIDGE_B_DYN = pl.dynamic("DSPARK_BRIDGE_B_DYN")
BRIDGE_GROUP_CONTEXT_T_DYN = pl.dynamic("DSPARK_BRIDGE_GROUP_CONTEXT_T_DYN")


@pl.jit.inline(auto_scope=False)
def _allgather_metadata(
    local_metadata: pl.Tensor[[LOCAL_METADATA_ROWS, METADATA_WIDTH], pl.INT64],
    group_metadata: pl.Tensor[[GROUP_METADATA_ROWS, METADATA_WIDTH], pl.INT64],
    metadata_window: pld.DistributedTensor[
        [GROUP_METADATA_ROWS, METADATA_WIDTH], pl.INT64
    ],
    metadata_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
):
    target_row = cp_rank * LOCAL_METADATA_ROWS
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_metadata_push",
        allow_early_resolve=True,
    ) as push_tid:
        for peer in pl.range(DSPARK_CP_SIZE):
            pld.tensor.put(
                dst=metadata_window,
                peer=group_base + peer,
                src=local_metadata,
                dst_offsets=[target_row, 0],
                src_offsets=[0, 0],
                shape=[LOCAL_METADATA_ROWS, METADATA_WIDTH],
                chunk_rows=META_COMM_ROWS,
                chunk_cols=METADATA_WIDTH,
            )
        for peer in pl.range(DSPARK_CP_SIZE):
            if peer != cp_rank:
                pld.system.notify(
                    target=metadata_signal,
                    peer=group_base + peer,
                    offsets=[cp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_metadata_payload_wait",
    ) as payload_wait_tid:
        for source in pl.range(DSPARK_CP_SIZE):
            if source != cp_rank:
                pld.system.defer_wait(
                    signal=metadata_signal,
                    offsets=[source, 0],
                    expected=pl.cast(1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_metadata_readback",
        deps=[push_tid, payload_wait_tid],
    ) as readback_tid:
        for row in pl.range(0, GROUP_METADATA_ROWS, META_COMM_ROWS):
            group_metadata[
                row : row + META_COMM_ROWS, 0:METADATA_WIDTH
            ] = metadata_window[row : row + META_COMM_ROWS, 0:METADATA_WIDTH]
        for peer in pl.range(DSPARK_CP_SIZE):
            if peer != cp_rank:
                pld.system.notify(
                    target=metadata_signal,
                    peer=group_base + peer,
                    offsets=[cp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_metadata_readback_wait",
    ) as readback_wait_tid:
        for source in pl.range(DSPARK_CP_SIZE):
            if source != cp_rank:
                pld.system.defer_wait(
                    signal=metadata_signal,
                    offsets=[source, 0],
                    expected=pl.cast(2, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_metadata_retire",
        deps=[readback_tid, readback_wait_tid],
    ):
        anchor = pl.read(group_metadata, [0, 0])
        for source in pl.range(DSPARK_CP_SIZE):
            if source != cp_rank:
                pld.system.notify(
                    target=metadata_signal,
                    peer=group_base + cp_rank,
                    offsets=[source, 0],
                    value=pl.cast(-2, pl.INT32),
                    op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(group_metadata, [0, 0], anchor)
    return group_metadata, metadata_signal


@pl.jit.inline(auto_scope=False)
def _allgather_rope(
    local_rope: pl.Tensor[[LOCAL_METADATA_ROWS, ROPE_DIM], pl.BF16],
    group_rope: pl.Tensor[[GROUP_METADATA_ROWS, ROPE_DIM], pl.BF16],
    rope_window: pld.DistributedTensor[[GROUP_METADATA_ROWS, ROPE_DIM], pl.BF16],
    rope_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
):
    target_row = cp_rank * LOCAL_METADATA_ROWS
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_rope_push",
        allow_early_resolve=True,
    ) as push_tid:
        for peer in pl.range(DSPARK_CP_SIZE):
            pld.tensor.put(
                dst=rope_window,
                peer=group_base + peer,
                src=local_rope,
                dst_offsets=[target_row, 0],
                src_offsets=[0, 0],
                shape=[LOCAL_METADATA_ROWS, ROPE_DIM],
                chunk_rows=ROPE_COMM_ROWS,
                chunk_cols=ROPE_DIM,
            )
        for peer in pl.range(DSPARK_CP_SIZE):
            if peer != cp_rank:
                pld.system.notify(
                    target=rope_signal,
                    peer=group_base + peer,
                    offsets=[cp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_rope_payload_wait",
    ) as payload_wait_tid:
        for source in pl.range(DSPARK_CP_SIZE):
            if source != cp_rank:
                pld.system.defer_wait(
                    signal=rope_signal,
                    offsets=[source, 0],
                    expected=pl.cast(1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_rope_readback",
        deps=[push_tid, payload_wait_tid],
    ) as readback_tid:
        for row in pl.range(0, GROUP_METADATA_ROWS, ROPE_COMM_ROWS):
            group_rope[row : row + ROPE_COMM_ROWS, 0:ROPE_DIM] = rope_window[
                row : row + ROPE_COMM_ROWS, 0:ROPE_DIM
            ]
        for peer in pl.range(DSPARK_CP_SIZE):
            if peer != cp_rank:
                pld.system.notify(
                    target=rope_signal,
                    peer=group_base + peer,
                    offsets=[cp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_rope_readback_wait",
    ) as readback_wait_tid:
        for source in pl.range(DSPARK_CP_SIZE):
            if source != cp_rank:
                pld.system.defer_wait(
                    signal=rope_signal,
                    offsets=[source, 0],
                    expected=pl.cast(2, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dspark_bridge_rope_retire",
        deps=[readback_tid, readback_wait_tid],
    ):
        anchor = pl.read(group_rope, [0, 0])
        for source in pl.range(DSPARK_CP_SIZE):
            if source != cp_rank:
                pld.system.notify(
                    target=rope_signal,
                    peer=group_base + cp_rank,
                    offsets=[source, 0],
                    value=pl.cast(-2, pl.INT32),
                    op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(group_rope, [0, 0], anchor)
    return group_rope, rope_signal


@pl.jit.inline(auto_scope=False)
def prepare_drafter_after_target(
    drafter_ready: pl.Tensor[[1], pl.INT32],
    state_slot_ids: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_meta: pl.Tensor[[LOCAL_BATCH * DSPARK_CP_SIZE, STATE_META_WIDTH], pl.INT32],
    accepted_counts: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    context_positions: pl.Tensor[[CONTEXT_T], pl.INT32],
    context_valid: pl.Tensor[[CONTEXT_T], pl.INT32],
    last_sampled: pl.Tensor[[LOCAL_BATCH], pl.INT64],
    anchor_positions: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    row_offsets: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    block_tables: pl.Tensor[
        [DSPARK_DRAFT_LAYERS, BRIDGE_B_DYN, ORI_MAX_BLOCKS], pl.INT32
    ],
    rope_cos_candidates: pl.Tensor[
        [BRIDGE_B_DYN, ROPE_CANDIDATE_ROWS, ROPE_DIM], pl.BF16
    ],
    rope_sin_candidates: pl.Tensor[
        [BRIDGE_B_DYN, ROPE_CANDIDATE_ROWS, ROPE_DIM], pl.BF16
    ],
    num_sampled: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT32]],
    compact_last_sampled: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT64]],
    next_prefill_tokens: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT64]],
    compact_anchor_positions: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT32]],
    compact_state_slot_ids: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT32]],
    compact_state_generations: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT32]],
    logit_row_indices: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    context_group_position_ids: pl.Out[
        pl.Tensor[[BRIDGE_GROUP_CONTEXT_T_DYN], pl.INT32]
    ],
    context_group_slot_mapping: pl.Out[
        pl.Tensor[[DSPARK_DRAFT_LAYERS, BRIDGE_GROUP_CONTEXT_T_DYN], pl.INT64]
    ],
    query_group_position_ids: pl.Out[
        pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT32]
    ],
    query_group_slot_mapping: pl.Out[
        pl.Tensor[
            [DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * T_QUERY], pl.INT64
        ]
    ],
    context_group_freqs_cos: pl.Out[
        pl.Tensor[[BRIDGE_GROUP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16]
    ],
    context_group_freqs_sin: pl.Out[
        pl.Tensor[[BRIDGE_GROUP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16]
    ],
    query_freqs_cos: pl.Out[pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16]],
    query_freqs_sin: pl.Out[pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16]],
    query_group_freqs_cos: pl.Out[
        pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16]
    ],
    query_group_freqs_sin: pl.Out[
        pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16]
    ],
    metadata_window: pld.DistributedTensor[
        [GROUP_METADATA_ROWS, METADATA_WIDTH], pl.INT64
    ],
    metadata_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    rope_cos_window: pld.DistributedTensor[
        [GROUP_METADATA_ROWS, ROPE_DIM], pl.BF16
    ],
    rope_sin_window: pld.DistributedTensor[
        [GROUP_METADATA_ROWS, ROPE_DIM], pl.BF16
    ],
    rope_cos_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    rope_sin_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
):
    """Compact accepted rows and publish rank-major drafter metadata."""
    block_tables.bind_dynamic(1, BRIDGE_B_DYN)
    rope_cos_candidates.bind_dynamic(0, BRIDGE_B_DYN)
    rope_sin_candidates.bind_dynamic(0, BRIDGE_B_DYN)
    num_sampled.bind_dynamic(0, BRIDGE_B_DYN)
    compact_last_sampled.bind_dynamic(0, BRIDGE_B_DYN)
    next_prefill_tokens.bind_dynamic(0, BRIDGE_B_DYN)
    compact_anchor_positions.bind_dynamic(0, BRIDGE_B_DYN)
    compact_state_slot_ids.bind_dynamic(0, BRIDGE_B_DYN)
    compact_state_generations.bind_dynamic(0, BRIDGE_B_DYN)
    context_group_position_ids.bind_dynamic(0, BRIDGE_GROUP_CONTEXT_T_DYN)
    context_group_slot_mapping.bind_dynamic(1, BRIDGE_GROUP_CONTEXT_T_DYN)
    context_group_freqs_cos.bind_dynamic(0, BRIDGE_GROUP_CONTEXT_T_DYN)
    context_group_freqs_sin.bind_dynamic(0, BRIDGE_GROUP_CONTEXT_T_DYN)
    batch = pl.tensor.dim(num_sampled, 0)
    group_context_tokens = pl.tensor.dim(context_group_position_ids, 0)
    local_context_tokens = group_context_tokens // DSPARK_CP_SIZE
    local_metadata = pl.create_tensor(
        [LOCAL_METADATA_ROWS, METADATA_WIDTH], dtype=pl.INT64
    )
    local_rope_cos = pl.create_tensor([LOCAL_METADATA_ROWS, ROPE_DIM], dtype=pl.BF16)
    local_rope_sin = pl.create_tensor([LOCAL_METADATA_ROWS, ROPE_DIM], dtype=pl.BF16)
    group_metadata = pl.create_tensor(
        [GROUP_METADATA_ROWS, METADATA_WIDTH], dtype=pl.INT64
    )
    group_rope_cos = pl.create_tensor([GROUP_METADATA_ROWS, ROPE_DIM], dtype=pl.BF16)
    group_rope_sin = pl.create_tensor([GROUP_METADATA_ROWS, ROPE_DIM], dtype=pl.BF16)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_bridge_prepare"):
        ready_offset = pl.read(drafter_ready, [0])
        for row in pl.range(LOCAL_METADATA_ROWS):
            pl.write(local_metadata, [row, 0], pl.cast(0, pl.INT64))
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(local_metadata, [row, 1 + layer], pl.cast(-1, pl.INT64))
            local_rope_cos[row : row + 1, 0:ROPE_DIM] = pl.full(
                [1, ROPE_DIM], dtype=pl.BF16, value=0.0
            )
            local_rope_sin[row : row + 1, 0:ROPE_DIM] = pl.full(
                [1, ROPE_DIM], dtype=pl.BF16, value=0.0
            )
        for request in pl.range(batch):
            pl.write(num_sampled, [request], pl.cast(0, pl.INT32))
            pl.write(compact_last_sampled, [request], pl.cast(0, pl.INT64))
            pl.write(next_prefill_tokens, [request], pl.cast(0, pl.INT64))
            pl.write(compact_anchor_positions, [request], pl.cast(0, pl.INT32))
            pl.write(compact_state_slot_ids, [request], pl.cast(-1, pl.INT32))
            pl.write(compact_state_generations, [request], pl.cast(-1, pl.INT32))
        for row in pl.range(MAX_LOGIT_ROWS):
            pl.write(logit_row_indices, [row], pl.cast(-1, pl.INT32))

        for request in pl.range(LOCAL_BATCH):
            destination_raw = pl.read(row_offsets, [request])
            if destination_raw >= 0:
                destination = pl.cast(destination_raw // S, pl.INDEX)
                accepted = pl.read(accepted_counts, [request])
                pl.write(num_sampled, [destination], accepted)
                pl.write(
                    compact_last_sampled,
                    [destination],
                    pl.read(last_sampled, [request]),
                )
                pl.write(
                    compact_anchor_positions,
                    [destination],
                    pl.read(anchor_positions, [request]),
                )
                pl.write(
                    compact_state_slot_ids,
                    [destination],
                    pl.read(state_slot_ids, [request]),
                )
                pl.write(
                    compact_state_generations,
                    [destination],
                    pl.read(state_generations, [request]),
                )
                for offset in pl.range(S):
                    source_row = request * S + offset
                    destination_row = destination * S + offset
                    valid = pl.read(context_valid, [source_row])
                    if valid == 1:
                        position = (
                            pl.read(context_positions, [source_row]) + ready_offset
                        )
                        pl.write(
                            local_metadata,
                            [destination_row, 0],
                            pl.cast(position, pl.INT64),
                        )
                        for layer in pl.range(DSPARK_DRAFT_LAYERS):
                            logical_block = position // BLOCK_SIZE
                            physical_block = pl.read(
                                block_tables,
                                [layer, request, pl.cast(logical_block, pl.INDEX)],
                            )
                            slot = physical_block * BLOCK_SIZE + position % BLOCK_SIZE
                            pl.write(
                                local_metadata,
                                [destination_row, 1 + layer],
                                pl.cast(slot, pl.INT64),
                            )
                        local_rope_cos[
                            destination_row : destination_row + 1, 0:ROPE_DIM
                        ] = pl.reshape(
                            rope_cos_candidates[
                                request : request + 1,
                                offset : offset + 1,
                                0:ROPE_DIM,
                            ],
                            [1, ROPE_DIM],
                        )
                        local_rope_sin[
                            destination_row : destination_row + 1, 0:ROPE_DIM
                        ] = pl.reshape(
                            rope_sin_candidates[
                                request : request + 1,
                                offset : offset + 1,
                                0:ROPE_DIM,
                            ],
                            [1, ROPE_DIM],
                        )
                for query in pl.range(DSPARK_QUERY_WIDTH):
                    query_row = destination * DSPARK_QUERY_WIDTH + query
                    query_position = (
                        pl.read(anchor_positions, [request])
                        + 1
                        + query
                        + ready_offset
                    )
                    metadata_row = CONTEXT_T + query_row
                    pl.write(
                        local_metadata,
                        [metadata_row, 0],
                        pl.cast(query_position, pl.INT64),
                    )
                    for layer in pl.range(DSPARK_DRAFT_LAYERS):
                        logical_block = query_position // BLOCK_SIZE
                        physical_block = pl.read(
                            block_tables,
                            [layer, request, pl.cast(logical_block, pl.INDEX)],
                        )
                        slot = physical_block * BLOCK_SIZE + query_position % BLOCK_SIZE
                        pl.write(
                            local_metadata,
                            [metadata_row, 1 + layer],
                            pl.cast(slot, pl.INT64),
                        )
                    candidate_row = accepted + query
                    local_rope_cos[
                        metadata_row : metadata_row + 1, 0:ROPE_DIM
                    ] = pl.reshape(
                        rope_cos_candidates[
                            request : request + 1,
                            candidate_row : candidate_row + 1,
                            0:ROPE_DIM,
                        ],
                        [1, ROPE_DIM],
                    )
                    local_rope_sin[
                        metadata_row : metadata_row + 1, 0:ROPE_DIM
                    ] = pl.reshape(
                        rope_sin_candidates[
                            request : request + 1,
                            candidate_row : candidate_row + 1,
                            0:ROPE_DIM,
                        ],
                        [1, ROPE_DIM],
                    )
                    pl.write(
                        logit_row_indices,
                        [query_row],
                        pl.cast(query_row, pl.INT32),
                    )

    group_metadata, metadata_signal = _allgather_metadata(
        local_metadata,
        group_metadata,
        metadata_window,
        metadata_signal,
        group_base,
        cp_rank,
    )
    group_rope_cos, rope_cos_signal = _allgather_rope(
        local_rope_cos,
        group_rope_cos,
        rope_cos_window,
        rope_cos_signal,
        group_base,
        cp_rank,
    )
    group_rope_sin, rope_sin_signal = _allgather_rope(
        local_rope_sin,
        group_rope_sin,
        rope_sin_window,
        rope_sin_signal,
        group_base,
        cp_rank,
    )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_bridge_unpack"):
        for row in pl.range(group_context_tokens):
            source_rank = row // local_context_tokens
            source_row = row % local_context_tokens
            metadata_row = source_rank * LOCAL_METADATA_ROWS + source_row
            pl.write(
                context_group_position_ids,
                [row],
                pl.cast(pl.read(group_metadata, [metadata_row, 0]), pl.INT32),
            )
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(
                    context_group_slot_mapping,
                    [layer, row],
                    pl.read(group_metadata, [metadata_row, 1 + layer]),
                )
            context_group_freqs_cos[row : row + 1, 0:ROPE_DIM] = group_rope_cos[
                metadata_row : metadata_row + 1, 0:ROPE_DIM
            ]
            context_group_freqs_sin[row : row + 1, 0:ROPE_DIM] = group_rope_sin[
                metadata_row : metadata_row + 1, 0:ROPE_DIM
            ]
        for row in pl.range(DSPARK_CP_SIZE * T_QUERY):
            source_rank = row // T_QUERY
            source_row = row % T_QUERY
            metadata_row = source_rank * LOCAL_METADATA_ROWS + CONTEXT_T + source_row
            pl.write(
                query_group_position_ids,
                [row],
                pl.cast(pl.read(group_metadata, [metadata_row, 0]), pl.INT32),
            )
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(
                    query_group_slot_mapping,
                    [layer, row],
                    pl.read(group_metadata, [metadata_row, 1 + layer]),
                )
            query_group_freqs_cos[row : row + 1, 0:ROPE_DIM] = group_rope_cos[
                metadata_row : metadata_row + 1, 0:ROPE_DIM
            ]
            query_group_freqs_sin[row : row + 1, 0:ROPE_DIM] = group_rope_sin[
                metadata_row : metadata_row + 1, 0:ROPE_DIM
            ]
        for row in pl.range(T_QUERY):
            metadata_row = CONTEXT_T + row
            query_freqs_cos[row : row + 1, 0:ROPE_DIM] = local_rope_cos[
                metadata_row : metadata_row + 1, 0:ROPE_DIM
            ]
            query_freqs_sin[row : row + 1, 0:ROPE_DIM] = local_rope_sin[
                metadata_row : metadata_row + 1, 0:ROPE_DIM
            ]
    return (
        num_sampled,
        compact_last_sampled,
        next_prefill_tokens,
        compact_anchor_positions,
        compact_state_slot_ids,
        compact_state_generations,
        logit_row_indices,
        context_group_position_ids,
        context_group_slot_mapping,
        query_group_position_ids,
        query_group_slot_mapping,
        context_group_freqs_cos,
        context_group_freqs_sin,
        query_freqs_cos,
        query_freqs_sin,
        query_group_freqs_cos,
        query_group_freqs_sin,
    )
