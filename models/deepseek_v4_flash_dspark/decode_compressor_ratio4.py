# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=4 overlap).

Uses overlapping state layout with 8 slots.
Front slots 0-3 at columns [0:HEAD_DIM], back slots 4-7 at columns [HEAD_DIM:OUT_DIM].
Tree reduction for softmax+pool. State shift after compression."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_LOCAL_REQUESTS,
    DECODE_SEQ,
    BLOCK_SIZE,
    FP32_NEG_INF,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_PAGES_PER_REQUEST,
    CSA_STATE_ROWS_PER_REQUEST,
)


# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
B = DECODE_LOCAL_REQUESTS
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim

# kernel-local (ratio-4 overlapping compressor)
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("CSA_STATE_BLOCK_NUM_DYN")
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
EVENT_DYN = pl.dynamic("CSA_MAIN_EVENT_DYN")

# tiling
K_TILE = 512
OUT_TILE = 64
MM_B_TILE = 16
RMS_PAD_TILE = 16  # 16-row block of B (min M for FP32 vec ops)
STATE_COMMIT_TOKEN_TILE = 8
POOL_HEAD_TILE = 128


@pl.jit.inline
def compressor_ratio4(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[
        [COMPRESS_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
        pl.FP32,
    ],
    state_page_ids: pl.Tensor[[B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32],
    state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    state_page_epochs: pl.Tensor[[B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    event_query_ids: pl.Tensor[[EVENT_DYN], pl.INT32],
    cos: pl.Tensor[[EVENT_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[EVENT_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    event_write_slots: pl.Tensor[[EVENT_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Produce main CSA rows from an eight-row recurrent-state transaction.

    Persistent state is read only after page, epoch, and logical-range checks.
    Current-step projections remain an overlay until every boundary pool has
    completed, then a separate task overwrites the four-page state ring.
    """
    b_dim = pl.tensor.dim(state_page_ids, 0)
    token_count = pl.tensor.dim(x, 0)
    s_dim = token_count // b_dim
    state_page_count = pl.tensor.dim(compress_state, 0)
    cache_block_count = pl.tensor.dim(cmp_kv_cache, 0)
    event_count = pl.tensor.dim(event_query_ids, 0)
    t_matmul = ((token_count + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE

    kv_proj_pad = pl.create_tensor([t_matmul, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([t_matmul, OUT_DIM], dtype=pl.FP32)
    with pl.spmd(
        t_matmul * OUT_DIM // (MM_B_TILE * OUT_TILE),
        name_hint="csa_main_state_projection",
        deps=[late_dep],
    ) as _projection_tid:
        task = pl.tile.get_block_idx()
        row0 = (task // (OUT_DIM // OUT_TILE)) * MM_B_TILE
        out0 = (task % (OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            valid_rows = pl.min(MM_B_TILE, token_count - row0)
            x_tile = pl.slice(
                x,
                [MM_B_TILE, K_TILE],
                [row0, k0],
                valid_shape=[valid_rows, K_TILE],
            )
            wkv_tile = wkv[out0 : out0 + OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[out0 : out0 + OUT_TILE, k0 : k0 + K_TILE]
            if kb == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(
                    x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True
                )
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(
                    score_acc, x_tile, wgate_tile, b_trans=True
                )
        kv_proj_pad[row0 : row0 + MM_B_TILE, out0 : out0 + OUT_TILE] = kv_acc
        score_proj_pad[
            row0 : row0 + MM_B_TILE, out0 : out0 + OUT_TILE
        ] = score_acc

    state_rows = state_page_count * CSA_STATE_BLOCK_SIZE
    compress_state_rows = pl.reshape(
        compress_state, [state_rows, COMPRESS_STATE_DIM]
    )
    pooled_kv = pl.create_tensor([event_count, HEAD_DIM], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="csa_main_boundary_pool",
        deps=[_projection_tid],
    ) as pool_tid:
        for event_index in pl.range(event_count):
            pooled_kv[event_index : event_index + 1, :] = pl.full(
                [1, HEAD_DIM], dtype=pl.FP32, value=0.0
            )
            event_query = pl.read(event_query_ids, [event_index])
            if event_query >= 0:
                if event_query < token_count:
                    request = event_query // s_dim
                    if request < b_dim:
                        first_query = request * s_dim
                        first_position = pl.read(
                            state_valid_ranges, [request, 1]
                        )
                        event_position = (
                            first_position + event_query - first_query
                        )
                        if (event_position + 1) % COMPRESS_RATIO == 0:
                            compression_start = (
                                event_position - (COMPRESS_RATIO - 1)
                            )
                            window_begin = compression_start - COMPRESS_RATIO
                            state_valid_begin = pl.read(
                                state_valid_ranges, [request, 0]
                            )
                            state_valid_end = pl.read(
                                state_valid_ranges, [request, 1]
                            )
                            request_epoch = pl.read(request_epochs, [request])
                            for h0 in pl.range(0, HEAD_DIM, POOL_HEAD_TILE):
                                pool_scores = pl.create_tensor(
                                    [STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32
                                )
                                pool_values = pl.create_tensor(
                                    [STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32
                                )
                                for slot in pl.range(STATE_LEN):
                                    logical_position = window_begin + slot
                                    state_ring_row = (
                                        logical_position
                                        % CSA_STATE_ROWS_PER_REQUEST
                                    )
                                    state_relative_page = (
                                        state_ring_row // CSA_STATE_BLOCK_SIZE
                                    )
                                    state_page_raw = pl.read(
                                        state_page_ids,
                                        [request, state_relative_page],
                                    )
                                    state_page_epoch = pl.read(
                                        state_page_epochs,
                                        [request, state_relative_page],
                                    )
                                    value = pl.full(
                                        [1, POOL_HEAD_TILE],
                                        dtype=pl.FP32,
                                        value=0.0,
                                    )
                                    score = pl.full(
                                        [1, POOL_HEAD_TILE],
                                        dtype=pl.FP32,
                                        value=FP32_NEG_INF,
                                    )
                                    if state_page_raw >= 0:
                                        if state_page_raw < state_page_count:
                                            if state_page_epoch == request_epoch:
                                                if logical_position >= state_valid_begin:
                                                    if logical_position < state_valid_end:
                                                        state_page = pl.cast(
                                                            state_page_raw,
                                                            target_type=pl.INDEX,
                                                        )
                                                        state_row = (
                                                            state_page
                                                            * CSA_STATE_BLOCK_SIZE
                                                            + state_ring_row
                                                            % CSA_STATE_BLOCK_SIZE
                                                        )
                                                        state_half = 0
                                                        if slot >= COMPRESS_RATIO:
                                                            state_half = HEAD_DIM
                                                        value = compress_state_rows[
                                                            state_row : state_row + 1,
                                                            state_half
                                                            + h0 : state_half
                                                            + h0
                                                            + POOL_HEAD_TILE,
                                                        ]
                                                        score = compress_state_rows[
                                                            state_row : state_row + 1,
                                                            OUT_DIM
                                                            + state_half
                                                            + h0 : OUT_DIM
                                                            + state_half
                                                            + h0
                                                            + POOL_HEAD_TILE,
                                                        ]
                                    if logical_position >= first_position:
                                        if logical_position <= event_position:
                                            overlay_query = (
                                                first_query
                                                + logical_position
                                                - first_position
                                            )
                                            projection_half = 0
                                            if slot >= COMPRESS_RATIO:
                                                projection_half = HEAD_DIM
                                            ape_row = pl.cast(
                                                logical_position
                                                % COMPRESS_RATIO,
                                                target_type=pl.INDEX,
                                            )
                                            value = kv_proj_pad[
                                                overlay_query
                                                : overlay_query + 1,
                                                projection_half
                                                + h0 : projection_half
                                                + h0
                                                + POOL_HEAD_TILE,
                                            ]
                                            score = pl.add(
                                                score_proj_pad[
                                                    overlay_query
                                                    : overlay_query + 1,
                                                    projection_half
                                                    + h0 : projection_half
                                                    + h0
                                                    + POOL_HEAD_TILE,
                                                ],
                                                ape[
                                                    ape_row : ape_row + 1,
                                                    projection_half
                                                    + h0 : projection_half
                                                    + h0
                                                    + POOL_HEAD_TILE,
                                                ],
                                            )
                                    pool_values[slot : slot + 1, :] = value
                                    pool_scores[slot : slot + 1, :] = score
                                score_max = pl.col_max(pool_scores)
                                score_exp = pl.col_expand_expdif(
                                    pool_scores, score_max
                                )
                                score_sum = pl.col_sum(score_exp)
                                score_prob = pl.col_expand_mul(
                                    score_exp, pl.recip(score_sum)
                                )
                                pooled_kv[
                                    event_index : event_index + 1,
                                    h0 : h0 + POOL_HEAD_TILE,
                                ] = pl.col_sum(
                                    pl.mul(pool_values, score_prob)
                                )

    state_commit_blocks = (
        token_count + STATE_COMMIT_TOKEN_TILE - 1
    ) // STATE_COMMIT_TOKEN_TILE
    with pl.spmd(
        state_commit_blocks,
        name_hint="csa_main_state_ring_commit",
        deps=[pool_tid],
    ) as state_commit_tid:
        task = pl.tile.get_block_idx()
        token0 = task * STATE_COMMIT_TOKEN_TILE
        valid_tokens = pl.min(STATE_COMMIT_TOKEN_TILE, token_count - token0)
        for local_token in pl.range(valid_tokens):
            token = token0 + local_token
            state_row_i64 = pl.read(state_slot_mapping, [token])
            if state_row_i64 >= 0:
                if state_row_i64 < state_rows:
                    state_row = pl.cast(state_row_i64, target_type=pl.INDEX)
                    request = token // s_dim
                    token_position = (
                        pl.read(state_valid_ranges, [request, 1])
                        + token
                        - request * s_dim
                    )
                    ape_row = pl.cast(
                        token_position % COMPRESS_RATIO, target_type=pl.INDEX
                    )
                    compress_state_rows[
                        state_row : state_row + 1, 0 : OUT_DIM
                    ] = kv_proj_pad[token : token + 1, 0 : OUT_DIM]
                    compress_state_rows[
                        state_row : state_row + 1, OUT_DIM : COMPRESS_STATE_DIM
                    ] = pl.add(
                        score_proj_pad[token : token + 1, 0 : OUT_DIM],
                        ape[ape_row : ape_row + 1, 0 : OUT_DIM],
                    )

    norm_w_row = pl.reshape(norm_w, [1, HEAD_DIM])
    cache_rows = cache_block_count * BLOCK_SIZE
    cache_flat = pl.reshape(cmp_kv_cache, [cache_rows, HEAD_DIM])
    event_blocks = (event_count + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    with pl.spmd(
        event_blocks,
        name_hint="csa_main_event_commit",
        deps=[pool_tid],
    ) as cache_commit_tid:
        event0 = pl.tile.get_block_idx() * RMS_PAD_TILE
        valid_events = pl.min(RMS_PAD_TILE, event_count - event0)
        pooled_block = pl.full(
            [RMS_PAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0
        )
        cos_block = pl.full(
            [RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0
        )
        sin_block = pl.full(
            [RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0
        )
        for local_event in pl.range(valid_events):
            load_event_index = event0 + local_event
            load_event_query = pl.read(event_query_ids, [load_event_index])
            if load_event_query >= 0:
                pooled_block[local_event : local_event + 1, :] = pooled_kv[
                    load_event_index : load_event_index + 1, :
                ]
                cos_block[local_event : local_event + 1, :] = cos[
                    load_event_index : load_event_index + 1, :
                ]
                sin_block[local_event : local_event + 1, :] = sin[
                    load_event_index : load_event_index + 1, :
                ]
        variance = pl.add(
            pl.mul(pl.row_sum(pl.mul(pooled_block, pooled_block)), HEAD_DIM_INV),
            EPS,
        )
        normed = pl.col_expand_mul(
            pl.row_expand_mul(pooled_block, pl.recip(pl.sqrt(variance))),
            pl.cast(norm_w_row, target_type=pl.FP32),
        )
        rope_normed = normed[:, NOPE_HEAD_DIM:HEAD_DIM]
        rope_ones = pl.full(
            [RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0
        )
        rope_col = pl.col_expand_mul(
            rope_ones,
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        rope_pair = pl.cast(
            pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        rope_lane = pl.sub(rope_col, pl.mul(rope_pair, 2.0))
        rope_swap_idx = pl.cast(
            pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)),
            target_type=pl.INT32,
        )
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(
            pl.mul(rope_normed, cos_block), pl.mul(swapped, sin_block)
        )
        normed = pl.concat(normed[:, :NOPE_HEAD_DIM], rope_rot)
        for local_event in pl.range(valid_events):
            store_event_index = event0 + local_event
            store_event_query = pl.read(event_query_ids, [store_event_index])
            cache_row_i64 = pl.read(event_write_slots, [store_event_index])
            if store_event_query >= 0:
                if store_event_query < token_count:
                    if cache_row_i64 >= 0:
                        if cache_row_i64 < cache_rows:
                            cache_row = pl.cast(
                                cache_row_i64, target_type=pl.INDEX
                            )
                            kv[
                                store_event_query : store_event_query + 1, :
                            ] = normed[local_event : local_event + 1, :]
                            cache_flat[
                                cache_row : cache_row + 1, :
                            ] = pl.cast(
                                normed[local_event : local_event + 1, :],
                                target_type=pl.BF16,
                                mode="rint",
                            )

    return cache_commit_tid, state_commit_tid


@pl.jit
def compressor_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[
        [COMPRESS_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
        pl.FP32,
    ]],
    state_page_ids: pl.Tensor[[B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32],
    state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    state_page_epochs: pl.Tensor[[B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    event_query_ids: pl.Tensor[[EVENT_DYN], pl.INT32],
    cos: pl.Tensor[[EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.InOut[pl.Tensor[
        [CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    event_write_slots: pl.Tensor[[EVENT_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    x.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, COMPRESS_STATE_BLOCK_NUM_DYN)
    state_page_ids.bind_dynamic(0, B_DYN)
    state_valid_ranges.bind_dynamic(0, B_DYN)
    state_page_epochs.bind_dynamic(0, B_DYN)
    request_epochs.bind_dynamic(0, B_DYN)
    event_query_ids.bind_dynamic(0, EVENT_DYN)
    cos.bind_dynamic(0, EVENT_DYN)
    sin.bind_dynamic(0, EVENT_DYN)
    cmp_kv_cache.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    event_write_slots.bind_dynamic(0, EVENT_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)

    event_count = pl.tensor.dim(event_query_ids, 0)
    cos_il = pl.create_tensor([event_count, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([event_count, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="csa_main_event_rope_interleave",
        allow_early_resolve=True,
    ):
        rope_ones = pl.full([1, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(
            rope_ones,
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        rope_dup_f = pl.cast(
            pl.cast(
                pl.mul(rope_col, 0.5),
                target_type=pl.INT32,
                mode="trunc",
            ),
            target_type=pl.FP32,
        )
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        for event in pl.range(event_count):
            cos_il[event : event + 1, :] = pl.gather(
                cos[event : event + 1, :], dim=-1, index=rope_dup_idx
            )
            sin_signed[event : event + 1, :] = pl.mul(
                pl.gather(
                    sin[event : event + 1, :],
                    dim=-1,
                    index=rope_dup_idx,
                ),
                rope_sign,
            )
    late_dep = pl.system.task_dummy(deps=[])
    _cache_done, _state_done = compressor_ratio4(
        x,
        kv,
        compress_state,
        state_page_ids,
        state_valid_ranges,
        state_page_epochs,
        request_epochs,
        wkv,
        wgate,
        ape,
        norm_w,
        event_query_ids,
        cos_il,
        sin_signed,
        cmp_kv_cache,
        event_write_slots,
        state_slot_mapping,
        late_dep,
    )
    return kv, compress_state, cmp_kv_cache


def golden_compressor(tensors):
    """Torch reference for the main ratio-4 transaction."""
    import torch

    x = tensors["x"].float()
    kv_projection = x @ tensors["wkv"].float().t()
    score_projection = x @ tensors["wgate"].float().t()
    ape = tensors["ape"].float()
    norm_w = tensors["norm_w"].float()
    state = tensors["compress_state"]
    state_pages = tensors["state_page_ids"].to(torch.int64)
    state_ranges = tensors["state_valid_ranges"].to(torch.int64)
    state_epochs = tensors["state_page_epochs"].to(torch.int64)
    request_epochs = tensors["request_epochs"].to(torch.int64)
    event_queries = tensors["event_query_ids"].to(torch.int64)
    event_slots = tensors["event_write_slots"].to(torch.int64)
    state_slots = tensors["state_slot_mapping"].to(torch.int64)
    cache = tensors["cmp_kv_cache"]
    token_count = x.shape[0]
    batch = state_pages.shape[0]
    seq = token_count // batch
    positions = (
        state_ranges[:, 1, None]
        + torch.arange(seq, dtype=torch.int64)[None, :]
    ).reshape(-1)
    pooled = torch.zeros(len(event_queries), HEAD_DIM, dtype=torch.float32)

    for event_index, event_query_tensor in enumerate(event_queries):
        event_query = int(event_query_tensor.item())
        if event_query < 0 or event_query >= token_count:
            continue
        request = event_query // seq
        event_position = int(positions[event_query].item())
        if (event_position + 1) % COMPRESS_RATIO != 0:
            continue
        compression_start = event_position - (COMPRESS_RATIO - 1)
        window_begin = compression_start - COMPRESS_RATIO
        first_query = request * seq
        first_position = int(positions[first_query].item())
        values = []
        scores = []
        for slot in range(STATE_LEN):
            logical_position = window_begin + slot
            ring_row = logical_position % CSA_STATE_ROWS_PER_REQUEST
            relative_page = ring_row // CSA_STATE_BLOCK_SIZE
            half = 0 if slot < COMPRESS_RATIO else HEAD_DIM
            value = torch.zeros(HEAD_DIM, dtype=torch.float32)
            score = torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32)
            if (
                int(state_pages[request, relative_page].item()) >= 0
                and int(state_epochs[request, relative_page].item())
                == int(request_epochs[request].item())
                and int(state_ranges[request, 0].item()) <= logical_position
                < int(state_ranges[request, 1].item())
            ):
                state_row = (
                    int(state_pages[request, relative_page].item())
                    * CSA_STATE_BLOCK_SIZE
                    + ring_row % CSA_STATE_BLOCK_SIZE
                )
                value = state[state_row // CSA_STATE_BLOCK_SIZE,
                              state_row % CSA_STATE_BLOCK_SIZE,
                              half : half + HEAD_DIM].float()
                score = state[state_row // CSA_STATE_BLOCK_SIZE,
                              state_row % CSA_STATE_BLOCK_SIZE,
                              OUT_DIM + half : OUT_DIM + half + HEAD_DIM].float()
            if first_position <= logical_position <= event_position:
                query = first_query + logical_position - first_position
                value = kv_projection[query, half : half + HEAD_DIM]
                score = (
                    score_projection[query, half : half + HEAD_DIM]
                    + ape[logical_position % COMPRESS_RATIO, half : half + HEAD_DIM]
                )
            values.append(value)
            scores.append(score)
        values = torch.stack(values)
        scores = torch.stack(scores)
        pooled[event_index] = (values * scores.softmax(dim=0)).sum(dim=0)

    for token in range(token_count):
        slot = int(state_slots[token].item())
        if slot < 0:
            continue
        position = int(positions[token].item())
        page, row = divmod(slot, CSA_STATE_BLOCK_SIZE)
        state[page, row, :OUT_DIM] = kv_projection[token]
        state[page, row, OUT_DIM:] = (
            score_projection[token] + ape[position % COMPRESS_RATIO]
        )

    tensors["compress_state"][:] = state
    for event_index, query_tensor in enumerate(event_queries):
        query = int(query_tensor.item())
        slot = int(event_slots[event_index].item())
        if query < 0 or slot < 0:
            continue
        row = pooled[event_index : event_index + 1]
        row = row * torch.rsqrt(row.square().mean(-1, keepdim=True) + EPS)
        row = row * norm_w
        pair = row[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        row0, row1 = pair[..., 0], pair[..., 1]
        cos = tensors["cos"][event_index].view(-1)
        sin = tensors["sin"][event_index].view(-1)
        rope = torch.stack(
            [row0 * cos - row1 * sin, row0 * sin + row1 * cos], dim=-1
        ).flatten(-2)
        row = torch.cat([row[..., :-ROPE_HEAD_DIM], rope], dim=-1)
        tensors["kv"][query : query + 1] = row
        block, intra = divmod(slot, BLOCK_SIZE)
        cache[block, intra, 0] = row[0].to(torch.bfloat16)
    tensors["cmp_kv_cache"][:] = cache


def build_tensor_specs(start_pos=None, batch=B):
    import torch
    from golden import TensorSpec
    from utils import (
        csa_decode_start_set,
        csa_event_local_rope,
        resolve_start_positions,
    )

    starts = resolve_start_positions(
        start_pos,
        batch=batch,
        seq=S,
        max_seq_len=M.max_position_embeddings,
        default_fn=lambda: csa_decode_start_set(
            batch=batch, seq=S, compress_ratio=COMPRESS_RATIO,
            state_block_size=CSA_STATE_BLOCK_SIZE,
        ),
    ).to(torch.int64)
    positions = starts[:, None] + torch.arange(S, dtype=torch.int64)[None, :]

    event_queries: list[int] = []
    for request in range(batch):
        for local_query in range(S):
            query = request * S + local_query
            if (int(positions[request, local_query].item()) + 1) % COMPRESS_RATIO == 0:
                event_queries.append(query)
    has_events = bool(event_queries)
    if not has_events:
        event_queries = [-1]
    event_query_ids = torch.tensor(event_queries, dtype=torch.int32)
    event_rows = torch.zeros(len(event_queries), dtype=torch.int64)
    for event_index, query in enumerate(event_queries):
        if query >= 0:
            event_rows[event_index] = positions.reshape(-1)[query] // COMPRESS_RATIO
    cos_full, sin_full = csa_event_local_rope(M, event_rows, dtype=torch.float32)
    cos = cos_full[:, : ROPE_HEAD_DIM // 2].contiguous()
    sin = sin_full[:, : ROPE_HEAD_DIM // 2].contiguous()

    state_page_ids = torch.arange(
        batch * CSA_STATE_PAGES_PER_REQUEST - 1, -1, -1, dtype=torch.int32
    ).reshape(batch, CSA_STATE_PAGES_PER_REQUEST)
    state_valid_ranges = torch.zeros(batch, 2, dtype=torch.int32)
    for request, start in enumerate(starts.tolist()):
        state_valid_ranges[request, 0] = max(0, start - CSA_STATE_ROWS_PER_REQUEST)
        state_valid_ranges[request, 1] = start
    request_epochs = torch.arange(batch, dtype=torch.int32) + 17
    state_page_epochs = request_epochs[:, None].expand(
        batch, CSA_STATE_PAGES_PER_REQUEST
    ).clone()
    state_slots = torch.empty(batch, S, dtype=torch.int64)
    for request in range(batch):
        for local_query in range(S):
            position = int(positions[request, local_query].item())
            ring_row = position % CSA_STATE_ROWS_PER_REQUEST
            relative_page = ring_row // CSA_STATE_BLOCK_SIZE
            page = int(state_page_ids[request, relative_page].item())
            state_slots[request, local_query] = (
                page * CSA_STATE_BLOCK_SIZE + ring_row % CSA_STATE_BLOCK_SIZE
            )

    cache_blocks = max(4, len(event_queries) if has_events else 4)
    event_write_slots = torch.full((len(event_queries),), -1, dtype=torch.int64)
    if has_events:
        for event_index, query in enumerate(event_queries):
            position = int(positions.reshape(-1)[query].item())
            candidate = position // COMPRESS_RATIO
            page = (3 * event_index + 1) % cache_blocks
            event_write_slots[event_index] = page * BLOCK_SIZE + candidate % BLOCK_SIZE
        valid_event_slots = event_write_slots[event_write_slots >= 0]
        if torch.unique(valid_event_slots).numel() != valid_event_slots.numel():
            raise AssertionError("CSA event write slots must not alias")

    state_pages = batch * CSA_STATE_PAGES_PER_REQUEST

    def init_state():
        state = torch.randn(
            state_pages, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM
        ) * 0.05
        return state

    return [
        TensorSpec("x", [batch * S, D], torch.bfloat16,
                   init_value=lambda: torch.rand(batch * S, D)),
        TensorSpec("kv", [batch * S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("compress_state",
                   [state_pages, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
                   torch.float32, init_value=init_state, is_output=True),
        TensorSpec("state_page_ids", [batch, CSA_STATE_PAGES_PER_REQUEST], torch.int32,
                   init_value=lambda: state_page_ids.clone()),
        TensorSpec("state_valid_ranges", [batch, 2], torch.int32,
                   init_value=lambda: state_valid_ranges.clone()),
        TensorSpec("state_page_epochs", [batch, CSA_STATE_PAGES_PER_REQUEST], torch.int32,
                   init_value=lambda: state_page_epochs.clone()),
        TensorSpec("request_epochs", [batch], torch.int32,
                   init_value=lambda: request_epochs.clone()),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16,
                   init_value=lambda: torch.randn(OUT_DIM, D) * 0.0245),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16,
                   init_value=lambda: torch.randn(OUT_DIM, D) * 0.0388),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32,
                   init_value=lambda: torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1243),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16,
                   init_value=lambda: 0.9666 + 0.1929 * torch.randn(HEAD_DIM)),
        TensorSpec("event_query_ids", [event_query_ids.numel()], torch.int32,
                   init_value=lambda: event_query_ids.clone()),
        TensorSpec("cos", list(cos.shape), torch.float32,
                   init_value=lambda: cos.clone()),
        TensorSpec("sin", list(sin.shape), torch.float32,
                   init_value=lambda: sin.clone()),
        TensorSpec("cmp_kv_cache", [cache_blocks, BLOCK_SIZE, 1, HEAD_DIM],
                   torch.bfloat16, init_value=lambda: torch.zeros(
                       cache_blocks, BLOCK_SIZE, 1, HEAD_DIM
                   ), is_output=True),
        TensorSpec("event_write_slots", [event_write_slots.numel()],
                   torch.int64,
                   init_value=lambda: event_write_slots.clone()),
        TensorSpec("state_slot_mapping", [batch * S], torch.int64,
                   init_value=lambda: state_slots.reshape(-1).clone()),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count; up to {B} (the compile-time "
                             "upper bound). The batch axis is pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch CSA set.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos, batch=args.batch),
        golden_fn=golden_compressor,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv":          ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state":    ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
