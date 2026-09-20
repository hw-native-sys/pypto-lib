# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-side metadata lowering for recurrent DeepSeek-V4 DSpark decode."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FLASH as M,
    HCA_CMP_STORAGE_BLOCK_SIZE,
    MOE_TOKENS,
    TP,
)


# Dynamic shape variables.
ORI_TABLE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_ORI_TABLE_BLOCKS_DYN")
HCA_CMP_TABLE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_HCA_CMP_TABLE_BLOCKS_DYN")
CSA_CMP_TABLE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_CSA_CMP_TABLE_BLOCKS_DYN")
IDX_TABLE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_IDX_TABLE_BLOCKS_DYN")
HCA_GROUP_STATE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_HCA_GROUP_STATE_BLOCKS_DYN")
CSA_GROUP_STATE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_CSA_GROUP_STATE_BLOCKS_DYN")
ROPE_ROWS_DYN = pl.dynamic("DSPARK_PREPARE_ROPE_ROWS_DYN")
DRAFTER_B_DYN = pl.dynamic("DSPARK_PREPARE_DRAFTER_B_DYN")
DRAFTER_HEAD_B_DYN = pl.dynamic("DSPARK_PREPARE_DRAFTER_HEAD_B_DYN")
COMMIT_B_DYN = pl.dynamic("DSPARK_STATE_COMMIT_B_DYN")

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
# Each rank owns one frozen MoE slab of tokens, not the TP split of the group
# batch: MOE_TOKENS keeps its EP=16 value, so the two agree only at TP=4, and
# the target forward takes its local token count off the slab.
LOCAL_T = MOE_TOKENS
LOCAL_B = LOCAL_T // S
LOCAL_BATCH = LOCAL_B          # the name the device-state stages use
D = M.hidden_size
MAIN_HIDDEN_DIM = 3 * D
DSPARK_QUERY_WIDTH = S - 1
WIN = M.sliding_window
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
HCA_COMPRESS_RATIO = 128
CSA_COMPRESS_RATIO = 4

# One state replica per TP rank, indexed by the group-local drafter lease; the
# fused wrapper publishes each updated row to its three peers.
STATE_CAPACITY = DECODE_BATCH
STATE_VALID = 0
STATE_GENERATION = 1
STATE_ANCHOR_POSITION = 2
STATE_COMMITTED_COUNT = 3
STATE_DRAFT_COUNT = 4
STATE_POSITION_LIMIT = 5
STATE_META_WIDTH = 6
STATE_CURRENT_TOKEN = 0
STATE_FIRST_DRAFT = 1
STATE_TOKEN_WIDTH = 1 + DSPARK_QUERY_WIDTH

# tiling
CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
HCA_STATE_TABLE_BLOCKS = M.max_position_embeddings // C128_COMPRESSOR_BLOCK_SIZE
HCA_HISTORY_PAGES = HCA_COMPRESS_RATIO // C128_COMPRESSOR_BLOCK_SIZE
CSA_STATE_STORAGE_LEN = 8 + S
CSA_STATE_TABLE_BLOCKS = (CSA_STATE_STORAGE_LEN + C4A_COMPRESSOR_BLOCK_SIZE - 1) // C4A_COMPRESSOR_BLOCK_SIZE
DRAFTER_CANDIDATE_ROWS = 16

assert S == STATE_TOKEN_WIDTH
assert S == DSPARK_QUERY_WIDTH + 1


@pl.jit.inline(auto_scope=False)
def gather_group_decode_rope_rows(
    swa_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    swa_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio4_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio4_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio128_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio128_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    active_widths: pl.Tensor[[B], pl.INT32],
    swa_cos: pl.Tensor[[LOCAL_T, ROPE_DIM], pl.BF16],
    swa_sin: pl.Tensor[[LOCAL_T, ROPE_DIM], pl.BF16],
    compressed_cos: pl.Tensor[[LOCAL_T, ROPE_DIM], pl.BF16],
    compressed_sin: pl.Tensor[[LOCAL_T, ROPE_DIM], pl.BF16],
    csa_cmp_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    csa_cmp_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    hca_cmp_cos: pl.Tensor[[B, HALF_ROPE], pl.FP32],
    hca_cmp_sin: pl.Tensor[[B, HALF_ROPE], pl.FP32],
    drafter_cos_candidates: pl.Tensor[[DRAFTER_B_DYN, DRAFTER_CANDIDATE_ROWS, ROPE_DIM], pl.BF16],
    drafter_sin_candidates: pl.Tensor[[DRAFTER_B_DYN, DRAFTER_CANDIDATE_ROWS, ROPE_DIM], pl.BF16],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather target and drafter RoPE rows from resident full tables."""
    rope_rows = pl.tensor.dim(swa_cos_table, 0)
    draft_batch = pl.tensor.dim(drafter_cos_candidates, 0)
    local_token_begin = tp_rank * LOCAL_T
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_decode_rope_rows"):
        for local_token in pl.range(LOCAL_T):
            group_token = local_token_begin + local_token
            position = pl.cast(pl.read(position_ids, [group_token]), pl.INDEX)
            swa_cos[local_token : local_token + 1, :] = swa_cos_table[
                position : position + 1, :
            ]
            swa_sin[local_token : local_token + 1, :] = swa_sin_table[
                position : position + 1, :
            ]
            compressed_cos[local_token : local_token + 1, :] = ratio128_cos_table[
                position : position + 1, :
            ]
            compressed_sin[local_token : local_token + 1, :] = ratio128_sin_table[
                position : position + 1, :
            ]
        for token in pl.range(T):
            group_position = pl.read(position_ids, [token])
            boundary = pl.cast(0, pl.INDEX)
            if (group_position + 1) % CSA_COMPRESS_RATIO == 0:
                boundary = pl.cast(group_position - CSA_COMPRESS_RATIO + 1, pl.INDEX)
            csa_cmp_cos[token : token + 1, :] = ratio4_cos_table[
                boundary : boundary + 1, :
            ]
            csa_cmp_sin[token : token + 1, :] = ratio4_sin_table[
                boundary : boundary + 1, :
            ]
        for request in pl.range(B):
            first_position = pl.read(position_ids, [request * S])
            boundary = pl.cast(first_position - first_position % HCA_COMPRESS_RATIO, pl.INDEX)
            hca_cmp_cos[request : request + 1, :] = pl.cast(
                ratio128_cos_table[boundary : boundary + 1, 0:HALF_ROPE],
                target_type=pl.FP32,
            )
            hca_cmp_sin[request : request + 1, :] = pl.cast(
                ratio128_sin_table[boundary : boundary + 1, 0:HALF_ROPE],
                target_type=pl.FP32,
            )
        for local_request in pl.range(draft_batch):
            group_request = tp_rank * LOCAL_B + local_request
            active_width = pl.read(active_widths, [group_request])
            anchor = pl.read(position_ids, [group_request * S])
            candidate_cos = pl.create_tensor([DRAFTER_CANDIDATE_ROWS, ROPE_DIM], dtype=pl.BF16)
            candidate_sin = pl.create_tensor([DRAFTER_CANDIDATE_ROWS, ROPE_DIM], dtype=pl.BF16)
            for offset in pl.range(DRAFTER_CANDIDATE_ROWS):
                position = pl.cast(0, pl.INDEX)
                if active_width > 0:
                    position = pl.cast(pl.min(anchor + offset, rope_rows - 1), pl.INDEX)
                candidate_cos[offset : offset + 1, :] = swa_cos_table[
                    position : position + 1, :
                ]
                candidate_sin[offset : offset + 1, :] = swa_sin_table[
                    position : position + 1, :
                ]
            drafter_cos_candidates[local_request, :, :] = candidate_cos
            drafter_sin_candidates[local_request, :, :] = candidate_sin
    return (
        swa_cos,
        swa_sin,
        compressed_cos,
        compressed_sin,
        csa_cmp_cos,
        csa_cmp_sin,
        hca_cmp_cos,
        hca_cmp_sin,
        drafter_cos_candidates,
        drafter_sin_candidates,
    )


@pl.jit.inline(auto_scope=False)
def build_group_decode_metadata(
    position_ids: pl.Tensor[[T], pl.INT32],
    active_widths: pl.Tensor[[B], pl.INT32],
    ori_block_table: pl.Tensor[[B, ORI_TABLE_BLOCKS_DYN], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[B, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[B, CSA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    idx_block_table: pl.Tensor[[B, IDX_TABLE_BLOCKS_DYN], pl.INT32],
    hca_state_block_table: pl.Tensor[[B, HCA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    csa_state_block_table: pl.Tensor[[B, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    csa_inner_state_block_table: pl.Tensor[[B, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    group_hca_state_block_table: pl.Tensor[[B, HCA_STATE_TABLE_BLOCKS], pl.INT32],
    group_csa_state_block_table: pl.Tensor[[B, CSA_STATE_TABLE_BLOCKS], pl.INT32],
    group_csa_inner_state_block_table: pl.Tensor[[B, CSA_STATE_TABLE_BLOCKS], pl.INT32],
    swa_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[LOCAL_T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[LOCAL_T], pl.INT32],
    hca_ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_swa_indices: pl.Tensor[[LOCAL_T, WIN], pl.INT32],
    hca_swa_lens: pl.Tensor[[LOCAL_T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_swa_indices: pl.Tensor[[LOCAL_T, WIN], pl.INT32],
    csa_swa_lens: pl.Tensor[[LOCAL_T], pl.INT32],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Build every acceptance-dependent TP-group cache descriptor."""
    hca_state_blocks = pl.tensor.dim(hca_state_block_table, 1)
    csa_state_blocks = pl.tensor.dim(csa_state_block_table, 1)
    for core in pl.spmd(1, name_hint="dspark_build_local_state_tables"):
        # HCA/CSA compressors consume the TP-gathered group stream, so every
        # TP rank needs the same full group request table.  This deliberately
        # differs from the attention query metadata below, which is local.
        for group_request in pl.range(B):
            active_width = pl.read(active_widths, [group_request])
            if active_width > 0:
                anchor = pl.read(position_ids, [group_request * S])
                last_hca_page = (anchor + active_width - 1) // C128_COMPRESSOR_BLOCK_SIZE
                for delta in pl.range(HCA_HISTORY_PAGES + 1):
                    logical_page = last_hca_page - delta
                    if logical_page >= 0 and logical_page < HCA_STATE_TABLE_BLOCKS:
                        pl.write(
                            group_hca_state_block_table,
                            [group_request, pl.cast(logical_page, pl.INDEX)],
                            pl.read(
                                hca_state_block_table,
                                [
                                    group_request,
                                    pl.cast(logical_page % hca_state_blocks, pl.INDEX),
                                ],
                            ),
                        )
                for table_index in pl.range(CSA_STATE_TABLE_BLOCKS):
                    pl.write(group_csa_state_block_table, [group_request, table_index], pl.cast(-1, pl.INT32))
                    pl.write(group_csa_inner_state_block_table, [group_request, table_index], pl.cast(-1, pl.INT32))
                last_page = (anchor + S - 1) // C4A_COMPRESSOR_BLOCK_SIZE
                for delta in pl.range(CSA_STATE_TABLE_BLOCKS):
                    logical_page = last_page - delta
                    if logical_page >= 0:
                        table_index = pl.cast(logical_page % CSA_STATE_TABLE_BLOCKS, pl.INDEX)
                        source_index = pl.cast(logical_page % csa_state_blocks, pl.INDEX)
                        pl.write(
                            group_csa_state_block_table,
                            [group_request, table_index],
                            pl.read(csa_state_block_table, [group_request, source_index]),
                        )
                        pl.write(
                            group_csa_inner_state_block_table,
                            [group_request, table_index],
                            pl.read(csa_inner_state_block_table, [group_request, source_index]),
                        )
    local_token_begin = tp_rank * LOCAL_T
    for local_token in pl.spmd(LOCAL_T, name_hint="dspark_build_swa_metadata"):
        group_token = local_token_begin + local_token
        request = group_token // S
        request_offset = group_token % S
        active_width = pl.read(active_widths, [request])
        position = pl.read(position_ids, [group_token])
        index_row = pl.create_tensor([1, WIN], dtype=pl.INT32)
        index_row[:, :] = pl.full([1, WIN], dtype=pl.INT32, value=-1)
        visible_len = pl.cast(0, pl.INT32)
        if request_offset < active_width:
            visible_len = pl.cast(pl.min(position + 1, WIN), pl.INT32)
            start = position - visible_len + 1
            for offset in pl.range(WIN):
                if offset < visible_len:
                    visible_position = start + offset
                    logical_block = visible_position // BLOCK_SIZE
                    block_offset = visible_position % BLOCK_SIZE
                    physical_block = pl.read(ori_block_table, [request, pl.cast(logical_block, pl.INDEX)])
                    pl.write(index_row, [0, offset], pl.cast(physical_block * BLOCK_SIZE + block_offset, pl.INT32))
        swa_indices[local_token : local_token + 1, :] = index_row
        hca_swa_indices[local_token : local_token + 1, :] = index_row
        csa_swa_indices[local_token : local_token + 1, :] = index_row

    for core in pl.spmd(1, name_hint="dspark_build_cache_metadata"):
        for token in pl.range(core, T):
            request = token // S
            request_offset = token % S
            active_width = pl.read(active_widths, [request])
            active = request_offset < active_width
            position = pl.read(position_ids, [token])
            visible_len = pl.cast(0, pl.INT32)
            if active:
                visible_len = pl.cast(pl.min(position + 1, WIN), pl.INT32)
            if token >= local_token_begin and token < local_token_begin + LOCAL_T:
                local_output_token = pl.cast(token - local_token_begin, pl.INDEX)
                pl.write(swa_lens, [local_output_token], visible_len)
                pl.write(hca_swa_lens, [local_output_token], visible_len)
                pl.write(csa_swa_lens, [local_output_token], visible_len)
            ori_slot = pl.cast(-1, pl.INT64)
            hca_cmp_slot = pl.cast(-1, pl.INT64)
            hca_state_slot = pl.cast(-1, pl.INT64)
            csa_cmp_slot = pl.cast(-1, pl.INT64)
            csa_idx_slot = pl.cast(-1, pl.INT64)
            csa_state_slot = pl.cast(-1, pl.INT64)
            csa_inner_state_slot = pl.cast(-1, pl.INT64)
            if active:
                logical_block = position // BLOCK_SIZE
                block_offset = position % BLOCK_SIZE
                ori_physical_block = pl.read(ori_block_table, [request, pl.cast(logical_block, pl.INDEX)])
                ori_slot = pl.cast(ori_physical_block * BLOCK_SIZE + block_offset, pl.INT64)

                if (position + 1) % HCA_COMPRESS_RATIO == 0:
                    cache_row = position // HCA_COMPRESS_RATIO
                    cache_block = cache_row // HCA_CMP_STORAGE_BLOCK_SIZE
                    cache_offset = cache_row % HCA_CMP_STORAGE_BLOCK_SIZE
                    physical_block = pl.read(hca_cmp_block_table, [request, pl.cast(cache_block, pl.INDEX)])
                    hca_cmp_slot = pl.cast(physical_block * HCA_CMP_STORAGE_BLOCK_SIZE + cache_offset, pl.INT64)

                if (position + 1) % CSA_COMPRESS_RATIO == 0:
                    cache_row = position // CSA_COMPRESS_RATIO
                    cache_block = cache_row // CSA_CMP_STORAGE_BLOCK_SIZE
                    cache_offset = cache_row % CSA_CMP_STORAGE_BLOCK_SIZE
                    csa_physical_block = pl.read(csa_cmp_block_table, [request, pl.cast(cache_block, pl.INDEX)])
                    idx_physical_block = pl.read(idx_block_table, [request, pl.cast(cache_block, pl.INDEX)])
                    csa_cmp_slot = pl.cast(csa_physical_block * CSA_CMP_STORAGE_BLOCK_SIZE + cache_offset, pl.INT64)
                    csa_idx_slot = pl.cast(idx_physical_block * CSA_CMP_STORAGE_BLOCK_SIZE + cache_offset, pl.INT64)

                hca_state_page = position // C128_COMPRESSOR_BLOCK_SIZE
                hca_state_offset = position % C128_COMPRESSOR_BLOCK_SIZE
                hca_state_physical_block = pl.read(
                    hca_state_block_table,
                    [
                        request,
                        pl.cast(hca_state_page % hca_state_blocks, pl.INDEX),
                    ],
                )
                hca_state_slot = pl.cast(
                    hca_state_physical_block * C128_COMPRESSOR_BLOCK_SIZE
                    + hca_state_offset,
                    pl.INT64,
                )

                csa_state_page = position // C4A_COMPRESSOR_BLOCK_SIZE
                csa_state_offset = position % C4A_COMPRESSOR_BLOCK_SIZE
                csa_state_physical_block = pl.read(
                    csa_state_block_table,
                    [
                        request,
                        pl.cast(csa_state_page % csa_state_blocks, pl.INDEX),
                    ],
                )
                csa_inner_state_physical_block = pl.read(
                    csa_inner_state_block_table,
                    [
                        request,
                        pl.cast(csa_state_page % csa_state_blocks, pl.INDEX),
                    ],
                )
                csa_state_slot = pl.cast(
                    csa_state_physical_block * C4A_COMPRESSOR_BLOCK_SIZE
                    + csa_state_offset,
                    pl.INT64,
                )
                csa_inner_state_slot = pl.cast(
                    csa_inner_state_physical_block * C4A_COMPRESSOR_BLOCK_SIZE
                    + csa_state_offset,
                    pl.INT64,
                )
            pl.write(swa_slot_mapping, [token], ori_slot)
            pl.write(hca_ori_slot_mapping, [token], ori_slot)
            pl.write(csa_ori_slot_mapping, [token], ori_slot)
            pl.write(hca_cmp_slot_mapping, [token], hca_cmp_slot)
            pl.write(hca_state_slot_mapping, [token], hca_state_slot)
            pl.write(csa_cmp_slot_mapping, [token], csa_cmp_slot)
            pl.write(csa_idx_slot_mapping, [token], csa_idx_slot)
            pl.write(csa_state_slot_mapping, [token], csa_state_slot)
            pl.write(csa_inner_state_slot_mapping, [token], csa_inner_state_slot)
    return (
        swa_slot_mapping,
        swa_indices,
        swa_lens,
        hca_ori_slot_mapping,
        hca_swa_indices,
        hca_swa_lens,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_ori_slot_mapping,
        csa_swa_indices,
        csa_swa_lens,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
        group_csa_state_block_table,
        group_csa_inner_state_block_table,
    )

@pl.jit.inline(auto_scope=False)
def prepare_target_group_from_device_state(
    group_state_slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    group_state_generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    state_tokens: pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.Tensor[[LOCAL_T], pl.INT64],
    position_ids_local: pl.Tensor[[LOCAL_T], pl.INT32],
    position_ids_group: pl.Tensor[[T], pl.INT32],
    csa_kv_seq_lens: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    hca_kv_seq_lens: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    logit_row_indices: pl.Tensor[[LOCAL_T], pl.INT32],
    sampled_row_offsets: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    active_widths: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    group_active_widths: pl.Tensor[[DECODE_BATCH], pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Resolve the TP-group target window from replicated persistent state."""
    for core in pl.spmd(1, name_hint="dspark_group_state_prepare"):
        local_begin = tp_rank * LOCAL_BATCH
        local_end = local_begin + LOCAL_BATCH
        for local_request in pl.range(LOCAL_BATCH):
            pl.write(csa_kv_seq_lens, [local_request], pl.cast(0, pl.INT32))
            pl.write(hca_kv_seq_lens, [local_request], pl.cast(0, pl.INT32))
            pl.write(sampled_row_offsets, [local_request], pl.cast(-1, pl.INT32))
            pl.write(active_widths, [local_request], pl.cast(0, pl.INT32))
            for offset in pl.range(S):
                local_row = local_request * S + offset
                pl.write(input_ids, [local_row], pl.cast(0, pl.INT64))
                pl.write(position_ids_local, [local_row], pl.cast(offset, pl.INT32))
                pl.write(logit_row_indices, [local_row], pl.cast(-1, pl.INT32))
        for request in pl.range(DECODE_BATCH):
            pl.write(group_active_widths, [request], pl.cast(0, pl.INT32))
            group_row = request * S
            for offset in pl.range(S):
                pl.write(position_ids_group, [group_row + offset], pl.cast(offset, pl.INT32))
            slot_raw = pl.read(group_state_slot_ids, [request])
            slot = pl.cast(pl.max(pl.min(slot_raw, STATE_CAPACITY - 1), 0), pl.INDEX)
            if slot_raw >= 0 and slot_raw < STATE_CAPACITY:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(group_state_generations, [request])
                if valid == 1 and generation == expected:
                    anchor = pl.read(state_meta, [slot, STATE_ANCHOR_POSITION])
                    draft_count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                    position_limit = pl.read(state_meta, [slot, STATE_POSITION_LIMIT])
                    active_width = pl.cast(1, pl.INT32)
                    if anchor + draft_count < position_limit:
                        active_width = pl.cast(draft_count + 1, pl.INT32)
                    pl.write(group_active_widths, [request], active_width)
                    for offset in pl.range(S):
                        pl.write(
                            position_ids_group,
                            [group_row + offset],
                            pl.cast(pl.min(anchor + offset, position_limit - 1), pl.INT32),
                        )
                    if request >= local_begin and request < local_end:
                        local_request = pl.cast(request - local_begin, pl.INDEX)
                        local_row = local_request * S
                        for offset in pl.range(S):
                            token = pl.read(state_tokens, [slot, offset])
                            pl.write(input_ids, [local_row + offset], token)
                            pl.write(
                                position_ids_local,
                                [local_row + offset],
                                pl.cast(pl.min(anchor + offset, position_limit - 1), pl.INT32),
                            )
                            if offset < active_width:
                                pl.write(
                                    logit_row_indices,
                                    [local_row + offset],
                                    pl.cast(local_row + offset, pl.INT32),
                                )
                        pl.write(csa_kv_seq_lens, [local_request], pl.cast(anchor + active_width, pl.INT32))
                        pl.write(hca_kv_seq_lens, [local_request], pl.cast(anchor + active_width, pl.INT32))
                        pl.write(sampled_row_offsets, [local_request], pl.cast(local_row, pl.INT32))
                        pl.write(active_widths, [local_request], active_width)
    return (
        input_ids,
        position_ids_local,
        position_ids_group,
        csa_kv_seq_lens,
        hca_kv_seq_lens,
        logit_row_indices,
        sampled_row_offsets,
        active_widths,
        group_active_widths,
    )


@pl.jit.inline(auto_scope=False)
def accept_target_into_device_state(
    state_slot_ids: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    sampled_row_offsets: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    hidden_row_offsets: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32]],
    sampled_ids: pl.Tensor[[LOCAL_T, S], pl.INT32],
    target_hidden: pl.Tensor[[LOCAL_T, MAIN_HIDDEN_DIM], pl.BF16],
    accepted_token_ids: pl.Out[pl.Tensor[[LOCAL_BATCH, S], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
    drafter_target_hidden: pl.Out[pl.Tensor[[LOCAL_T, MAIN_HIDDEN_DIM], pl.BF16]],
    drafter_context_positions: pl.Out[pl.Tensor[[LOCAL_T], pl.INT32]],
    drafter_context_valid: pl.Out[pl.Tensor[[LOCAL_T], pl.INT32]],
    drafter_last_sampled: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT64]],
    drafter_anchor_positions: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
    drafter_row_offsets: pl.Out[pl.Tensor[[LOCAL_BATCH], pl.INT32]],
    drafter_ready: pl.Out[pl.Tensor[[1], pl.INT32]],
):
    """Accept the longest matching prefix and prepare the next drafter inputs."""
    for core in pl.spmd(1, name_hint="dspark_state_accept"):
        next_drafter_row = pl.cast(0, pl.INT32)
        for request in pl.range(core, LOCAL_BATCH):
            pl.write(accepted_counts, [request], pl.cast(0, pl.INT32))
            pl.write(drafter_last_sampled, [request], pl.cast(0, pl.INT64))
            pl.write(drafter_anchor_positions, [request], pl.cast(0, pl.INT32))
            pl.write(drafter_row_offsets, [request], pl.cast(-1, pl.INT32))
            for offset in pl.range(S):
                pl.write(accepted_token_ids, [request, offset], pl.cast(-1, pl.INT32))
                context_row = request * S + offset
                pl.write(drafter_context_positions, [context_row], pl.cast(0, pl.INT32))
                pl.write(drafter_context_valid, [context_row], pl.cast(0, pl.INT32))
            slot_raw = pl.read(state_slot_ids, [request])
            sampled_row_raw = pl.read(sampled_row_offsets, [request])
            slot = pl.cast(pl.max(pl.min(slot_raw, STATE_CAPACITY - 1), 0), pl.INDEX)
            if slot_raw >= 0 and slot_raw < STATE_CAPACITY and sampled_row_raw >= 0:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    sampled_row = pl.cast(sampled_row_raw, pl.INDEX)
                    old_anchor = pl.read(state_meta, [slot, STATE_ANCHOR_POSITION])
                    draft_count = pl.read(state_meta, [slot, STATE_DRAFT_COUNT])
                    position_limit = pl.read(state_meta, [slot, STATE_POSITION_LIMIT])
                    effective_draft_count = pl.cast(0, pl.INT32)
                    if old_anchor + draft_count < position_limit:
                        effective_draft_count = draft_count
                    matched = pl.cast(0, pl.INT32)
                    still_matching = pl.cast(1, pl.INT32)
                    for draft_offset in pl.range(DSPARK_QUERY_WIDTH):
                        draft = pl.read(state_tokens, [slot, STATE_FIRST_DRAFT + draft_offset])
                        predicted = pl.cast(pl.read(sampled_ids, [sampled_row + draft_offset, 0]), pl.INT64)
                        if (
                            draft_offset < effective_draft_count
                            and still_matching == 1
                            and draft == predicted
                        ):
                            matched = pl.cast(draft_offset + 1, pl.INT32)
                        else:
                            still_matching = pl.cast(0, pl.INT32)
                    accepted = pl.cast(matched + 1, pl.INT32)
                    next_token = pl.cast(pl.read(sampled_ids, [sampled_row + matched, 0]), pl.INT64)
                    committed = pl.read(state_meta, [slot, STATE_COMMITTED_COUNT])
                    pl.write(accepted_counts, [request], accepted)
                    pl.write(drafter_last_sampled, [request], next_token)
                    pl.write(drafter_anchor_positions, [request], pl.cast(old_anchor + accepted - 1, pl.INT32))
                    for offset in pl.range(S):
                        if offset < accepted:
                            accepted_token = pl.read(sampled_ids, [sampled_row + offset, 0])
                            pl.write(accepted_token_ids, [request, offset], accepted_token)
                            context_row = request * S + offset
                            pl.write(drafter_context_positions, [context_row], pl.cast(old_anchor + offset, pl.INT32))
                            pl.write(drafter_context_valid, [context_row], pl.cast(1, pl.INT32))
                    pl.write(state_tokens, [slot, STATE_CURRENT_TOKEN], next_token)
                    pl.write(state_meta, [slot, STATE_ANCHOR_POSITION], pl.cast(old_anchor + accepted, pl.INT32))
                    pl.write(state_meta, [slot, STATE_COMMITTED_COUNT], pl.cast(committed + accepted, pl.INT32))
                    # Old drafts are consumed by this verification.  Publish a
                    # compact drafter destination only when another full K=7
                    # query window fits; commit_drafts restores draft_count.
                    pl.write(state_meta, [slot, STATE_DRAFT_COUNT], pl.cast(0, pl.INT32))
                    next_anchor = pl.cast(old_anchor + accepted - 1, pl.INT32)
                    next_target_anchor = pl.cast(old_anchor + accepted, pl.INT32)
                    if next_target_anchor + DSPARK_QUERY_WIDTH < position_limit:
                        pl.write(drafter_row_offsets, [request], pl.cast(next_drafter_row * S, pl.INT32))
                        next_drafter_row = pl.cast(next_drafter_row + 1, pl.INT32)

    with pl.spmd(LOCAL_T, name_hint="dspark_state_pack_hidden") as pack_hidden_tid:
        token = pl.tile.get_block_idx()
        request = token // S
        offset = token % S
        accepted = pl.read(accepted_counts, [request])
        destination_base = pl.read(drafter_row_offsets, [request])
        if destination_base >= 0 and offset < accepted:
            source_base = pl.read(hidden_row_offsets, [request])
            source = pl.cast(source_base + offset, pl.INDEX)
            destination = pl.cast(destination_base + offset, pl.INDEX)
            drafter_target_hidden[destination : destination + 1, 0:MAIN_HIDDEN_DIM] = (
                target_hidden[source : source + 1, 0:MAIN_HIDDEN_DIM]
            )
        elif destination_base >= 0:
            destination = pl.cast(destination_base + offset, pl.INDEX)
            drafter_target_hidden[destination : destination + 1, 0:MAIN_HIDDEN_DIM] = pl.full(
                [1, MAIN_HIDDEN_DIM],
                dtype=pl.BF16,
                value=0.0,
            )
    with pl.spmd(
        1,
        name_hint="dspark_state_publish_drafter_ready",
        deps=[pack_hidden_tid],
    ):
        publish_core = pl.tile.get_block_idx()
        pl.write(drafter_ready, [publish_core], pl.cast(0, pl.INT32))
    return (
        state_tokens,
        state_meta,
        accepted_token_ids,
        accepted_counts,
        drafter_target_hidden,
        drafter_context_positions,
        drafter_context_valid,
        drafter_last_sampled,
        drafter_anchor_positions,
        drafter_row_offsets,
        drafter_ready,
    )


@pl.jit.inline(auto_scope=False)
def commit_drafts_to_device_state(
    state_slot_ids: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32]],
    draft_token_ids: pl.Tensor[[COMMIT_B_DYN, DSPARK_QUERY_WIDTH], pl.INT32],
):
    """Commit the Markov output as the next target verification window."""
    draft_token_ids.bind_dynamic(0, COMMIT_B_DYN)
    batch = pl.tensor.dim(draft_token_ids, 0)
    for core in pl.spmd(1, name_hint="dspark_state_commit_drafts"):
        for request in pl.range(core, batch):
            slot_raw = pl.read(state_slot_ids, [request])
            slot = pl.cast(pl.max(pl.min(slot_raw, STATE_CAPACITY - 1), 0), pl.INDEX)
            if slot_raw >= 0 and slot_raw < STATE_CAPACITY:
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    anchor = pl.read(state_meta, [slot, STATE_ANCHOR_POSITION])
                    position_limit = pl.read(state_meta, [slot, STATE_POSITION_LIMIT])
                    if anchor + DSPARK_QUERY_WIDTH < position_limit:
                        for offset in pl.range(DSPARK_QUERY_WIDTH):
                            token = pl.cast(pl.read(draft_token_ids, [request, offset]), pl.INT64)
                            pl.write(state_tokens, [slot, STATE_FIRST_DRAFT + offset], token)
                        pl.write(state_meta, [slot, STATE_DRAFT_COUNT], pl.cast(DSPARK_QUERY_WIDTH, pl.INT32))
    return state_tokens, state_meta


@pl.jit.inline(auto_scope=False)
def fence_drafter_head_hidden(
    head_hidden: pl.InOut[pl.Tensor[[DRAFTER_HEAD_B_DYN, DSPARK_QUERY_WIDTH, D], pl.BF16]],
    drafter_ready: pl.Scalar[pl.TASK_ID],
):
    """Order the Markov sampler behind the drafter's last head-hidden write."""
    head_hidden.bind_dynamic(0, DRAFTER_HEAD_B_DYN)
    batch = pl.tensor.dim(head_hidden, 0)
    active_tokens = batch * DSPARK_QUERY_WIDTH
    head_hidden_flat = pl.reshape(head_hidden, [active_tokens, D])
    with pl.spmd(
        active_tokens,
        name_hint="dspark_drafter_markov_bridge",
        deps=[drafter_ready],
    ):
        token = pl.tile.get_block_idx()
        head_hidden_flat[token : token + 1, :] = head_hidden_flat[token : token + 1, :]


# ---------------------------------------------------------------------------
# Drafter bridge: compact the accepted rows and publish rank-major drafter
# metadata.  The geometry below mirrors the dspark_drafter contract, derived
# from config so that this module never imports the drafter (the drafter
# rewrites config.TP/EP while it is imported).
# ---------------------------------------------------------------------------
DSPARK_DRAFT_LAYERS = 3
DSPARK_QUERY_PAD = S
DSPARK_MAX_BATCH = MOE_TOKENS // DSPARK_QUERY_PAD
DSPARK_CP_SIZE = TP
T_QUERY = DSPARK_MAX_BATCH * DSPARK_QUERY_WIDTH
ORI_MAX_BLOCKS = (M.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
MAX_LOGIT_ROWS = MOE_TOKENS

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
    metadata_window: pld.DistributedTensor[[GROUP_METADATA_ROWS, METADATA_WIDTH], pl.INT64],
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
    accepted_counts: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    context_positions: pl.Tensor[[CONTEXT_T], pl.INT32],
    context_valid: pl.Tensor[[CONTEXT_T], pl.INT32],
    last_sampled: pl.Tensor[[LOCAL_BATCH], pl.INT64],
    anchor_positions: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    row_offsets: pl.Tensor[[LOCAL_BATCH], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, BRIDGE_B_DYN, ORI_MAX_BLOCKS], pl.INT32],
    rope_cos_candidates: pl.Tensor[[BRIDGE_B_DYN, ROPE_CANDIDATE_ROWS, ROPE_DIM], pl.BF16],
    rope_sin_candidates: pl.Tensor[[BRIDGE_B_DYN, ROPE_CANDIDATE_ROWS, ROPE_DIM], pl.BF16],
    num_sampled: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT32]],
    compact_last_sampled: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT64]],
    next_prefill_tokens: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT64]],
    compact_anchor_positions: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT32]],
    compact_state_slot_ids: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT32]],
    compact_state_generations: pl.Out[pl.Tensor[[BRIDGE_B_DYN], pl.INT32]],
    logit_row_indices: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    context_group_position_ids: pl.Out[pl.Tensor[[BRIDGE_GROUP_CONTEXT_T_DYN], pl.INT32]],
    context_group_slot_mapping: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, BRIDGE_GROUP_CONTEXT_T_DYN], pl.INT64]],
    query_group_position_ids: pl.Out[pl.Tensor[[DSPARK_CP_SIZE * T_QUERY], pl.INT32]],
    query_group_slot_mapping: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, DSPARK_CP_SIZE * T_QUERY], pl.INT64]],
    context_group_freqs_cos: pl.Out[pl.Tensor[[BRIDGE_GROUP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16]],
    context_group_freqs_sin: pl.Out[pl.Tensor[[BRIDGE_GROUP_CONTEXT_T_DYN, ROPE_DIM], pl.BF16]],
    query_freqs_cos: pl.Out[pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16]],
    query_freqs_sin: pl.Out[pl.Tensor[[T_QUERY, ROPE_DIM], pl.BF16]],
    query_group_freqs_cos: pl.Out[pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16]],
    query_group_freqs_sin: pl.Out[pl.Tensor[[DSPARK_CP_SIZE * T_QUERY, ROPE_DIM], pl.BF16]],
    metadata_window: pld.DistributedTensor[[GROUP_METADATA_ROWS, METADATA_WIDTH], pl.INT64],
    metadata_signal: pld.DistributedTensor[[DSPARK_CP_SIZE, 1], pl.INT32],
    rope_cos_window: pld.DistributedTensor[[GROUP_METADATA_ROWS, ROPE_DIM], pl.BF16],
    rope_sin_window: pld.DistributedTensor[[GROUP_METADATA_ROWS, ROPE_DIM], pl.BF16],
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
    local_metadata = pl.create_tensor([LOCAL_METADATA_ROWS, METADATA_WIDTH], dtype=pl.INT64)
    local_rope_cos = pl.create_tensor([LOCAL_METADATA_ROWS, ROPE_DIM], dtype=pl.BF16)
    local_rope_sin = pl.create_tensor([LOCAL_METADATA_ROWS, ROPE_DIM], dtype=pl.BF16)
    group_metadata = pl.create_tensor([GROUP_METADATA_ROWS, METADATA_WIDTH], dtype=pl.INT64)
    group_rope_cos = pl.create_tensor([GROUP_METADATA_ROWS, ROPE_DIM], dtype=pl.BF16)
    group_rope_sin = pl.create_tensor([GROUP_METADATA_ROWS, ROPE_DIM], dtype=pl.BF16)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_bridge_prepare"):
        ready_offset = pl.read(drafter_ready, [0])
        for row in pl.range(LOCAL_METADATA_ROWS):
            pl.write(local_metadata, [row, 0], pl.cast(0, pl.INT64))
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(local_metadata, [row, 1 + layer], pl.cast(-1, pl.INT64))
            local_rope_cos[row : row + 1, 0:ROPE_DIM] = pl.full([1, ROPE_DIM], dtype=pl.BF16, value=0.0)
            local_rope_sin[row : row + 1, 0:ROPE_DIM] = pl.full([1, ROPE_DIM], dtype=pl.BF16, value=0.0)
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
                pl.write(compact_last_sampled, [destination], pl.read(last_sampled, [request]))
                pl.write(compact_anchor_positions, [destination], pl.read(anchor_positions, [request]))
                pl.write(compact_state_slot_ids, [destination], pl.read(state_slot_ids, [request]))
                pl.write(compact_state_generations, [destination], pl.read(state_generations, [request]))
                for offset in pl.range(S):
                    source_row = request * S + offset
                    destination_row = destination * S + offset
                    valid = pl.read(context_valid, [source_row])
                    if valid == 1:
                        position = pl.read(context_positions, [source_row]) + ready_offset
                        pl.write(local_metadata, [destination_row, 0], pl.cast(position, pl.INT64))
                        for layer in pl.range(DSPARK_DRAFT_LAYERS):
                            logical_block = position // BLOCK_SIZE
                            physical_block = pl.read(block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)])
                            slot = physical_block * BLOCK_SIZE + position % BLOCK_SIZE
                            pl.write(local_metadata, [destination_row, 1 + layer], pl.cast(slot, pl.INT64))
                        context_cos_row = rope_cos_candidates[request : request + 1, offset : offset + 1, 0:ROPE_DIM]
                        context_cos_flat = pl.reshape(context_cos_row, [1, ROPE_DIM])
                        local_rope_cos[destination_row : destination_row + 1, 0:ROPE_DIM] = context_cos_flat
                        context_sin_row = rope_sin_candidates[request : request + 1, offset : offset + 1, 0:ROPE_DIM]
                        context_sin_flat = pl.reshape(context_sin_row, [1, ROPE_DIM])
                        local_rope_sin[destination_row : destination_row + 1, 0:ROPE_DIM] = context_sin_flat
                for query in pl.range(DSPARK_QUERY_WIDTH):
                    query_row = destination * DSPARK_QUERY_WIDTH + query
                    query_position = pl.read(anchor_positions, [request]) + 1 + query + ready_offset
                    metadata_row = CONTEXT_T + query_row
                    pl.write(local_metadata, [metadata_row, 0], pl.cast(query_position, pl.INT64))
                    for layer in pl.range(DSPARK_DRAFT_LAYERS):
                        logical_block = query_position // BLOCK_SIZE
                        physical_block = pl.read(block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)])
                        slot = physical_block * BLOCK_SIZE + query_position % BLOCK_SIZE
                        pl.write(local_metadata, [metadata_row, 1 + layer], pl.cast(slot, pl.INT64))
                    candidate_row = accepted + query
                    query_cos_row = rope_cos_candidates[request : request + 1, candidate_row : candidate_row + 1, 0:ROPE_DIM]
                    query_cos_flat = pl.reshape(query_cos_row, [1, ROPE_DIM])
                    local_rope_cos[metadata_row : metadata_row + 1, 0:ROPE_DIM] = query_cos_flat
                    query_sin_row = rope_sin_candidates[request : request + 1, candidate_row : candidate_row + 1, 0:ROPE_DIM]
                    query_sin_flat = pl.reshape(query_sin_row, [1, ROPE_DIM])
                    local_rope_sin[metadata_row : metadata_row + 1, 0:ROPE_DIM] = query_sin_flat
                    pl.write(logit_row_indices, [query_row], pl.cast(query_row, pl.INT32))

    group_metadata, metadata_signal = _allgather_metadata(
        local_metadata, group_metadata,
        metadata_window, metadata_signal,
        group_base, cp_rank,
    )
    group_rope_cos, rope_cos_signal = _allgather_rope(
        local_rope_cos, group_rope_cos,
        rope_cos_window, rope_cos_signal,
        group_base, cp_rank,
    )
    group_rope_sin, rope_sin_signal = _allgather_rope(
        local_rope_sin, group_rope_sin,
        rope_sin_window, rope_sin_signal,
        group_base, cp_rank,
    )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_bridge_unpack"):
        for row in pl.range(group_context_tokens):
            source_rank = row // local_context_tokens
            source_row = row % local_context_tokens
            metadata_row = source_rank * LOCAL_METADATA_ROWS + source_row
            pl.write(context_group_position_ids, [row], pl.cast(pl.read(group_metadata, [metadata_row, 0]), pl.INT32))
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(context_group_slot_mapping, [layer, row], pl.read(group_metadata, [metadata_row, 1 + layer]))
            group_cos_src = group_rope_cos[metadata_row : metadata_row + 1, 0:ROPE_DIM]
            context_group_freqs_cos[row : row + 1, 0:ROPE_DIM] = group_cos_src
            group_sin_src = group_rope_sin[metadata_row : metadata_row + 1, 0:ROPE_DIM]
            context_group_freqs_sin[row : row + 1, 0:ROPE_DIM] = group_sin_src
        for row in pl.range(DSPARK_CP_SIZE * T_QUERY):
            source_rank = row // T_QUERY
            source_row = row % T_QUERY
            metadata_row = source_rank * LOCAL_METADATA_ROWS + CONTEXT_T + source_row
            pl.write(query_group_position_ids, [row], pl.cast(pl.read(group_metadata, [metadata_row, 0]), pl.INT32))
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(query_group_slot_mapping, [layer, row], pl.read(group_metadata, [metadata_row, 1 + layer]))
            query_group_cos_src = group_rope_cos[metadata_row : metadata_row + 1, 0:ROPE_DIM]
            query_group_freqs_cos[row : row + 1, 0:ROPE_DIM] = query_group_cos_src
            query_group_sin_src = group_rope_sin[metadata_row : metadata_row + 1, 0:ROPE_DIM]
            query_group_freqs_sin[row : row + 1, 0:ROPE_DIM] = query_group_sin_src
        for row in pl.range(T_QUERY):
            metadata_row = CONTEXT_T + row
            local_cos_src = local_rope_cos[metadata_row : metadata_row + 1, 0:ROPE_DIM]
            query_freqs_cos[row : row + 1, 0:ROPE_DIM] = local_cos_src
            local_sin_src = local_rope_sin[metadata_row : metadata_row + 1, 0:ROPE_DIM]
            query_freqs_sin[row : row + 1, 0:ROPE_DIM] = local_sin_src
    return (
        num_sampled, compact_last_sampled, next_prefill_tokens, compact_anchor_positions,
        compact_state_slot_ids, compact_state_generations, logit_row_indices,
        context_group_position_ids, context_group_slot_mapping,
        query_group_position_ids, query_group_slot_mapping,
        context_group_freqs_cos, context_group_freqs_sin,
        query_freqs_cos, query_freqs_sin,
        query_group_freqs_cos, query_group_freqs_sin,
    )
