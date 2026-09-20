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
    EP as N_RANKS,
    FLASH as M,
    TP,
)
from dspark_device_state import (
    STATE_CAPACITY,
    STATE_META_WIDTH,
    STATE_TOKEN_WIDTH,
    build_group_prepare_tensor_specs,
    prepare_target_group_from_device_state,
)


ORI_TABLE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_ORI_TABLE_BLOCKS_DYN")
HCA_CMP_TABLE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_HCA_CMP_TABLE_BLOCKS_DYN")
CSA_CMP_TABLE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_CSA_CMP_TABLE_BLOCKS_DYN")
IDX_TABLE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_IDX_TABLE_BLOCKS_DYN")
HCA_GROUP_STATE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_HCA_GROUP_STATE_BLOCKS_DYN")
CSA_GROUP_STATE_BLOCKS_DYN = pl.dynamic("DSPARK_PREPARE_CSA_GROUP_STATE_BLOCKS_DYN")
ROPE_ROWS_DYN = pl.dynamic("DSPARK_PREPARE_ROPE_ROWS_DYN")
DRAFTER_B_DYN = pl.dynamic("DSPARK_PREPARE_DRAFTER_B_DYN")

B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
LOCAL_B = B // TP
LOCAL_T = LOCAL_B * S
WIN = M.sliding_window
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
HCA_COMPRESS_RATIO = 128
CSA_COMPRESS_RATIO = 4
HCA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
HCA_STATE_TABLE_BLOCKS = M.max_position_embeddings // C128_COMPRESSOR_BLOCK_SIZE
HCA_HISTORY_PAGES = HCA_COMPRESS_RATIO // C128_COMPRESSOR_BLOCK_SIZE
CSA_STATE_STORAGE_LEN = 8 + S
CSA_STATE_TABLE_BLOCKS = (
    CSA_STATE_STORAGE_LEN + C4A_COMPRESSOR_BLOCK_SIZE - 1
) // C4A_COMPRESSOR_BLOCK_SIZE
DRAFTER_CANDIDATE_ROWS = 16


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
    drafter_cos_candidates: pl.Tensor[
        [DRAFTER_B_DYN, DRAFTER_CANDIDATE_ROWS, ROPE_DIM], pl.BF16
    ],
    drafter_sin_candidates: pl.Tensor[
        [DRAFTER_B_DYN, DRAFTER_CANDIDATE_ROWS, ROPE_DIM], pl.BF16
    ],
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
                boundary = pl.cast(
                    group_position - CSA_COMPRESS_RATIO + 1,
                    pl.INDEX,
                )
            csa_cmp_cos[token : token + 1, :] = ratio4_cos_table[
                boundary : boundary + 1, :
            ]
            csa_cmp_sin[token : token + 1, :] = ratio4_sin_table[
                boundary : boundary + 1, :
            ]
        for request in pl.range(B):
            first_position = pl.read(position_ids, [request * S])
            boundary = pl.cast(
                first_position - first_position % HCA_COMPRESS_RATIO,
                pl.INDEX,
            )
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
            candidate_cos = pl.create_tensor(
                [DRAFTER_CANDIDATE_ROWS, ROPE_DIM],
                dtype=pl.BF16,
            )
            candidate_sin = pl.create_tensor(
                [DRAFTER_CANDIDATE_ROWS, ROPE_DIM],
                dtype=pl.BF16,
            )
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


gather_group_decode_rope_rows_l2 = pl.jit(auto_scope=False)(
    gather_group_decode_rope_rows._func
)


@pl.jit.host
def l3_gather_group_decode_rope_rows(
    swa_cos_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    swa_sin_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio4_cos_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio4_sin_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio128_cos_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio128_sin_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    active_widths: pl.Tensor[[N_RANKS, B], pl.INT32],
    swa_cos: pl.InOut[pl.Tensor[[N_RANKS, LOCAL_T, ROPE_DIM], pl.BF16]],
    swa_sin: pl.InOut[pl.Tensor[[N_RANKS, LOCAL_T, ROPE_DIM], pl.BF16]],
    compressed_cos: pl.InOut[
        pl.Tensor[[N_RANKS, LOCAL_T, ROPE_DIM], pl.BF16]
    ],
    compressed_sin: pl.InOut[
        pl.Tensor[[N_RANKS, LOCAL_T, ROPE_DIM], pl.BF16]
    ],
    csa_cmp_cos: pl.InOut[pl.Tensor[[N_RANKS, T, ROPE_DIM], pl.BF16]],
    csa_cmp_sin: pl.InOut[pl.Tensor[[N_RANKS, T, ROPE_DIM], pl.BF16]],
    hca_cmp_cos: pl.InOut[pl.Tensor[[N_RANKS, B, HALF_ROPE], pl.FP32]],
    hca_cmp_sin: pl.InOut[pl.Tensor[[N_RANKS, B, HALF_ROPE], pl.FP32]],
    drafter_cos_candidates: pl.InOut[
        pl.Tensor[
            [N_RANKS, DRAFTER_B_DYN, DRAFTER_CANDIDATE_ROWS, ROPE_DIM],
            pl.BF16,
        ]
    ],
    drafter_sin_candidates: pl.InOut[
        pl.Tensor[
            [N_RANKS, DRAFTER_B_DYN, DRAFTER_CANDIDATE_ROWS, ROPE_DIM],
            pl.BF16,
        ]
    ],
):
    swa_cos_table.bind_dynamic(1, ROPE_ROWS_DYN)
    swa_sin_table.bind_dynamic(1, ROPE_ROWS_DYN)
    ratio4_cos_table.bind_dynamic(1, ROPE_ROWS_DYN)
    ratio4_sin_table.bind_dynamic(1, ROPE_ROWS_DYN)
    ratio128_cos_table.bind_dynamic(1, ROPE_ROWS_DYN)
    ratio128_sin_table.bind_dynamic(1, ROPE_ROWS_DYN)
    drafter_cos_candidates.bind_dynamic(1, DRAFTER_B_DYN)
    drafter_sin_candidates.bind_dynamic(1, DRAFTER_B_DYN)
    for rank in pl.range(pld.world_size()):
        gather_group_decode_rope_rows_l2(
            swa_cos_table[rank],
            swa_sin_table[rank],
            ratio4_cos_table[rank],
            ratio4_sin_table[rank],
            ratio128_cos_table[rank],
            ratio128_sin_table[rank],
            position_ids[rank],
            active_widths[rank],
            swa_cos[rank],
            swa_sin[rank],
            compressed_cos[rank],
            compressed_sin[rank],
            csa_cmp_cos[rank],
            csa_cmp_sin[rank],
            hca_cmp_cos[rank],
            hca_cmp_sin[rank],
            drafter_cos_candidates[rank],
            drafter_sin_candidates[rank],
            rank % TP,
            device=rank,
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
    group_hca_state_block_table: pl.Tensor[
        [B, HCA_STATE_TABLE_BLOCKS], pl.INT32
    ],
    group_csa_state_block_table: pl.Tensor[[B, CSA_STATE_TABLE_BLOCKS], pl.INT32],
    group_csa_inner_state_block_table: pl.Tensor[
        [B, CSA_STATE_TABLE_BLOCKS], pl.INT32
    ],
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
                                    pl.cast(
                                        logical_page % hca_state_blocks,
                                        pl.INDEX,
                                    ),
                                ],
                            ),
                        )
                for table_index in pl.range(CSA_STATE_TABLE_BLOCKS):
                    pl.write(
                        group_csa_state_block_table,
                        [group_request, table_index],
                        pl.cast(-1, pl.INT32),
                    )
                    pl.write(
                        group_csa_inner_state_block_table,
                        [group_request, table_index],
                        pl.cast(-1, pl.INT32),
                    )
                last_page = (anchor + S - 1) // C4A_COMPRESSOR_BLOCK_SIZE
                for delta in pl.range(CSA_STATE_TABLE_BLOCKS):
                    logical_page = last_page - delta
                    if logical_page >= 0:
                        table_index = pl.cast(
                            logical_page % CSA_STATE_TABLE_BLOCKS,
                            pl.INDEX,
                        )
                        source_index = pl.cast(
                            logical_page % csa_state_blocks,
                            pl.INDEX,
                        )
                        pl.write(
                            group_csa_state_block_table,
                            [group_request, table_index],
                            pl.read(
                                csa_state_block_table,
                                [group_request, source_index],
                            ),
                        )
                        pl.write(
                            group_csa_inner_state_block_table,
                            [group_request, table_index],
                            pl.read(
                                csa_inner_state_block_table,
                                [group_request, source_index],
                            ),
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
                    physical_block = pl.read(
                        ori_block_table,
                        [request, pl.cast(logical_block, pl.INDEX)],
                    )
                    pl.write(
                        index_row,
                        [0, offset],
                        pl.cast(physical_block * BLOCK_SIZE + block_offset, pl.INT32),
                    )
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
                ori_physical_block = pl.read(
                    ori_block_table,
                    [request, pl.cast(logical_block, pl.INDEX)],
                )
                ori_slot = pl.cast(
                    ori_physical_block * BLOCK_SIZE + block_offset,
                    pl.INT64,
                )

                if (position + 1) % HCA_COMPRESS_RATIO == 0:
                    cache_row = position // HCA_COMPRESS_RATIO
                    cache_block = cache_row // HCA_CMP_STORAGE_BLOCK_SIZE
                    cache_offset = cache_row % HCA_CMP_STORAGE_BLOCK_SIZE
                    physical_block = pl.read(
                        hca_cmp_block_table,
                        [request, pl.cast(cache_block, pl.INDEX)],
                    )
                    hca_cmp_slot = pl.cast(
                        physical_block * HCA_CMP_STORAGE_BLOCK_SIZE + cache_offset,
                        pl.INT64,
                    )

                if (position + 1) % CSA_COMPRESS_RATIO == 0:
                    cache_row = position // CSA_COMPRESS_RATIO
                    cache_block = cache_row // CSA_CMP_STORAGE_BLOCK_SIZE
                    cache_offset = cache_row % CSA_CMP_STORAGE_BLOCK_SIZE
                    csa_physical_block = pl.read(
                        csa_cmp_block_table,
                        [request, pl.cast(cache_block, pl.INDEX)],
                    )
                    idx_physical_block = pl.read(
                        idx_block_table,
                        [request, pl.cast(cache_block, pl.INDEX)],
                    )
                    csa_cmp_slot = pl.cast(
                        csa_physical_block * CSA_CMP_STORAGE_BLOCK_SIZE + cache_offset,
                        pl.INT64,
                    )
                    csa_idx_slot = pl.cast(
                        idx_physical_block * CSA_CMP_STORAGE_BLOCK_SIZE + cache_offset,
                        pl.INT64,
                    )

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


# Standalone L2 wrapper for validating the metadata portion independently of
# target compute.  Fused decode continues to call the inline form above.
build_group_decode_metadata_l2 = pl.jit(auto_scope=False)(
    build_group_decode_metadata._func
)


@pl.jit(auto_scope=False)
def l2_decode_prepare_probe(
    group_state_slot_ids: pl.Tensor[[B], pl.INT32],
    group_state_generations: pl.Tensor[[B], pl.INT32],
    state_tokens: pl.Tensor[[STATE_CAPACITY, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[STATE_CAPACITY, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.InOut[pl.Tensor[[LOCAL_T], pl.INT64]],
    position_ids_local: pl.InOut[pl.Tensor[[LOCAL_T], pl.INT32]],
    position_ids: pl.InOut[pl.Tensor[[T], pl.INT32]],
    csa_kv_seq_lens: pl.InOut[pl.Tensor[[LOCAL_B], pl.INT32]],
    hca_kv_seq_lens: pl.InOut[pl.Tensor[[LOCAL_B], pl.INT32]],
    logit_row_indices: pl.InOut[pl.Tensor[[LOCAL_T], pl.INT32]],
    sampled_row_offsets: pl.InOut[pl.Tensor[[LOCAL_B], pl.INT32]],
    ori_block_table: pl.Tensor[[B, ORI_TABLE_BLOCKS_DYN], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[B, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[B, CSA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    idx_block_table: pl.Tensor[[B, IDX_TABLE_BLOCKS_DYN], pl.INT32],
    hca_state_block_table: pl.Tensor[
        [B, HCA_GROUP_STATE_BLOCKS_DYN], pl.INT32
    ],
    csa_state_block_table: pl.Tensor[
        [B, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32
    ],
    csa_inner_state_block_table: pl.Tensor[
        [B, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32
    ],
    group_hca_state_block_table: pl.InOut[
        pl.Tensor[[B, HCA_STATE_TABLE_BLOCKS], pl.INT32]
    ],
    group_csa_state_block_table: pl.InOut[
        pl.Tensor[[B, CSA_STATE_TABLE_BLOCKS], pl.INT32]
    ],
    group_csa_inner_state_block_table: pl.InOut[
        pl.Tensor[[B, CSA_STATE_TABLE_BLOCKS], pl.INT32]
    ],
    swa_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    swa_indices: pl.InOut[pl.Tensor[[LOCAL_T, WIN], pl.INT32]],
    swa_lens: pl.InOut[pl.Tensor[[LOCAL_T], pl.INT32]],
    hca_ori_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    hca_swa_indices: pl.InOut[pl.Tensor[[LOCAL_T, WIN], pl.INT32]],
    hca_swa_lens: pl.InOut[pl.Tensor[[LOCAL_T], pl.INT32]],
    hca_cmp_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    hca_state_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    csa_ori_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    csa_swa_indices: pl.InOut[pl.Tensor[[LOCAL_T, WIN], pl.INT32]],
    csa_swa_lens: pl.InOut[pl.Tensor[[LOCAL_T], pl.INT32]],
    csa_cmp_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    csa_idx_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    csa_state_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    csa_inner_state_slot_mapping: pl.InOut[pl.Tensor[[T], pl.INT64]],
    swa_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    swa_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio4_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio4_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio128_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    ratio128_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM], pl.BF16],
    swa_cos: pl.InOut[pl.Tensor[[LOCAL_T, ROPE_DIM], pl.BF16]],
    swa_sin: pl.InOut[pl.Tensor[[LOCAL_T, ROPE_DIM], pl.BF16]],
    compressed_cos: pl.InOut[pl.Tensor[[LOCAL_T, ROPE_DIM], pl.BF16]],
    compressed_sin: pl.InOut[pl.Tensor[[LOCAL_T, ROPE_DIM], pl.BF16]],
    csa_cmp_cos: pl.InOut[pl.Tensor[[T, ROPE_DIM], pl.BF16]],
    csa_cmp_sin: pl.InOut[pl.Tensor[[T, ROPE_DIM], pl.BF16]],
    hca_cmp_cos: pl.InOut[pl.Tensor[[B, HALF_ROPE], pl.FP32]],
    hca_cmp_sin: pl.InOut[pl.Tensor[[B, HALF_ROPE], pl.FP32]],
    drafter_cos_candidates: pl.InOut[
        pl.Tensor[[DRAFTER_B_DYN, DRAFTER_CANDIDATE_ROWS, ROPE_DIM], pl.BF16]
    ],
    drafter_sin_candidates: pl.InOut[
        pl.Tensor[[DRAFTER_B_DYN, DRAFTER_CANDIDATE_ROWS, ROPE_DIM], pl.BF16]
    ],
):
    ori_block_table.bind_dynamic(1, ORI_TABLE_BLOCKS_DYN)
    hca_cmp_block_table.bind_dynamic(1, HCA_CMP_TABLE_BLOCKS_DYN)
    csa_cmp_block_table.bind_dynamic(1, CSA_CMP_TABLE_BLOCKS_DYN)
    idx_block_table.bind_dynamic(1, IDX_TABLE_BLOCKS_DYN)
    hca_state_block_table.bind_dynamic(1, HCA_GROUP_STATE_BLOCKS_DYN)
    csa_state_block_table.bind_dynamic(1, CSA_GROUP_STATE_BLOCKS_DYN)
    csa_inner_state_block_table.bind_dynamic(1, CSA_GROUP_STATE_BLOCKS_DYN)
    swa_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    swa_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    ratio4_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    ratio4_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    ratio128_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    ratio128_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    drafter_cos_candidates.bind_dynamic(0, DRAFTER_B_DYN)
    drafter_sin_candidates.bind_dynamic(0, DRAFTER_B_DYN)
    with pl.scope():
        local_active_widths = pl.create_tensor([LOCAL_B], dtype=pl.INT32)
        group_active_widths = pl.create_tensor([B], dtype=pl.INT32)
        prepare_target_group_from_device_state(
            group_state_slot_ids,
            group_state_generations,
            state_tokens,
            state_meta,
            input_ids,
            position_ids_local,
            position_ids,
            csa_kv_seq_lens,
            hca_kv_seq_lens,
            logit_row_indices,
            sampled_row_offsets,
            local_active_widths,
            group_active_widths,
            0,
        )
        build_group_decode_metadata(
            position_ids,
            group_active_widths,
            ori_block_table,
            hca_cmp_block_table,
            csa_cmp_block_table,
            idx_block_table,
            hca_state_block_table,
            csa_state_block_table,
            csa_inner_state_block_table,
            group_hca_state_block_table,
            group_csa_state_block_table,
            group_csa_inner_state_block_table,
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
            0,
        )
        gather_group_decode_rope_rows(
            swa_cos_table,
            swa_sin_table,
            ratio4_cos_table,
            ratio4_sin_table,
            ratio128_cos_table,
            ratio128_sin_table,
            position_ids,
            group_active_widths,
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
            0,
        )
    return input_ids, position_ids, swa_slot_mapping


@pl.jit.host
def l3_build_group_decode_metadata(
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    active_widths: pl.Tensor[[N_RANKS, B], pl.INT32],
    ori_block_table: pl.Tensor[[N_RANKS, B, ORI_TABLE_BLOCKS_DYN], pl.INT32],
    hca_cmp_block_table: pl.Tensor[
        [N_RANKS, B, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32
    ],
    csa_cmp_block_table: pl.Tensor[
        [N_RANKS, B, CSA_CMP_TABLE_BLOCKS_DYN], pl.INT32
    ],
    idx_block_table: pl.Tensor[[N_RANKS, B, IDX_TABLE_BLOCKS_DYN], pl.INT32],
    hca_state_block_table: pl.Tensor[
        [N_RANKS, B, HCA_GROUP_STATE_BLOCKS_DYN], pl.INT32
    ],
    csa_state_block_table: pl.Tensor[
        [N_RANKS, B, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32
    ],
    csa_inner_state_block_table: pl.Tensor[
        [N_RANKS, B, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32
    ],
    group_hca_state_block_table: pl.InOut[
        pl.Tensor[[N_RANKS, B, HCA_STATE_TABLE_BLOCKS], pl.INT32]
    ],
    group_csa_state_block_table: pl.InOut[
        pl.Tensor[[N_RANKS, B, CSA_STATE_TABLE_BLOCKS], pl.INT32]
    ],
    group_csa_inner_state_block_table: pl.InOut[
        pl.Tensor[[N_RANKS, B, CSA_STATE_TABLE_BLOCKS], pl.INT32]
    ],
    swa_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    swa_indices: pl.InOut[pl.Tensor[[N_RANKS, LOCAL_T, WIN], pl.INT32]],
    swa_lens: pl.InOut[pl.Tensor[[N_RANKS, LOCAL_T], pl.INT32]],
    hca_ori_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    hca_swa_indices: pl.InOut[pl.Tensor[[N_RANKS, LOCAL_T, WIN], pl.INT32]],
    hca_swa_lens: pl.InOut[pl.Tensor[[N_RANKS, LOCAL_T], pl.INT32]],
    hca_cmp_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    hca_state_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    csa_ori_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    csa_swa_indices: pl.InOut[pl.Tensor[[N_RANKS, LOCAL_T, WIN], pl.INT32]],
    csa_swa_lens: pl.InOut[pl.Tensor[[N_RANKS, LOCAL_T], pl.INT32]],
    csa_cmp_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    csa_idx_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    csa_state_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    csa_inner_state_slot_mapping: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
):
    ori_block_table.bind_dynamic(2, ORI_TABLE_BLOCKS_DYN)
    hca_cmp_block_table.bind_dynamic(2, HCA_CMP_TABLE_BLOCKS_DYN)
    csa_cmp_block_table.bind_dynamic(2, CSA_CMP_TABLE_BLOCKS_DYN)
    idx_block_table.bind_dynamic(2, IDX_TABLE_BLOCKS_DYN)
    hca_state_block_table.bind_dynamic(2, HCA_GROUP_STATE_BLOCKS_DYN)
    csa_state_block_table.bind_dynamic(2, CSA_GROUP_STATE_BLOCKS_DYN)
    csa_inner_state_block_table.bind_dynamic(2, CSA_GROUP_STATE_BLOCKS_DYN)
    for rank in pl.range(pld.world_size()):
        build_group_decode_metadata_l2(
            position_ids[rank],
            active_widths[rank],
            ori_block_table[rank],
            hca_cmp_block_table[rank],
            csa_cmp_block_table[rank],
            idx_block_table[rank],
            hca_state_block_table[rank],
            csa_state_block_table[rank],
            csa_inner_state_block_table[rank],
            group_hca_state_block_table[rank],
            group_csa_state_block_table[rank],
            group_csa_inner_state_block_table[rank],
            swa_slot_mapping[rank],
            swa_indices[rank],
            swa_lens[rank],
            hca_ori_slot_mapping[rank],
            hca_swa_indices[rank],
            hca_swa_lens[rank],
            hca_cmp_slot_mapping[rank],
            hca_state_slot_mapping[rank],
            csa_ori_slot_mapping[rank],
            csa_swa_indices[rank],
            csa_swa_lens[rank],
            csa_cmp_slot_mapping[rank],
            csa_idx_slot_mapping[rank],
            csa_state_slot_mapping[rank],
            csa_inner_state_slot_mapping[rank],
            rank % TP,
            device=rank,
        )


def build_metadata_tensor_specs():
    """Build a small but active TP4/EP16 metadata-lowering fixture."""
    import torch
    from golden import TensorSpec

    position_ids = torch.arange(S, dtype=torch.int32).repeat(N_RANKS, B)
    position_ids[:, :S] = torch.arange(64, 64 + S, dtype=torch.int32)
    active_widths = torch.zeros((N_RANKS, B), dtype=torch.int32)
    active_widths[:, 0] = S

    def spec(name, shape, dtype, init_value=0):
        return TensorSpec(name, list(shape), dtype, init_value=init_value)

    i32 = torch.int32
    i64 = torch.int64
    table4 = (N_RANKS, B, 4)
    hca_state = (N_RANKS, B, HCA_HISTORY_PAGES + 2)
    csa_state = (N_RANKS, B, CSA_STATE_TABLE_BLOCKS)
    return [
        spec("position_ids", position_ids.shape, i32, position_ids),
        spec("active_widths", active_widths.shape, i32, active_widths),
        spec("ori_block_table", table4, i32, 1),
        spec("hca_cmp_block_table", table4, i32, 2),
        spec("csa_cmp_block_table", table4, i32, 3),
        spec("idx_block_table", table4, i32, 4),
        spec("hca_state_block_table", hca_state, i32, 5),
        spec("csa_state_block_table", csa_state, i32, 6),
        spec("csa_inner_state_block_table", csa_state, i32, 7),
        spec(
            "group_hca_state_block_table",
            (N_RANKS, B, HCA_STATE_TABLE_BLOCKS),
            i32,
            -1,
        ),
        spec(
            "group_csa_state_block_table",
            (N_RANKS, B, CSA_STATE_TABLE_BLOCKS),
            i32,
            -1,
        ),
        spec(
            "group_csa_inner_state_block_table",
            (N_RANKS, B, CSA_STATE_TABLE_BLOCKS),
            i32,
            -1,
        ),
        spec("swa_slot_mapping", (N_RANKS, T), i64, -1),
        spec("swa_indices", (N_RANKS, LOCAL_T, WIN), i32, -1),
        spec("swa_lens", (N_RANKS, LOCAL_T), i32),
        spec("hca_ori_slot_mapping", (N_RANKS, T), i64, -1),
        spec("hca_swa_indices", (N_RANKS, LOCAL_T, WIN), i32, -1),
        spec("hca_swa_lens", (N_RANKS, LOCAL_T), i32),
        spec("hca_cmp_slot_mapping", (N_RANKS, T), i64, -1),
        spec("hca_state_slot_mapping", (N_RANKS, T), i64, -1),
        spec("csa_ori_slot_mapping", (N_RANKS, T), i64, -1),
        spec("csa_swa_indices", (N_RANKS, LOCAL_T, WIN), i32, -1),
        spec("csa_swa_lens", (N_RANKS, LOCAL_T), i32),
        spec("csa_cmp_slot_mapping", (N_RANKS, T), i64, -1),
        spec("csa_idx_slot_mapping", (N_RANKS, T), i64, -1),
        spec("csa_state_slot_mapping", (N_RANKS, T), i64, -1),
        spec("csa_inner_state_slot_mapping", (N_RANKS, T), i64, -1),
    ]


def build_rope_tensor_specs():
    """Build an active-plus-padding fixture for all decode RoPE outputs."""
    import torch
    from golden import TensorSpec

    rope_rows = 1024
    draft_batch = 4
    position_ids = torch.arange(S, dtype=torch.int32).repeat(N_RANKS, B)
    position_ids[:, :S] = torch.arange(64, 64 + S, dtype=torch.int32)
    active_widths = torch.zeros((N_RANKS, B), dtype=torch.int32)
    active_widths[:, 0] = S

    def spec(name, shape, dtype, init_value=0):
        return TensorSpec(name, list(shape), dtype, init_value=init_value)

    table_shape = (N_RANKS, rope_rows, ROPE_DIM)
    local_rope_shape = (N_RANKS, LOCAL_T, ROPE_DIM)
    group_rope_shape = (N_RANKS, T, ROPE_DIM)
    candidate_shape = (
        N_RANKS,
        draft_batch,
        DRAFTER_CANDIDATE_ROWS,
        ROPE_DIM,
    )
    return [
        spec("swa_cos_table", table_shape, torch.bfloat16, 1.0),
        spec("swa_sin_table", table_shape, torch.bfloat16, 2.0),
        spec("ratio4_cos_table", table_shape, torch.bfloat16, 3.0),
        spec("ratio4_sin_table", table_shape, torch.bfloat16, 4.0),
        spec("ratio128_cos_table", table_shape, torch.bfloat16, 5.0),
        spec("ratio128_sin_table", table_shape, torch.bfloat16, 6.0),
        spec("position_ids", position_ids.shape, torch.int32, position_ids),
        spec("active_widths", active_widths.shape, torch.int32, active_widths),
        spec("swa_cos", local_rope_shape, torch.bfloat16),
        spec("swa_sin", local_rope_shape, torch.bfloat16),
        spec("compressed_cos", local_rope_shape, torch.bfloat16),
        spec("compressed_sin", local_rope_shape, torch.bfloat16),
        spec("csa_cmp_cos", group_rope_shape, torch.bfloat16),
        spec("csa_cmp_sin", group_rope_shape, torch.bfloat16),
        spec("hca_cmp_cos", (N_RANKS, B, HALF_ROPE), torch.float32),
        spec("hca_cmp_sin", (N_RANKS, B, HALF_ROPE), torch.float32),
        spec("drafter_cos_candidates", candidate_shape, torch.bfloat16),
        spec("drafter_sin_candidates", candidate_shape, torch.bfloat16),
    ]


def build_decode_prepare_probe_specs():
    """Reuse the distributed fixtures as one rank-local fused-L2 probe."""
    from golden import TensorSpec

    candidates = {}
    for source in (
        build_group_prepare_tensor_specs(),
        build_metadata_tensor_specs(),
        build_rope_tensor_specs(),
    ):
        for item in source:
            value = item.create_tensor()[0].contiguous()
            candidates[item.name] = TensorSpec(
                item.name,
                list(value.shape),
                item.dtype,
                init_value=value,
            )
    return [
        candidates[name]
        for name in l2_decode_prepare_probe._param_names()
    ]


def main():
    """Compile or execute one prepare stage without loading model weights."""
    import argparse

    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="Validate DSpark decode device preparation")
    parser.add_argument(
        "--stage",
        choices=("metadata", "rope", "prepare-all"),
        default="metadata",
    )
    parser.add_argument(
        "-p", "--platform", choices=("a2a3", "a2a3sim"), default="a2a3"
    )
    parser.add_argument(
        "-d", "--device", default=",".join(str(rank) for rank in range(N_RANKS))
    )
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != N_RANKS:
        parser.error(f"expected exactly {N_RANKS} device ids, got {device_ids}")
    distributed = args.stage != "prepare-all"
    if args.stage == "metadata":
        fn = l3_build_group_decode_metadata
        specs = build_metadata_tensor_specs()
    elif args.stage == "rope":
        fn = l3_gather_group_decode_rope_rows
        specs = build_rope_tensor_specs()
    else:
        fn = l2_decode_prepare_probe
        specs = build_decode_prepare_probe_specs()
    run_config = dict(platform=args.platform, ring_heap=512 * 1024 * 1024)
    if distributed:
        run_config["distributed_config"] = DistributedConfig(
            device_ids=device_ids, num_sub_workers=0
        )
    else:
        run_config["device_id"] = device_ids[0]
    result = run(
        fn=fn,
        specs=specs,
        compile_only=args.compile_only,
        config=run_config,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
