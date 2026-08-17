# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-side DeepSeek-V4 decode preamble: paged-cache metadata lowering and input packing."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FLASH as M,
    IDX_CACHE_MAX_BLOCKS,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_TABLE_MAX_BLOCKS,
)


# Dynamic shape variables.
VOCAB_DYN = pl.dynamic("PACK_X_HC_VOCAB_DYN")

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
WIN = M.sliding_window

# paged block-table extents, one per cache
ORI_TABLE_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
HCA_STATE_MAX_BLOCKS = 2048
CSA_STATE_MAX_BLOCKS = 4096
CSA_INNER_STATE_MAX_BLOCKS = 4096

# block_counts columns
GROUP_ORI = 0
GROUP_HCA_CMP = 1
GROUP_CSA_CMP = 2
GROUP_IDX = 3
GROUP_HCA_STATE = 4
GROUP_CSA_STATE = 5
GROUP_CSA_INNER_STATE = 6
N_CACHE_GROUPS = 7
HCA_COMPRESS_RATIO = 128
CSA_COMPRESS_RATIO = 4
HCA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // HCA_COMPRESS_RATIO
CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // CSA_COMPRESS_RATIO

# tiling
X_HC_HIDDEN_TILE = 512
MTP_HIDDEN_TILE = 1024
SPMD_BLOCKS = 48


@pl.jit.inline
def build_swa_metadata(
    # Inputs: bare Tensor parameters have PyPTO's default In direction.
    position_ids: pl.Tensor[[T], pl.INT32],
    ori_block_table: pl.Tensor[[B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
    # Outputs.
    swa_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    swa_indices: pl.Out[pl.Tensor[[T, WIN], pl.INT32]],
    swa_lens: pl.Out[pl.Tensor[[T], pl.INT32]],
):
    """Lower paged write slots and visible SWA rows for each decode token."""
    for token in pl.spmd(T, name_hint="decode_build_swa_metadata"):
        request = token // S
        position = pl.read(position_ids, [token])
        valid_len = pl.min(position + 1, WIN)
        start = position - valid_len + 1
        index_row = pl.create_tensor([1, WIN], dtype=pl.INT32)
        index_row[:, :] = pl.full([1, WIN], dtype=pl.INT32, value=-1)
        for offset in pl.range(WIN):
            if offset < valid_len:
                visible_position = start + offset
                visible_block = visible_position // BLOCK_SIZE
                visible_offset = visible_position % BLOCK_SIZE
                visible_physical_block = pl.read(
                    ori_block_table,
                    [request, pl.cast(visible_block, pl.INDEX)],
                )
                pl.write(
                    index_row,
                    [0, offset],
                    pl.cast(
                        visible_physical_block * BLOCK_SIZE + visible_offset,
                        pl.INT32,
                    ),
                )
        swa_indices[token : token + 1, :] = index_row

    for metadata_core in pl.spmd(1, name_hint="decode_build_swa_scalar_metadata"):
        for token in pl.range(metadata_core, T):
            request = token // S
            position = pl.read(position_ids, [token])
            logical_block = position // BLOCK_SIZE
            block_offset = position % BLOCK_SIZE
            physical_block = pl.read(
                ori_block_table,
                [request, pl.cast(logical_block, pl.INDEX)],
            )
            pl.write(
                swa_slot_mapping,
                [token],
                pl.cast(physical_block * BLOCK_SIZE + block_offset, pl.INT64),
            )
            pl.write(
                swa_lens,
                [token],
                pl.cast(pl.min(position + 1, WIN), pl.INT32),
            )


@pl.jit.inline
def build_decode_metadata(
    # Inputs: bare Tensor parameters have PyPTO's default In direction.
    position_ids: pl.Tensor[[T], pl.INT32],
    ori_block_table: pl.Tensor[[B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[B, IDX_MAX_BLOCKS], pl.INT32],
    hca_state_block_table: pl.Tensor[[B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_state_block_table: pl.Tensor[[B, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_state_block_table: pl.Tensor[
        [B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    block_counts: pl.Tensor[[B, N_CACHE_GROUPS], pl.INT32],
    # Outputs.
    ori_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    swa_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    swa_indices: pl.Out[pl.Tensor[[T, WIN], pl.INT32]],
    swa_lens: pl.Out[pl.Tensor[[T], pl.INT32]],
    hca_cmp_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    hca_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_cmp_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_idx_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_inner_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
):
    """Build every position-dependent metadata tensor consumed by decode_fwd."""
    build_swa_metadata(
        position_ids,
        ori_block_table,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
    )
    for metadata_core in pl.spmd(1, name_hint="decode_build_cache_metadata"):
        for token in pl.range(metadata_core, T):
            request = token // S
            position = pl.read(position_ids, [token])
            logical_block = position // BLOCK_SIZE
            block_offset = position % BLOCK_SIZE
            ori_physical_block = pl.read(
                ori_block_table,
                [request, pl.cast(logical_block, pl.INDEX)],
            )
            pl.write(
                ori_slot_mapping,
                [token],
                pl.cast(ori_physical_block * BLOCK_SIZE + block_offset, pl.INT64),
            )

            hca_cmp_slot = pl.cast(-1, pl.INT64)
            if (position + 1) % HCA_COMPRESS_RATIO == 0:
                source_block = position // BLOCK_SIZE
                storage_offset = position % BLOCK_SIZE // HCA_COMPRESS_RATIO
                count = pl.read(block_counts, [request, GROUP_HCA_CMP])
                physical_block = pl.read(
                    hca_cmp_block_table,
                    [request, pl.cast(source_block % count, pl.INDEX)],
                )
                hca_cmp_slot = pl.cast(
                    physical_block * HCA_CMP_STORAGE_BLOCK_SIZE + storage_offset,
                    pl.INT64,
                )
            pl.write(hca_cmp_slot_mapping, [token], hca_cmp_slot)

            csa_cmp_slot = pl.cast(-1, pl.INT64)
            csa_idx_slot = pl.cast(-1, pl.INT64)
            if (position + 1) % CSA_COMPRESS_RATIO == 0:
                source_block = position // BLOCK_SIZE
                storage_offset = position % BLOCK_SIZE // CSA_COMPRESS_RATIO
                cmp_count = pl.read(block_counts, [request, GROUP_CSA_CMP])
                cmp_physical_block = pl.read(
                    csa_cmp_block_table,
                    [request, pl.cast(source_block % cmp_count, pl.INDEX)],
                )
                csa_cmp_slot = pl.cast(
                    cmp_physical_block * CSA_CMP_STORAGE_BLOCK_SIZE + storage_offset,
                    pl.INT64,
                )
                idx_count = pl.read(block_counts, [request, GROUP_IDX])
                idx_physical_block = pl.read(
                    idx_block_table,
                    [request, pl.cast(source_block % idx_count, pl.INDEX)],
                )
                csa_idx_slot = pl.cast(
                    idx_physical_block * CSA_CMP_STORAGE_BLOCK_SIZE + storage_offset,
                    pl.INT64,
                )
            pl.write(csa_cmp_slot_mapping, [token], csa_cmp_slot)
            pl.write(csa_idx_slot_mapping, [token], csa_idx_slot)

            hca_state_logical = position // C128_COMPRESSOR_BLOCK_SIZE
            hca_state_count = pl.read(block_counts, [request, GROUP_HCA_STATE])
            hca_state_physical_block = pl.read(
                hca_state_block_table,
                [
                    request,
                    pl.cast(hca_state_logical % hca_state_count, pl.INDEX),
                ],
            )
            pl.write(
                hca_state_slot_mapping,
                [token],
                pl.cast(
                    hca_state_physical_block * C128_COMPRESSOR_BLOCK_SIZE
                    + position % C128_COMPRESSOR_BLOCK_SIZE,
                    pl.INT64,
                ),
            )

            csa_state_logical = position // C4A_COMPRESSOR_BLOCK_SIZE
            csa_state_count = pl.read(block_counts, [request, GROUP_CSA_STATE])
            csa_state_physical_block = pl.read(
                csa_state_block_table,
                [
                    request,
                    pl.cast(csa_state_logical % csa_state_count, pl.INDEX),
                ],
            )
            pl.write(
                csa_state_slot_mapping,
                [token],
                pl.cast(
                    csa_state_physical_block * C4A_COMPRESSOR_BLOCK_SIZE
                    + position % C4A_COMPRESSOR_BLOCK_SIZE,
                    pl.INT64,
                ),
            )

            inner_state_count = pl.read(
                block_counts,
                [request, GROUP_CSA_INNER_STATE],
            )
            inner_state_physical_block = pl.read(
                csa_inner_state_block_table,
                [
                    request,
                    pl.cast(csa_state_logical % inner_state_count, pl.INDEX),
                ],
            )
            pl.write(
                csa_inner_state_slot_mapping,
                [token],
                pl.cast(
                    inner_state_physical_block * C4A_COMPRESSOR_BLOCK_SIZE
                    + position % C4A_COMPRESSOR_BLOCK_SIZE,
                    pl.INT64,
                ),
            )
    return (
        ori_slot_mapping,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
    )


@pl.jit.inline
def pack_x_hc(
    input_ids: pl.Tensor[[T], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    x_hc_flat = pl.reshape(x_hc, [T * HC_MULT, D])
    for block in pl.spmd(SPMD_BLOCKS, name_hint="pack_x_hc"):
        for work_idx in pl.range(block, T * (D // X_HC_HIDDEN_TILE), SPMD_BLOCKS):
            token_idx = work_idx // (D // X_HC_HIDDEN_TILE)
            hidden_offset = (work_idx % (D // X_HC_HIDDEN_TILE)) * X_HC_HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            embed_chunk = embed_weight[token_row : token_row + 1, hidden_offset : hidden_offset + X_HC_HIDDEN_TILE]
            hidden_chunk = pl.cast(embed_chunk, target_type=pl.FP32)
            # Every HC lane of a token starts from the same embedding row.
            for hc_idx in pl.range(HC_MULT):
                x_hc_row = token_idx * HC_MULT + hc_idx
                x_hc_flat[x_hc_row : x_hc_row + 1, hidden_offset : hidden_offset + X_HC_HIDDEN_TILE] = hidden_chunk
    return x_hc


@pl.jit.inline
def pack_mtp_hidden(
    main_pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    tail_pre_hc_pool: pl.Tensor[[B, HC_MULT, D], pl.FP32],
    accepted_counts: pl.Tensor[[B], pl.INT32],
    tail_slot_ids: pl.Tensor[[B], pl.INT32],
    fallback_hidden: pl.Tensor[[S, HC_MULT, D], pl.FP32],
    packed_hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    for block in pl.spmd(SPMD_BLOCKS, name_hint="pack_mtp_hidden"):
        for work_idx in pl.range(block, B * HC_MULT * (D // MTP_HIDDEN_TILE), SPMD_BLOCKS):
            batch_idx = work_idx // (HC_MULT * (D // MTP_HIDDEN_TILE))
            local_idx = work_idx % (HC_MULT * (D // MTP_HIDDEN_TILE))
            hc_idx = local_idx // (D // MTP_HIDDEN_TILE)
            hidden_offset = (local_idx % (D // MTP_HIDDEN_TILE)) * MTP_HIDDEN_TILE
            row0 = batch_idx * S
            row1 = row0 + 1

            slot_raw = pl.read(tail_slot_ids, [batch_idx])
            if slot_raw >= 0:
                accepted_count = pl.read(accepted_counts, [batch_idx])
                last_row = row0 + pl.cast(accepted_count, target_type=pl.INDEX) - 1
                last_hidden = main_pre_hc_hidden[
                    last_row : last_row + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                ]
                slot = pl.cast(slot_raw, target_type=pl.INDEX)
                # Rejected draft replays the previous step's tail out of the pool.
                if accepted_count == 1:
                    pool_hidden = tail_pre_hc_pool[
                        slot : slot + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                    ]
                    packed_hidden[
                        row0 : row0 + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                    ] = pool_hidden
                else:
                    main_hidden = main_pre_hc_hidden[
                        row0 : row0 + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                    ]
                    packed_hidden[
                        row0 : row0 + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                    ] = main_hidden
                packed_hidden[
                    row1 : row1 + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                ] = last_hidden
                tail_pre_hc_pool[
                    slot : slot + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                ] = last_hidden
            else:
                for seq_idx in pl.range(S):
                    fallback_row = fallback_hidden[
                        seq_idx : seq_idx + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                    ]
                    pack_row = row0 + seq_idx
                    packed_hidden[
                        pack_row : pack_row + 1, hc_idx : hc_idx + 1, hidden_offset : hidden_offset + MTP_HIDDEN_TILE
                    ] = fallback_row
    return packed_hidden
