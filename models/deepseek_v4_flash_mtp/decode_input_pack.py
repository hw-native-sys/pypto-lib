# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone device operators for DeepSeek-V4 Flash decode input packing."""

import pypto.language as pl

from config import DECODE_BATCH, DECODE_SEQ, DECODE_TOKENS, FLASH as M


VOCAB_DYN = pl.dynamic("PACK_X_HC_VOCAB_DYN")

D = M.hidden_size
HC_MULT = M.hc_mult

X_HC_HIDDEN_TILE = 512
MTP_HIDDEN_TILE = 1024
SPMD_BLOCKS = 48

assert D % X_HC_HIDDEN_TILE == 0
assert D % MTP_HIDDEN_TILE == 0


@pl.jit.inline
def pack_x_hc(
    input_ids: pl.Tensor[[DECODE_TOKENS], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32]:
    x_hc_flat = pl.reshape(x_hc, [DECODE_TOKENS * HC_MULT, D])
    for block in pl.spmd(SPMD_BLOCKS, name_hint="pack_x_hc"):
        for work_idx in pl.range(
            block,
            DECODE_TOKENS * (D // X_HC_HIDDEN_TILE),
            SPMD_BLOCKS,
        ):
            token_idx = work_idx // (D // X_HC_HIDDEN_TILE)
            hidden_offset = (work_idx % (D // X_HC_HIDDEN_TILE)) * X_HC_HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            hidden_chunk = pl.cast(
                embed_weight[
                    token_row : token_row + 1,
                    hidden_offset : hidden_offset + X_HC_HIDDEN_TILE,
                ],
                target_type=pl.FP32,
            )
            for hc_idx in pl.range(HC_MULT):
                x_hc_row = token_idx * HC_MULT + hc_idx
                x_hc_flat[
                    x_hc_row : x_hc_row + 1,
                    hidden_offset : hidden_offset + X_HC_HIDDEN_TILE,
                ] = hidden_chunk
    return x_hc


@pl.jit.inline
def pack_mtp_hidden(
    main_pre_hc_hidden: pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32],
    tail_pre_hc_pool: pl.Tensor[[DECODE_BATCH, HC_MULT, D], pl.FP32],
    accepted_counts: pl.Tensor[[DECODE_BATCH], pl.INT32],
    tail_slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    fallback_hidden: pl.Tensor[[DECODE_SEQ, HC_MULT, D], pl.FP32],
    packed_hidden: pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32],
) -> pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32]:
    for block in pl.spmd(SPMD_BLOCKS, name_hint="pack_mtp_hidden"):
        for work_idx in pl.range(
            block,
            DECODE_BATCH * HC_MULT * (D // MTP_HIDDEN_TILE),
            SPMD_BLOCKS,
        ):
            batch_idx = work_idx // (HC_MULT * (D // MTP_HIDDEN_TILE))
            local_idx = work_idx % (HC_MULT * (D // MTP_HIDDEN_TILE))
            hc_idx = local_idx // (D // MTP_HIDDEN_TILE)
            hidden_offset = (local_idx % (D // MTP_HIDDEN_TILE)) * MTP_HIDDEN_TILE
            row0 = batch_idx * DECODE_SEQ
            row1 = row0 + 1

            slot_raw = pl.read(tail_slot_ids, [batch_idx])
            if slot_raw >= 0:
                accepted_count = pl.read(accepted_counts, [batch_idx])
                last_row = row0 + pl.cast(accepted_count, target_type=pl.INDEX) - 1
                last_hidden = main_pre_hc_hidden[
                    last_row : last_row + 1,
                    hc_idx : hc_idx + 1,
                    hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                ]
                slot = pl.cast(slot_raw, target_type=pl.INDEX)
                if accepted_count == 1:
                    packed_hidden[
                        row0 : row0 + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ] = tail_pre_hc_pool[
                        slot : slot + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ]
                else:
                    packed_hidden[
                        row0 : row0 + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ] = main_pre_hc_hidden[
                        row0 : row0 + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ]
                packed_hidden[
                    row1 : row1 + 1,
                    hc_idx : hc_idx + 1,
                    hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                ] = last_hidden
                tail_pre_hc_pool[
                    slot : slot + 1,
                    hc_idx : hc_idx + 1,
                    hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                ] = last_hidden
            else:
                for seq_idx in pl.range(DECODE_SEQ):
                    packed_hidden[
                        row0 + seq_idx : row0 + seq_idx + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ] = fallback_hidden[
                        seq_idx : seq_idx + 1,
                        hc_idx : hc_idx + 1,
                        hidden_offset : hidden_offset + MTP_HIDDEN_TILE,
                    ]
    return packed_hidden
