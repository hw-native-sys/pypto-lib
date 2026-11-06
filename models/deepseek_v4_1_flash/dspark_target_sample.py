# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sample the V4.1 DSpark target rows in one L3 call."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import DECODE_ROWS_PER_REQUEST, FLASH, MAX_BATCH_PER_DP, TP_SIZE
from hc import hc_head
from lm_head import GROUP_LOGIT_ROWS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD, VOCAB, VOCAB_PER_TP, lm_head_with_sampling
from rmsnorm import rms_norm_tiled_compensated


D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
ROWS = MAX_BATCH_PER_DP * DECODE_ROWS_PER_REQUEST
CHUNKS = (ROWS + MAX_LOGIT_ROWS - 1) // MAX_LOGIT_ROWS
TARGET_T_DYN = pl.dynamic("V41_DSPARK_TARGET_T_DYN")

assert CHUNKS == 5

if TP_SIZE != 1:
    raise ValueError("DSpark target row sampling currently requires TP1")


@pl.jit
def sample_target_rank(
    x_hc: pl.Tensor[[TARGET_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TARGET_T_DYN, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    active_widths: pl.Tensor[[MAX_BATCH_PER_DP], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, TARGET_T_DYN)
    pre_mix.bind_dynamic(0, TARGET_T_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_head(x_hc, pre_mix, collapsed)
    normalized = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd((tokens + 7) // 8, name_hint="dspark_target_final_norm") as ready:
        block = pl.tile.get_block_idx()
        rms_norm_tiled_compensated(collapsed, final_norm_weight, normalized, block, tokens)
    row_indices = pl.create_tensor([MAX_LOGIT_ROWS], dtype=pl.INT32)
    logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    chunk_ids = pl.create_tensor([MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.INT32)
    for chunk in pl.range(CHUNKS):
        # A subsequent writer must wait for the previous chunk's scratch and window readers.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_target_select_rows", deps=[ready]):
            for row in pl.range(MAX_LOGIT_ROWS):
                global_row = chunk * MAX_LOGIT_ROWS + row
                selected = pl.cast(-1, pl.INT32)
                if global_row < ROWS:
                    request = global_row // DECODE_ROWS_PER_REQUEST
                    offset = global_row % DECODE_ROWS_PER_REQUEST
                    width = pl.read(active_widths, [request])
                    if offset < width and global_row < tokens:
                        selected = pl.cast(global_row, pl.INT32)
                pl.write(row_indices, [row], selected)
        lm_head_with_sampling(
            normalized, lm_head_weight, row_indices, logits, chunk_ids,
            hidden_window, hidden_done, logits_window, logits_done,
            rank, pl.cast(0, pl.INT32), pl.cast(chunk + 1, pl.INT32), pl.cast(CHUNKS, pl.INT32),
        )
        # The last chunk has eight real rows. Keep its padded rows in scratch.
        with pl.spmd(pl.min(MAX_LOGIT_ROWS, ROWS - chunk * MAX_LOGIT_ROWS),
                     name_hint="dspark_target_publish_ids") as published:
            row = pl.tile.get_block_idx()
            global_row = chunk * MAX_LOGIT_ROWS + row
            sampled_ids[global_row : global_row + 1, :] = chunk_ids[row : row + 1, :]
        ready = published
    return sampled_ids
