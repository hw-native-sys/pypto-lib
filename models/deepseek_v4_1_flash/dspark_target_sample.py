# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sample one 16-row V4.1 DSpark target chunk per completed L3 call."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import DECODE_ROWS_PER_REQUEST, EP_SIZE, FLASH, MAX_BATCH_PER_DP, TP_SIZE
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
def sample_target_chunk_rank(
    x_hc: pl.Tensor[[TARGET_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TARGET_T_DYN, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    active_widths: pl.Tensor[[MAX_BATCH_PER_DP], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    chunk: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, TARGET_T_DYN)
    pre_mix.bind_dynamic(0, TARGET_T_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_head(x_hc, pre_mix, collapsed)
    normalized = pl.create_tensor([tokens, D], dtype=pl.BF16)
    for block in pl.spmd((tokens + 7) // 8, name_hint="dspark_target_final_norm"):
        rms_norm_tiled_compensated(collapsed, final_norm_weight, normalized, block, tokens)
    row_indices = pl.create_tensor([MAX_LOGIT_ROWS], dtype=pl.INT32)
    for _core in pl.spmd(1, name_hint="dspark_target_select_rows"):
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
    logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    lm_head_with_sampling(
        normalized, lm_head_weight, row_indices, logits, sampled_ids,
        hidden_window, hidden_done, logits_window, logits_done,
        rank, pl.cast(0, pl.INT32), pl.cast(1, pl.INT32), pl.cast(1, pl.INT32),
    )
    return sampled_ids


@pl.jit.host
def l3_sample_target_chunk(
    x_hc: pl.Tensor[[EP_SIZE, TARGET_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[EP_SIZE, TARGET_T_DYN, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[EP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    active_widths: pl.Tensor[[EP_SIZE, MAX_BATCH_PER_DP], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    chunk: pl.Scalar[pl.INT32],
):
    """Call with chunk 0 through 4 in order, reusing target state after each call completes."""
    x_hc.bind_dynamic(1, TARGET_T_DYN)
    pre_mix.bind_dynamic(1, TARGET_T_DYN)
    hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    hidden_done_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    logits_done_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        sample_target_chunk_rank(
            x_hc[rank], pre_mix[rank], final_norm_weight[rank], lm_head_weight[rank],
            active_widths[rank], sampled_ids[rank], hidden_window, hidden_done,
            logits_window, logits_done, rank, chunk, device=rank,
        )


@pl.jit
def merge_target_chunks_rank(
    chunk_0: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    chunk_1: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    chunk_2: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    chunk_3: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    chunk_4: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    """Discard the eight padded rows after the five 16-row LM-head calls."""
    for _core in pl.spmd(1, name_hint="dspark_merge_target_chunks"):
        for row in pl.range(ROWS):
            chunk = row // MAX_LOGIT_ROWS
            offset = row % MAX_LOGIT_ROWS
            for lane in pl.range(SAMPLED_IDS_PAD):
                value = pl.cast(0, pl.INT32)
                if chunk == 0:
                    value = pl.read(chunk_0, [offset, lane])
                elif chunk == 1:
                    value = pl.read(chunk_1, [offset, lane])
                elif chunk == 2:
                    value = pl.read(chunk_2, [offset, lane])
                elif chunk == 3:
                    value = pl.read(chunk_3, [offset, lane])
                else:
                    value = pl.read(chunk_4, [offset, lane])
                pl.write(sampled_ids, [row, lane], value)
    return sampled_ids


@pl.jit.host
def l3_merge_target_chunks(
    chunk_0: pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    chunk_1: pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    chunk_2: pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    chunk_3: pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    chunk_4: pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    for rank in pl.range(pld.world_size()):
        merge_target_chunks_rank(
            chunk_0[rank], chunk_1[rank], chunk_2[rank], chunk_3[rank], chunk_4[rank],
            sampled_ids[rank], device=rank,
        )
