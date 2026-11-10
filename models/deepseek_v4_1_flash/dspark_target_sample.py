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

from config import DECODE_ROWS_PER_REQUEST, FLASH, MAX_BATCH_PER_DP, TP_SIZE
from hc import hc_head
from lm_head import LM_HEAD_M_TILE, SAMPLED_IDS_PAD, VOCAB_PER_TP, greedy_sample, lm_head_project, lm_head_sample
from rmsnorm import rms_norm_tiled_compensated


D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
ROWS = MAX_BATCH_PER_DP * DECODE_ROWS_PER_REQUEST
TARGET_T_DYN = pl.dynamic("V41_DSPARK_TARGET_T_DYN")

if TP_SIZE != 1:
    raise ValueError("DSpark target row sampling currently requires TP1")


@pl.jit.inline(auto_scope=False)
def select_target_rows(
    normalized: pl.Tensor,
    active_widths: pl.Tensor[[MAX_BATCH_PER_DP], pl.INT32],
    selected_hidden: pl.Tensor[[LM_HEAD_M_TILE, D], pl.BF16],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Select active verification rows and zero the remaining head rows."""
    tokens = pl.tensor.dim(normalized, 0)
    for worker in pl.spmd(64, name_hint="dspark_target_select_rows", deps=[ready]):
        for row in pl.range(worker, LM_HEAD_M_TILE, 64):
            selected_row = pl.full([1, D], dtype=pl.BF16, value=0.0)
            if row < ROWS and row < tokens:
                request = row // DECODE_ROWS_PER_REQUEST
                offset = row % DECODE_ROWS_PER_REQUEST
                width = pl.read(active_widths, [request])
                if offset < width:
                    selected_row = normalized[row : row + 1, :]
            selected_hidden[row : row + 1, :] = selected_row
    return selected_hidden


@pl.jit.inline(auto_scope=False)
def sample_normalized_target(
    normalized: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    active_widths: pl.Tensor[[MAX_BATCH_PER_DP], pl.INT32],
    sampled_ids: pl.Tensor[[ROWS, SAMPLED_IDS_PAD], pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Sample dense verification rows directly or select a masked batch."""
    tokens = pl.tensor.dim(normalized, 0)
    full_batch = pl.cast(1, pl.INT32)
    if tokens < ROWS:
        full_batch = pl.cast(0, pl.INT32)
    for request in pl.range(MAX_BATCH_PER_DP):
        if pl.read(active_widths, [request]) < DECODE_ROWS_PER_REQUEST:
            full_batch = pl.cast(0, pl.INT32)
    if full_batch > 0:
        logits = pl.create_tensor([ROWS, VOCAB_PER_TP], dtype=pl.FP32)
        projected = lm_head_project(normalized, lm_head_weight, logits, LM_HEAD_M_TILE)
        greedy_sample(logits, sampled_ids, projected)
    else:
        selected_hidden = pl.create_tensor([LM_HEAD_M_TILE, D], dtype=pl.BF16)
        select_target_rows(normalized, active_widths, selected_hidden, ready)
        lm_head_sample(selected_hidden, lm_head_weight, sampled_ids)
    return sampled_ids


@pl.jit
def sample_target_rank(
    x_hc: pl.Tensor[[TARGET_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TARGET_T_DYN, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    active_widths: pl.Tensor[[MAX_BATCH_PER_DP], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    x_hc.bind_dynamic(0, TARGET_T_DYN)
    pre_mix.bind_dynamic(0, TARGET_T_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_head(x_hc, pre_mix, collapsed, worker_limit=64)
    normalized = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd((tokens + 7) // 8, name_hint="dspark_target_final_norm") as ready:
        block = pl.tile.get_block_idx()
        rms_norm_tiled_compensated(collapsed, final_norm_weight, normalized, block, tokens)
    sample_normalized_target(normalized, lm_head_weight, active_widths, sampled_ids, ready)
    return sampled_ids
