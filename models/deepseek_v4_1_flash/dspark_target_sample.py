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

from config import DECODE_SEQ, FLASH, DECODE_BATCH, LM_HEAD_TP_SIZE
from hc import hc_head
from lm_head import (
    SAMPLED_IDS_PAD, VOCAB, VOCAB_PER_TP, GROUP_LOGIT_ROWS, MAX_LOGIT_ROWS,
    lm_head_sample, lm_head, greedy_sample,
)
from rmsnorm import rms_norm


D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
ROWS = DECODE_BATCH * DECODE_SEQ
TARGET_T_DYN = pl.dynamic("DSPARK_TARGET_T_DYN")

@pl.jit.inline(auto_scope=False)
def sample_target(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    active_widths: pl.Tensor[[DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    collapsed: pl.Tensor,
    normalized: pl.Tensor,
):
    tokens = pl.tensor.dim(x_hc, 0)
    hc_head(x_hc, pre_mix, collapsed, worker_limit=64)
    ready = rms_norm(collapsed, final_norm_weight, normalized, tokens, width=D, zero_tail=False)
    if LM_HEAD_TP_SIZE == 1:
        lm_head_sample(normalized, lm_head_weight, active_widths, sampled_ids, ready)
    else:
        for chunk in pl.unroll((ROWS + MAX_LOGIT_ROWS - 1) // MAX_LOGIT_ROWS):
            row_indices = pl.create_tensor([MAX_LOGIT_ROWS], dtype=pl.INT32)
            with pl.spmd(1, name_hint="target_logit_rows", deps=[ready]):
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, MAX_LOGIT_ROWS):
                    source = chunk * MAX_LOGIT_ROWS + row
                    index = pl.cast(-1, pl.INT32)
                    if source < ROWS and source < tokens:
                        request = source // DECODE_SEQ
                        width = pl.read(active_widths, [request])
                        if source % DECODE_SEQ < width:
                            index = pl.cast(source, pl.INT32)
                    pl.write(row_indices, [row], index)
            logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
            logits, projected = lm_head(
                normalized, lm_head_weight, row_indices, logits,
                hidden_window, hidden_done, logits_window, logits_done,
                rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE, rank % LM_HEAD_TP_SIZE,
                chunk + 1, (ROWS + MAX_LOGIT_ROWS - 1) // MAX_LOGIT_ROWS,
            )
            sampled_chunk = pl.create_tensor([MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.INT32)
            greedy_sample(logits, sampled_chunk, projected)
            with pl.spmd(MAX_LOGIT_ROWS, name_hint="target_sample_rows") as ready:
                row = pl.tile.get_block_idx()
                target = chunk * MAX_LOGIT_ROWS + row
                if target < ROWS:
                    sampled_ids[target:target + 1, :] = sampled_chunk[row:row + 1, :]
    return sampled_ids

@pl.jit(auto_scope=False)
def sample_target_rank(
    x_hc: pl.Tensor[[TARGET_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TARGET_T_DYN, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    active_widths: pl.Tensor[[DECODE_BATCH], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, TARGET_T_DYN)
    pre_mix.bind_dynamic(0, TARGET_T_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normalized = pl.create_tensor([tokens, D], dtype=pl.BF16)
    return sample_target(
        x_hc, pre_mix, final_norm_weight, lm_head_weight,
        active_widths, sampled_ids, hidden_window, hidden_done,
        logits_window, logits_done, rank,
        collapsed, normalized,
    )
