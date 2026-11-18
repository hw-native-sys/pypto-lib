# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TP8 prefill target taps and the existing DSpark final head."""

import pypto.language as pl
import pypto.language.distributed as pld

from hc_head import hc_head
from lm_head import (
    D, GRAMMAR_SEGMENT_WORDS, GREEDY_GRID_ROWS, GROUP_LOGIT_ROWS, MAX_LOGIT_ROWS,
    SAMPLED_IDS_PAD, TP_SIZE, VOCAB, VOCAB_PER_TP, greedy_sample, lm_head,
)
from rmsnorm import rms_norm


HC_MULT = 4
HC_DIM = HC_MULT * D
PREFILL_ROWS = pl.dynamic("LOWLAT_PREFILL_HEAD_ROWS")
LOCAL_ROWS = pl.dynamic("LOWLAT_PREFILL_HEAD_LOCAL_ROWS")


@pl.jit(auto_scope=False)
def l2_prefill_tap(
    x_hc: pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32],
    target_hidden: pl.InOut[pl.Tensor[[LOCAL_ROWS, 3 * D], pl.BF16]],
    tap_index: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, PREFILL_ROWS)
    target_hidden.bind_dynamic(0, LOCAL_ROWS)
    local_rows = pl.tensor.dim(target_hidden, 0)
    first_row = pl.cast(rank, pl.INDEX) * local_rows
    column = pl.cast(tap_index, pl.INDEX) * D
    for token in pl.spmd(local_rows, name_hint="prefill_tp_target_tap"):
        row = first_row + token
        if row < group_tokens:
            hc_row = x_hc[row : row + 1, 0:HC_MULT, 0:D]
            mean = pl.mul(pl.col_sum(pl.reshape(hc_row, [HC_MULT, D])), 1.0 / HC_MULT)
            target_hidden[token : token + 1, column : column + D] = pl.cast(mean, pl.BF16, mode="rint")
        else:
            target_hidden[token : token + 1, column : column + D] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    return target_hidden


@pl.jit.host
def l3_prefill_tap(
    x_hc: pl.Tensor[[TP_SIZE, PREFILL_ROWS, HC_MULT, D], pl.FP32],
    target_hidden: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_ROWS, 3 * D], pl.BF16]],
    tap_index: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, PREFILL_ROWS)
    target_hidden.bind_dynamic(1, LOCAL_ROWS)
    for rank in pl.range(pld.world_size()):
        l2_prefill_tap(x_hc[rank], target_hidden[rank], tap_index, group_tokens, rank, device=rank)
    return target_hidden


@pl.jit(auto_scope=False)
def l2_prefill_head(
    x_hc: pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    grammar_mask: pl.Tensor[[MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    x_out: pl.Out[pl.Tensor[[PREFILL_ROWS, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, PREFILL_ROWS)
    x_out.bind_dynamic(0, PREFILL_ROWS)
    rows = pl.tensor.dim(x_hc, 0)
    hidden = pl.create_tensor([rows, D], dtype=pl.BF16)
    hidden = hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, hidden)
    ready = rms_norm(hidden, final_norm_w, x_out)
    lm_head(x_out, lm_head_weight, logit_row_indices, logits,
            hidden_window, hidden_done, logits_window, logits_done,
            pl.const(0, pl.INT32), rank, pl.const(1, pl.INT32), ready)
    greedy_sample(logits, logit_row_indices, grammar_mask, sampled_ids)
    return x_out


@pl.jit.host
def l3_prefill_head(
    x_hc: pl.Tensor[[TP_SIZE, PREFILL_ROWS, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[TP_SIZE, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[TP_SIZE, 1], pl.FP32],
    hc_head_base: pl.Tensor[[TP_SIZE, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[TP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[TP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[TP_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    grammar_mask: pl.Tensor[[TP_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    x_out: pl.Out[pl.Tensor[[TP_SIZE, PREFILL_ROWS, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[TP_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[TP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    x_hc.bind_dynamic(1, PREFILL_ROWS)
    x_out.bind_dynamic(1, PREFILL_ROWS)
    hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    hidden_done_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    logits_done_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        l2_prefill_head(x_hc[rank], hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank],
                        final_norm_w[rank], lm_head_weight[rank], logit_row_indices[rank], grammar_mask[rank],
                        x_out[rank], logits[rank], sampled_ids[rank], hidden_window, hidden_done,
                        logits_window, logits_done, rank, device=rank)
    return x_out
