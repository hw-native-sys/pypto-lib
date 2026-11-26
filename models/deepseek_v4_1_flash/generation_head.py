# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Shared final HC reduction, vocabulary projection and sampling for generation."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import FLASH as M, LM_HEAD_TP_SIZE
from hc import hc_head
from lm_head import MAX_LOGIT_ROWS, clear_lm_head_signals, lm_head
from rmsnorm import rms_norm
from sample import sample

D = M.hidden_size
VOCAB = M.vocab_size


@pl.jit.inline(auto_scope=False)
def generation_head(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16, pl.NZ],
    logit_rows: pl.Tensor[[16], pl.INT32],
    sampling_temperatures: pl.Tensor[[16], pl.FP32],
    sampling_top_ks: pl.Tensor[[16], pl.INT32],
    sampling_top_ps: pl.Tensor[[16], pl.FP32],
    sampling_seeds: pl.Tensor[[16], pl.INT32],
    sampling_positions: pl.Tensor[[16], pl.INT32],
    logits: pl.Tensor,
    sampled_ids: pl.Tensor,
    head_hidden: pld.DistributedTensor[[LM_HEAD_TP_SIZE * 16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.BF16],
    head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x_hc, 0)
    collapsed = pl.create_tensor([rows, D], dtype=pl.BF16)
    hc_head(x_hc, pre_mix, collapsed, worker_limit=64)
    normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
    rms_norm(collapsed, final_norm_weight, normalized, rows, width=D, zero_tail=False)
    lm_head(
        normalized, lm_head_weight, logit_rows, logits,
        head_hidden, head_hidden_done, head_logits, head_logits_done, rank, MAX_LOGIT_ROWS,
    )
    clear_lm_head_signals(logits, head_hidden_done, head_logits_done)
    sample(
        logits, sampling_temperatures, sampling_top_ks, sampling_top_ps,
        sampling_seeds, sampling_positions, sampled_ids,
    )
