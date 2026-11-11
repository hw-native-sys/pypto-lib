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
from lm_head import SAMPLED_IDS_PAD, VOCAB_PER_TP, lm_head_sample
from rmsnorm import rms_norm_tiled_compensated


D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
ROWS = MAX_BATCH_PER_DP * DECODE_ROWS_PER_REQUEST
TARGET_T_DYN = pl.dynamic("V41_DSPARK_TARGET_T_DYN")

if TP_SIZE != 1:
    raise ValueError("DSpark target row sampling currently requires TP1")


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
    lm_head_sample(normalized, lm_head_weight, active_widths, sampled_ids, ready)
    return sampled_ids
