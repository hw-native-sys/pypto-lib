# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 entry for the shared C2 projected-pool/state ABI."""
import pypto.language as pl
from .compressor_core import compressor_ratio2, RING_ROWS_DYN
from .config import T_DYN, B_DYN, Q_START_DYN, STATE_BLOCKS_DYN

WIDTH = 512


@pl.jit
def pool_projected(
    kv: pl.Tensor[[T_DYN, WIDTH], pl.FP32],
    scores: pl.Tensor[[T_DYN, WIDTH], pl.FP32],
    query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT32],
    requests: pl.Tensor[[T_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    norm: pl.Tensor[[WIDTH], pl.BF16],
    state: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * WIDTH], pl.FP32]],
    latent: pl.InOut[pl.Tensor[[T_DYN, WIDTH], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    kv.bind_dynamic(0, T_DYN)
    scores.bind_dynamic(0, T_DYN)
    positions.bind_dynamic(0, T_DYN)
    requests.bind_dynamic(0, T_DYN)
    latent.bind_dynamic(0, T_DYN)
    copied_kv = pl.create_tensor([pl.tensor.dim(kv, 0), WIDTH], dtype=pl.FP32)
    copied_scores = pl.create_tensor([pl.tensor.dim(kv, 0), WIDTH], dtype=pl.FP32)
    with pl.spmd(active, name_hint='a2_compressor_inputs') as ready:
        row = pl.tile.get_block_idx()
        copied_kv[row:row + 1, :WIDTH] = kv[row:row + 1, :WIDTH]
        copied_scores[row:row + 1, :WIDTH] = scores[row:row + 1, :WIDTH]
    compressor_ratio2(copied_kv, copied_scores, query_start, positions, requests,
                      block_table, state, norm, latent, active, ready)
    return state, latent
