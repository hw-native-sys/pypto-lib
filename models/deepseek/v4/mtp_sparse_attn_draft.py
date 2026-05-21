# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 mtp sparse_attn scaffold.

Kernel body is intentionally empty; golden follows the torch reference for this stage.
"""

import pypto.language as pl

from decode_sparse_attn import *  # noqa: F401,F403
from decode_sparse_attn import build_tensor_specs as _build_tensor_specs


@pl.jit
def mtp_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM],                               pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS],                            pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS],                            pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK],                                      pl.INT32],
    attn_sink: pl.Tensor[[H],                                            pl.FP32],
    seqused_kv: pl.Tensor[[B],                                            pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    even_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],            pl.BF16],
    odd_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK],            pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],                 pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA],                         pl.INT8],
    wo_b_scale: pl.Tensor[[D],                                            pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D],                                  pl.BF16]],
):
    # TODO: kernel implementation
    return attn_out


def golden_mtp_sparse_attn(tensors):
    from prefill_sparse_attn_draft import golden_prefill_sparse_attn

    return golden_prefill_sparse_attn(tensors)


def build_tensor_specs(*args, **kwargs):
    return _build_tensor_specs(*args, **kwargs)
