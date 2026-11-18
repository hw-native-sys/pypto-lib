# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import BLOCK_SIZE, FLASH as M

from .qkv_proj_rope import kv_proj_rope, rope_prepare

T_DYN = pl.dynamic("NATIVE_D_DSPARK_CONTEXT_KV_T_DYN")

ORI_BLOCK_NUM_DYN = pl.dynamic("NATIVE_D_DSPARK_CONTEXT_KV_ORI_BLOCK_NUM_DYN")

DSPARK_CONTEXT_LAYERS = 3

D = M.hidden_size

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim


@pl.jit.inline
def dspark_context_kv(
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[DSPARK_CONTEXT_LAYERS, T_DYN], pl.INT64],
    layer_index: pl.Scalar[pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(position_ids, 0)

    rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos, freqs_sin, rope_cos_il, rope_sin_signed, rope_swap_idx)

    # This no-work source task seeds the explicit kv_proj dependency chain.
    late_dep = pl.system.task_dummy(deps=[])
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(main_x, wkv, gamma_ckv, rope_cos_il, rope_sin_signed, rope_swap_idx, kv, late_dep)

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_context_kv_scatter"):
        for write_t in pl.range(t_dim):
            write_row_i64 = pl.read(slot_mapping, [layer_index, write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

    return kv_cache
