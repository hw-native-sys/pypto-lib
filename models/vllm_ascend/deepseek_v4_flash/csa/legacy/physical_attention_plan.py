# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""已验证单SPMD的选中页检查与INT32物理行转换，供多task plan内联使用。

调用方统一证明T>=64、0<页表列<=2048、0<物理页<=2**24；不满足则整体
使用旧逻辑表示。索引转换用INT32位移加法，地址不经FP32；范围比较沿原
FP32精确整数域。外部Top-K保持不变，内部无效物理行写-1，由MGATHER填零。
"""
import pypto.language as pl

from ..common.layout import COMPRESSED_TABLE_COLUMNS_DYN

B_DYN = pl.dynamic("B_DYN")
TOPK = 512
PHYSICAL_PLAN_COLUMNS = 2048
PHYSICAL_PLAN_MAX_PAGES = 1 << 24
PHYSICAL_PLAN_MIN_TOKENS = 64


@pl.jit.inline(auto_scope=False)
def compressed_physical_plan(
    p_raw: pl.Tensor[[1, TOPK], pl.INT32],
    compressed_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    upper: pl.Scalar[pl.INDEX],
    compressed_pages: pl.Scalar[pl.INDEX],
):
    p_nonnegative = pl.maximum(p_raw, 0)
    p_columns = pl.shrs(p_nonnegative, 5)
    p_column_count = pl.tensor.dim(compressed_table, 1)
    p_safe_columns = pl.minimum(p_columns, pl.cast(p_column_count - 1, pl.INT32))
    p_table = pl.load(
        compressed_table, [request, 0], [1, PHYSICAL_PLAN_COLUMNS],
        valid_shape=[1, p_column_count], target_memory=pl.MemorySpace.Vec,
    )
    p_pages = pl.gather(p_table, p_safe_columns)
    # clip只防UB gather和位移溢出；合法mask仍用原始逻辑列和页值。
    # page数<=2**24，因此下述FP32范围比较不会把边界坏页判为好页。
    p_raw_f = pl.cast(p_raw, pl.FP32)
    p_column_f = pl.cast(p_columns, pl.FP32)
    p_page_f = pl.cast(p_pages, pl.FP32)
    p_upper_f = pl.cast(pl.cast(upper, pl.INT32), pl.FP32)
    p_column_count_f = pl.cast(pl.cast(p_column_count, pl.INT32), pl.FP32)
    p_page_count_f = pl.cast(pl.cast(compressed_pages, pl.INT32), pl.FP32)
    p_raw_ge = pl.minimum(pl.maximum(pl.add(p_raw_f, 1.0), 0.0), 1.0)
    p_raw_lt = pl.minimum(pl.maximum(pl.add(pl.neg(p_raw_f), p_upper_f), 0.0), 1.0)
    p_col_lt = pl.minimum(pl.maximum(pl.add(pl.neg(p_column_f), p_column_count_f), 0.0), 1.0)
    p_page_ge = pl.minimum(pl.maximum(pl.add(p_page_f, 1.0), 0.0), 1.0)
    p_page_lt = pl.minimum(pl.maximum(pl.add(pl.neg(p_page_f), p_page_count_f), 0.0), 1.0)
    p_valid = pl.mul(pl.mul(pl.mul(p_raw_ge, p_raw_lt), p_col_lt), pl.mul(p_page_ge, p_page_lt))
    p_safe_pages = pl.minimum(
        pl.maximum(p_pages, 0), pl.cast(compressed_pages - 1, pl.INT32)
    )
    p_remainders = pl.sub(p_nonnegative, pl.shls(p_columns, 5))
    p_physical = pl.add(pl.shls(p_safe_pages, 5), p_remainders)
    p_valid_i32 = pl.cast(p_valid, pl.INT32)
    p_output = pl.sub(pl.add(pl.mul(p_physical, p_valid_i32), p_valid_i32), 1)
    return p_output, p_valid
