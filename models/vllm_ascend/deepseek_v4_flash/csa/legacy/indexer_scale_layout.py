# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""单SPMD已验证连续页证明的S6多task适配；唯一Scalar写者，每次覆盖全部实际请求。

只读Native页表/长度，不读取cache payload或改变cache布局。证明包括全部有效页，
严格升序/降序连续才返回非零方向，碎片/失活/容量外均返回(0,0)。Score显式等待本
task完成后读取；不能仅依赖函数返回别名或把首尾连续当作全表连续。
"""
import pypto.language as pl

from ..common.config import DECODE_BATCH, DECODE_SEQ, TP
from ..common.layout import INDEXER_TABLE_COLUMNS_DYN

B_DYN = pl.dynamic("B_DYN")
SCALE_LAYOUT_CAPACITY = 2 * (DECODE_BATCH // TP)

LAYOUT_WORDS_PER_REQUEST = 2
LAYOUT_PROOF_PAGES = 1024
LAYOUT_PROOF_ROWS = 8
LAYOUT_EXACT_INTEGER_LIMIT = 16777216
LAYOUT_COMPRESS_RATIO = 4
LAYOUT_ROWS_PER_PAGE = 32



@pl.jit.inline(auto_scope=False)
def prove_scale_layout(
    lengths: pl.Tensor[[B_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    workspace: pl.InOut[pl.Tensor[[SCALE_LAYOUT_CAPACITY], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    physical_pages: pl.Scalar[pl.INDEX],
):
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    base = 0
    for request in pl.range(requests):
        first_page = pl.cast(0, pl.INT32)
        direction = pl.cast(0, pl.INT32)
        # 多task入口原有S6合同；不从capacity猜活动性，dummy长度必须非正。
        active = pl.cast(0, pl.INT32)
        if request * DECODE_SEQ < tokens:
            if pl.read(lengths, [request]) > 0:
                active = pl.cast(1, pl.INT32)
        if active != 0:
            length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
            compressed = length // LAYOUT_COMPRESS_RATIO
            pages = (compressed + LAYOUT_ROWS_PER_PAGE - 1) // LAYOUT_ROWS_PER_PAGE
            if pages > 0 and pages <= pl.tensor.dim(block_table, 1):
                if physical_pages > 0 and physical_pages <= LAYOUT_EXACT_INTEGER_LIMIT:
                    first = pl.cast(pl.read(block_table, [request, 0]), pl.INDEX)
                    last = pl.cast(pl.read(block_table, [request, pages - 1]), pl.INDEX)
                    trial_direction = pl.cast(1, pl.INT32)
                    if last < first:
                        trial_direction = pl.cast(-1, pl.INT32)
                    expected_last = first + pl.cast(trial_direction, pl.INDEX) * (pages - 1)
                    if last == expected_last:
                        if pl.min(first, expected_last) >= 0 and pl.max(first, expected_last) < physical_pages:
                            first_page = pl.cast(first, pl.INT32)
                            direction = trial_direction
                            # 只在首尾及完整合法域成立后做逐页证明。保留 direction
                            # 为循环携带 Scalar；任何一块失败，本轮所有叶都回退。
                            for block in pl.range((pages + LAYOUT_PROOF_PAGES - 1) // LAYOUT_PROOF_PAGES):
                                if direction != 0:
                                    page_begin = block * LAYOUT_PROOF_PAGES
                                    valid_pages = pl.min(pages - page_begin, LAYOUT_PROOF_PAGES)
                                    page_ids = pl.load(
                                        block_table, [request, page_begin], [1, LAYOUT_PROOF_PAGES],
                                        valid_shape=[1, valid_pages],
                                    )
                                    expected_begin = first + pl.cast(trial_direction, pl.INDEX) * page_begin
                                    expected = pl.set_validshape(
                                        pl.add(
                                            pl.mul(
                                                pl.cast(
                                                    pl.tile.ci(0, [1, LAYOUT_PROOF_PAGES], dtype=pl.INT32), pl.FP32
                                                ),
                                                pl.cast(trial_direction, pl.FP32),
                                            ),
                                            pl.cast(pl.cast(expected_begin, pl.INT32), pl.FP32),
                                        ),
                                        1,
                                        valid_pages,
                                    )
                                    difference = pl.abs(pl.sub(pl.cast(page_ids, pl.FP32), expected))
                                    ones = pl.set_validshape(
                                        pl.tile.full([LAYOUT_PROOF_ROWS, LAYOUT_PROOF_PAGES], dtype=pl.FP32, value=1.0),
                                        LAYOUT_PROOF_ROWS, valid_pages,
                                    )
                                    repeated = pl.set_validshape(
                                        pl.col_expand_mul(ones, difference), LAYOUT_PROOF_ROWS, valid_pages,
                                    )
                                    scratch = pl.create_tile([LAYOUT_PROOF_ROWS, LAYOUT_PROOF_PAGES], dtype=pl.FP32)
                                    max_difference = pl.row_max(repeated, scratch)
                                    if pl.tile.read(max_difference, [0, 0]) != 0.0:
                                        direction = pl.cast(0, pl.INT32)
        if direction == 0:
            first_page = pl.cast(0, pl.INT32)
        # 两项都无条件覆盖。相邻请求共享缓存行，由唯一 Scalar 写者发布。
        row = base + LAYOUT_WORDS_PER_REQUEST * request
        pl.write(workspace, [row], first_page)
        pl.write(workspace, [row + 1], direction)
    # 对应单SPMD发布前的Scalar缓存维护；跨task依赖另由调用方显式建立。
    pl.system.cacheinvalid()
    return workspace
