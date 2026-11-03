# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""在前置 META 内证明 Indexer Native scale 的连续页布局，供 Score 复用。

整体思路与阅读顺序
    indexer_scale_layout_base 定义已有 group_offsets 工作区的新增尾分区；
    indexer_scale_layout_plan_stage 由既有唯一 META planner 执行。它检查每个
    活动请求完整的压缩历史页表是否严格正序或逆序连续，记录首物理页和方向。
    Score 的两条 AIV 及全部叶读取这一结果；方向非零时用二维搬运批量读取
    Native 页内原有的 FP16 scale，方向为零时保留原逐页 gather。
    本模块不读 scale 数值、不改变 cache 布局、不执行 Score 数学。

动态输入、内存布局和回退边界
    bounds、lengths、block_table 是 Native device Tensor，B、T、页表宽度与
    物理页容量均动态。request_bounds 过滤 dummy/非法 query 区间；实际页数
    为 ceil(floor(length/4)/32)，不是历史档位常量。仅数学 tile 固定为 1024
    页，最后一块按实际有效列归约。不存在固定 B/S 白名单或 Host 数值回读。
    原 group_offsets 的前缀、组描述符、48 个 writer 链表头和 4*T 项地址
    描述符保持原偏移；尾部追加 2*B 项，每请求 (first_page, direction)。
    每轮都覆盖这两项，失活、越界、碎片或方向混合均写 (0,0)，不会读旧证明。
    页号域不超过 2^24 时，合法整数可无损转 FP32 比较；更大容量直接回退。
    首尾仅用于筛选方向与范围，仍需检查全部有效页，不能据首尾猜中间连续。

分核、同步和资源
    由 vector_schedule 选择的一名 AIV 在已有 META 发布前运行，其他核继续
    HC/QKV 前段。相邻 Scalar 项只由该写者写，不存在同缓存行多核覆盖。
    既有 PHASE_META 负责业务写回、DCCI/DSB 发布和消费者获取；本文件不新增
    task、SPMD、notify/wait，也不以普通 GM 写入代替缓存可见性保证。
    pl.range(requests) 遍历实际请求；每请求按 1024 页的动态块循环，失败后
    不再发后续 Vector 检查。A3 的行归约使用 8 行副本，所有广播和归约均保留
    动态有效列，不让尾部未加载数据进入判断。Tile 临时量仅活跃于 META。

验证约束
    这是保守的布局证明，不负责修复非法输入。只对合法 Native 输入承诺原
    Score 语义；必须检查连续/碎片页、长度变化、padding、图重放和完整 L1。
    证明前移可能推迟其它 META 消费者，是否采用只能依据同轮真机全链结果。
"""

import pypto.language as pl

from .indexer_group_stages import OFFSET_ROWS_DYN
from .indexer_scale_plan_stages import SCALE_JOB_WORDS, indexer_scale_plan_base
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

LAYOUT_TABLE_COLS = pl.dynamic("CSA_SCALE_LAYOUT_TABLE_COLS")
LAYOUT_WORDS_PER_REQUEST = 2
LAYOUT_PROOF_PAGES = 1024
LAYOUT_PROOF_ROWS = 8
LAYOUT_EXACT_INTEGER_LIMIT = 16777216
LAYOUT_COMPRESS_RATIO = 4
LAYOUT_ROWS_PER_PAGE = 32


@pl.jit.inline(auto_scope=False)
def indexer_scale_layout_base(
    requests: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """追加到容量分区尾部；不能用本轮实际 scale job 数缩短前面的保留区。"""
    base = indexer_scale_plan_base(requests, tokens)
    return base + WORKERS + SCALE_JOB_WORDS * tokens


@pl.jit.inline(auto_scope=False)
def indexer_scale_layout_plan_stage(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    block_table: pl.Tensor[[REQUESTS_DYN, LAYOUT_TABLE_COLS], pl.INT32],
    workspace: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    physical_pages: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """只由既有唯一 META planner 调用，检查结果通过同一 PHASE_META 发布。"""
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    base = indexer_scale_layout_base(requests, tokens, WORKERS)
    for request in pl.range(requests):
        first_page = pl.cast(0, pl.INT32)
        direction = pl.cast(0, pl.INT32)
        begin, end, active = request_bounds(bounds, lengths, request, tokens)
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
                                        block_table,
                                        [request, page_begin],
                                        [1, LAYOUT_PROOF_PAGES],
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
                                        LAYOUT_PROOF_ROWS,
                                        valid_pages,
                                    )
                                    repeated = pl.set_validshape(
                                        pl.col_expand_mul(ones, difference),
                                        LAYOUT_PROOF_ROWS,
                                        valid_pages,
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
    return workspace
