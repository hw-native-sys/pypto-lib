# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Indexer Native FP16 scale的设备端页所有权计划；在已有META阶段内执行。

整体思路与阅读顺序
    indexer_scale_page_plan_stage由前置metadata的唯一AIV扫描本轮Q/L、位置、
    compact offsets及slots，把合法scale写入分给page%WORKERS对应的AIV。
    输出是每worker的链表头及每项(page,offset,compact_row,next)；量化结束后，
    本文件的scale提交沿自己的链执行原64B读改写。
    多个请求共享物理页时，全部scale更新仍只在一个核上；不同页可并行。

动态布局与范围
    workspace复用group_offsets这一个INT32动态Tensor。前面仍是pad16(B+1)
    请求组前缀和4*T组描述符，向上对齐16项后为WORKERS个链表头，再接最多T
    项四字描述符。T是positions实际容量，合法关闭数不超过有效query总数。
    链表项按合法输入的枚举次序编号，next总是指向较早项或-1，故必然终止。
    这里只缓存地址及所有权，量化值在之后读取；不以Host基本类型代替device值。

同步、缓存与生命周期
    全计划仅由调用方选定的一个Scalar写；即使相邻项共享缓存行也没有多写者。
    首先覆盖全部head为-1，只从本轮可达链读取，不清空也不读取工作区陈旧尾部。
    外层PHASE_META的发布/获取维护DCache；scale消费者另须等待量化完成。
    本文件不发task、不创建SPMD、不自行notify；完整调用期间输入metadata不变。
    page_bytes必须至少4160且为64的倍数，由绑定层保证；这让不同物理页的
    64B scale区不重叠。合法slot应唯一；重复slot的输入合同错误不由此计划修复。

验证边界
    本计划不改变量化/FP16转换数学，但改变scale提交顺序与并行度；需验证
    同页不同offset、不同页、dummy/陈旧位置、未写区域和图重放。源码存在不等于
    跨核可见性或完整CSA性能已经通过，证据统一记在SPMD优化日志。
"""

import pypto.language as pl

from .indexer_cache_stages import CACHE_BYTES_DYN, CACHE_PAGES_DYN, SLOT_ROWS_DYN, T_DYN
from .indexer_group_stages import OFFSET_ROWS_DYN
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

SCALE_PLAN_ALIGNMENT = 16
SCALE_JOB_WORDS = 4
GROUP_WORDS_PER_TOKEN = 4
SCALE_PAGE_ROWS = 32
SCALE_CACHE_MIN_BYTES = 4160


@pl.jit.inline(auto_scope=False)
def indexer_scale_plan_base(requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]):
    """与Host容量公式相同的设备地址表达式；不使用有效group数量缩短固定区。"""
    group_base = (requests + 1 + SCALE_PLAN_ALIGNMENT - 1) // SCALE_PLAN_ALIGNMENT * SCALE_PLAN_ALIGNMENT
    group_end = group_base + GROUP_WORDS_PER_TOKEN * tokens
    return (group_end + SCALE_PLAN_ALIGNMENT - 1) // SCALE_PLAN_ALIGNMENT * SCALE_PLAN_ALIGNMENT


@pl.jit.inline(auto_scope=False)
def indexer_scale_page_plan_stage(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    compact_offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    slots: pl.Tensor[[SLOT_ROWS_DYN, 2], pl.INT32],
    cache: pl.Tensor[[CACHE_PAGES_DYN, CACHE_BYTES_DYN], pl.INT8],
    workspace: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    scale_capacity: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """仅允许前置唯一planner调用；描述符仅覆盖真实合法的关闭行。"""
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    tokens = pl.tensor.dim(positions, 0)
    base = indexer_scale_plan_base(requests, tokens)
    for owner in pl.range(WORKERS):
        pl.write(workspace, [base + owner], pl.cast(-1, pl.INT32))
    jobs = pl.cast(0, pl.INDEX)
    for request in pl.range(requests):
        begin, end, active = request_bounds(bounds, lengths, request, tokens)
        if active != 0:
            length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
            history = length - (end - begin)
            count = length // 4 - history // 4
            first_token = begin + 3 - history % 4
            prefix = pl.cast(pl.read(compact_offsets, [request]), pl.INDEX) + history // 4 + 1
            if prefix >= 0 and prefix + count <= scale_capacity and prefix + count <= pl.tensor.dim(slots, 0):
                for local_row in pl.range(count):
                    token = first_token + local_row * 4
                    position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                    if position == history + token - begin:
                        page = pl.cast(pl.read(slots, [prefix + local_row, 0]), pl.INDEX)
                        offset = pl.cast(pl.read(slots, [prefix + local_row, 1]), pl.INDEX)
                        if page >= 0 and page < pl.tensor.dim(cache, 0) and offset >= 0 and offset < SCALE_PAGE_ROWS:
                            if pl.tensor.dim(cache, 1) >= SCALE_CACHE_MIN_BYTES:
                                owner = page % WORKERS
                                head = pl.read(workspace, [base + owner])
                                row = base + WORKERS + jobs * SCALE_JOB_WORDS
                                pl.write(workspace, [row], pl.cast(page, pl.INT32))
                                pl.write(workspace, [row + 1], pl.cast(offset, pl.INT32))
                                pl.write(workspace, [row + 2], pl.cast(prefix + local_row, pl.INT32))
                                pl.write(workspace, [row + 3], head)
                                pl.write(workspace, [base + owner], pl.cast(jobs, pl.INT32))
                                jobs = jobs + 1
    return workspace


@pl.jit.inline(auto_scope=False)
def indexer_scale_page_commit_stage(
    scale_values: pl.Tensor,
    workspace: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    cache: pl.InOut[pl.Tensor[[CACHE_PAGES_DYN, CACHE_BYTES_DYN], pl.INT8]],
    tokens: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """每物理页只有一个writer；逐项保持原Scalar FP16转换及64B Tile读改写。"""
    base = indexer_scale_plan_base(requests, tokens)
    initial_node = pl.cast(pl.read(workspace, [base + worker]), pl.INDEX)
    # 链表只有本轮合法项，下一项编号严格减小；空worker读到-1直接结束。
    # 不能改成按request均分，否则共享prefix可能落在不同核同时RMW同一64B行。
    # 显式携带node；当前隔离编译器会丢掉自然while中仅供下轮条件使用的更新，
    # 使非空链无法退出。pl.while_把条件与回边值写明，仍是设备端动态循环。
    for (node,) in pl.while_(init_values=(initial_node,)):
        pl.cond(node >= 0)
        row = base + WORKERS + node * SCALE_JOB_WORDS
        page = pl.cast(pl.read(workspace, [row]), pl.INDEX)
        offset = pl.cast(pl.read(workspace, [row + 1]), pl.INDEX)
        compact_row = pl.cast(pl.read(workspace, [row + 2]), pl.INDEX)
        scale = pl.cast(pl.read(scale_values, [compact_row, 0]), pl.FP16)
        scale_bytes = pl.tile.load(cache, [page, 4096], [1, 64])
        scale_half = pl.tile.reinterpret_view(scale_bytes, pl.FP16)
        pl.tile.write(scale_half, [0, offset], scale)
        updated = pl.tile.reinterpret_view(scale_half, pl.INT8)
        pl.tile.store(updated, [page, 4096], cache)
        next_node = pl.cast(pl.read(workspace, [row + 3]), pl.INDEX)
        completed_node = pl.yield_(next_node)  # noqa: F841 — DSL回边的结果绑定
    return cache
