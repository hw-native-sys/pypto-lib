# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 的紧凑 C4 元数据：请求前缀偏移与核内 RoPE 取行。

入参、输出与动态轴
    Q 是 Native INT32[B+1] query 前缀和，L 是含本轮 query 的 INT32[B]
    KV 长度，positions 是 INT64[T]。这些已有 device Tensor 原样传入，
    B、T、Q 容量彼此独立；Q 的正常逻辑长度为 B+1，但边界访问仍以它自己
    的实际维度校验。请求长度来自 Q[r+1]-Q[r]，不用固定 S 整除。
    token_capacity 是本轮 token Tensor 的实际行数 T，其中可含 dummy；
    活动请求仍由 Q/L 判断。不能换成更大的 staging 容量；RoPE 取行 helper
    直接从 positions 的 T 轴取得此界，不额外要求 Host 读取有效 token 数。
    offsets 是 INT32[B] 小工作区；压缩 cos/sin 保持 Native FP32[C,64]
    紧凑行，不展开成 T 行。C 可以为零，不能用 B 或容量推断有效关闭组数。
    需要全局压缩行循环的调用者使用 INT32[1] valid_count 工作区，保存 Q/L
    导出的实际关闭数。它由设备写入、设备读取，不需要 Host .item()。

整体数据流与阅读顺序
    1. _scan_compact_offsets 定义唯一的前缀算法。compact_offsets_stage
       只提供偏移；compact_offsets_and_count_stage 在同一次扫描后同时写
       最终行数。调用者二选一，不为了得到 count 再扫描一遍请求。
       指定集合的 worker 0 执行，其他 worker 不写，也不等待全核集合。
    2. load_request_compact_rope_rows 在已有压缩消费者的核内执行；它已知
       request 与其 token 子区间，使用该请求偏移只加载真正关闭 C4 的行。
       返回两个 UB tile，未关闭行用 cos=1/sin=0；不新增 GM 展开缓存。

分工、同步与工作区
    紧凑偏移扫描只做 Scalar 地址运算，可在选定的 AIC 或 AIV 集合内调用；
    必须只选一个集合的 worker 0 写 offsets，不能让两个角色各自的 0 并发写。
    取行 helper 则在压缩消费者的 AIV 内返回 UB Tile，不跨核返回 tile 存储。
    offsets 的前缀有跨请求依赖，首版保持单写者，避免每个请求再扫描全部
    前驱形成 O(B²)。调用者安排它与数学并行，并在完成后发布阶段信号。
    消费者先等 offsets/count 可见再调用取行或 Hadamard；各函数不创建 task 或
    notify。RoPE 输入由 Native 生产，其 stream/event 依赖仍由外层保证。
    offsets 即使为负也是合法中间量，必须加 (position+1)//4 后才作下标。
    设请求历史起点 H=L-(Q[r+1]-Q[r])，前面请求已输出 prefix 个紧凑行，
    本请求关闭数为 floor(L/4)-floor(H/4)，offset=prefix-floor(H/4)-1。
    减 1 将第一次关闭组映射到 prefix：如 H=6、L=12、prefix=0，offset=-2，
    position=7/11 两个关闭行分别读取 compact_row=0/1；不能预留每请求两行。

有效边界与数值合同
    request_bounds 先检查 L 与 Q；dummy 不读取残留 position，不推进前缀。
    活动请求允许零关闭组或多个关闭组；count=0 不等于 inactive。取行仅
    接受单请求内至多 16 行，并要求 position 与 causal decode 的 L-Qlen
    起点一致。闭合行号同时检查 cos/sin 的实际容量及非负性，防止 capture
    的陈旧 dummy 位置越界。拒绝条件留下单位旋转，不把非法 metadata
    当成支持的数学场景。调用方仍负责正确 Native metadata 与页表合同。

当前范围
    本段不改压缩数学、state 或 KV 数据，只定义紧凑地址。它能够表达
    odd/mixed query 的地址，但不代表后续压缩/Indexer 已全部接入或上卡
    验证。tile 行数 16 是当前消费者的 UB 分块，不是固定 batch/sequence。
"""

import pypto.language as pl

from .request_geometry import COMPRESS_RATIO, QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

T_DYN = pl.dynamic("CSA_METADATA_T")
COMPACT_ROWS_DYN = pl.dynamic("CSA_METADATA_COMPACT_ROWS")
ROPE_DIM = 64
ROPE_TILE_ROWS = 16


@pl.jit.inline(auto_scope=False)
def _scan_compact_offsets(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Out[pl.Tensor[[REQUESTS_DYN], pl.INT32]],
    token_capacity: pl.Scalar[pl.INDEX],
):
    """只供指定的单写者内联，返回 offsets 别名和核内最终前缀 Scalar。"""
    prefix = pl.cast(0, pl.INDEX)
    # 请求顺序来自 Native Q，必须串行保留 prefix；不是逐 token 展开任务。
    for request in pl.range(pl.tensor.dim(lengths, 0)):
        begin, end, active = request_bounds(bounds, lengths, request, token_capacity)
        value = pl.cast(0, pl.INT32)
        if active != 0:
            length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
            history_start = length - (end - begin)
            value = pl.cast(prefix - history_start // COMPRESS_RATIO - 1, pl.INT32)
            prefix = prefix + length // COMPRESS_RATIO - history_start // COMPRESS_RATIO
        # 每次覆盖所有 B 槽；dummy 不遗留上一轮偏移，但消费者仍须检查 active。
        pl.write(offsets, [request], value)
    return offsets, prefix


@pl.jit.inline(auto_scope=False)
def compact_offsets_stage(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Out[pl.Tensor[[REQUESTS_DYN], pl.INT32]],
    token_capacity: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
):
    """指定集合的 worker 0 写 offsets[B]；所有输入均为当前调用的 device 值。

    其它 worker 的空返回不表示 offsets 已生成；真正的生产者完成整个请求
    扫描后，外层才能发布该阶段。dummy 槽写零仅清除旧值，不把它变成活动请求。
    """
    if worker_id == 0:
        offsets, _count = _scan_compact_offsets(bounds, lengths, offsets, token_capacity)
    return offsets


@pl.jit.inline(auto_scope=False)
def compact_offsets_and_count_stage(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Out[pl.Tensor[[REQUESTS_DYN], pl.INT32]],
    valid_count: pl.Out[pl.Tensor[[1], pl.INT32]],
    token_capacity: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
):
    """同一扫描写 offsets 与实际行数，外层一次发布这两个 GM 输出。

    count 来自合法 Q/L 的关闭候选数，不能因 stale position 缩减，否则会
    改变后续请求的 Native compact 偏移。norm 先初始化各请求的候选行，再
    只覆盖 causal 行；cache 副作用仍逐行核对 position/slot。调用者保证
    count 不大于 compact 表容量，不能把容量不足的输入称为有效支持场景。
    """
    if worker_id == 0:
        offsets, count = _scan_compact_offsets(bounds, lengths, offsets, token_capacity)
        # T 是INT32 query前缀覆盖的本轮 token 数，C4关闭数不超过T，可无损存INT32。
        pl.write(valid_count, [0], pl.cast(count, pl.INT32))
    return offsets, valid_count


@pl.jit.inline(auto_scope=False)
def load_request_compact_rope_rows(
    cos: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    first: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
):
    """在消费者 AIV 内返回 cos/sin UB tile；first/rows 必须属于同一请求。"""
    # 显式创建UB Tile：helper返回给同一核继续计算，不能被解析为需要GM
    # 存储的高层Tensor。这样调用者可以直接对返回值设置valid_shape并store。
    cosine = pl.tile.full([ROPE_TILE_ROWS, ROPE_DIM], dtype=pl.FP32, value=1.0)
    sine = pl.tile.full([ROPE_TILE_ROWS, ROPE_DIM], dtype=pl.FP32, value=0.0)
    begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
    if active != 0 and rows > 0 and rows <= ROPE_TILE_ROWS and first >= begin and first + rows <= end:
        length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
        history_start = length - (end - begin)
        compact_offset = pl.cast(pl.read(offsets, [request]), pl.INDEX)
        # rows<=16 且 first..first+rows 已限定在同一请求内，不会将取行范围
        # 跨入另一请求。仅迭代真实 rows，UB 其余行保持单位旋转；返回者须在消费或
        # store 时限制有效行，不能把未填充的单位旋转行当作额外 query。
        # 每行同时检查位置连续性与 C4 关闭条件，未关闭行不触及紧凑表。
        for row in pl.range(rows):
            token = first + row
            position = pl.cast(pl.read(positions, [token]), pl.INDEX)
            if position == history_start + token - begin and (position + 1) % COMPRESS_RATIO == 0:
                compact_row = compact_offset + (position + 1) // COMPRESS_RATIO
                if compact_row >= 0 and compact_row < pl.tensor.dim(cos, 0) and compact_row < pl.tensor.dim(sin, 0):
                    cosine = pl.gather_row(cosine, cos, [row, 0], [compact_row, 0], [1, ROPE_DIM])
                    sine = pl.gather_row(sine, sin, [row, 0], [compact_row, 0], [1, ROPE_DIM])
    return cosine, sine
