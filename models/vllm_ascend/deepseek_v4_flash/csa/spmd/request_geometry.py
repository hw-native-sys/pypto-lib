# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CSA 的动态请求几何：直接消费 Native 的原始 TND 边界与 KV 长度。

query_start_loc[r:r+2] 表示本轮请求 r 的原始 query 区间，seq_lens[r]
表示包含本轮 query 的未压缩总长度。它们已经是 Native 的 device Tensor；
本模块不把它们回读到 Host，也不建立逐 token 的 request_id 展开缓冲。
不同请求可以有不同 query 数，不能用 token // S 或 T // B 推导请求边界。

这些 helper 在调用者的核内 inline，既不单独提交 task，也不承担核间同步。
Q 和 B 是独立动态轴；调用者通过已有 Tensor 的动态 dim 传入 token_capacity。
地址、循环变量和计数统一使用 INDEX，active 使用 INT32 的 0/1。返回的
compact_offset 可以为负，它是地址计算的中间偏移，不能单独用作 GM 下标。

失活请求、越界 request、反向/负边界、超出 token 容量的边界，以及 query
长度超过总 KV 长度的请求均返回 active=0。先检查 seq_lens 再读 Q，可避免
padding 请求残留的边界或 position 被误用。全局请求顺序、Q 与各组页表的
一致性仍由 Native 输入合同保证；query_request 额外拒绝多个活动区间覆盖。

压缩比固定为模型的 C4 几何。压缩前缀只累计合法活动请求，在原始边界上
计算实际关闭的压缩组数；不假定每请求六行，也不预留每请求恰好两行输出。
本模块只定义地址与有效范围，不改变排序、attention 或压缩数学。

阅读顺序：request_bounds 先定义合法请求；query_request 在只有 token 编号
时反查所属请求；compression_geometry 再计算 C4 输出计数与紧凑偏移。
返回值均为核内 Scalar，不分配 GM 工作区。AIC/AIV 都可按需内联；通常应
保留请求驱动的分工，用已有 request 直接取边界，避免每行重复 O(B) 反查。
目前仅有同源 CPU 逻辑与编译证据；这些 helper 尚不能代表所有消费者已经
完成 mixed-query、padding 和请求生命周期的设备验证。
"""

import pypto.language as pl

QUERY_BOUNDS_DYN = pl.dynamic("CSA_GEOMETRY_QUERY_BOUNDS")
REQUESTS_DYN = pl.dynamic("CSA_GEOMETRY_REQUESTS")
COMPRESS_RATIO = 4


@pl.jit.inline(auto_scope=False)
def request_bounds(
    query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    token_capacity: pl.Scalar[pl.INDEX],
) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]]:
    """返回 (begin, end, active)；失活时 begin=end=0，禁止据其读取 payload。"""
    begin = pl.cast(0, pl.INDEX)
    end = pl.cast(0, pl.INDEX)
    active = pl.cast(0, pl.INT32)
    # 两个 Tensor 的轴分别检查；即使传入长度不匹配也不能越界读 Q[r+1]。
    if request >= 0 and request < pl.tensor.dim(seq_lens, 0):
        if request + 1 < pl.tensor.dim(query_start_loc, 0):
            length = pl.cast(pl.tensor.read(seq_lens, [request]), pl.INDEX)
            if length > 0:
                raw_begin = pl.cast(pl.tensor.read(query_start_loc, [request]), pl.INDEX)
                raw_end = pl.cast(pl.tensor.read(query_start_loc, [request + 1]), pl.INDEX)
                if raw_begin >= 0 and raw_begin < raw_end and raw_end <= token_capacity:
                    if raw_end - raw_begin <= length:
                        begin = raw_begin
                        end = raw_end
                        active = pl.cast(1, pl.INT32)
    return begin, end, active


@pl.jit.inline(auto_scope=False)
def query_request(
    query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    query: pl.Scalar[pl.INDEX],
    token_capacity: pl.Scalar[pl.INDEX],
) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]]:
    """返回 (request, begin, end, active)；未找到或多重覆盖时 request=-1。"""
    owner = pl.cast(-1, pl.INDEX)
    owner_begin = pl.cast(0, pl.INDEX)
    owner_end = pl.cast(0, pl.INDEX)
    matches = pl.cast(0, pl.INDEX)
    active = pl.cast(0, pl.INT32)
    if query >= 0 and query < token_capacity:
        # 本地扫描已有 B+1 个边界，不在 GM 展开 T 个映射。请求驱动的消费者
        # 已持有 request，可直接用 request_bounds，避免重复反查。
        for request in pl.range(pl.tensor.dim(seq_lens, 0)):
            begin, end, request_active = request_bounds(query_start_loc, seq_lens, request, token_capacity)
            if request_active != 0 and query >= begin and query < end:
                owner = request
                owner_begin = begin
                owner_end = end
                matches = matches + 1
        if matches == 1:
            active = pl.cast(1, pl.INT32)
        else:
            owner = pl.cast(-1, pl.INDEX)
            owner_begin = pl.cast(0, pl.INDEX)
            owner_end = pl.cast(0, pl.INDEX)
    return owner, owner_begin, owner_end, active


@pl.jit.inline(auto_scope=False)
def compression_geometry(
    query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    token_capacity: pl.Scalar[pl.INDEX],
) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INT32]]:
    """返回 (count, compact_offset, active)，适用于 Native C4 紧凑元数据。

    合法关闭行的下标是 compact_offset + (position+1)//4；只有 active=1、
    position 对应本请求且满足关闭边界时才能使用，还须核对目标 Tensor 容量。
    活动请求没有关闭组时 count=0 但 active=1，两者不能混为一谈。
    """
    count = pl.cast(0, pl.INDEX)
    offset = pl.cast(0, pl.INDEX)
    begin, end, active = request_bounds(query_start_loc, seq_lens, request, token_capacity)
    if active != 0:
        prefix = pl.cast(0, pl.INDEX)
        # 只统计本请求之前的真实关闭组数；dummy 和非法 Q 不推进前缀。
        for previous in pl.range(request):
            prior_begin, prior_end, prior_active = request_bounds(
                query_start_loc, seq_lens, previous, token_capacity
            )
            if prior_active != 0:
                prior_length = pl.cast(pl.tensor.read(seq_lens, [previous]), pl.INDEX)
                prior_start = prior_length - (prior_end - prior_begin)
                prefix = prefix + prior_length // COMPRESS_RATIO - prior_start // COMPRESS_RATIO
        length = pl.cast(pl.tensor.read(seq_lens, [request]), pl.INDEX)
        start = length - (end - begin)
        count = length // COMPRESS_RATIO - start // COMPRESS_RATIO
        offset = prefix - start // COMPRESS_RATIO - 1
    return count, offset, active
