# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 的 Indexer 压缩 cache：真实紧凑行量化、key 写入及单写者 scale 提交。

整体计算与阅读顺序
    indexer_compact_quant_cache_stage 等待 Hadamard 后，按请求的真实 C4 关闭数
    量化紧凑 FP32 行，保存 FP32 scale，并按 Native 紧凑 slots 写 INT8 key。
    indexer_compact_scale_commit_stage 再由 worker 0 串行更新同一 cache 的
    FP16 scale。两阶段复用 Q/L 和 offsets 算出的关闭行地址，没有固定每请求
    两行布局，也不需要 C→request 或 token→request 的展开映射。

输入输出与动态容量
    values 是 Hadamard 输出 FP32[VALUE_ROWS,128]，scale_values 为
    FP32[SCALE_ROWS,1]，两者容量至少覆盖 Q/L 推导的实际关闭数 C_valid。
    Q/L、INT64 positions[T]、INT32 offsets[B] 都来自同一请求顺序。紧凑
    slots 为 INT32[C_CAP,2]，C_CAP 是容量，可能大于有效数且允许为零。
    cache 直接接 Native INT8[pages,page_bytes]：每页 32 个 key 各 128B，
    后面是 32 个 FP16 scale 共 64B；page_bytes 至少 4160，可能带其他尾部。
    kv_debug 是 FP32[T,128] 原调试输出别名，只写合法关闭行，未写区保留。
    Tensor 容量和 T/B 均动态。128列、C4、每页32行是模型/Native布局常量。

分工、工作区和核间同步
    量化 AIV 按 request % WORKERS 领取请求，内层以最多16个真实关闭行做块。
    block起点可不对齐16；load/store 均限制有效行，不能覆盖下一请求或pad。
    外层先等待 Hadamard 和 metadata，量化函数写 scale_values/key 后发布；
    scale 提交必须等待所有量化生产者并完成业务缓存维护，再由唯一 worker0
    执行。32个FP16 scale共用一次64B读改写，即使slot各不相同也不能多核并发。
    helper 不创建task/spmd、不隐藏notify/wait。所有worker都由外层维护调用
    代际；return只有GM别名语义。完整 Indexer 读取需等待key与scale均可见。

数值顺序与边界
    保留当前canonical顺序：FP32→BF16(rint)→FP32，再乘128**(-0.5)，
    row_max(abs)→amax下限→127/amax→recip；INT8经INT32(rint)、FP16(round)、
    INT8(trunc)转换。不能把BF16舍入挪到缩放后。scale最后按原Scalar cast
    转FP16存入Native页；邻近历史scale通过单写者64B RMW保持。
    request_bounds先排除dummy，再枚举真实关闭token并核对causal position。
    compact行上下界先于slot读取，page/offset和页字节容量先于cache写入。
    stale position 的候选行可有确定的中间scale，但不能产生cache副作用。
    有效写slot唯一、Q全局无重叠、工作区容量充足仍由外层保证；越界保护不代表
    不一致metadata具有正确数学含义。这里不含Hadamard、state或完整CSA入口，
    当前源码/CPU编译结果不代替设备精度、图重放或性能验收。
"""

import pypto.language as pl

from ..common.config import INT8_AMAX_EPS, INT8_SCALE_MAX
from .request_geometry import COMPRESS_RATIO, QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

T_DYN = pl.dynamic("CSA_INDEX_CACHE_T")
VALUE_ROWS_DYN = pl.dynamic("CSA_INDEX_CACHE_VALUE_ROWS")
SCALE_ROWS_DYN = pl.dynamic("CSA_INDEX_CACHE_SCALE_ROWS")
SLOT_ROWS_DYN = pl.dynamic("CSA_INDEX_CACHE_SLOT_ROWS")
CACHE_PAGES_DYN = pl.dynamic("CSA_INDEX_CACHE_PAGES")
CACHE_BYTES_DYN = pl.dynamic("CSA_INDEX_CACHE_BYTES")
HEAD_DIM = 128
ROW_TILE = 16
PAGE_TOKENS = 32
KEY_BYTES = PAGE_TOKENS * HEAD_DIM
SCALE_BYTES = PAGE_TOKENS * 2
HADAMARD_SCALE = HEAD_DIM**-0.5


@pl.jit.inline(auto_scope=False)
def indexer_compact_quant_cache_stage(
    values: pl.Tensor[[VALUE_ROWS_DYN, HEAD_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    slots: pl.Tensor[[SLOT_ROWS_DYN, 2], pl.INT32],
    scale_values: pl.InOut[pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32]],
    kv_debug: pl.InOut[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
    cache: pl.InOut[pl.Tensor[[CACHE_PAGES_DYN, CACHE_BYTES_DYN], pl.INT8]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV按请求量化；只把causal且slot合法的关闭行写到Native key。"""
    for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), WORKERS):
        begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
        if active != 0:
            length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
            history_start = length - (end - begin)
            count = length // COMPRESS_RATIO - history_start // COMPRESS_RATIO
            first_token = begin + COMPRESS_RATIO - 1 - history_start % COMPRESS_RATIO
            prefix = pl.cast(pl.read(offsets, [request]), pl.INDEX) + history_start // COMPRESS_RATIO + 1
            # 容量仅作保护；有效范围由Q/L决定，不处理无用pad行。
            if prefix >= 0 and prefix + count <= pl.tensor.dim(values, 0):
                if prefix + count <= pl.tensor.dim(scale_values, 0) and prefix + count <= pl.tensor.dim(slots, 0):
                    for local_first in pl.range(0, count, ROW_TILE):
                        first = prefix + local_first
                        rows = pl.min(ROW_TILE, count - local_first)
                        raw = pl.load(values, [first, 0], [ROW_TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
                        kv_f32 = pl.cast(pl.cast(raw, target_type=pl.BF16, mode="rint"), target_type=pl.FP32)
                        kv_f32 = pl.mul(kv_f32, HADAMARD_SCALE)
                        # 显式Tile归约需UB暂存，仍是同一次128列row_max，未拆归约树。
                        reduce_tmp = pl.create_tile([ROW_TILE, HEAD_DIM], dtype=pl.FP32)
                        amax = pl.reshape(pl.row_max(pl.abs(kv_f32), reduce_tmp), [1, ROW_TILE])
                        # 行归约reshape后有效形状是[1,rows]，常数也须匹配该尾形状；
                        # 只改变UB有效掩码，不能把未加载的尾部当成额外有效行。
                        floor = pl.set_validshape(
                            pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS), 1, rows
                        )
                        numerator = pl.set_validshape(
                            pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), 1, rows
                        )
                        amax = pl.maximum(amax, floor)
                        scale_q_row = pl.div(numerator, amax)
                        scale_dq_col = pl.reshape(pl.recip(scale_q_row), [ROW_TILE, 1])
                        scale_q_col = pl.reshape(scale_q_row, [ROW_TILE, 1])
                        # 相邻请求可由另一核处理；短尾不能按16行覆盖其scale。
                        pl.store(pl.set_validshape(scale_dq_col, rows, 1), [first, 0], scale_values)
                        scaled = pl.row_expand_mul(kv_f32, scale_q_col)
                        quant_i32 = pl.cast(scaled, target_type=pl.INT32, mode="rint")
                        quant_half = pl.cast(quant_i32, target_type=pl.FP16, mode="round")
                        quant_i8 = pl.cast(quant_half, target_type=pl.INT8, mode="trunc")
                        for inner in pl.range(rows):
                            token = first_token + (local_first + inner) * COMPRESS_RATIO
                            position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                            if position == history_start + token - begin:
                                # compact行已在上层检查；只有causal行才读取Native slot。
                                page = pl.cast(pl.read(slots, [first + inner, 0]), pl.INDEX)
                                offset = pl.cast(pl.read(slots, [first + inner, 1]), pl.INDEX)
                                if (
                                    page >= 0
                                    and page < pl.tensor.dim(cache, 0)
                                    and offset >= 0
                                    and offset < PAGE_TOKENS
                                ):
                                    if pl.tensor.dim(cache, 1) >= KEY_BYTES + SCALE_BYTES:
                                        # inner是运行时行号，显式Tile slice并声明一行有效；
                                        # 避免Python下标把动态边界当成静态子Tile要求。
                                        debug_row = pl.slice(
                                            kv_f32, [1, HEAD_DIM], [inner, 0], valid_shape=[1, HEAD_DIM]
                                        )
                                        key_row = pl.slice(
                                            quant_i8, [1, HEAD_DIM], [inner, 0], valid_shape=[1, HEAD_DIM]
                                        )
                                        pl.store(debug_row, [token, 0], kv_debug)
                                        pl.store(key_row, [page, offset * HEAD_DIM], cache)
    return scale_values, kv_debug, cache


@pl.jit.inline(auto_scope=False)
def indexer_compact_scale_commit_stage(
    scale_values: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    slots: pl.Tensor[[SLOT_ROWS_DYN, 2], pl.INT32],
    cache: pl.InOut[pl.Tensor[[CACHE_PAGES_DYN, CACHE_BYTES_DYN], pl.INT8]],
    worker_id: pl.Scalar[pl.INDEX],
):
    """唯一AIV worker0提交FP16 scale；调用前必须等所有量化写回可见。"""
    if worker_id == 0:
        for request in pl.range(pl.tensor.dim(lengths, 0)):
            begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
            if active != 0:
                length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
                history_start = length - (end - begin)
                count = length // COMPRESS_RATIO - history_start // COMPRESS_RATIO
                first_token = begin + COMPRESS_RATIO - 1 - history_start % COMPRESS_RATIO
                prefix = pl.cast(pl.read(offsets, [request]), pl.INDEX) + history_start // COMPRESS_RATIO + 1
                if prefix >= 0 and prefix + count <= pl.tensor.dim(scale_values, 0):
                    if prefix + count <= pl.tensor.dim(slots, 0):
                        for local_row in pl.range(count):
                            token = first_token + local_row * COMPRESS_RATIO
                            position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                            if position == history_start + token - begin:
                                page = pl.cast(pl.read(slots, [prefix + local_row, 0]), pl.INDEX)
                                offset = pl.cast(pl.read(slots, [prefix + local_row, 1]), pl.INDEX)
                                if (
                                    page >= 0
                                    and page < pl.tensor.dim(cache, 0)
                                    and offset >= 0
                                    and offset < PAGE_TOKENS
                                ):
                                    if pl.tensor.dim(cache, 1) >= KEY_BYTES + SCALE_BYTES:
                                        scale = pl.cast(pl.read(scale_values, [prefix + local_row, 0]), pl.FP16)
                                        # 64B块是共享读改写区域，不能凭slot唯一拆到多个核。
                                        scale_bytes = pl.tile.load(cache, [page, KEY_BYTES], [1, SCALE_BYTES])
                                        scale_half = pl.tile.reinterpret_view(scale_bytes, pl.FP16)
                                        pl.tile.write(scale_half, [0, offset], scale)
                                        updated = pl.tile.reinterpret_view(scale_half, pl.INT8)
                                        pl.tile.store(updated, [page, KEY_BYTES], cache)
    return cache
