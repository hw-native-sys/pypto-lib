# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 的原始 KV 页写回：直接使用 Native slot 和 cache。

输入输出及布局
    kv 是前一阶段得到的 BF16[T,512]，slots 是 Native INT32[T,2]，两列
    分别是 page 和页内 offset。cache_rows 是原始 BF16[页数,32,1,512]
    cache 的二维连续 GM 别名 [页数*32,512]；外层只绑定描述符、不搬运数据。
    T 与 cache 行数都是 pl.dynamic，页大小 32 和 head_dim 512 是模型合同。
    cache 总行数必须为 32 的整数倍；二维别名须由外层在 incore 之外绑定。
    slot 的两列来自 device Tensor，既不是 Host 的两个整数，也不是 token
    在页表中的隐含连续编号；本函数不将位置或请求号换算为 slot。

算法与 AIV 分工
    按原多 task 数学的八个 token 一块，由显式 worker 按 WORKERS 条带
    领取块；每块按实际剩余行数逐行写入 slot 指定的页。tile 是工作划分，
    不会把离散 slot 误当作连续八行 cache 写。token t 的唯一写者是
    (t//8)%WORKERS；该编号指写回 worker，不等于前一段 KV RMS 的 worker。
    读 slot 后先验证页号和页内
    offset，再加载对应 KV；负数 dummy 与超容量位置均不访问 cache。

同步、所有权与生命周期
    此 helper 在外层单 SPMD 的 AIV 内 inline，没有派发或独立完成信号。
    调用前必须等待本核使用的 KV 生产者；返回后由外层完成写回/缓存发布，
    attention 读页前等待对应的写回生产者，不能只等待更早的 KV 计算完成。
    本段不使用额外 GM 工作区或 Host metadata；单行 BF16 拷贝不改变数值或
    舍入次序。kv 与 cache 不得重叠到会覆盖其它 token 尚未读取的源行。
    Native 合同要求本次所有有效 slot 唯一，dummy 的 slot 为负；共享 prefix
    可以读取，但不能让两个活动 token 并发覆盖同一个写槽。本函数不做全卡
    去重，也不从 B/S 猜请求归属。若 dummy 保留有效 slot，须先在上游修复
    metadata 合同，不能因输出形状相同就把它当成有效请求。

阅读顺序与当前边界
    唯一入口 original_kv_writeback_stage：动态块数→核内领取→slot 检查→
    单行写回。它只写原始 KV 页；压缩 cache、Indexer INT8/scale 和 state
    由各自消费者处理。CPU 编译不证明整层设备同步、prefix 生命周期或性能。
"""

import pypto.language as pl

T_DYN = pl.dynamic("CSA_CACHE_WRITE_T")
CACHE_ROWS_DYN = pl.dynamic("CSA_ORIGINAL_CACHE_ROWS")
BLOCK_SIZE = 32
HEAD_DIM = 512
WRITE_TILE = 8


@pl.jit.inline(auto_scope=False)
def original_kv_writeback_stage(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS_DYN, HEAD_DIM], pl.BF16]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV 按真实 T 写 Native 原始页；WORKERS>0，worker_id∈[0,WORKERS)。"""
    tokens = pl.tensor.dim(kv, 0)
    cache_pages = pl.tensor.dim(cache_rows, 0) // BLOCK_SIZE
    blocks = (tokens + WRITE_TILE - 1) // WRITE_TILE
    # 编号是当前单 SPMD 内的工作块，不产生新 task；向上取整保留 T%8 尾块。
    for block in pl.range(worker_id, blocks, WORKERS):
        first = block * WRITE_TILE
        # 八行块只是分工单位：尾块只迭代真实剩余行；每行独立解析离散目标。
        # 无 token 可领的 worker 完全不读 slot，仍由外层决定是否发布空阶段完成。
        for row_in_block in pl.range(pl.min(WRITE_TILE, tokens - first)):
            token = first + row_in_block
            page = pl.cast(pl.read(slots, [token, 0]), pl.INDEX)
            offset = pl.cast(pl.read(slots, [token, 1]), pl.INDEX)
            # 页和页内偏移分别检查，防止 offset>=32 悄悄写到相邻合法页。
            # cache_rows 的总行数必须由外层保证是 BLOCK_SIZE 的整数倍。
            if page >= 0 and page < cache_pages and offset >= 0 and offset < BLOCK_SIZE:
                target = page * BLOCK_SIZE + offset
                cache_rows[target : target + 1, 0:HEAD_DIM] = kv[token : token + 1, 0:HEAD_DIM]
    return cache_rows
