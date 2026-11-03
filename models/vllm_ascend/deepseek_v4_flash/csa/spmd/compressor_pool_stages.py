# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 的 C4 池化与 Native state 写回：按真实请求区间读取历史和本轮投影。

整体思路与阅读顺序
    1. compressor_pool_stage 为主 CSA 的 512 列 KV 做八位置在线 softmax；
       indexer_compressor_pool_stage 为 Indexer 的 128 列 KV 做相同窗口计算。
       两者保留当前 canonical 的数学块：主分支一次处理 512 列，Indexer
       分两次处理 64 列。函数分开写，便于直接核对各自原代码的数值顺序。
    2. 主/Indexer 的 state_commit_stage 把本轮投影直接写回 Native state。
       它们必须在所有 pool 的历史读取完成后执行，不能合并进请求内循环。
       其他请求可能共享历史页，局部请求读完不代表整层已经不再读该物理页。

输入输出、布局与动态范围
    Q: INT32[B+1] 是本轮 Native query 前缀和，L: INT32[B] 包含本轮 query；
    positions: INT64[T]。B、T、Q 长度独立动态，不以 token//S 或 T//B 反算。
    主投影 kv/score 是 FP32[PAD_MAIN,1024]，Indexer 是 FP32[PAD_IDX,256]；
    两个半区分别保存重叠压缩窗口的前/后四位置，kv/score 两份容量均须>=T；
    不读取投影padding。APE 是 FP32[4,2*head]。
    pool 输出 FP32[POOL_ROWS,head]，POOL_ROWS>=T；仅活动请求的原始 token
    行被写入，非关闭行置零，关闭行得到 pooled 值。inactive/dummy、Q 未覆盖
    区域和容量尾部不读不写，消费者必须按同一 Q/L 仅消费活动关闭行。不能把
    本工作区交给旧版无条件全 T 的 norm：旧版依赖 dummy 行预先清零。
    Native state 是 FP32[pages,page_elements] 连续二维别名，不建立搬运窗口。
    每个逻辑位置包含 [kv的2*head, score的2*head]；state_table 给出历史页，
    state_slots[T,2] 给出本轮写入的物理页/页内位置。页大小是模型配置中的 2。
    所有容量轴保留 pl.dynamic；固定的 head/窗口宽度是模型数学，不是 B/S。

请求分工与同步
    worker_id 是本阶段 AIV 集合内的编号，0<=worker_id<WORKERS，WORKERS>0。
    pl.range(worker_id,B,WORKERS) 分配完整请求；核内按 Q 的实际 query 数循环。
    当前保持原请求所有权以便核对数学，尚未按请求长短优化分核。空核直接返回，
    仍由外层按阶段协议发布完成；helper 内不创建 task/spmd 或 notify/wait。
    pool 先等投影可见；所有 pool 结束并发布后，commit 才能覆盖历史 state。
    返回 Tensor 只是同一 GM 别名，不表示其他核完成。不同 helper 不复用临时
    GM，原 state 与投影/pooled 不能别名，UB tile 只在当前核/当前窗口内存活。

关键数学和边界
    活动请求先检查 Q/L，再计算 H=L-(Q[r+1]-Q[r])；不读取 dummy 的残留位置。
    实际 position 必须等于 H+token-Q[r]，逐行检查只屏蔽该行，不把某个旧行的
    异常位置解释为整个请求失活；合法请求仍须满足全段causal合同。
    窗口按当前位置→其余七位置的原顺序
    累加，保留 mi/li/oi 的 maximum、exp、乘法、加法和最后 div，不能替换为
    八行批量 softmax 或改变归约树。前四位置取投影前半，后四位置取后半。
    历史位置通过 Native 页表读 state；本轮位置直接读投影并加 APE。负历史
    位置使用 value=0/score=负无穷；负页保持原 gather 的 value=0/score=0。
    新增容量检查仅防止非法 metadata 越界，非法页不是已支持的有效数学输入。
    commit 检查活动/causal/页和页内位置后才写；有效写槽唯一、页表和 Q/L
    同轮同序仍是外层合同。本文件只覆盖 pool/commit，不含紧凑 norm/cache、
    完整 CSA 入口或设备数值/性能结论；编译或 CPU 模型不能代替设备验收。
"""

import pypto.language as pl

from ..common.config import C4A_COMPRESSOR_BLOCK_SIZE, FP32_NEG_INF
from .request_geometry import COMPRESS_RATIO, QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

T_DYN = pl.dynamic("CSA_POOL_T")
PROJECTION_ROWS_DYN = pl.dynamic("CSA_POOL_PROJECTION_ROWS")
POOL_ROWS_DYN = pl.dynamic("CSA_POOL_ROWS")
STATE_PAGES_DYN = pl.dynamic("CSA_POOL_STATE_PAGES")
STATE_ELEMENTS_DYN = pl.dynamic("CSA_POOL_STATE_ELEMENTS")
TABLE_COLUMNS_DYN = pl.dynamic("CSA_POOL_TABLE_COLUMNS")
HEAD_DIM = 512
OUT_DIM = 2 * HEAD_DIM
IDX_HEAD_DIM = 128
IDX_OUT_DIM = 2 * IDX_HEAD_DIM
IDX_HEAD_TILE = 64
STATE_LEN = 2 * COMPRESS_RATIO
STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE


@pl.jit.inline(auto_scope=False)
def compressor_pool_stage(
    kv_proj: pl.Tensor[[PROJECTION_ROWS_DYN, OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[PROJECTION_ROWS_DYN, OUT_DIM], pl.FP32],
    state: pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32],
    state_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    pooled: pl.InOut[pl.Tensor[[POOL_ROWS_DYN, HEAD_DIM], pl.FP32]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV 主压缩：请求内串行处理关闭窗口，整个 512 列保持原计算粒度。"""
    for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), WORKERS):
        begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
        if active != 0:
            first_position = pl.cast(pl.read(lengths, [request]), pl.INDEX) - (end - begin)
            for token in pl.range(begin, end):
                # 同一请求只有一个写者，初始化与后续覆盖均在该核有序执行。
                pl.store(pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0), [token, 0], pooled)
                position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                if position == first_position + token - begin and (position + 1) % COMPRESS_RATIO == 0:
                    window_start = position - STATE_LEN + 1
                    last_ape = position % COMPRESS_RATIO
                    # 以当前位置后半区起步；不能把它移到七项归并之后。
                    mi = pl.add(
                        pl.load(score_proj, [token, HEAD_DIM], [1, HEAD_DIM]),
                        pl.load(ape, [last_ape, HEAD_DIM], [1, HEAD_DIM]),
                    )
                    li = pl.exp(pl.sub(mi, mi))
                    oi = pl.load(kv_proj, [token, HEAD_DIM], [1, HEAD_DIM])
                    for state_index in pl.range(STATE_LEN - 1):
                        logical_position = window_start + state_index
                        value = pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                        score = pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=FP32_NEG_INF)
                        half = 0
                        if state_index >= COMPRESS_RATIO:
                            half = HEAD_DIM
                        if logical_position >= 0 and logical_position < first_position:
                            # 历史负页原来是零行，不得误当成负无穷 mask 改变 softmax。
                            score = pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                            table_column = logical_position // STATE_BLOCK_SIZE
                            if table_column < pl.tensor.dim(state_table, 1):
                                page = pl.cast(pl.read(state_table, [request, table_column]), pl.INDEX)
                                column = logical_position % STATE_BLOCK_SIZE * (2 * OUT_DIM) + half
                                if page >= 0 and page < pl.tensor.dim(state, 0):
                                    if column + OUT_DIM + HEAD_DIM <= pl.tensor.dim(state, 1):
                                        value = pl.load(state, [page, column], [1, HEAD_DIM])
                                        score = pl.load(state, [page, column + OUT_DIM], [1, HEAD_DIM])
                        if logical_position >= first_position and logical_position <= position:
                            # 请求局部坐标映射回 TND；不会跨到前后请求的投影行。
                            overlay_token = begin + logical_position - first_position
                            ape_row = logical_position % COMPRESS_RATIO
                            value = pl.load(kv_proj, [overlay_token, half], [1, HEAD_DIM])
                            score = pl.add(
                                pl.load(score_proj, [overlay_token, half], [1, HEAD_DIM]),
                                pl.load(ape, [ape_row, half], [1, HEAD_DIM]),
                            )
                        mi_next = pl.maximum(mi, score)
                        alpha = pl.exp(pl.sub(mi, mi_next))
                        beta = pl.exp(pl.sub(score, mi_next))
                        li = pl.add(pl.mul(alpha, li), beta)
                        oi = pl.add(pl.mul(oi, alpha), pl.mul(value, beta))
                        mi = mi_next
                    pl.store(pl.div(oi, li), [token, 0], pooled)
    return pooled


@pl.jit.inline(auto_scope=False)
def indexer_compressor_pool_stage(
    kv_proj: pl.Tensor[[PROJECTION_ROWS_DYN, IDX_OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[PROJECTION_ROWS_DYN, IDX_OUT_DIM], pl.FP32],
    state: pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32],
    state_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    ape: pl.Tensor[[COMPRESS_RATIO, IDX_OUT_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    pooled: pl.InOut[pl.Tensor[[POOL_ROWS_DYN, IDX_HEAD_DIM], pl.FP32]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV Indexer 压缩：补齐 Q/L 活动判定，保留两个 64 列窗口的数值顺序。"""
    for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), WORKERS):
        begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
        if active != 0:
            first_position = pl.cast(pl.read(lengths, [request]), pl.INDEX) - (end - begin)
            for token in pl.range(begin, end):
                pl.store(pl.tile.full([1, IDX_HEAD_DIM], dtype=pl.FP32, value=0.0), [token, 0], pooled)
                position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                if position == first_position + token - begin and (position + 1) % COMPRESS_RATIO == 0:
                    window_start = position - STATE_LEN + 1
                    # 每次完整执行一个列片的七次在线归并；这是原128列算法的分块。
                    for h0 in pl.range(0, IDX_HEAD_DIM, IDX_HEAD_TILE):
                        last_ape = position % COMPRESS_RATIO
                        mi = pl.add(
                            pl.load(score_proj, [token, IDX_HEAD_DIM + h0], [1, IDX_HEAD_TILE]),
                            pl.load(ape, [last_ape, IDX_HEAD_DIM + h0], [1, IDX_HEAD_TILE]),
                        )
                        li = pl.exp(pl.sub(mi, mi))
                        oi = pl.load(kv_proj, [token, IDX_HEAD_DIM + h0], [1, IDX_HEAD_TILE])
                        for state_index in pl.range(STATE_LEN - 1):
                            logical_position = window_start + state_index
                            value = pl.tile.full([1, IDX_HEAD_TILE], dtype=pl.FP32, value=0.0)
                            score = pl.tile.full([1, IDX_HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            half = 0
                            if state_index >= COMPRESS_RATIO:
                                half = IDX_HEAD_DIM
                            if logical_position >= 0 and logical_position < first_position:
                                score = pl.tile.full([1, IDX_HEAD_TILE], dtype=pl.FP32, value=0.0)
                                table_column = logical_position // STATE_BLOCK_SIZE
                                if table_column < pl.tensor.dim(state_table, 1):
                                    page = pl.cast(pl.read(state_table, [request, table_column]), pl.INDEX)
                                    column = logical_position % STATE_BLOCK_SIZE * (2 * IDX_OUT_DIM) + half + h0
                                    if page >= 0 and page < pl.tensor.dim(state, 0):
                                        if column + IDX_OUT_DIM + IDX_HEAD_TILE <= pl.tensor.dim(state, 1):
                                            value = pl.load(state, [page, column], [1, IDX_HEAD_TILE])
                                            score = pl.load(state, [page, column + IDX_OUT_DIM], [1, IDX_HEAD_TILE])
                            if logical_position >= first_position and logical_position <= position:
                                overlay_token = begin + logical_position - first_position
                                ape_row = logical_position % COMPRESS_RATIO
                                value = pl.load(kv_proj, [overlay_token, half + h0], [1, IDX_HEAD_TILE])
                                score = pl.add(
                                    pl.load(score_proj, [overlay_token, half + h0], [1, IDX_HEAD_TILE]),
                                    pl.load(ape, [ape_row, half + h0], [1, IDX_HEAD_TILE]),
                                )
                            mi_next = pl.maximum(mi, score)
                            alpha = pl.exp(pl.sub(mi, mi_next))
                            beta = pl.exp(pl.sub(score, mi_next))
                            li = pl.add(pl.mul(alpha, li), beta)
                            oi = pl.add(pl.mul(oi, alpha), pl.mul(value, beta))
                            mi = mi_next
                        pl.store(pl.div(oi, li), [token, h0], pooled)
    return pooled


@pl.jit.inline(auto_scope=False)
def compressor_state_commit_stage(
    kv_proj: pl.Tensor[[PROJECTION_ROWS_DYN, OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[PROJECTION_ROWS_DYN, OUT_DIM], pl.FP32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    state_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    state: pl.InOut[pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV 主 state 写回；外层先等待所有读取旧 state 的 pool，不只等本核。"""
    for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), WORKERS):
        begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
        if active != 0:
            first_position = pl.cast(pl.read(lengths, [request]), pl.INDEX) - (end - begin)
            for token in pl.range(begin, end):
                position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                if position == first_position + token - begin:
                    page = pl.cast(pl.read(state_slots, [token, 0]), pl.INDEX)
                    offset = pl.cast(pl.read(state_slots, [token, 1]), pl.INDEX)
                    if page >= 0 and page < pl.tensor.dim(state, 0) and offset >= 0 and offset < STATE_BLOCK_SIZE:
                        column = offset * (2 * OUT_DIM)
                        if column + 2 * OUT_DIM <= pl.tensor.dim(state, 1):
                            # score 存入时就加 APE；后续历史消费不得再次加同一项。
                            ape_row = position % COMPRESS_RATIO
                            value = pl.load(kv_proj, [token, 0], [1, OUT_DIM])
                            score = pl.add(
                                pl.load(score_proj, [token, 0], [1, OUT_DIM]), pl.load(ape, [ape_row, 0], [1, OUT_DIM])
                            )
                            pl.store(value, [page, column], state)
                            pl.store(score, [page, column + OUT_DIM], state)
    return state


@pl.jit.inline(auto_scope=False)
def indexer_compressor_state_commit_stage(
    kv_proj: pl.Tensor[[PROJECTION_ROWS_DYN, IDX_OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[PROJECTION_ROWS_DYN, IDX_OUT_DIM], pl.FP32],
    ape: pl.Tensor[[COMPRESS_RATIO, IDX_OUT_DIM], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    state_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    state: pl.InOut[pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV Indexer state 写回；使用本组 Native slots，不能与主 state 混用。"""
    for request in pl.range(worker_id, pl.tensor.dim(lengths, 0), WORKERS):
        begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
        if active != 0:
            first_position = pl.cast(pl.read(lengths, [request]), pl.INDEX) - (end - begin)
            for token in pl.range(begin, end):
                position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                if position == first_position + token - begin:
                    page = pl.cast(pl.read(state_slots, [token, 0]), pl.INDEX)
                    offset = pl.cast(pl.read(state_slots, [token, 1]), pl.INDEX)
                    if page >= 0 and page < pl.tensor.dim(state, 0) and offset >= 0 and offset < STATE_BLOCK_SIZE:
                        column = offset * (2 * IDX_OUT_DIM)
                        if column + 2 * IDX_OUT_DIM <= pl.tensor.dim(state, 1):
                            ape_row = position % COMPRESS_RATIO
                            value = pl.load(kv_proj, [token, 0], [1, IDX_OUT_DIM])
                            score = pl.add(
                                pl.load(score_proj, [token, 0], [1, IDX_OUT_DIM]),
                                pl.load(ape, [ape_row, 0], [1, IDX_OUT_DIM]),
                            )
                            pl.store(value, [page, column], state)
                            pl.store(score, [page, column + IDX_OUT_DIM], state)
    return state
