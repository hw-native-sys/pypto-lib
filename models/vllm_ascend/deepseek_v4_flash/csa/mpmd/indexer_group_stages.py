# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA Indexer 的请求内分组、叶计划和 FP16 head 系数。

整体流程与阅读顺序
    group_plan_stage 单写者扫描 Native Q/L/positions，产生组前缀、组描述符和计划；
    decode_group / query_visibility 定义 AIC、两个 AIV 与最终归并共同使用
    的 query 范围和可见历史；work_item / leaf_range 定义工作顺序与分片；
    head_coefficients_stage 最后在 AIV 上生成 Cube head 归约所用对角矩阵。
    本文件没有 task/spmd/跨核通知，完整入口负责统一调度。它不执行 QK、
    FFTS 握手或最终 Top-K 发布，不能独立代替整个 Indexer。

布局、动态范围与策略
    Q=INT32[Q_ROWS]、L=INT32[B]、positions=INT64[T] 都是 Native device
    Tensor。Cube路径每个请求单独切 G=2 或 G=6 的小组，尾组 live=min(G,end-first)，
    不跨请求拼组、不使用 token//S。G 是当前 Cube 资源允许的策略常量，
    不是请求长度白名单；任意正 query 长度可以切成多个组。策略选择仍由
    上层完成，不能由最终 merge 按 T 再猜一遍 G 或根布局。
    更短历史的Vector路径可只复用Scalar几何，以G=1、roots_per_leaf=0
    标识直接单叶score发布，不调用本文件的Cube系数；其数学与流水另行实现。
    group_offsets=INT32[OFFSET_ROWS]，前B+1项保存各请求的组前缀，向上对齐
    16项后接至多T组的四项(request,first,live,max_visible)设备描述符；
    所需容量为pad16(B+1)+4*T。尾容量不读取，不能把上轮描述符当作本轮组。
    plan=INT32[8] 保存 groups/max_history/leaves/leaf_tiles/extra/G/balanced/
    roots_per_leaf。最大历史只决定循环上界，单 query 的 visible 还要结合
    真实位置。叶单位为1024个候选，常规8192；可选平衡方案保留原24核逻辑。
    模型最大候选数、64 heads、16个系数行是模型/矩阵 tile 常量，不是 B/T。
    scale 二维别名 FP32[SCALE_ROWS,64] 与 weights FP32[WEIGHT_ROWS,64]
    的有效前缀均须覆盖 T；外层直接绑定已有scale存储，不搬运或回读数值。
    coefficients FP16[COEFF_ROWS,COEFF_COLS] 要求 cols=G*64、rows>=16*groups；
    只写真实组的完整16行，其余容量保持不变。入口必须统一检查这些容量。

数值、分工与工作区寿命
    原 Native Cube 路径先分别将 query scale、head weight 转 FP16(rint)，
    再在 FP16 相乘；不得先乘 FP32 再转。每 query 的64项系数写入16×(G*64)
    矩阵的对角块，其他位置为零。使用等宽[16*G,64] UB行再reshape，避免
    A3非整列TMOV限制。尾组/失效位置的整块系数为零，且不读取其 payload。
    group%WORKERS 决定系数块唯一 AIV writer；消费者从四项描述符直接取组，
    不再对每叶重复扫描全部请求。描述符只缓存当前组几何，不替代逐lane的
    causal遮罩，不新增独立GM Tensor。计划只由所选集合 worker0
    写一次，不能同时让 AIC0 和 AIV0 写。其他角色先等待计划发布，再读取；
    系数还需等待 query scale/weights。helper 返回不代表跨核缓存已可见。
    Q/L/positions/group_offsets 必须来自同一轮且在整个 score 阶段保持不变；
    描述符由同一Scalar唯一写者写出，发布前清理DCache并完成DSB，消费者
    在既有PHASE_META等待后获取。相邻描述符可以共用缓存行，不能多核同时写。
    coeff 不能在 Cube 消费完之前复用。普通阶段通知不提供逐槽消费 ACK。

有效性与验证边界
    request_bounds 负责请求有效性；positions 必须等于 H+(query-begin)，
    其中 H=L-query_len。陈旧位置只被屏蔽，不被当作另一个合法 decode 场景。
    原始全局 Q 单调、不重叠及页表一致性仍是入口合同。正常组的计算顺序
    与原系数相同；新增 mixed/odd 分组及其设备数值/性能需要单独验证。

B1原半叶调度
    BALANCED且GROUP_SIZE=2表示把原8192叶的两个半叶独立分派。末叶
    半跨度为min(ceil(original_valid/1024)*512,4096)，来自本轮device
    visible；不能统一成4096。保留末尾空根及原前后半叶顺序，供最终归并
    精确复现旧同分规则。GROUP_SIZE=6仍使用原1024粒度平衡叶计划。
"""

import pypto.language as pl

from ..common.config import FLASH as M
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

TOKENS_DYN = pl.dynamic("CSA_INDEX_GROUP_T")
OFFSET_ROWS_DYN = pl.dynamic("CSA_INDEX_GROUP_OFFSETS")
SCALE_ROWS_DYN = pl.dynamic("CSA_INDEX_GROUP_SCALE_ROWS")
WEIGHT_ROWS_DYN = pl.dynamic("CSA_INDEX_GROUP_WEIGHT_ROWS")
COEFF_ROWS_DYN = pl.dynamic("CSA_INDEX_GROUP_COEFF_ROWS")
COEFF_COLS_DYN = pl.dynamic("CSA_INDEX_GROUP_COEFF_COLS")
HEADS = M.index_n_heads
COEFFICIENT_ROWS = 16
COMPRESS_RATIO = 4
CANDIDATES_PER_LEAF = 8192
SCORE_TILE = 1024
MAX_CANDIDATES = M.max_position_embeddings // COMPRESS_RATIO
MAX_LEAVES = (MAX_CANDIDATES + CANDIDATES_PER_LEAF - 1) // CANDIDATES_PER_LEAF
SCORE_WORKERS = 24
PLAN_WORDS = 8
GROUP_DESCRIPTOR_WORDS = 4
GROUP_DESCRIPTOR_ALIGNMENT = 16


@pl.jit.inline(auto_scope=False)
def indexer_long_leaf_plan(
    max_cache_count: pl.Scalar[pl.INDEX],
    query_groups: pl.Scalar[pl.INDEX],
):
    """沿用原1024粒度平衡公式；组数来自实际请求，不按固定S反算。"""
    tile_count = (max_cache_count + SCORE_TILE - 1) // SCORE_TILE
    leaf_count = pl.max((max_cache_count + CANDIDATES_PER_LEAF - 1) // CANDIDATES_PER_LEAF, 1)
    leaf_tiles = CANDIDATES_PER_LEAF // SCORE_TILE
    extra_leaves = 0
    leaf_multiple = 1
    if 3 * query_groups == 2 * SCORE_WORKERS:  # noqa: SIM114 - 保持原PyPTO显式Scalar分支
        leaf_multiple = 3
    elif 3 * query_groups == SCORE_WORKERS:
        leaf_multiple = 3
    elif 6 * query_groups == SCORE_WORKERS:
        leaf_multiple = 6
    if leaf_multiple > 1:
        balanced_count = (leaf_count + leaf_multiple - 1) // leaf_multiple * leaf_multiple
        if balanced_count <= MAX_LEAVES:
            leaf_count = balanced_count
            leaf_tiles = pl.max(tile_count // balanced_count, 1)
            extra_leaves = tile_count % balanced_count
    return leaf_count, leaf_tiles, extra_leaves


@pl.jit.inline(auto_scope=False)
def indexer_group_plan_stage(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    plan: pl.InOut[pl.Tensor[[PLAN_WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    GROUP_SIZE: pl.constexpr,
    BALANCED: pl.constexpr,
    ROOTS_PER_LEAF: pl.constexpr,
):
    """worker0写组前缀、描述符和叶计划；容量pad16(B+1)+4*T由入口保证。"""
    if worker == 0:
        groups = pl.cast(0, pl.INDEX)
        max_history = pl.cast(0, pl.INDEX)
        requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
        descriptor_base = (
            (requests + 1 + GROUP_DESCRIPTOR_ALIGNMENT - 1) // GROUP_DESCRIPTOR_ALIGNMENT * GROUP_DESCRIPTOR_ALIGNMENT
        )
        # 前缀有跨请求依赖，只扫描一次。描述符也在这名早期AIV上生成，
        # 消费者按组号读四个Scalar，避免AIC/双AIV对每个叶重复O(B)定位。
        for request in pl.range(requests):
            pl.write(offsets, [request], pl.cast(groups, pl.INT32))
            begin, end, active = request_bounds(bounds, lengths, request, tokens)
            if active != 0:
                length = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
                max_history = pl.max(max_history, pl.min(length // COMPRESS_RATIO, MAX_CANDIDATES))
                group_count = (end - begin + GROUP_SIZE - 1) // GROUP_SIZE
                history = length - (end - begin)
                for local_group in pl.range(group_count):
                    first = begin + local_group * GROUP_SIZE
                    live = pl.min(GROUP_SIZE, end - first)
                    visible = pl.cast(0, pl.INDEX)
                    # 真实位置逐lane校验；尾lane或陈旧位置不能扩大组可见上界。
                    # 这只是组循环上界，Score/系数仍保留各query自己的causal检查。
                    for lane in pl.range(live):
                        query = first + lane
                        position = pl.cast(pl.tensor.read(positions, [query]), pl.INDEX)
                        if position == history + query - begin:
                            lane_visible = pl.max(
                                pl.min(
                                    pl.min(length // COMPRESS_RATIO, (position + 1) // COMPRESS_RATIO), MAX_CANDIDATES
                                ),
                                0,
                            )
                            visible = pl.max(visible, lane_visible)
                    row = descriptor_base + (groups + local_group) * GROUP_DESCRIPTOR_WORDS
                    pl.write(offsets, [row], pl.cast(request, pl.INT32))
                    pl.write(offsets, [row + 1], pl.cast(first, pl.INT32))
                    pl.write(offsets, [row + 2], pl.cast(live, pl.INT32))
                    pl.write(offsets, [row + 3], pl.cast(visible, pl.INT32))
                groups = groups + group_count
        # 末端索引显式转成INDEX；保持descriptor动态维度，避免后端把裸dim
        # 当成缺失SSA的Scalar store下标，不读取任何Host数据。
        pl.write(offsets, [requests], pl.cast(groups, pl.INT32))
        leaves = pl.max((max_history + CANDIDATES_PER_LEAF - 1) // CANDIDATES_PER_LEAF, 1)
        leaf_tiles = pl.cast(CANDIDATES_PER_LEAF // SCORE_TILE, pl.INDEX)
        extra = pl.cast(0, pl.INDEX)
        if BALANCED:
            if GROUP_SIZE == 2:
                # B1把原每个8192叶的两个半叶独立调度，包含尾部空根。
                # 末叶边界依赖各组实际visible，不能把全部半叶静态等分为4096。
                leaf_tiles = pl.cast(CANDIDATES_PER_LEAF // (2 * SCORE_TILE), pl.INDEX)
                leaves = 2 * pl.max((max_history + CANDIDATES_PER_LEAF - 1)
                                    // CANDIDATES_PER_LEAF, 1)
            else:
                leaves, leaf_tiles, extra = indexer_long_leaf_plan(max_history, groups)
        pl.write(plan, [0], pl.cast(groups, pl.INT32))
        pl.write(plan, [1], pl.cast(max_history, pl.INT32))
        pl.write(plan, [2], pl.cast(leaves, pl.INT32))
        pl.write(plan, [3], pl.cast(leaf_tiles, pl.INT32))
        pl.write(plan, [4], pl.cast(extra, pl.INT32))
        pl.write(plan, [5], pl.cast(GROUP_SIZE, pl.INT32))
        pl.write(plan, [6], pl.cast(BALANCED, pl.INT32))
        pl.write(plan, [7], pl.cast(ROOTS_PER_LEAF, pl.INT32))
    return offsets, plan


@pl.jit.inline(auto_scope=False)
def indexer_query_visibility(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    request: pl.Scalar[pl.INDEX],
    query: pl.Scalar[pl.INDEX],
):
    """返回visible/causal；零历史的合法query仍causal=1，不能和dummy混淆。"""
    visible = pl.cast(0, pl.INDEX)
    causal = pl.cast(0, pl.INT32)
    begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
    if active != 0 and query >= begin and query < end:
        length = pl.cast(pl.tensor.read(lengths, [request]), pl.INDEX)
        position = pl.cast(pl.tensor.read(positions, [query]), pl.INDEX)
        if position == length - (end - begin) + query - begin:
            causal = pl.cast(1, pl.INT32)
            visible = pl.max(
                pl.min(pl.min(length // COMPRESS_RATIO, (position + 1) // COMPRESS_RATIO), MAX_CANDIDATES), 0
            )
    return visible, causal


@pl.jit.inline(auto_scope=False)
def indexer_decode_group(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    group: pl.Scalar[pl.INDEX],
    GROUP_SIZE: pl.constexpr,
):
    """从本轮已发布的组描述符返回几何；越界组不读取未初始化的容量尾部。"""
    owner = pl.cast(-1, pl.INDEX)
    first = pl.cast(0, pl.INDEX)
    live = pl.cast(0, pl.INDEX)
    visible = pl.cast(0, pl.INDEX)
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    groups = pl.cast(pl.read(offsets, [requests]), pl.INDEX)
    if group >= 0 and group < groups:
        descriptor_base = (
            (requests + 1 + GROUP_DESCRIPTOR_ALIGNMENT - 1) // GROUP_DESCRIPTOR_ALIGNMENT * GROUP_DESCRIPTOR_ALIGNMENT
        )
        row = descriptor_base + group * GROUP_DESCRIPTOR_WORDS
        owner = pl.cast(pl.read(offsets, [row]), pl.INDEX)
        first = pl.cast(pl.read(offsets, [row + 1]), pl.INDEX)
        live = pl.cast(pl.read(offsets, [row + 2]), pl.INDEX)
        visible = pl.cast(pl.read(offsets, [row + 3]), pl.INDEX)
    return owner, first, live, visible


@pl.jit.inline(auto_scope=False)
def indexer_work_item(
    item: pl.Scalar[pl.INDEX],
    groups: pl.Scalar[pl.INDEX],
    leaves: pl.Scalar[pl.INDEX],
):
    """少组时先铺满不同组，多组时同组连续叶；空输入返回-1而不除零。"""
    group = pl.cast(-1, pl.INDEX)
    leaf = pl.cast(-1, pl.INDEX)
    if groups > 0 and leaves > 0 and item >= 0 and item < groups * leaves:
        if groups < SCORE_WORKERS:
            group = item % groups
            leaf = item // groups
        else:
            group = item // leaves
            leaf = item % leaves
    return group, leaf


@pl.jit.inline(auto_scope=False)
def indexer_leaf_range(
    leaf: pl.Scalar[pl.INDEX],
    leaf_tiles: pl.Scalar[pl.INDEX],
    extra: pl.Scalar[pl.INDEX],
    visible: pl.Scalar[pl.INDEX],
    BALANCED: pl.constexpr,
    GROUP_SIZE: pl.constexpr,
):
    """普通叶沿1024分片；B1逐项复现原末叶半跨度，再返回begin/capacity/valid。"""
    if BALANCED and GROUP_SIZE == 2:
        original_begin = (leaf // 2) * CANDIDATES_PER_LEAF
        original_valid = pl.max(pl.min(visible - original_begin, CANDIDATES_PER_LEAF), 0)
        original_tiles = (original_valid + SCORE_TILE - 1) // SCORE_TILE
        capacity = pl.min(original_tiles * (SCORE_TILE // 2), CANDIDATES_PER_LEAF // 2)
        begin = original_begin + (leaf % 2) * capacity
    else:
        begin = (leaf * leaf_tiles + pl.min(leaf, extra)) * SCORE_TILE
        capacity = (leaf_tiles + pl.cast(leaf < extra, pl.INDEX)) * SCORE_TILE
    valid = pl.max(pl.min(visible - begin, capacity), 0)
    return begin, capacity, valid


@pl.jit.inline(auto_scope=False)
def indexer_head_coefficients_stage(
    scales: pl.Tensor[[SCALE_ROWS_DYN, HEADS], pl.FP32],
    weights: pl.Tensor[[WEIGHT_ROWS_DYN, HEADS], pl.FP32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    GROUP_SIZE: pl.constexpr,
):
    """AIV单写者生成每组完整对角系数块；输入scale/weight须已发布且覆盖T。"""
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    groups = pl.cast(pl.read(offsets, [requests]), pl.INDEX)
    # 前缀与描述符已由唯一planner发布；组号条带本身保证唯一writer。
    for group in pl.range(worker, groups, WORKERS):
        request, first, live, _visible = indexer_decode_group(bounds, lengths, positions, offsets, group, GROUP_SIZE)
        coefficient_rows = pl.tile.full([COEFFICIENT_ROWS * GROUP_SIZE, HEADS], dtype=pl.FP16, value=0.0)
        # 固定G只决定矩阵tile；尾lane分支先于任何输入load，非法
        # metadata也不把旧scale/weights误放到对角线上。
        for lane in pl.unroll(GROUP_SIZE):
            if lane < live:
                _visible, causal = indexer_query_visibility(bounds, lengths, positions, request, first + lane)
                if causal != 0:
                    query_scales = pl.load(scales, [first + lane, 0], [1, HEADS])
                    query_weights = pl.load(weights, [first + lane, 0], [1, HEADS])
                    query_scale_half = pl.cast(query_scales, pl.FP16, mode="rint")
                    query_weight_half = pl.cast(query_weights, pl.FP16, mode="rint")
                    head_coefficients = pl.mul(query_scale_half, query_weight_half)
                    coefficient_rows = pl.tile.assemble(
                        coefficient_rows, head_coefficients, [lane * (GROUP_SIZE + 1), 0]
                    )
        # reshape只作用于本核UB；完整零填矩阵一次发布，不改GM布局。
        coefficient_matrix = pl.tile.reshape(coefficient_rows, [COEFFICIENT_ROWS, GROUP_SIZE * HEADS])
        coefficients = pl.store(coefficient_matrix, [group * COEFFICIENT_ROWS, 0], coefficients)
    return coefficients
