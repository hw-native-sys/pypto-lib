# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""动态 CSA Indexer 的最终 Top-512 发布：统一计划、真实请求边界与原排序顺序。

整体流程和阅读顺序
    pair 路径先读 indexer_visible_root_count，再读 indexer_pair_publish_stage。
    上游每叶已产生一个或两个512项根；本模块按原顺序合并，拆出score/index。
    short-vector 路径读取 score_arena[query,:]，由文件末尾的单叶排序数学直接
    发布。两种路径由明确策略选择，不依据总T、固定S或历史测试档位猜测。
    本文件只执行最终AIV排序/发布，未实现score/QK，也不是完整CSA入口。

输入输出和动态布局
    原始 Native Q/L/positions 给出实际请求边界、总KV长度和因果位置；Q与B
    独立动态，T来自positions。pair在EARLY_ATTENTION_GEOMETRY开启时，
    读取本轮前段已验证的请求/可见范围，
    暂存在attention_indices行前缀；原ROPE phase负责发布，本行Top-K完成
    发布后plan才覆盖该前缀。关闭时在本阶段按原规则验证Q/L/位置，
    不读取这份前缀。输出scores和indices
    均为[T,512]且T相同，不用暂存容量反推实际query数。
    pair_arena FP32[PAIR_ROWS,1024] 每项是交错score/index位模式；每query
    固定64行跨度，即 MAX_LEAVES*2，与本轮叶数、G、roots_per_leaf无关。
    ROOTS_PER_LEAF=2：slot=q*64+leaf*2+lane；=1：slot=q*64+leaf。
    short score_arena FP32[SCORE_ROWS,8192] 直接按query取行，不是score worker
    的循环临时行。PAIR_ROWS>=T*64、SCORE_ROWS>=T均由外层统一验证。

统一计划和策略边界
    plan INT32[8] 与score生产者同一轮共享，依次保存 groups/max_history/
    leaves/leaf_tiles/extra/G/balanced/roots_per_leaf。BALANCED和ROOTS_PER_LEAF
    为调用者选择的常量，必须等于plan[6:8]。平衡叶按1024候选tile划分；常规
    叶为8192候选。每query只归并其visible真正覆盖的叶，不能把容量当根数。
    pair只接受roots=1/2；short-vector以roots=0、balanced=0、leaves=1明确
    标记，允许G=1的几何计划。G不参与最终排序，不能再从T推导它。

分工、同步与生命周期
    worker按query编号轮转，owner=query%WORKERS；每行恰好一个AIV写者。
    helper不创建task/spmd。plan、Q/L/position在调用前已就绪；pair路径每条
    query在读取根前只等待其真正的Score写者，随后获取业务缓存。其它query
    可以继续生产，不将48个AIV的全部Score完成设成统一集合点。
    非balanced路径逐根调用二路归并并原位更新本query根0，因此pair为InOut，
    不能与读取旧根0的其它消费者并发。balanced路径只读根；两者均不读本轮
    visible之外的未写根。阶段通知只能表示完成，不等于循环临时槽消费ACK。

数值、无效行与尾部
    保留canonical的sort32/mrgsort调用、二路incoming优先顺序、多路后根优先
    顺序及2048/4096/8192单叶分支，避免相同score时index顺序变化。失活、无
    唯一owner、陈旧causal或visible=0时，整行score写FP32_NEG_INF、index写-1，
    不读陈旧arena。可见候选少于512时，indices不足部分明确填-1；有效score
    的未满部分沿用原排序pad语义，不另换分数或排序算法。
    计划/容量不合法的防越界分支同样写空行，但这不是异常输入支持承诺；外层
    仍须统一拒绝不匹配策略和容量。本模块无Host数值回读，不使用冻结IR。

B1半叶根
    balanced/G2的根仍对应原8192叶的前/后半叶，包含最后的空后半根。
    累计根留在UB，逐child按incoming优先左折叠，只在末尾写根0；不能
    替换成balanced/G6使用的多路归并树。根数和实际等待写者来自同一
    device计划，保留本轮代际及业务数据发布/获取。
"""

import pypto.language as pl

from ..common.config import FP32_NEG_INF
from .attention_dependencies import wait_indexer_query_roots
from .attention_plan_stages import (
    EARLY_QUERY_BEGIN_COL,
    EARLY_QUERY_OWNER_COL,
    EARLY_QUERY_VISIBLE_COL,
    PLAN_CAP_DYN,
)
from .indexer_group_stages import (
    CANDIDATES_PER_LEAF,
    MAX_CANDIDATES,
    MAX_LEAVES,
    OFFSET_ROWS_DYN,
    PLAN_WORDS,
    SCORE_TILE,
    TOKENS_DYN,
    indexer_query_visibility,
)
from .indexer_topk_stages import (
    IDX_TOPK,
    PAIR_ROWS_DYN,
    SCORE_ROWS_DYN,
    TOPK_PAIR_WIDTH,
    merge2_top512_pairs,
    merge_top512_roots,
)
from .phase_sync import SIGNAL_WORDS
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, query_request

TOPK_CANDIDATES_PER_LEAF = CANDIDATES_PER_LEAF
TOPK_ROWS_PER_QUERY = MAX_LEAVES * 2


@pl.jit.inline(auto_scope=False)
def indexer_visible_root_count(
    visible: pl.Scalar[pl.INDEX],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    ROOTS_PER_LEAF: pl.constexpr,
    BALANCED: pl.constexpr,
):
    """返回本query连续有效根数；同源计划不匹配或不可安全读取时返回0。"""
    roots = pl.cast(0, pl.INDEX)
    leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    leaf_tiles = pl.cast(pl.read(plan, [3]), pl.INDEX)
    extra_leaves = pl.cast(pl.read(plan, [4]), pl.INDEX)
    max_history = pl.cast(pl.read(plan, [1]), pl.INDEX)
    # 不同策略的根布局不能靠有效总T猜。读取前先核对产生这些根的统一计划。
    if (
        ROOTS_PER_LEAF >= 1
        and ROOTS_PER_LEAF <= 2
        and pl.read(plan, [7]) == ROOTS_PER_LEAF
        and pl.read(plan, [6]) == pl.cast(BALANCED, pl.INT32)
    ):
        if visible > 0 and visible <= max_history and max_history <= MAX_CANDIDATES:
            if leaves > 0 and leaves <= MAX_LEAVES and leaf_tiles > 0:
                if extra_leaves >= 0 and extra_leaves < leaves:
                    leaf_count = (visible + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF
                    if BALANCED and pl.read(plan, [5]) == 2:
                        # B1独立调度原半叶，保留原2*ceil(visible/8192)个根。
                        # 包含末叶的空后半根，叶内与最终归并顺序均不改变。
                        if leaf_count > 0 and leaf_count * 2 <= leaves:
                            roots = leaf_count * 2
                    elif BALANCED:
                        # 与canonical一致：先覆盖extra个大叶，再覆盖其余普通叶。
                        # 用visible的1024向上取整计算叶前缀，不能用ceil(visible/8192)。
                        visible_tiles = (visible + SCORE_TILE - 1) // SCORE_TILE
                        larger_leaf_tiles = extra_leaves * (leaf_tiles + 1)
                        larger_leaf_count = pl.min(extra_leaves, (visible_tiles + leaf_tiles) // (leaf_tiles + 1))
                        remaining_tiles = pl.max(visible_tiles - larger_leaf_tiles, 0)
                        leaf_count = larger_leaf_count + (remaining_tiles + leaf_tiles - 1) // leaf_tiles
                        if leaf_count > 0 and leaf_count <= leaves:
                            roots = leaf_count * ROOTS_PER_LEAF
                    else:
                        if leaf_tiles == TOPK_CANDIDATES_PER_LEAF // SCORE_TILE and extra_leaves == 0:
                            if leaf_count > 0 and leaf_count <= leaves:
                                roots = leaf_count * ROOTS_PER_LEAF
    return roots


@pl.jit.inline(auto_scope=False)
def indexer_pair_publish_stage(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    attention_metadata: pl.Tensor[[PLAN_CAP_DYN, IDX_TOPK], pl.INT32],
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    topk_scores: pl.InOut[pl.Tensor[[TOKENS_DYN, IDX_TOPK], pl.FP32]],
    topk_indices: pl.InOut[pl.Tensor[[TOKENS_DYN, IDX_TOPK], pl.INT32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROOTS_PER_LEAF: pl.constexpr,
    BALANCED: pl.constexpr,
    generation: pl.Scalar[pl.INT32],
    QUERY_GROUP: pl.constexpr,
    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,
):
    """AIV发布pair根；保持非balanced根0原位归并及原多路平局优先顺序。"""
    tokens = pl.tensor.dim(positions, 0)
    for query in pl.range(worker, tokens, WORKERS):
        # 两条调度路径共用根数、等待与归并数学。前置路径读取原ROPE阶段
        # 已发布的本轮几何；后置路径只读Native Q/L/positions，不碰前缀。
        # 先声明跨constexpr分支的Scalar，保证SSA合并后的类型和定义域明确。
        request = pl.cast(-1, pl.INDEX)
        _begin = pl.cast(0, pl.INDEX)
        visible_count = pl.cast(0, pl.INDEX)
        if EARLY_ATTENTION_GEOMETRY:
            request = pl.cast(pl.read(attention_metadata, [query, EARLY_QUERY_OWNER_COL]), pl.INDEX)
            if request >= 0:
                _begin = pl.cast(pl.read(attention_metadata, [query, EARLY_QUERY_BEGIN_COL]), pl.INDEX)
                visible_count = pl.cast(pl.read(attention_metadata, [query, EARLY_QUERY_VISIBLE_COL]), pl.INDEX)
        else:
            request, _begin, _end, active = query_request(bounds, lengths, query, tokens)
            if active != 0:
                visible_count, _causal = indexer_query_visibility(bounds, lengths, positions, request, query)
        root_count = indexer_visible_root_count(visible_count, plan, ROOTS_PER_LEAF, BALANCED)
        arena_base = query * TOPK_ROWS_PER_QUERY
        if (
            root_count > 0
            and root_count <= TOPK_ROWS_PER_QUERY
            and arena_base + root_count <= pl.tensor.dim(pair_arena, 0)
        ):
            # 复用本query已算出的唯一request及begin，不额外扫描Q/L，也不把
            # request_begin替换成固定S的除法。根数与下方归并完全相同。
            local_query = query - _begin
            group = pl.cast(pl.read(offsets, [request]), pl.INDEX) + local_query // QUERY_GROUP
            signals = wait_indexer_query_roots(
                signals,
                plan,
                group,
                local_query % QUERY_GROUP,
                root_count,
                generation,
                QUERY_GROUP,
                BALANCED,
            )
            if BALANCED and QUERY_GROUP != 2:
                # G2独立半叶仍走下方原顺序左折叠，保持同分规则。
                # 多路归并helper保持incoming3/2/1/root顺序及最后不足四路分支。
                root_pairs = merge_top512_roots(pair_arena, arena_base, root_count)
            elif BALANCED and QUERY_GROUP == 2:
                # 单请求4K叶仍按child递增左折叠，incoming始终放在旧根之前。
                # 累积根在UB跨轮复用，不能换成另一棵归并树或交换同分优先级。
                root_pairs = pl.load(pair_arena, [arena_base, 0], [1, TOPK_PAIR_WIDTH])
                for child in pl.range(1, root_count):
                    next_pairs = pl.load(pair_arena, [arena_base + child, 0], [1, TOPK_PAIR_WIDTH])
                    resident_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                    resident_merged = pl.tile.mrgsort(next_pairs, root_pairs, tmp=resident_tmp)
                    # 实取1024元素，避免把整个2048结果的生命周期带进下一轮。
                    root_pairs = pl.tile.extract(
                        resident_merged,
                        0,
                        0,
                        [1, TOPK_PAIR_WIDTH],
                        target_memory=pl.MemorySpace.Vec,
                    )
                # 保留原根0最终内容；只消除无人消费的中间写回。
                pl.store(root_pairs, [arena_base, 0], pair_arena)
            else:
                # 原始短路径逐child向根0左折叠，且每次incoming先于旧root。
                # 保留原位写入及顺序；不能为了“统一接口”替换成另一棵merge树。
                for child in pl.range(1, root_count):
                    merge2_top512_pairs(pair_arena, arena_base, arena_base + child, arena_base)
                root_slot = arena_base
                root_pairs = pl.load(pair_arena, [root_slot, 0], [1, TOPK_PAIR_WIDTH])
            root_scores = pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
            pl.store(root_scores, [query, 0], topk_scores)
            root_indices = pl.tile.gather_mask(
                root_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            if visible_count >= IDX_TOPK:
                pl.store(root_indices, [query, 0], topk_indices)
            else:
                # 无效index不能依赖pair中负无穷的位解释；逐项保留真实候选前缀。
                output_indices = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
                for lane in pl.range(visible_count):
                    pl.tile.write(output_indices, [0, lane], pl.tile.read(root_indices, [0, lane]))
                pl.store(output_indices, [query, 0], topk_indices)
        else:
            # 每轮覆盖所有物理T行，包括dummy/gap，避免输出复用上一轮TopK。
            pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [query, 0], topk_scores)
            pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [query, 0], topk_indices)
    return topk_scores, topk_indices, signals


@pl.jit.inline(auto_scope=False)
def indexer_single_leaf_publish_stage(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    score_arena: pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    topk_scores: pl.InOut[pl.Tensor[[TOKENS_DYN, IDX_TOPK], pl.FP32]],
    topk_indices: pl.InOut[pl.Tensor[[TOKENS_DYN, IDX_TOPK], pl.INT32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """short-vector单叶直接排序；plan roots=0明确区分，没有隐式pair根回退。"""
    tokens = pl.tensor.dim(positions, 0)
    for query in pl.range(worker, tokens, WORKERS):
        request, _begin, _end, active = query_request(bounds, lengths, query, tokens)
        visible_count = pl.cast(0, pl.INDEX)
        if active != 0:
            visible_count, _causal = indexer_query_visibility(bounds, lengths, positions, request, query)
        plan_valid = pl.cast(0, pl.INT32)
        if pl.read(plan, [7]) == 0 and pl.read(plan, [6]) == 0 and pl.read(plan, [2]) == 1:
            if visible_count <= pl.read(plan, [1]) and pl.read(plan, [1]) <= TOPK_CANDIDATES_PER_LEAF:
                plan_valid = pl.cast(1, pl.INT32)
        if plan_valid != 0 and visible_count > 0 and query < pl.tensor.dim(score_arena, 0):
            indexer_topk_leaf_publish(score_arena, query, visible_count, topk_scores, topk_indices)
        else:
            pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF), [query, 0], topk_scores)
            pl.store(pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1), [query, 0], topk_indices)
    return topk_scores, topk_indices


@pl.jit.inline(auto_scope=False)
def indexer_topk_leaf_publish(
    score_arena: pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    query: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    topk_scores: pl.Tensor[[TOKENS_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[TOKENS_DYN, IDX_TOPK], pl.INT32],
) -> None:
    """单叶排序数学原样迁移；只有调用方验证过的1..8192个候选才能进入。"""
    # 2048及以下保持sort32后三级mrgsort；不足512的index单独填-1。
    if valid_count <= 2048:
        short_indices = pl.tile.arange(0, [1, 2048], dtype=pl.INT32)
        short_raw = pl.load(score_arena, [query, 0], [1, 2048], valid_shape=[1, valid_count])
        short_scores = pl.tile.fillpad(short_raw, pad_value=pl.PadValue.min)
        short_scores = pl.maximum(short_scores, FP32_NEG_INF)
        short_pairs = pl.sort32(short_scores, pl.reinterpret_view(short_indices, pl.UINT32))
        short_pairs = pl.mrgsort(short_pairs, block_len=64)
        short_pairs = pl.mrgsort(short_pairs, block_len=256)
        short_pairs = pl.mrgsort(short_pairs, block_len=1024)
        short_top = pl.tile.slice(short_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        short_values = pl.tile.gather_mask(short_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        short_selected = pl.tile.gather_mask(short_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(short_values, [query, 0], topk_scores)
        if valid_count >= IDX_TOPK:
            pl.store(short_selected, [query, 0], topk_indices)
        else:
            short_output = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(valid_count):
                pl.tile.write(short_output, [0, lane], pl.tile.read(short_selected, [0, lane]))
            pl.store(short_output, [query, 0], topk_indices)
    # 4096分支最后按right/left归并，次序决定相同score的index优先级。
    elif valid_count <= 4096:
        medium_indices = pl.tile.arange(0, [1, 4096], dtype=pl.INT32)
        medium_raw = pl.load(score_arena, [query, 0], [1, 4096], valid_shape=[1, valid_count])
        medium_scores = pl.tile.fillpad(medium_raw, pad_value=pl.PadValue.min)
        medium_scores = pl.maximum(medium_scores, FP32_NEG_INF)
        medium_pairs = pl.sort32(medium_scores, pl.reinterpret_view(medium_indices, pl.UINT32))
        medium_pairs = pl.mrgsort(medium_pairs, block_len=64)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=256)
        medium_pairs = pl.mrgsort(medium_pairs, block_len=1024)
        medium_left = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_right = pl.tile.slice(medium_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        medium_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        medium_merged = pl.tile.mrgsort(medium_right, medium_left, tmp=medium_tmp)
        medium_top = pl.tile.slice(medium_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        medium_values = pl.tile.gather_mask(medium_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        medium_selected = pl.tile.gather_mask(medium_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(medium_values, [query, 0], topk_scores)
        pl.store(medium_selected, [query, 0], topk_indices)
    else:
        # 8192分支保持fourth/third/second/first；不改为全量通用sort。
        full_indices = pl.tile.arange(0, [1, TOPK_CANDIDATES_PER_LEAF], dtype=pl.INT32)
        full_raw = pl.load(score_arena, [query, 0], [1, TOPK_CANDIDATES_PER_LEAF], valid_shape=[1, valid_count])
        full_scores = pl.tile.fillpad(full_raw, pad_value=pl.PadValue.min)
        full_min = pl.tile.full([1, TOPK_CANDIDATES_PER_LEAF], dtype=pl.FP32, value=FP32_NEG_INF)
        full_scores = pl.maximum(full_scores, full_min)
        full_pairs = pl.sort32(full_scores, pl.reinterpret_view(full_indices, pl.UINT32))
        full_pairs = pl.mrgsort(full_pairs, block_len=64)
        full_pairs = pl.mrgsort(full_pairs, block_len=256)
        full_pairs = pl.mrgsort(full_pairs, block_len=1024)
        full_first = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
        full_second = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 4096])
        full_third = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 8192])
        full_fourth = pl.tile.slice(full_pairs, [1, TOPK_PAIR_WIDTH], [0, 12288])
        full_tmp = pl.tile.create([1, 4 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
        full_merged = pl.tile.mrgsort(full_fourth, full_third, full_second, full_first, tmp=full_tmp)
        full_top = pl.tile.slice(full_merged, [1, TOPK_PAIR_WIDTH], [0, 0])
        full_values = pl.tile.gather_mask(full_top, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        full_selected = pl.tile.gather_mask(full_top, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        pl.store(full_values, [query, 0], topk_scores)
        pl.store(full_selected, [query, 0], topk_indices)
