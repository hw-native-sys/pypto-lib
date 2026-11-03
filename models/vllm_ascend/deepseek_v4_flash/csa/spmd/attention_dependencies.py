# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Score 到 Attention 的动态行依赖：按真正读取的数据选择生产者。

整体思路与阅读顺序
    wait_indexer_query_roots 保护单条 query 的 Top-K 根；
    wait_attention_plan_rows 保护一个 Attention 核组实际读取的计划行。
    Top-K 到八行 plan 的依赖直接在 attention_slot_plan_stage 内获取。
    这三条边延续历史静态 SPMD 的按行依赖思想，但生产者编号由本轮
    device group plan、query 边界和动态 T 计算，不读取冻结调度表。

数据、分工与动态边界
    signals 是既有独占 INT32 GM 阶段区，plan 是既有 INT32[8] 设备计划。
    group 是 group_offsets[request]+(query-request_begin)//G，local_query
    是同一请求内对 G 的余数；调用者已经确认唯一活动请求和因果位置。
    root_count 必须与最终归并实际读取的根数相同，不能用最大 arena 容量。
    少于24个group时，Score item=leaf*groups+group；否则为group*leaves+leaf。
    每个 item 归 item%24 的 AIC 组。balanced 每个query仅由一个 AIV lane
    写完整根；普通双根策略每叶由两个 lane 各写一根，两个都必须等待。
    Attention 的 query=core_group+k*24；它的计划行归 (query//8)%48。
    24/48/72是已有A3核拓扑，8是计划数学tile；T、group数、leaf数均动态。

发布、缓存与生命周期
    仍等待生产者完整阶段的原有信号，不提前把 FIFO ACK 当作根写回。
    每个 producer 在发布前排空流水并维护业务缓存；这里复用原 GM notify/wait
    协议，所有需要的信号到达后统一 fence/cacheinvalid/fence。没有省略 A3
    Scalar 非一致缓存所需的发布/获取，也不新增计数器、atomic 或信号槽。
    同一 producer 可能被重复等待，读取同一代完成信号是幂等的；不能清零它。
    各 query 根、Top-K 行、plan 行在本轮独占且写后不覆盖，因此别的 query
    仍在生产时可以安全读取已经发布的行。cache payload 的其它依赖由外层保留。

正确性约束
    helper 不改变数学、排序顺序、页保护或 FFTS 三槽流控。失活 query 不读
    根，失活 Attention 组不读计划，均无需虚假的生产者依赖。无效计划拒绝
    逻辑仍由既有归并与入口负责；此处不是对任意非法 plan 的恢复机制。
    源级依赖覆盖只能证明所有权关系，真机完整数值、图重放和性能另行记录。
"""

import pypto.language as pl

from .indexer_group_stages import PLAN_WORDS
from .phase_sync import SIGNAL_SLOT_WORDS, SIGNAL_WORDS
from .runtime_abi import CORES, GROUPS, PHASE_ATTN_PLAN, PHASE_SCORE_VECTOR, VECTORS
from .signals import csa_signal_wait


@pl.jit.inline(auto_scope=False)
def wait_indexer_query_roots(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    group: pl.Scalar[pl.INDEX],
    local_query: pl.Scalar[pl.INDEX],
    root_count: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    QUERY_GROUP: pl.constexpr,
    BALANCED: pl.constexpr,
):
    """只获取本次归并读取的根；先读完同代计划，再选择真实写者。"""
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    for root in pl.range(root_count):
        if BALANCED:
            leaf = root
            lane = local_query // (QUERY_GROUP // 2)
        else:
            leaf = root // 2
            lane = root % 2
        if groups < GROUPS:
            item = leaf * groups + group
        else:
            item = group * leaves + leaf
        producer = GROUPS + 2 * (item % GROUPS) + lane
        offset = (PHASE_SCORE_VECTOR * CORES + producer) * SIGNAL_SLOT_WORDS
        signals = csa_signal_wait(signals, offset, generation)
    # 原根归并的同代缓存获取仍保留，只是不再等不读取的其它query生产者。
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signals


@pl.jit.inline(auto_scope=False)
def wait_attention_plan_rows(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    core_group: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
):
    """AIC和双AIV读取相同query集合，获取同一组八行plan生产者。"""
    if core_group < tokens:
        for query in pl.range(core_group, tokens, GROUPS):
            producer = GROUPS + (query // 8) % VECTORS
            offset = (PHASE_ATTN_PLAN * CORES + producer) * SIGNAL_SLOT_WORDS
            signals = csa_signal_wait(signals, offset, generation)
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
    return signals
