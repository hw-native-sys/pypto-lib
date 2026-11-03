# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TND MPMD 的资源策略，数值分档取自冻结参考 84751e34。

输入仅为已有 Tensor/Native 描述符，不回读设备 Q/L。QUERY_GROUP 是
每请求内部数学 tile 上限，不表示请求等长，也不能用它反算请求数。

单 SPMD 的常驻核分工、phase 等待和 O_A/O_B 同核互补策略不直接沿用；
多 task 的顺序、并发与依赖由 entry.py 表达。以下核内策略在新多 task
路径中的正确性和完整 L1 收益仍以本轮设备证据为准。
"""

from dataclasses import dataclass

# 沿CSA已有投影行块选择窗口；不是按B/S/history测试点枚举。
EARLY_GEOMETRY_MIN_TOKENS = 64
# 与既有O_A N128输出区间一致；不是按七个测试点枚举。
LOCAL_STATS_MAX_TOKENS = 96
# 非balanced Score既有交接宽度；两AIV各消费一半。对应indexer_score_stages。
SHORT_SCORE_TILE = 768
# 既有TopK半叶直接归并的资源边界；更大范围用2560/3072/4096分支。
SHORT_PREFETCH_HALF_SORT_LIMIT = 2048
# 沿已有QA的小M16并行区间边界，两个完整M16行块起再启用Score新流水。
SCORE_SPLIT_MIN_TOKENS = 32
# 两个 M32 容量以内已验证删除空 worker 获益；更大容量保留原分工。
# 128K B28 的完整 L1 回退，不能因局部少派空任务而全域采用。
NONEMPTY_WORKER_MAX_TOKENS = 64


@dataclass(frozen=True)
class CSAMPMDPolicy:
    query_group: int
    qk_panel: int
    score_tile: int
    prefetch: int
    balanced: bool
    roots: int
    oa_cols: int
    ob_rows: int
    attention_query_metadata: bool = False
    hc_consumer_reduce: bool = False
    qa_m16_parallel: bool = False
    early_attention_geometry: bool = False
    local_attention_stats: bool = False
    scale_page_coalescing: bool = False
    plan_physical_rows: bool = False
    split_score_rows: bool = False
    limit_empty_workers: bool = False

    def constexprs(self) -> dict[str, int | bool]:
        return {
            "QUERY_GROUP": self.query_group,
            "QK_PANEL": self.qk_panel,
            "SCORE_TILE": self.score_tile,
            "PREFETCH": self.prefetch,
            "TO_L0": True,
            "BALANCED": self.balanced,
            "ROOTS": self.roots,
            "OA_COLS": self.oa_cols,
            "OB_ROWS": self.ob_rows,
            "ATTENTION_QUERY_METADATA": self.attention_query_metadata,
            "HC_CONSUMER_REDUCE": self.hc_consumer_reduce,
            "QA_M16_PARALLEL": self.qa_m16_parallel,
            "EARLY_ATTENTION_GEOMETRY": self.early_attention_geometry,
            "LOCAL_ATTENTION_STATS": self.local_attention_stats,
            "SCALE_PAGE_COALESCING": self.scale_page_coalescing,
            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,
            "SPLIT_SCORE_ROWS": self.split_score_rows,
            "LIMIT_EMPTY_WORKERS": self.limit_empty_workers,
        }


def select_policy(*, tokens: int, requests: int, max_query_tokens: int, max_seq_len: int) -> CSAMPMDPolicy:
    """仅依据已有Host信息选策略；有效分组数、叶数及全部访问地址由AIV/AIC计算。"""
    values = (tokens, requests, max_query_tokens, max_seq_len)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("tokens/requests/max_query_tokens/max_seq_len须为已有的正整数描述符")
    if tokens > requests * max_query_tokens:
        raise ValueError("token容量超过请求容量×query上界；请绑定Native同一档描述符")
    cache_rows = max_seq_len // 4
    if cache_rows < 2048:
        raise NotImplementedError("TND 多 task 的短历史 Vector Score 路径尚未接入")
    if cache_rows > 262144:
        raise ValueError("当前TopK工作区合同最多支持262144个压缩候选")
    long_history = cache_rows > 8192
    group = 2
    if long_history:
        # 24个query起可用多叶填满24组；query长度不足6时不为复用Key扩张M384。
        if tokens >= 24 and max_query_tokens >= 6:
            group = 6
    elif max_query_tokens >= 6 and requests >= 24:
        # 对工作量上界比较，而不是声明T/6就是实际请求数；不等长请求仍在device
        # 独立ceil分组。首版用已有原策略的最忙核增幅上限25%作为保守启发式。
        pair_waves = ((tokens + 1) // 2 + 23) // 24
        request_waves = (requests + 23) // 24
        if 4 * 6 * request_waves <= 5 * 2 * pair_waves:
            group = 6
    # visible受请求length//4钳制，不超过这里已有的Host长度上界。用上界
    # 保守估算半叶工作量，不新增device回读，不代替核内真实范围与尾块检查。
    # 该资源区间实测支持预取；较大半叶与短G6仍保持原路径。
    short_steps_upper = (cache_rows + SHORT_SCORE_TILE - 1) // SHORT_SCORE_TILE
    short_half_span_upper = short_steps_upper * (SHORT_SCORE_TILE // 2)
    short_key_prefetch = group == 2 and short_half_span_upper <= SHORT_PREFETCH_HALF_SORT_LIMIT
    # 用户允许单请求独立分档。仅改变已有G2的历史工作分配，
    # 保留动态Q/L及原半叶边界（末叶可小于4096）；较大容量及S1维持原策略。
    single_request_leaves = requests == 1 and long_history and cache_rows <= 65536 and 5 <= max_query_tokens <= 6
    ob_rows = 32 if tokens <= 32 else 96 if tokens <= 96 else 128
    parallel_small_qa = group == 6 and 16 < tokens < 32
    return CSAMPMDPolicy(
        limit_empty_workers=tokens <= NONEMPTY_WORKER_MAX_TOKENS,
        query_group=group,
        qk_panel=64 if group == 6 else 128,
        score_tile=1024 if long_history else SHORT_SCORE_TILE,
        prefetch=1 if long_history or short_key_prefetch else 0,
        balanced=(long_history and group == 6) or single_request_leaves,
        roots=1 if (long_history and group == 6) or single_request_leaves else 2,
        oa_cols=128 if tokens <= 96 else 256,
        ob_rows=ob_rows,
        # 首版沿用冻结参考的核内 QA tile 选择；MPMD 收益另行验证。
        qa_m16_parallel=parallel_small_qa,
        # 前置几何在小M16并行域及较大T区间有已测收益；中间单M32主体加尾块
        # 的域先保留后置算法，避免把metadata放到RoPE关键路径。64表示至少
        # 两个完整M32行块，仍须边界验证；不代表所有形状下都是理论最优。
        early_attention_geometry=parallel_small_qa or tokens >= EARLY_GEOMETRY_MIN_TOKENS,
        # 同一AIV生产和消费的mi/li留在UB三槽；只在长历史G6/N128域默认启用。
        # B4/B8/B15/B16重复测得收益；短历史T>96两点方向不一致，保留原GM。
        # 实际T/请求边界仍在device读取；依赖编译器删除关闭分支的死Tile状态。
        local_attention_stats=long_history and group == 6 and tokens <= LOCAL_STATS_MAX_TOKENS,
        # 连续scale读取需要先在META证明页布局；该工作并非所有负载都划算。
        # 沿已有小M16并行域和O_A N256的较大T域启用，中间域保留逐页读取。
        # 不按B/history验收点枚举；页是否连续仍由本轮device表完整证明。
        # 小M16与较大T边界已做完整同轮验证；其余验证边界见统一优化日志。
        scale_page_coalescing=(long_history and group == 6 and (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),
        # 沿已有至少两个M32行块的前置几何区间，统一在plan验证选中页，
        # 并把压缩逻辑索引转换成物理行；消费者不再重复Q/L扫描和页表查询。
        # 小T的计划生产者更少，搬移地址工作未稳定获益，保留原逻辑表示。
        # 此处只选策略；容量合法性和全部地址仍由device元数据决定，三阶段
        # 共享同一谓词，不能仅开启生产者或消费者。区间实测与限制见优化日志。
        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,
        # 小T保持原Score，中大T使用早FIX与单次WS写回。分界沿已有M16
        # 前段资源区间；依据完整实测取舍，不把回退唯一归因于QA或假定最优。
        # 实际有效query/长度/页表仍在device读取，不按B4/B8等测试点枚举。
        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,
    )
