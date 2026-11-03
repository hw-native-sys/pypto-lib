# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态CSA的区间策略：Host选择数学与调度策略，逐请求有效范围仍由device计算。

输入tokens/requests/max_query_tokens来自已有Tensor shape和Native描述符；
max_seq_len来自Native已有Host上界，禁止为了调用此函数新增device回读。
不同策略各自编译和缓存，档内T/B/Q/L/position保持动态。这里不出现七档
验收点白名单，亦不从T/B假定所有请求长度相等。

初始策略保留原CSA的Cube Score和输出投影数学：短/中历史双query或六query
分组；长历史有足够query时通过均衡历史叶增加并行度。分组6只是至多6个
query的数学tile，每请求可以拆成多组并保留尾组，不要求每请求恰好6行。
原小历史Vector Score尚待迁移，因此暂不接收max_seq_len//4<2048。这个
边界是当前实现能力，不是性能收益结论；所有策略仍须完整设备和性能验证。
输出尾段的组流水与列tile独立选择：每个输出组就绪后可先量化，O_B只等
自己读取的组。默认保留N128/N256原区间，启用按组通知；完整同进程对照
在T96长历史及T192短历史确认约2%/3%的相对收益，其他范围继续回归，
不能把该局部策略收益当作完整CSA已比原分task快30%。关闭开关仍复用数学体。
HC行依赖只按现有M16/K4与M8布局缩小等待，不改变数学、分工或动态范围。
七个固定S6回归点已通过完整精度/图/保护检查，默认开启；T96长/短历史
同进程开关对照分别降低完整CSA耗时2.13%/3.03%，不声称所有区间均有同轮消融。
关闭分支保留用于定位，完整动态场景与30%性能目标仍以统一日志为准。
阅读顺序：不可变策略→输入合同→Indexer分档→输出投影分档。
Attention查询几何复用候选使用已有mask行空余列，按次写入/发布，不缓存
上一轮请求信息。完整同进程对照已通过：128K/B16慢1.66%，8K/B32快1.33%，
收益不一致，默认保持关闭；候选和原始证据保留，不据此建立新的策略分界。
HC消费者归约候选让门值/Sinkhorn直接加载四份投影部分和，省去RAW交接。
128K/B16与8K/B32均与原分task精确一致；同进程五窗口完整L1分别慢1.06%、
快2.54%，因此默认关闭，尚不据两个点划定启用区间。见统一优化日志。
QA的M16并行仅在已有G6策略且16<tokens<32选用：此时原M32策略只有
两个M16尾块，改分核不增加数学或权重请求。长历史T24/T30同轮完整L1
降低1.33%/1.84%；G2的T18回退2.33%，因此沿既有G6边界保守收窄，
不把三点视为G6收益的因果证明或全历史普遍提速。这个区间实际为长历史、
query上界至少6、24<=T<32，混合Q与其它历史长度尚需继续回归。
开关进入constexpr签名，关闭编译旧QA路径；T和device metadata仍动态。
COOPERATIVE_PREFIX同时调整AIC/AIV前段：主压缩先生产并在Q_B期间消费，
4核Hadamard与20核Q_B重叠，原KV后处理退出Score前驱链。核资源比例来源于
M256 Hadamard与N256 Q_B的负载分工，不按七个shape白名单选择。七个S6回归点
同轮完整L1相对旧动态下降1.97%～6.49%；真实B1/S1、B5/S5相对旧动态精确
回归和B16同图metadata/padding通过，因此select_policy默认启用。其它区间的
性能、mixed-query及连续轨迹仍须继续验证，不将本次收益称为已达30%目标。
前置几何沿已存在的G6小M16并行域或T>=64区间开启，中间区间沿用后置
几何。T>=64对应至少两个完整M32行块，作为初始调度窗口分界；这不是
理论最优阈值，须结合边界实测调整。前/后置共用相同有效性与页保护规则，
只改变执行时点和同轮暂存，不能把关闭前置理解为跳过metadata检查。
首版前置七档精确/图通过且六档改善，B8回退；本区间版新增T30/T66边界与
长B16、短B32完整同轮检查，分别改善4.68%/2.58%/1.22%/1.76%。B8保持
原可执行C++，本轮仍有0.63%测量差，完整保留而不解释为确定的代码收益。
原前置版S1/S5和metadata/padding证据保留；区间版不重复声明已新测所有
动态场景。14项实际生成C++核对确认开关不增加额外算子运算或同步协议。
Attention同lane统计只在长历史G6且T<=96启用UB三槽，其余共用原GM数学。
修复独立PyPTO死状态闭环后，关闭路径生成代码与同编译器原路径一致。
七档及T90/T102边界的完整精确/图/保护检查通过，长B4/B8/B15/B16的
同轮完整L1改善6.01%/2.63%/1.31%/1.23%；短历史收益未稳定，默认关闭。
本轮未重测S1/S5/混合Q/连续轨迹，不把已有S6证据扩大为这些场景通过。
完整原始窗口及依赖修复在dead_state_dce_fix_evidence；30%目标仍0/7。
SCALE_PAGE_COALESCING独立控制META页布局证明与Score连续scale搬运。
只在长历史G6的小M16并行域或T>96的O_A N256域启用；中间域和G2仍走原
逐页路径，关闭时不分配证明尾区、不额外扫描页表做证明，也不读取证明。
选择依据为已有投影资源分区和整链实测取舍，不能理解成所有连续页输入都
一定更快。T30/T102两边界与B8关闭/B28开启均通过完整精确、状态、保护、
协议及图；边界完整L1分别降低2.00%/1.11%，B28复测降低2.42%。B8关闭
路径未回退，不将其测量差归因已关闭的连续读取。G2/G6短档关闭编译通过，
未重测短档性能。Native碎片页在启用区间仍逐轮证明失败后回退；未覆盖的
不等长query、其它历史和持续轨迹不能据这几项扩大为已验证。
Key预取在独立精确L1范围工具链下重新评估：长历史沿用提前一块；短历史
只在既有G2且半叶工作量上界<=2048项时开启。每步768列、每AIV取384列，
以已有max_seq_len估算最大步数，对应最多5步/3840压缩行。实际visible、
页址、循环与尾块仍在device计算；这不是B16/B32白名单。该范围的四个
S6验证点完整L1改善0.96%～3.16%，S1/S5精确回归通过；短G6及较大
半叶范围未获收益，保留原值。排序资源边界是实测分档依据，不等于已证明
回退由排序导致。收益证据基于PTOAS b7b4a82c隔离v3，旧工具链不作性能
保证，全部窗口及未覆盖范围见统一日志。
物理行plan沿已有T>=64的前置几何区间启用，消费者省去重复Q/L和页表查询。
容量是否支持在device判定；关闭或容量外保留原逻辑表示。全开七档六档
中位耗时下降，但小T的B4回退、B8收益不足1微秒，故保留小区间原路径。
这是基于完整L1实测的区间选择，不是证明所有输入都加速，也不改变精度门槛。
SPLIT_SCORE_ROWS在既有长G6域、T>=32时选择双QK早FIX和每panel单次WS
写回。32沿QA小M16区间的已有资源边界，不是B4/B8白名单或理论最优声明。
长B8/B16/B24/B28完整实测改善，小域保留原Score。关闭/开启代码等价、
真实S1/S5及T30/T36边界已核对后采用；范围和原始数据见统一优化日志。
这不是所有动态输入都已测过或五档30%目标已完成的声明。

B1原半叶区间
    单请求、query容量5或6且压缩历史上界在(8192,65536]时，保留G2，
    将原8192叶的两个动态半叶分别分派；不按固定S反推本轮有效query。
    半叶分数和累计归并根留在UB，保持原末叶跨度及六条排序分支同分规则。
    这是B1区间策略，其余请求数/容量继续使用原策略；不是固定128K模板。
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


@dataclass(frozen=True)
class CSASPMDPolicy:
    query_group: int
    qk_panel: int
    score_tile: int
    prefetch: int
    balanced: bool
    roots: int
    oa_cols: int
    ob_rows: int
    output_group_pipeline: bool = True
    hc_row_dependencies: bool = True
    attention_query_metadata: bool = False
    hc_consumer_reduce: bool = False
    qa_m16_parallel: bool = False
    cooperative_prefix: bool = False
    early_attention_geometry: bool = False
    local_attention_stats: bool = False
    scale_page_coalescing: bool = False
    plan_physical_rows: bool = False
    split_score_rows: bool = False
    output_work_balance: bool = False

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
            "OUTPUT_GROUP_PIPELINE": self.output_group_pipeline,
            "HC_ROW_DEPENDENCIES": self.hc_row_dependencies,
            "ATTENTION_QUERY_METADATA": self.attention_query_metadata,
            "HC_CONSUMER_REDUCE": self.hc_consumer_reduce,
            "QA_M16_PARALLEL": self.qa_m16_parallel,
            "COOPERATIVE_PREFIX": self.cooperative_prefix,
            "EARLY_ATTENTION_GEOMETRY": self.early_attention_geometry,
            "LOCAL_ATTENTION_STATS": self.local_attention_stats,
            "SCALE_PAGE_COALESCING": self.scale_page_coalescing,
            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,
            "SPLIT_SCORE_ROWS": self.split_score_rows,
            "OUTPUT_WORK_BALANCE": self.output_work_balance,
        }


def select_policy(*, tokens: int, requests: int, max_query_tokens: int, max_seq_len: int) -> CSASPMDPolicy:
    """仅依据已有Host信息选策略；有效分组数、叶数及全部访问地址由AIV/AIC计算。"""
    values = (tokens, requests, max_query_tokens, max_seq_len)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("tokens/requests/max_query_tokens/max_seq_len须为已有的正整数描述符")
    if tokens > requests * max_query_tokens:
        raise ValueError("token容量超过请求容量×query上界；请绑定Native同一档描述符")
    cache_rows = max_seq_len // 4
    if cache_rows < 2048:
        raise NotImplementedError("完整动态SPMD的小历史Vector Score路径尚未接入")
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
    single_request_leaves = (
        requests == 1 and long_history and cache_rows <= 65536
        and 5 <= max_query_tokens <= 6
    )
    ob_rows = 32 if tokens <= 32 else 96 if tokens <= 96 else 128
    parallel_small_qa = group == 6 and 16 < tokens < 32
    return CSASPMDPolicy(
        query_group=group,
        qk_panel=64 if group == 6 else 128,
        score_tile=1024 if long_history else SHORT_SCORE_TILE,
        prefetch=1 if long_history or short_key_prefetch else 0,
        balanced=(long_history and group == 6) or single_request_leaves,
        roots=1 if (long_history and group == 6) or single_request_leaves else 2,
        oa_cols=128 if tokens <= 96 else 256,
        ob_rows=ob_rows,
        # 沿已有G6分界选择两M16尾块的并行；G2已观察回退，不按B4/B5白名单。
        qa_m16_parallel=parallel_small_qa,
        cooperative_prefix=True,
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
        scale_page_coalescing=(long_history and group == 6 and
                               (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),
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
        # B1 reuses its existing balanced half-leaf domain. Multiple-request
        # redistribution is limited to long G6 and the existing 64..96-token
        # resource interval. Short G2/B16 regressed despite exact outputs;
        # neither equal tile counts nor source-level balance proves speedup.
        output_work_balance=(
            single_request_leaves
            or (long_history and group == 6
                and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS)
        ),
    )
