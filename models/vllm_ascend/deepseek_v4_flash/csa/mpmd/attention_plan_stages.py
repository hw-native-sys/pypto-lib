# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA Attention 的候选计划与逆 RoPE 正负号准备。

整体范围与阅读顺序
    当前物理行候选在C<=2048、0<物理页数<=2**24时复用indices发布INT32
    物理行；PLAN_PHYSICAL_ROWS由policy沿已有T>=64前置几何区间选择，
    三阶段用同一capacity谓词，关闭或能力外保留原逻辑表示。预置几何仍
    校验Q/L/位置但不重复做全页证明；真正选中的页在本plan逐项Vector检查。
    外部Top-K、bias和五个块mask不变；非法项写-1。压缩消费者在已获取该
    plan后只做最终物理行界检查，SWA及能力外路径仍用原Q/L和页表规则。
    原页合法性检查未跳过，而是移到最终选中地址；逐档真机证据及未验证范围
    记录在统一优化日志，不能仅据地址推导变少宣称完整算子加速。
    先读 attention_geometry_prepare_stage 的可选前段准备，再读
    attention_slot_plan_stage：将 Indexer 的逻辑 Top-K 转为可安全消费的
    压缩索引、640 列 softmax bias 和五个候选块的有效标志；再读
    attention_inverse_rope_sign_stage：准备最终 attention 输出所需的逆旋转
    sin 符号。两段来自 canonical decode_sparse_attn_csa.py 的 plan/rope_cs，
    不包括 QK/PV、softmax、环槽事件、输出投影，也不创建 task/SPMD/核间通知。

输入布局与真正动态的请求边界
    topk 是 INT32[T,512] 逻辑压缩行号，positions 为 Native INT64[T]；Q 是
    原始 INT32[B+1] query_start_loc，L 是包含本轮 query 的 INT32[B] 长度。
    T、B、Q 实际轴独立，允许同 T 对应不同 mixed-query 分组，禁止 token//S。
    两份 INT32[B,table_columns] 页表的列容量分别动态；两种 cache 的物理页数
    由外层已有 descriptor 的 dim 以 INDEX Scalar 传入。这里不接收 cache
    payload，所以不会读 cache；后续 gather 必须复用相同 Q/L、页表和容量。
    活动 token 还须满足 position=L-(Q[r+1]-Q[r])+(token-Q[r])。无 owner、
    多 owner、dummy、非法 Q 或陈旧 position 全部生成无效计划，不读页表。
    为保留八行向量数学，Top-K 可以读取实际 T 内的 dummy 行，但不据它访问页表。

输出、所有权与容量
    compressed_indices 为 INT32[PLAN_CAP,512]，bias 为 FP32[PLAN_CAP,640]，
    valid_blocks 为 INT32[PLAN_CAP,16]；PLAN_CAP>=T 由入口捕获前校验。
    每个实际 token 每次覆盖全部 indices/bias 及 mask 前五列。启用
    CACHE_QUERY_METADATA 时，再用第5/6列保存已验证的请求编号/causal位置，
    供双AIV的五个候选块直接复用；dummy写(-1,0)，其余九列及T后保持原值。
    关闭时第5列起仍保持原值。mask每行正好64B，八行tile独占各行，所有
    Scalar写由该行唯一owner完成；底层分配须保持64B对齐。未新增GM缓冲。
    sin 为 FP32[T,64]，signed_sin 为 FP32[ROPE_CAP,64]，ROPE_CAP>=T。
    两种容量不是实际 T，不可由容量反推活动请求或让消费者读取未写 pad。
    显式 worker_id/WORKERS 把第 tile 个八行块分给 tile%WORKERS；尾块用
    min(8,T-first) 限制 load/store。所有 UB 辅助 Tile 归本核，不额外分配 GM。

数学、保护与同步责任
    压缩候选保留 FP32 clamp、INT32 trunc 上界、index/bias 和 row_max 顺序；
    新增页表列/物理页上下界条件乘入同一 mask，不能只阻止 gather 却保留零 bias。
    压缩页表先按请求以Vector验证当前长度可能使用的全部列，尾tile的有效列
    必须贯穿load、广播与row_min，未加载的UB padding不参与合法性判断；同八行tile中的
    相邻同请求复用本核结果。全部页合法时批量设置候选页mask，含空洞、列容量
    不足或超出快速检查数值边界时保留逐候选Scalar回退；不跳过有效性检查。
    SWA 保留128行窗口、最多五个32-token页run；正常页走整块bias，页空洞及
    越界页才按合法run重建本行。非法候选 index=-1、bias=-1e20、块有效位=0。
    逆 RoPE 的偶数列乘+1、奇数列乘-1，与 query 正向 RoPE 的符号相反。
    EARLY_ATTENTION_GEOMETRY由policy按工作量区间选择。开启时，
    attention_geometry_prepare_stage在前段执行不依赖Top-K的请求/位置/页检查；
    attention_slot_plan_stage在Top-K后完成候选相关的工作。前段暂借indices
    每行前五项保存owner/begin/position/pages_ok/visible，由原PHASE_ROPE
    发布；Top-K读取这些几何，plan最后再覆盖为512项最终索引。不增加Host
    适配、根Tensor或跨调用缓存。preparer与plan均按八行tile%48分工，plan
    等本行Top-K发布后才覆盖。失活行每轮重写owner=-1及其它零值。
    关闭时，本阶段按原规则读取Q/L/position并执行同一个页检查函数，完全
    不读几何前缀。两路共用候选数学、页保护、最终store及后续同步协议。
    Native metadata在外层已就绪；每个八行tile只获取对应Top-K行的生产者，
    不等待无关query。plan完成后仍由外层发布整个worker阶段给QK/AIV。
    缓存的请求几何与五个块标志同次计算、同次发布，不跨调用沿用；Q/L/位置
    和页表在整个Attention阶段必须保持不变。attention_query_geometry读取
    这份已验证几何，或者在关闭分支保留原逐块Q/L扫描，不改页表容量保护。
    rope 可与 plan 独立运行，最终输出消费 signed_sin 前等实际 row owner。
    本模块不维护信号缓存；源码/CPU通过不能替代设备精度、同步或完整CSA验证。
"""

import pypto.language as pl

from .indexer_group_stages import MAX_CANDIDATES
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, query_request

T_DYN = pl.dynamic("CSA_ATTN_PLAN_T")
PLAN_CAP_DYN = pl.dynamic("CSA_ATTN_PLAN_CAP")
ROPE_CAP_DYN = pl.dynamic("CSA_ATTN_ROPE_CAP")
ORIGINAL_COLUMNS_DYN = pl.dynamic("CSA_ATTN_ORIGINAL_COLUMNS")
COMPRESSED_COLUMNS_DYN = pl.dynamic("CSA_ATTN_COMPRESSED_COLUMNS")
ROWS = 8
TOPK = 512
WIN = 128
BLOCK_SIZE = 32
ATTN_K_TILE = 128
SPARSE_BLOCKS = 5
BIAS_COLS = 640
MASK_COLS = 16
QUERY_OWNER_COL = 5
QUERY_POSITION_COL = 6
ROPE_DIM = 64
SWA_RUNS = 5
COMPRESS_RATIO_INV = 0.25
CSA_CMP_GE_BIAS = 1.0
NEG_INF = -1.0e20
PAGE_CHECK_TILE = 1024
# FP32可精确表达此范围内的页号和长度；超出时走原INT32/INDEX逐候选检查。
EXACT_FP32_INT = 16777216
PHYSICAL_PLAN_COLUMNS = 2048  # 单query页表UB上界；实际列数仍取device描述符。
# 最终indices产生前的短期行前缀；阶段37发布后，这五项不再是metadata。
EARLY_QUERY_OWNER_COL = 0
EARLY_QUERY_BEGIN_COL = 1
EARLY_QUERY_POSITION_COL = 2
EARLY_QUERY_PAGES_COL = 3
EARLY_QUERY_VISIBLE_COL = 4


@pl.jit.inline(auto_scope=False)
def physical_plan_supported(
    columns: pl.Scalar[pl.INDEX],
    physical_pages: pl.Scalar[pl.INDEX],
    ENABLED: pl.constexpr,
):
    """生产消费共用的表示判定，不按历史长度或batch猜indices含义。

    Native cache按32行整页提供；pages<=2**24允许页合法性用FP32精确边界，
    合法物理行仍用INT32位移加法，最大2**29-1，绝不转成FP32地址。
    不支持的页宽/容量整体退回原逻辑表示，不在同一调用内混用两种含义。
    """
    supported = pl.cast(0, pl.INT32)
    if ENABLED:
        if columns > 0 and columns <= PHYSICAL_PLAN_COLUMNS and physical_pages > 0 and physical_pages <= EXACT_FP32_INT:
            supported = pl.cast(1, pl.INT32)
    return supported


@pl.jit.inline(auto_scope=False)
def compressed_pages_all_valid(
    table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    length: pl.Scalar[pl.INDEX],
    physical_pages: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.INT32]:
    """保守快速判定：返回1才允许批量通过，返回0仍逐个检查候选，不丢合法页。

    一次读取1024个页号并做向量范围检查；最后只读取valid_shape内的列。
    当前请求所有query的压缩候选上界不超过length//4，故覆盖该前缀即可。
    本函数不接收候选，不改变索引/排序，不在GM保存跨调用可陈旧的判断。
    """
    needed = ((length // 4) + BLOCK_SIZE - 1) // BLOCK_SIZE
    all_valid = pl.cast(0, pl.INT32)
    if length <= EXACT_FP32_INT and physical_pages > 0 and physical_pages <= EXACT_FP32_INT:
        if needed <= pl.tensor.dim(table, 1):
            all_valid = pl.cast(1, pl.INT32)
            page_limit = pl.cast(pl.cast(physical_pages, pl.INT32), pl.FP32)
            reduce_tmp = pl.create_tile([ROWS, PAGE_CHECK_TILE], dtype=pl.FP32)
            for column in pl.range(0, needed, PAGE_CHECK_TILE):
                live = pl.min(PAGE_CHECK_TILE, needed - column)
                values = pl.cast(
                    pl.load(table, [request, column], [1, PAGE_CHECK_TILE], valid_shape=[1, live]),
                    pl.FP32,
                )
                lower_ok = pl.minimum(pl.maximum(pl.add(values, 1.0), 0.0), 1.0)
                upper_ok = pl.minimum(pl.maximum(pl.add(pl.neg(values), page_limit), 0.0), 1.0)
                # A3列向量结果要求32B物理步长，1×1 FP32归约结果不合法。
                # 同一行展开到8行后取第0个归约结果，避免不合法的4B列布局。
                # 有效列必须沿广播保留：短页表只加载live列，不能把对应1×live
                # 向量广播到全1024列后归约，否则会读取UB未初始化的padding，
                # 把全合法页误判成存在空洞。先限制左操作数，再显式约束归约输入；
                # 数学仍对[0,live)逐列检查，不把未加载列当作实际页表内容。
                ones = pl.set_validshape(pl.tile.full([ROWS, PAGE_CHECK_TILE], dtype=pl.FP32, value=1.0), ROWS, live)
                repeated = pl.set_validshape(pl.col_expand_mul(ones, pl.mul(lower_ok, upper_ok)), ROWS, live)
                block_ok = pl.row_min(repeated, reduce_tmp)
                if pl.read(block_ok, [0, 0]) < 1.0:
                    all_valid = pl.cast(0, pl.INT32)
    return all_valid


@pl.jit.inline(auto_scope=False)
def attention_geometry_prepare_stage(
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    compressed_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    indices: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, TOPK], pl.INT32]],
    compressed_pages: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
):
    """准备只读metadata，与投影重叠；不读Top-K或cache payload。

    保留原Q/L唯一owner、因果位置及页范围规则。页检查复用同一函数；返回0
    仍由晚期plan逐候选回退，不能据此把整行判为非法。每八行tile沿用原
    request缓存粒度。先在UB写五项，再用MTE3写八列对齐前缀（余三列为零），
    后续最终索引同样用MTE3，避免同一个GM根混用Scalar与MTE3写。仅写实际T。
    原ROPE phase负责发布；Top-K此前已获取该phase。整个调用的Q/L、位置、
    页表不变，压缩器更新cache payload不会改变这个只读判定。
    """
    tokens = pl.tensor.dim(positions, 0)
    # 三个阶段共用容量谓词；表示不靠Host猜测或逐query私自切换。
    physical_mode = physical_plan_supported(pl.tensor.dim(compressed_table, 1), compressed_pages, PLAN_PHYSICAL_ROWS)
    for first in pl.range(worker * ROWS, tokens, WORKERS * ROWS):
        rows = pl.min(ROWS, tokens - first)
        # 一行32B对齐；UB Scalar写由编译器接入MTE3依赖，不产生GM DCache脏副本。
        geometry = pl.tile.full([ROWS, 8], dtype=pl.INT32, value=0)
        checked_request = pl.cast(-1, pl.INDEX)
        checked_pages = pl.cast(0, pl.INT32)
        for row in pl.range(rows):
            token = first + row
            owner = pl.cast(-1, pl.INT32)
            saved_begin = pl.cast(0, pl.INT32)
            saved_position = pl.cast(0, pl.INT32)
            pages_ok = pl.cast(0, pl.INT32)
            visible = pl.cast(0, pl.INDEX)
            request, begin, end, active = query_request(bounds, lengths, token, tokens)
            if active != 0:
                position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
                if position == length - (end - begin) + token - begin:
                    owner = pl.cast(request, pl.INT32)
                    saved_begin = pl.cast(begin, pl.INT32)
                    saved_position = pl.cast(position, pl.INT32)
                    # 沿用indexer_query_visibility的整数边界，不由FP32值反推根数。
                    # 合法position<L且L为INT32，故可无损暂存。
                    visible = pl.max(pl.min(pl.min(length // 4, (position + 1) // 4), MAX_CANDIDATES), 0)
                    # 物理计划后续会逐个校验选中页，此处不重复扫描整个历史页表。
                    # pages_ok=0只是未作全页证明；Top-K不读取该列，物理plan也不用它。
                    if physical_mode == 0:
                        if request != checked_request:
                            checked_pages = compressed_pages_all_valid(
                                compressed_table, request, length, compressed_pages
                            )
                            checked_request = request
                        pages_ok = checked_pages
            # dummy和陈旧位置同样覆盖，不能把上轮最终索引解释为本轮几何。
            pl.tile.write(geometry, [row, EARLY_QUERY_OWNER_COL], owner)
            pl.tile.write(geometry, [row, EARLY_QUERY_BEGIN_COL], saved_begin)
            pl.tile.write(geometry, [row, EARLY_QUERY_POSITION_COL], saved_position)
            pl.tile.write(geometry, [row, EARLY_QUERY_PAGES_COL], pages_ok)
            pl.tile.write(geometry, [row, EARLY_QUERY_VISIBLE_COL], pl.cast(visible, pl.INT32))
        pl.store(pl.set_validshape(geometry, rows, 8), [first, 0], indices)
    return indices


@pl.jit.inline(auto_scope=False)
def attention_slot_plan_stage(
    topk: pl.Tensor[[T_DYN, TOPK], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    original_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    compressed_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    indices: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, TOPK], pl.INT32]],
    bias: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, BIAS_COLS], pl.FP32]],
    valid_blocks: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, MASK_COLS], pl.INT32]],
    original_pages: pl.Scalar[pl.INDEX],
    compressed_pages: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    CACHE_QUERY_METADATA: pl.constexpr = False,
    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
):
    """每个八行块独占 indices/bias/mask 行；只访问合法 owner 的 Native 页表。"""
    tokens = pl.tensor.dim(positions, 0)
    # 三个阶段共用容量谓词；表示不靠Host猜测或逐query私自切换。
    physical_mode = physical_plan_supported(pl.tensor.dim(compressed_table, 1), compressed_pages, PLAN_PHYSICAL_ROWS)
    # 固定8是向量tile，不是请求长度。一个tile可跨请求，逐行用Q/L判断owner。
    for first in pl.range(worker_id * ROWS, tokens, WORKERS * ROWS):
        rows = pl.min(ROWS, tokens - first)
        # Top-K按query%48写，plan按八行tile%48写。只等这个tile真正读取的
        # 最多八个owner；尾部仅rows行。wait内仍保留业务缓存的获取语义。
        # Producer completion and GM visibility are enforced by the MPMD DAG.
        c_raw_tile = pl.cast(
            pl.load(topk, [first, 0], [ROWS, TOPK], valid_shape=[rows, TOPK]),
            target_type=pl.FP32,
        )
        # 从一条32B连续行reshape成列向量，让物理布局为column-major；直接
        # full([8,1])的row-major每行仅4B，不满足A3 Vec的32B物理步长。
        c_pos = pl.reshape(pl.tile.full([1, ROWS], dtype=pl.FP32, value=0.0), [ROWS, 1])
        v_len_col = pl.reshape(pl.tile.full([1, ROWS], dtype=pl.FP32, value=0.0), [ROWS, 1])
        owners = pl.reshape(pl.tile.full([1, ROWS], dtype=pl.INT32, value=-1), [ROWS, 1])
        # 先读完同轮几何前缀，末尾才覆盖为最终索引。Top-K的读取已由上方
        # 依赖保护；保留原FP32数学转换，尾部未用行仍不写入物理pad。
        for row in pl.range(rows):
            token = first + row
            if EARLY_ATTENTION_GEOMETRY:
                request = pl.cast(pl.read(indices, [token, EARLY_QUERY_OWNER_COL]), pl.INDEX)
                if request >= 0:
                    position = pl.cast(pl.read(indices, [token, EARLY_QUERY_POSITION_COL]), pl.INDEX)
                    pl.tile.write(owners, [row, 0], pl.cast(request, pl.INT32))
                    pl.tile.write(c_pos, [row, 0], pl.cast(pl.cast(position, pl.INT32), pl.FP32))
                    pl.tile.write(v_len_col, [row, 0], pl.cast(pl.cast(pl.min(position + 1, WIN), pl.INT32), pl.FP32))
            else:
                request, begin, end, active = query_request(bounds, lengths, token, tokens)
                if active != 0:
                    position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                    length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
                    if position == length - (end - begin) + token - begin:
                        pl.tile.write(owners, [row, 0], pl.cast(request, pl.INT32))
                        # L为INT32且已证明position位于[0,L)，INT32中转不会损失
                        # 地址值；显式中转适配Scalar不支持INDEX直接转FP32的接口。
                        pl.tile.write(c_pos, [row, 0], pl.cast(pl.cast(position, pl.INT32), pl.FP32))
                        pl.tile.write(
                            v_len_col, [row, 0], pl.cast(pl.cast(pl.min(position + 1, WIN), pl.INT32), pl.FP32)
                        )
            if CACHE_QUERY_METADATA:
                # 复用本次已验证的owner，不能把未通过causal检查的request发布出去。
                # 正常position属于[0,L)，L为正INT32，所以这里无损存为INT32；
                # 不经c_pos的FP32中转，避免把数学用的舍入值当地址。无效行必写零
                # 位置和-1 owner，清除上轮descriptor；整个64B行只有本核写入。
                owner = pl.read(owners, [row, 0])
                saved_position = pl.cast(0, pl.INT32)
                if owner >= 0:
                    saved_position = pl.cast(pl.read(positions, [token]), pl.INT32)
                pl.write(valid_blocks, [token, QUERY_OWNER_COL], owner)
                pl.write(valid_blocks, [token, QUERY_POSITION_COL], saved_position)
        # canonical压缩上界仍先FP32再INT32 trunc；页检查读取同一个上界Tile，
        # 不另用整数除法悄悄改变浮点边界。正常模型范围内这些整数可由FP32精确表达。
        c_pos_q = pl.cast(
            pl.cast(pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        c_upper_b = pl.row_expand_mul(pl.tile.full([ROWS, TOPK], dtype=pl.FP32, value=1.0), c_pos_q)
        c_ge_tile = pl.minimum(pl.maximum(pl.add(c_raw_tile, CSA_CMP_GE_BIAS), 0.0), 1.0)
        c_lt_tile = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw_tile), 0.0), 1.0)
        c_mask_tile = pl.mul(c_ge_tile, c_lt_tile)
        # 页保护以消费者最终读取的INT32索引为准，而不是原始Top-K。在很大
        # 的整数边界上，FP32的(raw+1)-1可能舍入到相邻页；两者必须用同一地址。
        # 保存在本核Tile还避免逐候选再次Scalar读取Top-K GM。
        c_page_indices = pl.cast(pl.sub(pl.mul(c_mask_tile, pl.add(c_raw_tile, 1.0)), 1.0), target_type=pl.INT32)
        c_pages_valid = pl.tile.full([ROWS, TOPK], dtype=pl.FP32, value=0.0)
        v_columns = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
        v_col_tile = pl.col_expand_mul(pl.tile.full([ROWS, WIN], dtype=pl.FP32, value=1.0), v_columns)
        v_valid_tile = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(v_col_tile, v_len_col)), 0.0), 1.0)
        pl.store(pl.set_validshape(pl.mul(pl.sub(v_valid_tile, 1.0), -NEG_INF), rows, WIN), [first, 0], bias)
        # 前置策略读取同轮页检查；后置策略在这里调用同一检查函数，并保持
        # 同八行tile内的请求缓存。返回0都走原逐候选回退。前置路径的indices
        # 几何前缀直到末尾store才覆盖；后置路径从不读取未初始化前缀。
        checked_request = pl.cast(-1, pl.INDEX)
        checked_pages_valid = pl.cast(0, pl.INT32)
        for row in pl.range(rows):
            token = first + row
            request = pl.cast(pl.read(owners, [row, 0]), pl.INDEX)
            v_block_valid = pl.cast(0, pl.INT32)
            if request >= 0:
                upper = pl.cast(pl.cast(pl.read(c_pos_q, [row, 0]), pl.INT32), pl.INDEX)
                if physical_mode != 0:
                    # 只在plan当前query内驻留页表/512项地址，绝不把Tile带进Attention
                    # 五块流水。已有全页合法性扫描由选中页的逐项Vector检查替代，
                    # 不能在原fast-plan后再叠加512次Scalar查页。
                    p_raw = pl.slice(c_page_indices, [1, TOPK], [row, 0], valid_shape=[1, TOPK])
                    p_nonnegative = pl.tile.maximums(p_raw, 0)
                    p_columns = pl.tile.shrs(p_nonnegative, 5)
                    p_column_count = pl.tensor.dim(compressed_table, 1)
                    p_safe_columns = pl.tile.minimums(p_columns, pl.cast(p_column_count - 1, pl.INT32))
                    p_table = pl.load(
                        compressed_table,
                        [request, 0],
                        [1, PHYSICAL_PLAN_COLUMNS],
                        valid_shape=[1, p_column_count],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    p_scratch = pl.create_tile([1, TOPK], dtype=pl.INT32)
                    p_pages = pl.tile.gather(p_table, p_safe_columns, p_scratch)
                    # clip只防UB gather和位移溢出；合法mask仍用原始逻辑列和页值。
                    # page数<=2**24，因此下述FP32范围比较不会把边界坏页判为好页。
                    p_raw_f = pl.cast(p_raw, pl.FP32)
                    p_column_f = pl.cast(p_columns, pl.FP32)
                    p_page_f = pl.cast(p_pages, pl.FP32)
                    p_upper_f = pl.cast(pl.cast(upper, pl.INT32), pl.FP32)
                    p_column_count_f = pl.cast(pl.cast(p_column_count, pl.INT32), pl.FP32)
                    p_page_count_f = pl.cast(pl.cast(compressed_pages, pl.INT32), pl.FP32)
                    p_raw_ge = pl.minimum(pl.maximum(pl.add(p_raw_f, 1.0), 0.0), 1.0)
                    p_raw_lt = pl.minimum(pl.maximum(pl.add(pl.neg(p_raw_f), p_upper_f), 0.0), 1.0)
                    p_col_lt = pl.minimum(pl.maximum(pl.add(pl.neg(p_column_f), p_column_count_f), 0.0), 1.0)
                    p_page_ge = pl.minimum(pl.maximum(pl.add(p_page_f, 1.0), 0.0), 1.0)
                    p_page_lt = pl.minimum(pl.maximum(pl.add(pl.neg(p_page_f), p_page_count_f), 0.0), 1.0)
                    p_valid = pl.mul(pl.mul(pl.mul(p_raw_ge, p_raw_lt), p_col_lt), pl.mul(p_page_ge, p_page_lt))
                    c_pages_valid = pl.tile.assemble(c_pages_valid, p_valid, [row, 0])
                    p_safe_pages = pl.tile.minimums(
                        pl.tile.maximums(p_pages, 0), pl.cast(compressed_pages - 1, pl.INT32)
                    )
                    p_remainders = pl.tile.sub(p_nonnegative, pl.tile.shls(p_columns, 5))
                    p_physical = pl.tile.add(pl.tile.shls(p_safe_pages, 5), p_remainders)
                    p_valid_i32 = pl.cast(p_valid, pl.INT32)
                    p_output = pl.sub(pl.add(pl.mul(p_physical, p_valid_i32), p_valid_i32), 1)
                    # 逐行写既有GM indices，不另保留8×512物理地址Tile。非法候选
                    # 写-1，外部topk不变；bias仍在循环后由原逻辑c_out生成。
                    pl.store(p_output, [token, 0], indices)

                else:
                    if EARLY_ATTENTION_GEOMETRY:
                        checked_pages_valid = pl.read(indices, [token, EARLY_QUERY_PAGES_COL])
                    else:
                        if request != checked_request:
                            length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
                            checked_pages_valid = compressed_pages_all_valid(
                                compressed_table, request, length, compressed_pages
                            )
                            checked_request = request
                    if checked_pages_valid != 0:
                        # 最终INT32索引仍须满足原0<=index<upper检查；不能只验证原始Top-K，
                        # 因为原浮点(raw+1)-1可能改变极大数边界。这里只把逐元素判断向量化。
                        # row<rows已经由循环保证；显式声明该单行有效域，不能继承动态父Tile的rows。
                        final_indices = pl.cast(
                            pl.slice(c_page_indices, [1, TOPK], [row, 0], valid_shape=[1, TOPK]), pl.FP32
                        )
                        final_ge = pl.minimum(pl.maximum(pl.add(final_indices, 1.0), 0.0), 1.0)
                        upper_float = pl.cast(pl.cast(upper, pl.INT32), pl.FP32)
                        final_lt = pl.minimum(pl.maximum(pl.add(pl.neg(final_indices), upper_float), 0.0), 1.0)
                        c_pages_valid = pl.tile.assemble(c_pages_valid, pl.mul(final_ge, final_lt), [row, 0])
                    else:
                        # 请求存在页空洞或容量不足时，只屏蔽非法候选；合法候选继续保留。
                        for candidate in pl.range(TOPK):
                            raw = pl.cast(pl.read(c_page_indices, [row, candidate]), pl.INDEX)
                            if raw >= 0 and raw < upper:
                                column = raw // BLOCK_SIZE
                                if column < pl.tensor.dim(compressed_table, 1):
                                    page = pl.cast(pl.read(compressed_table, [request, column]), pl.INDEX)
                                    if page >= 0 and page < compressed_pages:
                                        pl.tile.write(c_pages_valid, [row, candidate], pl.cast(1.0, pl.FP32))
                position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                v_length = pl.min(position + 1, WIN)
                v_start = position - v_length + 1
                v_head = v_start % BLOCK_SIZE
                v_has_hole = pl.cast(0, pl.INT32)
                for v_run in pl.range(SWA_RUNS):
                    v_lo = pl.max(v_run * BLOCK_SIZE - v_head, 0)
                    v_hi = pl.min((v_run + 1) * BLOCK_SIZE - v_head, v_length)
                    if v_hi > v_lo:
                        column = (v_start + v_lo) // BLOCK_SIZE
                        page_valid = pl.cast(0, pl.INT32)
                        if column >= 0 and column < pl.tensor.dim(original_table, 1):
                            page = pl.cast(pl.read(original_table, [request, column]), pl.INDEX)
                            if page >= 0 and page < original_pages:
                                page_valid = pl.cast(1, pl.INT32)
                        if page_valid != 0:
                            v_block_valid = pl.cast(1, pl.INT32)
                        else:
                            v_has_hole = pl.cast(1, pl.INT32)
                # 仅空洞时重算本行，保留canonical常见全合法页的八行向量路径。
                if v_has_hole == 1:
                    v_columns_row = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
                    v_valid = pl.tile.full([1, WIN], dtype=pl.FP32, value=0.0)
                    for v_hole_run in pl.range(SWA_RUNS):
                        v_hole_lo = pl.max(v_hole_run * BLOCK_SIZE - v_head, 0)
                        v_hole_hi = pl.min((v_hole_run + 1) * BLOCK_SIZE - v_head, v_length)
                        if v_hole_hi > v_hole_lo:
                            column = (v_start + v_hole_lo) // BLOCK_SIZE
                            if column >= 0 and column < pl.tensor.dim(original_table, 1):
                                page = pl.cast(pl.read(original_table, [request, column]), pl.INDEX)
                                if page >= 0 and page < original_pages:
                                    v_lo_fp32 = pl.cast(pl.cast(v_hole_lo, pl.INT32), pl.FP32)
                                    v_hi_fp32 = pl.cast(pl.cast(v_hole_hi, pl.INT32), pl.FP32)
                                    v_ge = pl.minimum(
                                        pl.maximum(pl.add(pl.sub(v_columns_row, v_lo_fp32), 1.0), 0.0), 1.0
                                    )
                                    v_lt = pl.minimum(pl.maximum(pl.add(pl.neg(v_columns_row), v_hi_fp32), 0.0), 1.0)
                                    v_valid = pl.add(v_valid, pl.mul(v_ge, v_lt))
                    pl.store(pl.mul(pl.sub(v_valid, 1.0), -NEG_INF), [token, 0], bias)
            if physical_mode != 0:
                if request < 0:
                    # 每轮覆盖失活/dummy行，不能让图重放读到上次物理地址。
                    pl.store(pl.tile.full([1, TOPK], dtype=pl.INT32, value=-1), [token, 0], indices)
            pl.write(valid_blocks, [token, 0], v_block_valid)
        # 页合法性与数学有效性必须在写indices/bias/block flags之前合并。
        c_mask_tile = pl.mul(c_mask_tile, c_pages_valid)
        c_out_tile = pl.sub(pl.mul(c_mask_tile, pl.add(c_raw_tile, 1.0)), 1.0)
        if physical_mode == 0:
            pl.store(pl.set_validshape(pl.cast(c_out_tile, target_type=pl.INT32), rows, TOPK), [first, 0], indices)
        pl.store(pl.set_validshape(pl.mul(pl.minimum(c_out_tile, 0.0), -NEG_INF), rows, TOPK), [first, WIN], bias)
        # 每128候选形成一个块有效位；只写1..4，连同上面的SWA列0共五列。
        # 后续QK的事件谓词必须只读这五列，不能把几何列或其它poison解释成任务。
        # Tile归约显式借用UB暂存；仍是原来的128列row_max，不改变归约树。
        reduce_tmp = pl.create_tile([ROWS, ATTN_K_TILE], dtype=pl.FP32)
        for block in pl.range(1, SPARSE_BLOCKS):
            c_blk_valid = pl.row_max(
                pl.slice(
                    c_mask_tile, [ROWS, ATTN_K_TILE], [0, (block - 1) * ATTN_K_TILE], valid_shape=[rows, ATTN_K_TILE]
                ),
                reduce_tmp,
            )
            for row in pl.range(rows):
                c_valid = pl.cast(pl.read(c_blk_valid, [row, 0]), pl.INT32)
                pl.write(valid_blocks, [first + row, block], c_valid)
    return (indices, bias, valid_blocks)


@pl.jit.inline(auto_scope=False)
def attention_query_geometry(
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_CAP_DYN, MASK_COLS], pl.INT32],
    query: pl.Scalar[pl.INDEX],
    CACHE_QUERY_METADATA: pl.constexpr,
):
    """读取plan发布的本轮几何，避免两个AIV在五个块中重复O(B)查找。

    调用方已等待完整plan，且query在[0,T)中。缓存分支保留owner/位置基本
    范围检查；完整Q/L及causal关系由同次plan生产者验证。页表列/物理页保护
    仍由实际gather逐地址执行，不因缓存几何跳过。关闭分支保留旧算法供比较。
    """
    request = pl.cast(-1, pl.INDEX)
    position = pl.cast(0, pl.INDEX)
    active = pl.cast(0, pl.INT32)
    if CACHE_QUERY_METADATA:
        saved_owner = pl.cast(pl.read(plan, [query, QUERY_OWNER_COL]), pl.INDEX)
        saved_position = pl.cast(pl.read(plan, [query, QUERY_POSITION_COL]), pl.INDEX)
        if saved_owner >= 0 and saved_owner < pl.tensor.dim(lengths, 0) and saved_position >= 0:
            request = saved_owner
            position = saved_position
            active = pl.cast(1, pl.INT32)
    else:
        owner, begin, end, owner_active = query_request(bounds, lengths, query, pl.tensor.dim(positions, 0))
        if owner_active != 0:
            actual_position = pl.cast(pl.read(positions, [query]), pl.INDEX)
            length = pl.cast(pl.read(lengths, [owner]), pl.INDEX)
            if actual_position == length - (end - begin) + query - begin:
                request = owner
                position = actual_position
                active = pl.cast(1, pl.INT32)
    return request, position, active


@pl.jit.inline(auto_scope=False)
def attention_inverse_rope_sign_stage(
    sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    signed_sin: pl.InOut[pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """逐token逆RoPE预处理；与请求Q分组无关，输出容量尾部保持未写。"""
    tokens = pl.tensor.dim(sin, 0)
    # 每个worker重建小符号Tile，不能共享另一个核的UB地址。计算顺序与
    # canonical相同，只将原先等于S的tile改为8，并保留实际尾rows。
    for first in pl.range(worker_id * ROWS, tokens, WORKERS * ROWS):
        rows = pl.min(ROWS, tokens - first)
        cs_ones = pl.tile.full([ROWS, ROPE_DIM], dtype=pl.FP32, value=1.0)
        cs_idx_f = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_idx_f)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        cs_sin = pl.load(sin, [first, 0], [ROWS, ROPE_DIM], valid_shape=[rows, ROPE_DIM])
        cs_sign_rows = pl.set_validshape(cs_sign, rows, ROPE_DIM)
        pl.store(pl.mul(cs_sin, cs_sign_rows), [first, 0], signed_sin)
    return signed_sin
