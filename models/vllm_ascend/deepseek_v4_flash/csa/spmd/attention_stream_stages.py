# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""动态 CSA Attention 的成对 AIC / 双 AIV 三槽流水与最终头向量发布。

整体数据流与阅读顺序
    PLAN_PHYSICAL_ROWS开启时，压缩块读取本轮plan已经验证的物理行，删除重复的Q/L扫描、
    除32及Scalar查页。表示由plan与本函数共同的device容量谓词决定，不能
    只改消费者或拿原逻辑索引直接当物理行。几何/页合法性在同代plan执行，
    消费者由MGATHER(row, zero)执行最终行范围保护；一次载入64项物理索引，
    替代逐行Scalar读GM，ISA内部仍逐行DMA。SWA/能力外回退及所有事件保持。
    attention_qk_pv_cube_stage 执行QK与滞后两拍的PV；
    attention_qk_pv_vector_lane_stage 执行采KV、滞后一拍的局部softmax、
    滞后三拍的在线归并与最终归一化/逆RoPE/BF16和8组packed发布。
    LOCAL_ATTENTION_STATS由policy按已有历史/投影区间选择。开启时将每块
    softmax的mi/li保存在本AIV独占的三槽UB环中，隔两拍
    在线归并直接复用，删除只由本lane消费的mi/li GM写出与回读。
    两个GM形参保留入口ABI兼容；开启时不读写，关闭时执行原GM路径。
    无用的UB环和别名在关闭分支编译时消除，不能只凭源码if就声称没有额外成本。
    外部输出与cache/state不变，两条路径共用全部softmax和在线归并数学。
    不新增跨核信号，不改变 READY/ACK、三槽复用、失效块跳过及尾部排空。
    最终发布直接消费本核live Tile；当前inline接口不支持Tile参数，不为拆函数
    额外写GM。数学来自canonical decode_sparse_attn_csa，
    首版保留其5块/3槽/提前KV分支，不用全阶段barrier替代循环流水。

真实动态输入、布局与工作区
    q_flat为BF16[Q_ROWS,512]，Q_ROWS=64*T；两份cache直接使用Native连续
    BF16[pages*32,512]别名。positions INT64[T]、原始Q INT32[B+1]、L INT32[B]
    和两份INT32[B,columns]页表保持device输入，不用token//S或Host回读。
    plan必须来自同一批Q/L/positions/页表/页容量：indices INT32[PLAN_CAP,512]，
    bias FP32[PLAN_CAP,640]，mask INT32[PLAN_CAP,16]的前5列控制事件，PLAN_CAP>=T。
    CACHE_QUERY_METADATA开启时，第5/6列复用同次plan已验证的request/position，
    每个候选块直接读这两项，不让两AIV反复扫描Q/L。关闭时保持原扫描算法。
    sink FP32[64,1]，cos FP32[T,64]，逆RoPE signed_sin FP32[ROPE_CAP,64]。
    cos/sin/sink需覆盖所有实际T行且有限，包含约定写出零attention的dummy行。
    packed为BF16[PACK_ROWS,4096]，PACK_ROWS须整除8；P_PACK=PACK_ROWS//8>=T，
    第g组第t行地址为g*P_PACK+t。只写实际T，不触碰每组的pad。
    所有业务容量轴均为pl.dynamic；64头、512列、128候选块、5块和3槽是模型/流水
    常量，不是冻结batch/sequence。FFTS独立硬件控制区为INT64[256]；当前set_ffts接口要求静态长度，
    256不随T/B变化，也不影响其它业务动态轴。
    ring独立GM：KV BF16[KV_RING_ROWS,512]至少G*3*128行；score FP32[HEAD_RING_ROWS,128]、
    probability BF16[HEAD_RING_ROWS,128]、PV FP32[HEAD_RING_ROWS,512]、mi/li
    FP32[HEAD_RING_ROWS,1]共享相同的行容量且至少G*3*64行。相同dynamic符号
    必须绑定相同维度值；描述符连续，所有scratch互不非法alias。

核分工、事件与槽位生命周期
    调用者在同一个MIX内并发运行G个AIC和2G个AIV；core_id为组号，lane_id为0/1。
    query t归t%G；每组work跨全部query连续计数，slot=core_id*3+work%3。
    两AIV各管32heads/64个KV候选，不能先等待一个完整helper结束再启动另一个。
    EVENT_BASE..EVENT_BASE+3分别为KV_READY、SCORE_READY、PROB_READY、PV_READY；
    基址须让4个ID均落在0..13，14/15为编译器保留。生产者pipe沿用MTE3/FIX，
    消费者沿用MTE2，ffts_mode=2。调用者保证此期间其它阶段不复用这4个事件。
    两角色源级声明同一FFTS控制区；当前PyPTO会在最终wrapper删除直接set_ffts，
    实际硬件基址由Simpler运行时绑定，不把传入Tensor声称为最终硬件事件存储。
    同组两AIV的半块写入都就绪后AIC才可读，
    AIC通知需被两个lane分别消费。这里只保留canonical协议，不另造GM flag。
    AIC第tick拍做QK(tick)、PV(tick-2)；AIV做gather(tick)、softmax(tick-1)、
    merge(tick-3)。槽序跨query延续，只有最后一次排空。有效块由同一plan判断，
    失效块在两侧同时跳过事件与ring访存，但query末块仍发布/清零运行状态。
    T>=144时必须先consume前一块SCORE，才发布本块KV；小档先做前一块softmax，
    再发布KV。不要交换这些语句：它们也限制生产者前进速度，保护三槽复用。
    阶段级phase没有逐槽ACK，不能拿它替代这些通知；此处没有创建task或新SPMD。

算术边界、安全与外层责任
    KV进一次L1，QK用转置view，PV复用原L1行；PV N128/K128保留交替累加器。
    AIV保持FP32块内softmax和在线m/l/左右oi顺序，probability及最终输出BF16均
    rint。最终归一化仍加入sink，RoPE读取未cast的FP32值，再执行BF16边界。
    gather再次检查Q/L活动、causal position、页表列和物理页上下界；inactive
    不访问cache。plan与metadata必须保持一致，不能一侧私自改mask而破坏事件对数。
    调用前等待Q/plan/原压缩cache及cos/sin可见；各组完成尾排空后，外层再发布
    packed-ready供O_A读取。FFTS生命周期/事件共存、真实A3同步、精度与性能
    仍须后续整合设备验证；CPU编译和事件模型都不能替代该证据。
"""

import pypto.language as pl

from ..common.config import FLASH as M
from .attention_plan_stages import (
    COMPRESSED_COLUMNS_DYN,
    ORIGINAL_COLUMNS_DYN,
    PLAN_CAP_DYN,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    ROPE_CAP_DYN,
    T_DYN,
    attention_query_geometry,
    physical_plan_supported,
)

Q_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_Q_ROWS")
ORIGINAL_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_ORIGINAL_ROWS")
COMPRESSED_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_COMPRESSED_ROWS")
KV_RING_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_KV_RING_ROWS")
HEAD_RING_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_HEAD_RING_ROWS")
PACK_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_PACK_ROWS")
FFTS_WORDS = 256  # 硬件控制区常量，当前set_ffts API要求静态Tensor。
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = 448
WIN = 128
CMP_TOPK = 512
BLOCK_SIZE = 32
ATTN_K_TILE = 128
SPARSE_BLOCKS = 5
QK_PRE_LAUNCH = 2
QK_TRANSFER_SLOTS = 3
PV_N_TILE = 128
SWA_RUNS = 5
FINAL_HEAD_TILE = 16
O_GROUPS = 8
HEADS_PER_GROUP = 8
O_GROUP_IN = 4096
PUBLISH_GROUPS = 2
EARLY_KV_MIN_TOKENS = 144
SOFTMAX_SCALE = M.softmax_scale
QK_KV_READY_EVENT = 0
QK_SCORE_READY_EVENT = 1
QK_PROB_READY_EVENT = 2
QK_PV_READY_EVENT = 3


@pl.jit.inline(auto_scope=False)
def attention_qk_pv_cube_stage(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    score_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
):
    """AIC侧：QK领先PV两拍；必须与本组两个AIV helper并发。"""
    t_dim = pl.tensor.dim(q_flat, 0) // H
    qk_core = core_id
    # 固定硬件控制区声明，不搬运业务Tensor；最终Simpler包装使用运行时FFTS基址。
    pl.system.set_ffts(ffts_workspace)
    qk_query_count = (t_dim + CORE_GROUPS - 1 - qk_core) // CORE_GROUPS
    qk_block_count = qk_query_count * SPARSE_BLOCKS
    qk_q = pl.create_tile([H, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
    # KV 只进一次 L1：QK 用它的转置视图，PV 用同一份的行切片。
    # 原先 QK 与 PV 各自从 GM 搬一遍，同一块 KV 过两次 GM->L1。
    qk_l1 = pl.create_tile([QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
    # 多走两拍排空最后PV；空组仅走控制分支，不读plan或环槽。
    for qk_tick in pl.range(qk_block_count + QK_PRE_LAUNCH):
        if qk_tick < qk_block_count:
            qk_sb = qk_tick % SPARSE_BLOCKS
            qk_t = qk_core + (qk_tick // SPARSE_BLOCKS) * CORE_GROUPS
            if qk_sb == 0:
                qk_q = pl.gather_row(qk_q, q_flat, [0, 0], [qk_t * H, 0], [H, HEAD_DIM])
            if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS
                qk_kv_row = qk_slot * ATTN_K_TILE
                qk_transfer_row = qk_slot * H
                pl.system.sync_wait(EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                qk_l1_row = (qk_tick % QK_TRANSFER_SLOTS) * ATTN_K_TILE
                qk_l1 = pl.gather_row(qk_l1, kv_transfer, [qk_l1_row, 0], [qk_kv_row, 0], [ATTN_K_TILE, HEAD_DIM])
                qk_l1_t = pl.tile.transpose_view(qk_l1)
                qk_kv_t = pl.tile.slice(qk_l1_t, [HEAD_DIM, ATTN_K_TILE], [0, qk_l1_row])
                qk_scores = pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)
                pl.store(qk_scores, [qk_transfer_row, 0], score_transfer)
                pl.system.sync_set(
                    EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                )
        if qk_tick >= QK_PRE_LAUNCH:
            pv_work = qk_tick - QK_PRE_LAUNCH
            pv_sb = pv_work % SPARSE_BLOCKS
            pv_t = qk_core + (pv_work // SPARSE_BLOCKS) * CORE_GROUPS
            if pl.read(valid_block_mask, [pv_t, pv_sb]) > 0:
                pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_work % QK_TRANSFER_SLOTS
                pv_transfer_row = pv_slot * H
                pl.system.sync_wait(
                    EVENT_BASE + QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC
                )
                pv_probability = pl.load(
                    probability_transfer,
                    [pv_transfer_row, 0],
                    [H, ATTN_K_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                pv_l1_row = (pv_work % QK_TRANSFER_SLOTS) * ATTN_K_TILE
                # PV沿用N128/K128，保留每128候选一次softmax和query流水顺序。
                pv_probability_left = pl.tile.move(pv_probability, target_memory=pl.MemorySpace.Left)
                pv_first_right = pl.tile.extract(
                    qk_l1, pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right
                )
                pv_previous = pl.tile.matmul(pv_probability_left, pv_first_right)
                for pv_n in pl.unroll(1, HEAD_DIM // PV_N_TILE):
                    pv_next_right = pl.tile.extract(
                        qk_l1,
                        pv_l1_row,
                        pv_n * PV_N_TILE,
                        [ATTN_K_TILE, PV_N_TILE],
                        target_memory=pl.MemorySpace.Right,
                    )
                    # 同时保留前后两个累加器，让前块FIX写回与后块Cube重叠。
                    pv_current = pl.tile.matmul(pv_probability_left, pv_next_right)
                    pl.store(pv_previous, [pv_transfer_row, (pv_n - 1) * PV_N_TILE], pv_transfer)
                    pv_previous = pv_current
                pl.store(pv_previous, [pv_transfer_row, HEAD_DIM - PV_N_TILE], pv_transfer)
                pl.system.sync_set(
                    EVENT_BASE + QK_PV_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
                )

    return score_transfer, pv_transfer


@pl.jit.inline(auto_scope=False)
def attention_qk_pv_vector_lane_stage(
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[PLAN_CAP_DYN, CMP_TOPK], pl.INT32],
    sparse_bias: pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    attn_sink_col: pl.Tensor[[H, 1], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32],
    kv_transfer: pl.InOut[pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16]],
    score_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32],
    probability_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16]],
    pv_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32],
    mi_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    li_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    lane_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
    CACHE_QUERY_METADATA: pl.constexpr = False,
    LOCAL_ATTENTION_STATS: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
):
    """双AIV各32头/64候选；同一mask和事件次序保证与AIC配对。"""
    t_dim = pl.tensor.dim(position_ids, 0)
    physical_mode = physical_plan_supported(
        pl.tensor.dim(cmp_block_table, 1), pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE,
        PLAN_PHYSICAL_ROWS,
    )
    qk_core = core_id
    qk_aiv = lane_id
    qk_query_count = (t_dim + CORE_GROUPS - 1 - qk_core) // CORE_GROUPS
    qk_block_count = qk_query_count * SPARSE_BLOCKS
    # 固定硬件控制区声明，不搬运业务Tensor；最终Simpler包装使用运行时FFTS基址。
    pl.system.set_ffts(ffts_workspace)
    qk_lane_head = qk_aiv * (H // 2)
    qk_lane_kv = qk_aiv * (ATTN_K_TILE // 2)
    m_idx = pl.tile.ci(0, [1, ROPE_DIM], dtype=pl.INT32)
    m_rem_tmp = pl.create_tile([1, ROPE_DIM], dtype=pl.INT32)
    m_lane = pl.tile.rems(m_idx, 2, m_rem_tmp)
    m_swap_row = pl.tile.adds(pl.tile.sub(m_idx, pl.tile.muls(m_lane, 2)), NOPE_DIM + 1)
    m_swap_base = pl.create_tile([FINAL_HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
    m_swap_source = pl.col_expand(m_swap_base, m_swap_row)
    m_row_offsets = pl.tile.muls(pl.tile.ci(0, [1, FINAL_HEAD_TILE], dtype=pl.INT32), HEAD_DIM)
    m_swap_idx = pl.reshape(
        pl.row_expand_add(m_swap_source, pl.reshape(m_row_offsets, [FINAL_HEAD_TILE, 1])),
        [1, FINAL_HEAD_TILE * ROPE_DIM],
    )
    running_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1])
    # 上游口径：l 从 0 起算，sink 的那一项留到最终发布的分母里补。
    running_l = pl.tile.muls(running_m, 0.0)
    running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
    running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
    # mi/li 只由当前 lane 产生并由同一 lane 消费，无跨核可见性要求。
    # 每块 softmax 在 tick=work+1 写入，merge 在 tick=work+3 消费；
    # 同槽下一次写入为 tick=work+4，严格晚于旧值消费。跨 query 的 work
    # 连续编号和最终三拍 drain 均保留，不能在 query 边界清掉统计环。
    # 一个 [6,32] 连续行缓存交错保存每槽 mi/li，避免两个独立 loop-carried
    # 对象分别参与全循环干涉图。第2s行是mi，第2s+1行是li，逻辑共768 B。
    # 不用 [96,1] 列向量布局；无效块不写也不读，首次使用不需要清零。
    local_stats_ring = pl.create_tile([QK_TRANSFER_SLOTS * 2, H // 2], dtype=pl.FP32)
    # 多走三拍排空末块归并；循环携带的m/l/oi仅属于当前query，
    # 原始t_dim决定提前KV策略，不能用pad容量代替。
    for qk_tick, (m_iter, l_iter, left_iter, right_iter, stats_cache_iter) in pl.range(
        qk_block_count + QK_PRE_LAUNCH + 1,
        init_values=(running_m, running_l, running_left, running_right, local_stats_ring),
    ):
        qk_t = qk_core + (qk_tick // SPARSE_BLOCKS) * CORE_GROUPS
        qk_sb = qk_tick % SPARSE_BLOCKS
        if qk_tick < qk_block_count:
            if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS
                qk_kv_row = qk_slot * ATTN_K_TILE
                # mask是两角色共同事件谓词；额外地址guard只限制gather，不改变
                # 事件对数。合法调用必须使用与plan同一批Q/L/positions/页表。
                # 显式constexpr使关闭路径在Tile状态构建前裁掉此分支；仅靠
                # physical_mode最后化简为0仍可能留下额外loop-carried Tile。
                if PLAN_PHYSICAL_ROWS and physical_mode != 0 and qk_sb > 0:
                    # 本轮plan已验证唯一owner、causal位置和每个被选中页。外层已
                    # 获取其phase和业务缓存；mask来自同一plan，不是外部任意标记。
                    # 因此这里无需再次扫描Q/L或逐行查页，仍检查最终物理行容量。
                    # 无效候选为-1，保持本块初始化的零KV；顺序及64行分工不变。
                    # 索引来自已发布的本轮物理plan；一次DMA读64项到UB，避免
                    # 每行Scalar读GM。MGATHER仍逐行DMA，不能称为硬件单指令
                    # 并行采64行；zero模式保留负哨兵/容量外地址的零填充。
                    physical_column = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv
                    physical_rows = pl.load(
                        cmp_sparse_indices, [qk_t, physical_column], [1, ATTN_K_TILE // 2],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    qk_kv_half = pl.tile.mgather(
                        cmp_kv_flat, physical_rows, coalesce="row", gather_oob="zero"
                    )
                else:
                    # 只有原SWA/逻辑回退需要先清零；物理分支由MGATHER完整写出，
                    # 不让旧零Tile与新64KiB结果跨分支同时存活。
                    qk_kv_half = pl.tile.full([ATTN_K_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    qk_b, qk_pos, q_active = attention_query_geometry(
                        position_ids, bounds, lengths, valid_block_mask, qk_t, CACHE_QUERY_METADATA
                    )
                    if q_active != 0:
                        if qk_sb == 0:
                            qk_win_len = pl.min(qk_pos + 1, WIN)
                            qk_win_start = qk_pos - qk_win_len + 1
                            qk_head = qk_win_start % BLOCK_SIZE
                            qk_rows = pl.min(pl.max(qk_win_len - qk_lane_kv, 0), ATTN_K_TILE // 2)
                            # 最多五个页run，每个lane只采自己的64候选；尾run只读
                            # qk_hi-qk_lo行，先检查页表列再检查物理page容量。
                            for qk_run in pl.unroll(SWA_RUNS):
                                qk_lo = pl.max(qk_run * BLOCK_SIZE - qk_head - qk_lane_kv, 0)
                                qk_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head - qk_lane_kv, qk_rows)
                                if qk_hi > qk_lo:
                                    qk_absolute = qk_win_start + qk_lane_kv + qk_lo
                                    qk_column = qk_absolute // BLOCK_SIZE
                                    if qk_column >= 0 and qk_column < pl.tensor.dim(ori_block_table, 1):
                                        qk_raw_page = pl.cast(pl.read(ori_block_table, [qk_b, qk_column]), pl.INDEX)
                                        if (
                                            qk_raw_page >= 0
                                            and qk_raw_page < pl.tensor.dim(ori_kv_flat, 0) // BLOCK_SIZE
                                        ):
                                            qk_raw_row = qk_raw_page * BLOCK_SIZE + qk_absolute % BLOCK_SIZE
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half,
                                                ori_kv_flat,
                                                [qk_lo, 0],
                                                [qk_raw_row, 0],
                                                [ATTN_K_TILE // 2, HEAD_DIM],
                                                valid_shape=[qk_hi - qk_lo, HEAD_DIM],
                                            )
                        else:
                            # indices已经包含plan的causal/页mask，保留顺序并再做
                            # 地址上界保护；非法行留在本轮初始化的零KV中。
                            for qk_row in pl.range(ATTN_K_TILE // 2):
                                qk_cmp_k = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv + qk_row
                                if qk_cmp_k < CMP_TOPK:
                                    qk_ridx = pl.cast(pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k]), pl.INDEX)
                                    if qk_ridx >= 0:
                                        qk_column = qk_ridx // BLOCK_SIZE
                                        if qk_column < pl.tensor.dim(cmp_block_table, 1):
                                            qk_page = pl.cast(pl.read(cmp_block_table, [qk_b, qk_column]), pl.INDEX)
                                            if qk_page >= 0 and qk_page < pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE:
                                                qk_src = qk_page * BLOCK_SIZE + qk_ridx % BLOCK_SIZE
                                                qk_kv_half = pl.gather_row(
                                                    qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, HEAD_DIM]
                                                )
                pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)
        # 大档先消费上一块SCORE通知，再发布已经写好的本块KV；
        # 先让Cube推进，再做上一块softmax，同时限制SCORE覆盖速度。
        if t_dim >= EARLY_KV_MIN_TOKENS:
            if qk_tick > 0 and qk_tick <= qk_block_count:
                if (
                    pl.read(
                        valid_block_mask,
                        [
                            qk_core + ((qk_tick - 1) // SPARSE_BLOCKS) * CORE_GROUPS,
                            (qk_tick - 1) % SPARSE_BLOCKS,
                        ],
                    )
                    > 0
                ):
                    pl.system.sync_wait(
                        EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                    )
            if qk_tick < qk_block_count:
                if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                    pl.system.sync_set(
                        EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                    )
        if qk_tick > 0 and qk_tick <= qk_block_count:
            softmax_work = qk_tick - 1
            softmax_sb = softmax_work % SPARSE_BLOCKS
            softmax_t = qk_core + (softmax_work // SPARSE_BLOCKS) * CORE_GROUPS
            if pl.read(valid_block_mask, [softmax_t, softmax_sb]) > 0:
                qk_slot = qk_core * QK_TRANSFER_SLOTS + softmax_work % QK_TRANSFER_SLOTS
                qk_transfer_row = qk_slot * H
                qk_s0 = softmax_sb * ATTN_K_TILE
                if t_dim < EARLY_KV_MIN_TOKENS:
                    pl.system.sync_wait(
                        EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                    )
                # 整条 H//2 一次做完：原先按 SOFTMAX_HEAD_TILE=8 切成四段，
                # 是 ATTN_K_TILE=512 时 UB 放不下留下的，现在 128 宽已无必要。
                # 归约临时区不跨块保存状态，限制在此阶段，让末块发布复用UB。
                qk_reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                qk_scores_half = pl.load(score_transfer, [qk_transfer_row + qk_lane_head, 0], [H // 2, ATTN_K_TILE])
                qk_bias = pl.load(sparse_bias, [softmax_t, qk_s0], [1, ATTN_K_TILE])
                qk_masked = pl.col_expand_add(pl.mul(qk_scores_half, SOFTMAX_SCALE), qk_bias)
                # 块内取局部最大值，跨块的重标定挪到下面的归并里；
                # 这样每块 softmax 互不依赖，流水才排得开。
                qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
                qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
                qk_li = pl.row_sum(qk_exp, qk_reduce_tmp)
                # 性能版用 rint（就近偶数），与上游一致。已封存的精度版用 round
                # 是为了复刻 Native SAS 的 CAST_ROUND——半数远离零。
                qk_probability = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                pl.store(qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer)
                if LOCAL_ATTENTION_STATS:
                    # mi/li没有跨核消费者，完整CSA也不读取其GM副本；本轮直接
                    # 写入同lane UB环，省去两个MTE3写回及随后两个MTE2回读。
                    # 这里只做 FP32 reshape/搬运，不改变 row_max/row_sum 的归约树。
                    stats_cache_mi = pl.tile.assemble(
                        stats_cache_iter, pl.reshape(qk_mi, [1, H // 2]), [(softmax_work % QK_TRANSFER_SLOTS) * 2, 0]
                    )
                    stats_cache_updated = pl.tile.assemble(
                        stats_cache_mi, pl.reshape(qk_li, [1, H // 2]), [(softmax_work % QK_TRANSFER_SLOTS) * 2 + 1, 0]
                    )
                    stats_cache_updated_path = pl.yield_(stats_cache_updated)
                else:
                    # 关闭路径保留原GM写出/回读；未使用的UB环由constexpr消除。
                    pl.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
                    pl.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
                    stats_cache_updated_path = pl.yield_(stats_cache_iter)
                pl.system.sync_set(
                    EVENT_BASE + QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                )
                stats_cache_valid = pl.yield_(stats_cache_updated_path)
            else:
                # 与现有事件谓词一致：失效块不产生可消费的统计值。
                stats_cache_valid = pl.yield_(stats_cache_iter)
            stats_cache_ready = pl.yield_(stats_cache_valid)
        else:
            stats_cache_ready = pl.yield_(stats_cache_iter)
        if t_dim < EARLY_KV_MIN_TOKENS:
            if qk_tick < qk_block_count:
                if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                    pl.system.sync_set(
                        EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                    )
        if qk_tick >= QK_PRE_LAUNCH + 1:
            pv_work = qk_tick - QK_PRE_LAUNCH - 1
            pv_sb = pv_work % SPARSE_BLOCKS
            pv_t = qk_core + (pv_work // SPARSE_BLOCKS) * CORE_GROUPS
            if pl.read(valid_block_mask, [pv_t, pv_sb]) > 0:
                pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_work % QK_TRANSFER_SLOTS
                pv_transfer_row = pv_slot * H
                pl.system.sync_wait(EVENT_BASE + QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                # 同一 work 的 mi/li 已在两拍前产生；使用本 tick 更新后的环，
                # 当前 softmax 与 merge 相差两槽，互不覆盖。仍先等 PV_READY，
                # 此优化不代替 Cube 写回完成通知，也不缩短原流水排空。
                if LOCAL_ATTENTION_STATS:
                    # 动态slice后直接reshape存在已知运行时偏移保留限制；显式
                    # extract两个128B行到本地临时Tile后reshape，保留真实copy与
                    # 编译器依赖。不放开内存规划的安全门槛，也不经GM绕行。
                    pv_m = pl.reshape(
                        pl.tile.extract(
                            stats_cache_ready, (pv_work % QK_TRANSFER_SLOTS) * 2, 0,
                            [1, H // 2], target_memory=pl.MemorySpace.Vec,
                        ),
                        [H // 2, 1],
                    )
                    pv_l = pl.reshape(
                        pl.tile.extract(
                            stats_cache_ready, (pv_work % QK_TRANSFER_SLOTS) * 2 + 1, 0,
                            [1, H // 2], target_memory=pl.MemorySpace.Vec,
                        ),
                        [H // 2, 1],
                    )
                else:
                    pv_m = pl.load(mi_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1])
                    pv_l = pl.load(li_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1])
                next_m = pl.maximum(m_iter, pv_m)
                alpha = pl.exp(pl.sub(m_iter, next_m))
                beta = pl.exp(pl.sub(pv_m, next_m))
                next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
                pv_left = pl.load(pv_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, HEAD_DIM // 2])
                next_left = pl.add(pl.row_expand_mul(left_iter, alpha), pl.row_expand_mul(pv_left, beta))
                pv_right = pl.load(
                    pv_transfer, [pv_transfer_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2]
                )
                next_right = pl.add(pl.row_expand_mul(right_iter, alpha), pl.row_expand_mul(pv_right, beta))
                m_valid, l_valid, left_valid, right_valid = pl.yield_(next_m, next_l, next_left, next_right)
            else:
                m_valid, l_valid, left_valid, right_valid = pl.yield_(m_iter, l_iter, left_iter, right_iter)
            if pv_sb == SPARSE_BLOCKS - 1:
                # 末块直接消费本核live累加器；维持PTO数学，不把整T的FP32
                # mi/li/oi写GM再交另一个task归并。失效末块也必须发布/重置。
                packed_stride = pl.tensor.dim(o_packed_heads, 0) // O_GROUPS
                m_gather_tmp = pl.create_tile([1, FINAL_HEAD_TILE * ROPE_DIM], dtype=pl.INT32)
                # 每lane拆两段16头发布，控制UB峰值；每段对应两组4096列。
                for pub_h in pl.unroll((H // 2) // FINAL_HEAD_TILE):
                    pub_local_head = pub_h * FINAL_HEAD_TILE
                    m_h0 = qk_lane_head + pub_local_head
                    # 先将整列reshape成行，再显式ND extract，保留后16头偏移。
                    # 直接slice再列转行曾丢偏移；A3不支持ColMajor TEXTRACT。
                    m_mi = pl.reshape(
                        pl.tile.extract(
                            pl.reshape(m_valid, [1, H // 2]),
                            0,
                            pub_local_head,
                            [1, FINAL_HEAD_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        ),
                        [FINAL_HEAD_TILE, 1],
                    )
                    m_li = pl.reshape(
                        pl.tile.extract(
                            pl.reshape(l_valid, [1, H // 2]),
                            0,
                            pub_local_head,
                            [1, FINAL_HEAD_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        ),
                        [FINAL_HEAD_TILE, 1],
                    )
                    m_left = pl.slice(left_valid, [FINAL_HEAD_TILE, HEAD_DIM // 2], [pub_local_head, 0])
                    m_right = pl.slice(right_valid, [FINAL_HEAD_TILE, HEAD_DIM // 2], [pub_local_head, 0])
                    m_oi = pl.concat(m_left, m_right)
                    n_sink_bias = pl.load(attn_sink_col, [m_h0, 0], [FINAL_HEAD_TILE, 1])
                    n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
                    n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
                    n_full = pl.row_expand_div(m_oi, n_denom)
                    n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
                    m_rope = n_full[0:FINAL_HEAD_TILE, NOPE_DIM:HEAD_DIM]
                    m_cos_il = pl.load(freqs_cos, [pv_t, 0], [1, ROPE_DIM])
                    m_sin_signed = pl.load(rope_sin_signed, [pv_t, 0], [1, ROPE_DIM])
                    # 索引已含各头的绝对行偏移，一次连续Gather避免逐行向量barrier。
                    m_swapped_flat = pl.tile.gather(
                        pl.reshape(n_full, [1, FINAL_HEAD_TILE * HEAD_DIM]),
                        m_swap_idx,
                        m_gather_tmp,
                    )
                    m_swapped = pl.reshape(m_swapped_flat, [FINAL_HEAD_TILE, ROPE_DIM])
                    m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
                    n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
                    n_full_bf16 = pl.concat(n_bf16[0:FINAL_HEAD_TILE, 0:NOPE_DIM], n_rope_bf16)
                    n_group_bf16 = pl.reshape(n_full_bf16, [PUBLISH_GROUPS, O_GROUP_IN])
                    n_pack_first = n_group_bf16[0:1, 0:O_GROUP_IN]
                    n_pack_second = n_group_bf16[1:2, 0:O_GROUP_IN]
                    n_pack_row = (m_h0 // HEADS_PER_GROUP) * packed_stride + pv_t
                    n_pack_row_second = n_pack_row + packed_stride
                    pl.store(n_pack_first, [n_pack_row, 0], o_packed_heads)
                    pl.store(n_pack_second, [n_pack_row_second, 0], o_packed_heads)
                # 只重置刚完成query的在线数值状态；后续query的KV/QK/softmax
                # 仍可在飞行，不重置三槽或事件。
                reset_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1])
                reset_l = pl.tile.muls(reset_m, 0.0)
                reset_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                reset_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                m_published, l_published, left_published, right_published = pl.yield_(
                    reset_m, reset_l, reset_left, reset_right
                )
            else:
                m_published, l_published, left_published, right_published = pl.yield_(
                    m_valid, l_valid, left_valid, right_valid
                )
            m_after, l_after, left_after, right_after = pl.yield_(
                m_published, l_published, left_published, right_published
            )
        else:
            m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)
        running_m, running_l, running_left, running_right, local_stats_ring = pl.yield_(
            m_after, l_after, left_after, right_after, stats_cache_ready
        )
    return kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads
